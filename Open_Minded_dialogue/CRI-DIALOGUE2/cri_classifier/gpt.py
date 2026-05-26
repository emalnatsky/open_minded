"""
GPTIntentClassifier — GPT-4o-mini intent classifier with stub fallback.

Uses a system prompt + few-shot examples baked in here (no contract.json
dependency — that's Sherissa's separate module; this is the merged version
that keeps all logic in one readable place).

Falls back to StubIntentClassifier when:
  - The GPT call raises any exception
  - The returned confidence is below CONFIDENCE_THRESHOLD
  - The JSON response can't be parsed

When the first classify() attempt is low-confidence, classify() returns
REPEAT_SENTINEL (the dialogue asks the child to repeat).
classify_retry() then tries again and falls back to stub rather than
returning REPEAT_SENTINEL a second time.
"""

import os
import json
import logging
import re
from typing import Optional

from .intent_result import IntentResult, REPEAT_SENTINEL, VALID_INTENTS, EMBEDDED_VALID_FIELDS
from .stub import StubIntentClassifier

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD  = 0.7
GPT_INTENT_MODEL      = "gpt-4o-mini"
GPT_INTENT_MAX_TOKENS = 120
GPT_INTENT_TEMPERATURE = 0.0

# ── System prompt ─────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a child-robot interaction system. "
    "A NAO robot called Leo is talking to a Dutch child aged 8-11. "
    "The child speaks Dutch. Classify what the child said into exactly one intent, "
    "identify the UM field if relevant, extract the value if relevant, and return "
    "a confidence score 0.0-1.0. Naming: um_* intents touch the database, "
    "Only use um_add, um_update, or um_delete when the child explicitly gives, "
    "corrects, changes, or asks Leo to forget a memory value. Prefer um_update "
    "when the child corrects or replaces a value Leo just stated; use um_add for "
    "newly volunteered values. For ordinary answers "
    "to Leo's scripted questions, use dialogue_answer even if the answer mentions "
    "a hobby, sport, animal, school subject, food, or future job. Never invent a "
    "field or value that is not explicitly present in the child's utterance. "
    "Use dialogue_none only for silence, unintelligible speech, or filler sounds "
    "such as 'uh', 'um', or 'eh'. Short replies like 'ja', 'nee', 'ja tuurlijk', "
    "'weet ik niet', and 'geen idee' are dialogue_answer, not dialogue_none. "
    "dialogue_* intents are conversation only. Classify memory-access phrases such "
    "as 'wat weet je nog over mij', 'wat heb je onthouden', and "
    "'wat zei je net over mij' as um_inspect. Return ONLY valid JSON."
)

# ── Few-shot examples ─────────────────────────────────────────────────────────

INTENT_FEW_SHOT_EXAMPLES = [
    ("Mijn lievelingseten is pizza",             {"intent": "um_add",       "field": "fav_food",   "value": "pizza",   "confidence": 0.97}),
    ("Ik wil later dokter worden",               {"intent": "um_add",       "field": "aspiration", "value": "dokter",  "confidence": 0.95}),
    ("Eigenlijk niet meer pizza, nu is het sushi",{"intent": "um_update",   "field": "fav_food",   "value": "sushi",   "confidence": 0.95}),
    ("Mijn favoriete hobby is tekenen",          {"intent": "um_update",    "field": "hobby_fav",  "value": "tekenen", "confidence": 0.96}),
    ("Nee, mijn favoriete hobby is tekenen",     {"intent": "um_update",    "field": "hobby_fav",  "value": "tekenen", "confidence": 0.97}),
    ("Vergeet wat ik zei over mijn huisdier",    {"intent": "um_delete",    "field": "has_pet",    "value": None,      "confidence": 0.98}),
    ("Het is gezond en ik doe het graag met mijn vrienden", {"intent": "dialogue_answer", "field": None, "value": None, "confidence": 0.93}),
    ("Ik vind turnen leuk omdat ik het met vrienden doe", {"intent": "dialogue_answer", "field": None, "value": None, "confidence": 0.93}),
    ("Ja tuurlijk",                              {"intent": "dialogue_answer","field": None,       "value": None,      "confidence": 0.96}),
    ("Nee, niet echt",                           {"intent": "dialogue_answer","field": None,       "value": None,      "confidence": 0.94}),
    ("Weet ik niet",                             {"intent": "dialogue_answer","field": None,       "value": None,      "confidence": 0.92}),
    ("Wat weet je over mijn lievelingshobby?",   {"intent": "um_inspect",   "field": "hobby_fav",  "value": None,      "confidence": 0.96}),
    ("Wat weet je nog over mij?",                {"intent": "um_inspect",   "field": None,         "value": None,      "confidence": 0.95}),
    ("Wat heb je over mij onthouden?",           {"intent": "um_inspect",   "field": None,         "value": None,      "confidence": 0.94}),
    ("Pizza... nee wacht, ik bedoel sushi",      {"intent": "dialogue_update","field": "fav_food", "value": "sushi",   "confidence": 0.92}),
    ("Voetbal",                                  {"intent": "dialogue_answer","field": None,       "value": "voetbal", "confidence": 0.88}),
    ("Waarom wil je dat weten?",                 {"intent": "dialogue_question","field": None,     "value": None,      "confidence": 0.95}),
    ("Haha dat is grappig",                      {"intent": "dialogue_social","field": None,       "value": None,      "confidence": 0.99}),
    ("um eh ja nee",                             {"intent": "dialogue_none", "field": None,        "value": None,      "confidence": 0.91}),
]


class GPTIntentClassifier:
    """GPT-4o-mini classifier with automatic stub fallback."""

    def __init__(
        self,
        openai_key: Optional[str] = None,
        valid_fields: list = None,
        schema_path: str = None,    # accepted but ignored (kept for API compat)
        contract_path: str = None,  # accepted but ignored (kept for API compat)
    ):
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required for GPTIntentClassifier.")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for GPTIntentClassifier. Install it with: pip install openai")
        self.client = OpenAI(api_key=key)
        self.valid_fields = set(valid_fields or EMBEDDED_VALID_FIELDS)
        self.stub = StubIntentClassifier(valid_fields=list(self.valid_fields))

        self.system_prompt = (
            INTENT_SYSTEM_PROMPT
            + f"\n\nValid intents: {VALID_INTENTS}"
            + f"\nValid fields: {sorted(self.valid_fields)}"
            + "\nReturn ONLY a JSON object with keys: intent, field, value, confidence."
        )

        # Build the few-shot message list once at construction time
        self.few_shot_messages = []
        for utterance, result in INTENT_FEW_SHOT_EXAMPLES:
            self.few_shot_messages.append({"role": "user", "content": utterance})
            self.few_shot_messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> IntentResult:
        """First attempt. Returns REPEAT_SENTINEL if confidence is too low."""
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self._call_gpt(text.strip())
        if result is None:
            logger.warning("GPT hard failure on first attempt; using stub.")
            return self.stub.classify(text)

        intent, field, value, confidence = self._coerce_result(*result, text=text.strip())
        if confidence >= CONFIDENCE_THRESHOLD:
            return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

        return IntentResult(intent=REPEAT_SENTINEL, field=None, value=None, confidence=confidence)

    def classify_retry(self, text: str) -> IntentResult:
        """Second attempt after REPEAT_SENTINEL. Falls back to stub rather than repeating."""
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self._call_gpt(text.strip())
        if result is None:
            return self.stub.classify(text)

        intent, field, value, confidence = self._coerce_result(*result, text=text.strip())
        if confidence < CONFIDENCE_THRESHOLD:
            return self.stub.classify(text)

        return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_filler_only(self, text: str) -> bool:
        tokens = re.findall(r"[a-zA-Z]+", str(text or "").lower())
        if not tokens:
            return True
        filler = {"um", "uh", "eh", "hm", "hmm", "ah", "ahh", "uhh"}
        filler_with_backchannels = filler | {"ja", "nee"}
        return all(token in filler for token in tokens) or (
            tokens[0] in filler and all(token in filler_with_backchannels for token in tokens)
        )

    def _coerce_result(self, intent: str, field, value, confidence: float, text: str) -> tuple:
        if intent == "dialogue_none" and not self._is_filler_only(text):
            return "dialogue_answer", None, None, max(confidence, 0.85)
        return intent, field, value, confidence

    def _call_gpt(self, text: str) -> Optional[tuple]:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.few_shot_messages,
                {"role": "user", "content": text},
            ]
            response = self.client.chat.completions.create(
                model=GPT_INTENT_MODEL,
                messages=messages,
                max_tokens=GPT_INTENT_MAX_TOKENS,
                temperature=GPT_INTENT_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw, text)
        except Exception as e:
            logger.error("GPT intent classifier error: %s", e)
            return None

    def _parse_response(self, raw: str, original_text: str) -> Optional[tuple]:
        # Strip possible markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Invalid intent JSON for '%s': %s", original_text[:50], e)
            return None

        for key in ("intent", "field", "value", "confidence"):
            if key not in parsed:
                return None

        intent = str(parsed["intent"]).strip()
        field  = parsed["field"]
        value  = parsed["value"]
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if intent not in VALID_INTENTS:
            return None
        if field is not None and field not in self.valid_fields:
            field = None

        confidence = max(0.0, min(1.0, confidence))
        return intent, field, value, confidence
