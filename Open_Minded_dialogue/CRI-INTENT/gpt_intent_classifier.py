"""
intent classifier using GPT-4o-mini.

Structured as MVC:
    Model      —> intent_classification_contract.json (few-shot prompt + schema)
    Controller —> GPTIntentClassifier.classify() (GPT call + confidence check + routing)
    View       —> DialogueManager._generate_response() in intent_classifier.py (unchanged)

Same interface as StubIntentClassifier:
    clf = GPTIntentClassifier(openai_key="...", schema_path="um_field_schema.json")
    result = clf.classify("Mijn lievelingseten is pizza")
    # IntentResult(intent='um_add', field='fav_food', value='pizza', confidence=0.97)

Confidence threshold: 0.7
    >= 0.7  → use GPT result
    <  0.7  → signal caller to ask child to repeat (returns REPEAT_SENTINEL)
    second time < 0.7 → fall back to StubIntentClassifier (same input text)

Fallback chain:
    GPT call fails entirely    → StubIntentClassifier
    GPT returns invalid JSON   → StubIntentClassifier
    GPT returns unknown intent → StubIntentClassifier
    GPT confidence < 0.7       → ask child to repeat once, then StubIntentClassifier
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from stub_intent_classifier import IntentResult, StubIntentClassifier, _load_valid_fields

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------Config-----------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7
GPT_MODEL            = "gpt-4o-mini"
GPT_MAX_TOKENS       = 120
GPT_TEMPERATURE      = 0.0

# Special sentinel intent returned when confidence is low on first attempt.
# The dialogue loop should ask the child to repeat, then call classify() again.
# On the second low-confidence result the classifier falls back to stub.
REPEAT_SENTINEL = "dialogue_repeat"

VALID_INTENTS = [
    "um_add", "um_update", "um_delete", "um_inspect",
    "dialogue_update", "dialogue_answer", "dialogue_question",
    "dialogue_social", "dialogue_none",
]


# ---------------------------------------------------------------------------Load prompt from contract file-------------------------------------------------------------------------

def _load_contract(contract_path: str) -> dict:
    """Load the gpt_classifier section from the contract JSON."""
    path = Path(contract_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Contract file not found: {contract_path}. "
            "Make sure intent_classification_contract.json is in CRI-INTENT-MOC/"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gpt_classifier", {})


def _build_prompt(contract: dict, valid_intents: list, valid_fields: list) -> str:
    system = contract.get("system_prompt", "")
    system += (
        f"\n\nValid intents: {valid_intents}"
        f"\nValid fields: {valid_fields}"
        f"\n\n{contract.get('output_format', '')}"
    )
    return system


def _build_few_shot_messages(contract: dict) -> list[dict]:
    messages = []
    for ex in contract.get("few_shot_examples", []):
        messages.append({"role": "user",      "content": ex["utterance"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["result"], ensure_ascii=False)})
    return messages


# ---------------------------------------------------------------------------GPT Intent Classifier--------------------------------------------------------------------------

class GPTIntentClassifier:
    """
    Production intent classifier using GPT-4o-mini.

    Same interface as StubIntentClassifier:
        clf.classify(text: str) -> IntentResult

    Confidence flow:
        1. GPT classifies the utterance.
        2. If confidence >= 0.7 → return the result.
        3. If confidence < 0.7 on FIRST attempt:
               → return IntentResult(intent=REPEAT_SENTINEL, ...)
               The dialogue loop should ask the child to repeat and call
               classify_retry(new_text) with the new utterance.
        4. If confidence < 0.7 on SECOND attempt (or any hard failure):
               → fall back to StubIntentClassifier.

    Usage in dialogue loop:
        result = clf.classify(transcript)
        if result.intent == REPEAT_SENTINEL:
            self.say("Kun je dat nog een keer zeggen?")
            new_transcript = self.listen()
            result = clf.classify_retry(new_transcript)
        # now handle result normally

    Fallback chain (hard failures — no repeat, straight to stub):
        GPT call fails entirely    → StubIntentClassifier
        GPT returns invalid JSON   → StubIntentClassifier
        GPT returns unknown intent → StubIntentClassifier

    Parameters
    ----------
    openai_key : str
        OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
    schema_path : str
        Path to um_field_schema.json (for valid field names).
    contract_path : str
        Path to intent_classification_contract.json (for prompt + few-shot examples).
    """

    def __init__(
        self,
        openai_key:    Optional[str] = None,
        schema_path:   str = "um_field_schema.json",
        contract_path: str = "intent_classification_contract.json",
    ):
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass openai_key= or set OPENAI_API_KEY env var."
            )

        self.client       = OpenAI(api_key=key)
        self.valid_fields = _load_valid_fields(schema_path)
        self._stub        = StubIntentClassifier(schema_path)

        contract            = _load_contract(contract_path)
        self._system_prompt = _build_prompt(contract, VALID_INTENTS, self.valid_fields)
        self._few_shot      = _build_few_shot_messages(contract)

        logger.info(
            "GPTIntentClassifier ready. Model=%s, threshold=%.1f, few-shot=%d examples.",
            GPT_MODEL, CONFIDENCE_THRESHOLD, len(contract.get("few_shot_examples", []))
        )

    # ------------------------------------------------------------------Public interface------------------------------------------------------------------

    def classify(self, text: str) -> IntentResult:
        """
        First-attempt classification of a Dutch child utterance.

        If GPT confidence < 0.7 returns IntentResult(intent=REPEAT_SENTINEL).
        The dialogue loop should ask the child to repeat and call classify_retry().

        Never raises — always returns a valid IntentResult.
        """
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        gpt_result = self._call_gpt(text.strip())

        # Hard failure (network, JSON, unknown intent) → stub immediately
        if gpt_result is None:
            logger.warning("GPT hard failure on first attempt — falling back to stub.")
            return self._stub.classify(text)

        intent, field, value, confidence = gpt_result

        # Good confidence → return GPT result
        if confidence >= CONFIDENCE_THRESHOLD:
            logger.info(
                "GPT classified '%s' → intent=%s field=%s conf=%.2f",
                text[:50], intent, field, confidence
            )
            return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

        # Low confidence → signal caller to ask child to repeat
        logger.info(
            "GPT confidence %.2f < %.1f for '%s' — returning REPEAT_SENTINEL.",
            confidence, CONFIDENCE_THRESHOLD, text[:50]
        )
        return IntentResult(
            intent=REPEAT_SENTINEL,
            field=None,
            value=None,
            confidence=confidence,
        )

    def classify_retry(self, text: str) -> IntentResult:
        """
        Second-attempt classification after the child repeated themselves.

        If GPT confidence is still < 0.7, falls back to StubIntentClassifier.
        Never returns REPEAT_SENTINEL — always resolves to a final result.

        Never raises — always returns a valid IntentResult.
        """
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        gpt_result = self._call_gpt(text.strip())

        # Hard failure → stub
        if gpt_result is None:
            logger.warning("GPT hard failure on retry — falling back to stub.")
            return self._stub.classify(text)

        intent, field, value, confidence = gpt_result

        # Still low confidence after retry → stub
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "GPT confidence %.2f still < %.1f after retry — falling back to stub.",
                confidence, CONFIDENCE_THRESHOLD
            )
            return self._stub.classify(text)

        logger.info(
            "GPT retry classified '%s' → intent=%s field=%s conf=%.2f",
            text[:50], intent, field, confidence
        )
        return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

    # ------------------------------------------------------------------Internal:  GPT call + parsing------------------------------------------------------------------

    def _call_gpt(self, text: str) -> Optional[tuple]:
        """
        Call GPT and parse the JSON response.
        Returns (intent, field, value, confidence) or None on hard failure.
        """
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                *self._few_shot,
                {"role": "user", "content": text},
            ]
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=GPT_MAX_TOKENS,
                temperature=GPT_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw, text)

        except Exception as e:
            logger.error("GPT API error: %s", e)
            return None

    def _parse_response(self, raw: str, original_text: str) -> Optional[tuple]:
        """
        Parse GPT's JSON response.
        Returns (intent, field, value, confidence) or None on parse failure.
        """
        # Strip markdown fences if GPT adds them despite the prompt
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw   = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "GPT returned invalid JSON for '%s': %s | raw: %s",
                original_text[:50], e, raw[:100]
            )
            return None

        for key in ("intent", "field", "value", "confidence"):
            if key not in parsed:
                logger.warning("GPT response missing key '%s': %s", key, parsed)
                return None

        intent     = str(parsed["intent"]).strip()
        field      = parsed["field"]
        value      = parsed["value"]
        confidence = float(parsed.get("confidence", 0.0))

        if intent not in VALID_INTENTS:
            logger.warning("GPT returned unknown intent '%s' — hard failure.", intent)
            return None

        if field is not None and field not in self.valid_fields:
            logger.warning("GPT returned unknown field '%s' — setting to None.", field)
            field = None

        confidence = max(0.0, min(1.0, confidence))
        return intent, field, value, confidence


# ---------------------------------------------------------------------------Smoke test---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    key  = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Set OPENAI_API_KEY first.  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    clf = GPTIntentClassifier(
        openai_key=key,
        schema_path=os.path.join(here, "um_field_schema.json"),
        contract_path=os.path.join(here, "intent_classification_contract.json"),
    )

    tests = [
        "Mijn lievelingseten is pizza",
        "Ik wil later dokter worden",
        "Eigenlijk niet meer pizza, nu is het sushi",
        "Vergeet wat ik zei over mijn huisdier",
        "Wat weet je over mijn lievelingshobby?",
        "Pizza... nee wacht, ik bedoel sushi",
        "Waarom wil je dat weten?",
        "Haha dat is grappig",
        "Voetbal",
        "um eh ja nee",
    ]

    print(f"\n{'Utterance':<52} {'intent':<22} {'field':<18} {'conf':>6}  value")
    print("-" * 122)
    for t in tests:
        r = clf.classify(t)
        flag = " ← REPEAT" if r.intent == REPEAT_SENTINEL else ""
        print(f"{t!r:<52} {r.intent:<22} {str(r.field):<18} {r.confidence:>6.2f}  {r.value!r}{flag}")
