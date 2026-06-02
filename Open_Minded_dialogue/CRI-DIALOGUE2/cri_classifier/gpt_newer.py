"""
v2: Now accepts optional turn context (Leo's previous utterance, current
topic, relevant UM fields) alongside the child's transcript. This
dramatically improves classification accuracy because the classifier
can see WHAT Leo just said, e.g., if Leo stated a wrong hobby and
the child says "Nee, tekenen!", the classifier knows it's um_update
on hobby_fav, not just a dialogue_answer.

Falls back to StubIntentClassifier when:
  - The GPT call raises any exception
  - The returned confidence is below CONFIDENCE_THRESHOLD
  - The JSON response can't be parsed

When the first classify() attempt is low-confidence, classify() returns
REPEAT_SENTINEL (the dialogue asks the child to repeat).
classify_retry() then tries again and falls back to stub rather than
returning REPEAT_SENTINEL a second time.

Contract: output always matches intent_classification_contract.json v3.0.0.
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
GPT_INTENT_MAX_TOKENS = 150
GPT_INTENT_TEMPERATURE = 0.0

MEMORY_ACCESS_PATTERN = re.compile(
    r"\b("
    r"wat weet je|wat weet jij|weet je nog|wat heb je onthouden|wat onthoud je|"
    r"wat herinner je|wat staat er in je geheugen|wat staat er in jouw geheugen|"
    r"geheugenboek|geheugen boek|"
    r"kan ik je geheugen|kan ik jouw geheugen|mag ik je geheugen|mag ik jouw geheugen|"
    r"kan ik in je geheugen|kan ik in jouw geheugen|mag ik in je geheugen|mag ik in jouw geheugen|"
    r"je geheugen zien|jouw geheugen zien|geheugen zien|"
    r"geheugenboek zien|geheugen boek zien|"
    r"laat je geheugen zien|laat jouw geheugen zien|laat eens zien wat je onthoudt"
    r")\b",
    re.IGNORECASE,
)

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the core identity + task definition. Turn-specific context is
# injected into the user message, not here.

INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a child-robot interaction system.

Context:
- A NAO robot called Leo is having a one-on-one conversation with a Dutch child aged 8-11 in a primary school setting.
- Leo has stored knowledge about the child (hobbies, interests, school subjects, aspirations) from previous interactions.
- The child speaks Dutch. The child may speak informally, briefly, use slang, switch topics, or ask questions back.
- You will receive Leo's previous utterance alongside the child's response. Use Leo's line to understand what the child is responding to.

Your task:
- Classify the child's latest utterance into exactly ONE intent.
- Identify which UM (User Model) field is targeted, if any.
- Extract the value the child stated, if any.
- Return a confidence score (0.0-1.0).

Intent definitions:
- um_add: Child provides NEW information to store. ("Mijn lievelingseten is pizza")
- um_update: Child CORRECTS something Leo said or something already stored. ("Nee, niet bakken, tekenen!")
- um_delete: Child wants Leo to FORGET something. ("Vergeet wat ik zei over mijn huisdier")
- um_inspect: Child asks what Leo REMEMBERS or asks to see/open/read/hear Leo's memory or memory book. ("Wat weet je over mij?", "Wat heb je onthouden?", "Kan ik je geheugen zien?")
- dialogue_update: Child corrects THEMSELVES mid-turn (not correcting Leo). ("Pizza... nee wacht, sushi")
- dialogue_answer: Child answers Leo's question normally. ("Voetbal", "Ja", "Negen jaar")
- dialogue_question: Child asks Leo a question. ("Waarom wil je dat weten?", "Ben jij echt een robot?")
- dialogue_social: Social/emotional response. ("Haha dat is grappig", "Oké!", "Cool")
- dialogue_none: Silence, gibberish, or unintelligible. ("um eh ja nee", "")

Critical classification rules:
- If Leo just stated something WRONG about the child (e.g., "Volgens mij is bakken jouw lievelingshobby") and the child disagrees or provides a different value, that is um_update — the child is correcting Leo's memory.
- If the child says "Nee" or "Klopt niet" after Leo stated a memory value, that is um_update (rejection of stored value), even without a new value.
- If the user message includes "Expected correction field", Leo has just asked for the replacement value for that exact field. A short answer like "Gym" or "hockey" should be um_update for that field with the child's answer as value.
- If the child gives both a pet type and pet name in one utterance, return um_update for one explicit pet field; prefer pet_name when a name is given. The action handler will use the full transcript to extract the paired pet field.
- If the child says "Ja" or "Klopt" after Leo stated a memory value, that is dialogue_answer (confirmation).
- Memory-access phrases like "wat weet je nog over mij", "wat heb je onthouden", "kun je vertellen wat je weet", "kan ik je geheugen zien", "mag ik je geheugenboek zien", and "kan ik in je geheugen kijken" are always um_inspect.
- Do NOT classify memory/geheugen/geheugenboek requests as dialogue_question just because they start with "kan ik", "mag ik", or "kun je".
- Only return um_add/um_update if the child explicitly states information. Never invent values.
- Use Leo's previous utterance to determine which field the child is talking about.
- If Response mode is school_joke_transition, do NOT return a UM-changing intent. Return intent="dialogue_answer", field=null, and put exactly one of these labels in value:
  plays_along_yes, rejects_joke_no, likes_school, dislikes_school, mixed_or_depends, unclear.
  Use plays_along_yes when the child accepts the joke or jokes along; use likes_school only when they really say school is fun or nice.
  Use rejects_joke_no when they simply reject the joke; use dislikes_school when they clearly say school is bad, boring, stupid, or unpleasant.
- If Response mode is robot_school_guess, do NOT return a UM-changing intent. Return intent="dialogue_answer", field=null, and put exactly one of these labels in value:
  correct_listening, wrong_guess, unclear.
  Use correct_listening only when the child guesses listening/hearing/paying attention. Use unclear for "weet ik niet", no idea, silence, or no real guess. Use wrong_guess for any other guess.

Output: Return ONLY a valid JSON object with exactly these four keys:
  intent (string from the list above)
  field (string from valid_fields, or null)
  value (string or null)
  confidence (float 0.0-1.0)
No markdown, no explanation, no extra keys."""

# ── Few-shot examples ─────────────────────────────────────────────────────────
# These now include Leo's previous utterance for context, showing the model
# how to use it for better classification.

INTENT_FEW_SHOT_EXAMPLES = [
    # === um_add ===
    {
        "leo": "Wat doe jij graag in je vrije tijd?",
        "child": "Mijn lievelingseten is pizza",
        "result": {"intent": "um_add", "field": "fav_food", "value": "pizza", "confidence": 0.97},
    },
    {
        "leo": "Wat wil je later worden?",
        "child": "Ik wil later dokter worden",
        "result": {"intent": "um_add", "field": "aspiration", "value": "dokter", "confidence": 0.95},
    },

    # === um_update (child corrects Leo) ===
    {
        "leo": "En volgens mij is bakken jouw allerliefste hobby.",
        "child": "Nee, tekenen!",
        "result": {"intent": "um_update", "field": "hobby_fav", "value": "tekenen", "confidence": 0.96},
    },
    {
        "leo": "Ik weet nog dat jouw lievelingseten pizza is.",
        "child": "Nee, pannenkoeken!",
        "result": {"intent": "um_update", "field": "fav_food", "value": "pannenkoeken", "confidence": 0.95},
    },
    {
        "leo": "En volgens mij ben jij vooral goed in rekenen.",
        "child": "Nee hoor, taal!",
        "result": {"intent": "um_update", "field": "school_strength", "value": "taal", "confidence": 0.94},
    },
    {
        "leo": "Oeps, wat moet ik dan onthouden over je huisdier?",
        "child": "Dat hij lulu heet en het is een poes",
        "result": {"intent": "um_update", "field": "pet_name", "value": "lulu", "confidence": 0.95},
    },
    {
        "leo": "Oeps, wat moet ik dan onthouden over je huisdier?",
        "child": "Dat het een poes is die lulu heet",
        "result": {"intent": "um_update", "field": "pet_name", "value": "lulu", "confidence": 0.95},
    },
    {
        "leo": "En volgens mij is bakken jouw allerliefste hobby.",
        "child": "Dat klopt niet",
        "result": {"intent": "um_update", "field": "hobby_fav", "value": None, "confidence": 0.90},
    },

    # === um_delete ===
    {
        "leo": "Ik weet ook nog dat jij een kat hebt.",
        "child": "Vergeet wat ik zei over mijn huisdier",
        "result": {"intent": "um_delete", "field": "has_pet", "value": None, "confidence": 0.98},
    },

    # === um_inspect ===
    {
        "leo": "Ik heb al best veel over jou onthouden.",
        "child": "Wat weet je nog over mij?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.95},
    },
    {
        "leo": "We hebben al veel besproken vandaag.",
        "child": "Wat heb je over mij onthouden?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.94},
    },
    {
        "leo": "Ik probeer alles goed te onthouden.",
        "child": "Kun je precies vertellen wat je nog weet?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.93},
    },
    {
        "leo": "We kunnen samen kijken wat ik over jou onthoud.",
        "child": "Wat weet je eigenlijk allemaal van mij?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.94},
    },
    {
        "leo": "Ik weet al best veel over jou.",
        "child": "Wat weet je over mijn lievelingshobby?",
        "result": {"intent": "um_inspect", "field": "hobby_fav", "value": None, "confidence": 0.96},
    },
    {
        "leo": "Ik weet ook nog dat jij zelf basketbal doet.",
        "child": "Kan ik je geheugen zien?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.96},
    },
    {
        "leo": "Ik probeer alles goed te onthouden.",
        "child": "Mag ik je geheugenboek zien?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.96},
    },
    {
        "leo": "We hebben al veel besproken vandaag.",
        "child": "Kan ik in je geheugen kijken?",
        "result": {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.96},
    },

    # === dialogue_update (child corrects themselves) ===
    {
        "leo": "Wat is jouw lievelingseten?",
        "child": "Pizza... nee wacht, ik bedoel sushi",
        "result": {"intent": "dialogue_update", "field": "fav_food", "value": "sushi", "confidence": 0.92},
    },

    # === dialogue_answer ===
    {
        "leo": "In welke positie speel jij?",
        "child": "Voorin meestal",
        "result": {"intent": "dialogue_answer", "field": None, "value": "voorin", "confidence": 0.88},
    },
    {
        "leo": "Ik weet ook nog dat jij aan hockey doet.",
        "child": "Ja klopt!",
        "result": {"intent": "dialogue_answer", "field": None, "value": None, "confidence": 0.90},
    },
    {
        "leo": "Doe jij dat ook wel eens?",
        "child": "Ja soms wel",
        "result": {"intent": "dialogue_answer", "field": None, "value": None, "confidence": 0.88},
    },

    # === school_joke_transition categories ===
    {
        "leo": "School is vast ook jouw allergrootste hobby ooit, toch?",
        "child": "Ja dat klopt",
        "result": {"intent": "dialogue_answer", "field": None, "value": "plays_along_yes", "confidence": 0.92},
    },
    {
        "leo": "School is vast ook jouw allergrootste hobby ooit, toch?",
        "child": "Nee echt niet",
        "result": {"intent": "dialogue_answer", "field": None, "value": "rejects_joke_no", "confidence": 0.94},
    },
    {
        "leo": "School is vast ook jouw allergrootste hobby ooit, toch?",
        "child": "Ik vind school eigenlijk best leuk",
        "result": {"intent": "dialogue_answer", "field": None, "value": "likes_school", "confidence": 0.93},
    },
    {
        "leo": "School is vast ook jouw allergrootste hobby ooit, toch?",
        "child": "School is stom",
        "result": {"intent": "dialogue_answer", "field": None, "value": "dislikes_school", "confidence": 0.93},
    },
    {
        "leo": "School is vast ook jouw allergrootste hobby ooit, toch?",
        "child": "Soms wel, hangt ervan af",
        "result": {"intent": "dialogue_answer", "field": None, "value": "mixed_or_depends", "confidence": 0.91},
    },

    # === robot_school_guess categories ===
    {
        "leo": "Kun jij raden waar ik op robotschool juist goed in was?",
        "child": "Luisteren?",
        "result": {"intent": "dialogue_answer", "field": None, "value": "correct_listening", "confidence": 0.94},
    },
    {
        "leo": "Kun jij raden waar ik op robotschool juist goed in was?",
        "child": "Rekenen?",
        "result": {"intent": "dialogue_answer", "field": None, "value": "wrong_guess", "confidence": 0.93},
    },
    {
        "leo": "Kun jij raden waar ik op robotschool juist goed in was?",
        "child": "Weet ik niet",
        "result": {"intent": "dialogue_answer", "field": None, "value": "unclear", "confidence": 0.95},
    },

    # === dialogue_question ===
    {
        "leo": "Ik vind het gewoon leuk om nieuwe dingen te proberen.",
        "child": "Waarom wil je dat weten?",
        "result": {"intent": "dialogue_question", "field": None, "value": None, "confidence": 0.95},
    },

    # === dialogue_social ===
    {
        "leo": "Mijn lama-vrienden vonden het wél een succes.",
        "child": "Haha dat is grappig",
        "result": {"intent": "dialogue_social", "field": None, "value": None, "confidence": 0.99},
    },
    {
        "leo": "Dat snap ik wel.",
        "child": "Oké!",
        "result": {"intent": "dialogue_social", "field": None, "value": None, "confidence": 0.96},
    },

    # === dialogue_none ===
    {
        "leo": "Wat vind jij daar zo leuk aan?",
        "child": "um eh ja nee",
        "result": {"intent": "dialogue_none", "field": None, "value": None, "confidence": 0.91},
    },

    # === Memory change request (informal phrasing) ===
    {
        "leo": "Ik weet ook nog dat jij aan hockey doet.",
        "child": "Ik zit niet meer op hockey, kun je dat veranderen naar ijsschaatsen?",
        "result": {"intent": "um_update", "field": "sports_fav_play", "value": "ijsschaatsen", "confidence": 0.96},
    },
    {
        "leo": "Klopt alles een beetje?",
        "child": "Ik wil mijn informatie veranderen",
        "result": {"intent": "um_update", "field": None, "value": None, "confidence": 0.88},
    },
    {
        "leo": "Is er nog iets dat je wilt veranderen?",
        "child": "Mag ik een paar dingen aanpassen in je geheugen?",
        "result": {"intent": "um_update", "field": None, "value": None, "confidence": 0.90},
    },
]


class GPTIntentClassifier:
    """GPT-4o-mini classifier with automatic stub fallback and turn-context support."""

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
            raise ImportError("openai package is required. Install with: pip install openai")
        self.client = OpenAI(api_key=key)
        self.valid_fields = set(valid_fields or EMBEDDED_VALID_FIELDS)
        self.stub = StubIntentClassifier(valid_fields=list(self.valid_fields))

        self.system_prompt = (
            INTENT_SYSTEM_PROMPT
            + f"\n\nValid intents: {list(VALID_INTENTS)}"
            + f"\nValid fields: {sorted(self.valid_fields)}"
        )

        # Build the few-shot message list once at construction time.
        # Each example now includes Leo's previous line as context.
        self.few_shot_messages = []
        for ex in INTENT_FEW_SHOT_EXAMPLES:
            user_content = self._format_user_message(
                child_text=ex["child"],
                leo_previous=ex.get("leo"),
            )
            self.few_shot_messages.append({"role": "user", "content": user_content})
            self.few_shot_messages.append({
                "role": "assistant",
                "content": json.dumps(ex["result"], ensure_ascii=False),
            })

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, turn_context: dict = None) -> IntentResult:
        """
        Classify a child utterance. Returns REPEAT_SENTINEL if confidence is too low.

        Args:
            text:         The child's transcript (from Whisper STT).
            turn_context: Optional dict with keys:
                - leo_previous:      Leo's immediately previous utterance
                - topic:             Current topic (e.g., "sport", "hobby")
                - script_phase:      Current script phase
                - relevant_fields:   List of UM fields in scope for this turn
                - response_mode:     e.g., "mistake_interpretation"
                - expected_correction_field: UM field Leo is explicitly asking the child to correct
                - correction_question: The correction question Leo just asked
        """
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self._call_gpt(text.strip(), turn_context)
        if result is None:
            logger.warning("GPT hard failure on first attempt; using stub.")
            return self.stub.classify(text)

        intent, field, value, confidence = self._coerce_result(*result, text=text.strip())
        if confidence >= CONFIDENCE_THRESHOLD:
            return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

        return IntentResult(intent=REPEAT_SENTINEL, field=None, value=None, confidence=confidence)

    def classify_retry(self, text: str, turn_context: dict = None) -> IntentResult:
        """Second attempt after REPEAT_SENTINEL. Falls back to stub rather than repeating."""
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self._call_gpt(text.strip(), turn_context)
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

    def _looks_like_memory_access(self, text: str) -> bool:
        return bool(MEMORY_ACCESS_PATTERN.search(str(text or "")))

    def _coerce_result(self, intent: str, field, value, confidence: float, text: str) -> tuple:
        if self._looks_like_memory_access(text) and intent not in {
            "um_add",
            "um_update",
            "um_delete",
            "dialogue_update",
        }:
            return "um_inspect", field, None, max(confidence, 0.95)
        if intent == "dialogue_none" and not self._is_filler_only(text):
            return "dialogue_answer", None, None, max(confidence, 0.85)
        return intent, field, value, confidence

    def _format_user_message(
        self,
        child_text: str,
        leo_previous: str = None,
        topic: str = None,
        relevant_fields: list = None,
        response_mode: str = None,
        expected_correction_field: str = None,
        correction_question: str = None,
    ) -> str:
        """Build the user message with as much context as available."""
        parts = []

        if leo_previous:
            parts.append(f"Leo said: \"{leo_previous}\"")

        if topic:
            parts.append(f"Topic: {topic}")

        if response_mode:
            parts.append(f"Response mode: {response_mode}")

        if relevant_fields:
            parts.append(f"Fields in scope: {', '.join(relevant_fields)}")

        if expected_correction_field:
            parts.append(f"Expected correction field: {expected_correction_field}")
            if correction_question:
                parts.append(f"Correction question Leo asked: \"{correction_question}\"")
            parts.append(
                "Correction instruction: interpret the child answer as the replacement value "
                f"for {expected_correction_field}, unless the child clearly refuses, says they do not know, "
                "or asks to inspect memory."
            )

        parts.append(f"Child said: \"{child_text}\"")

        return "\n".join(parts)

    def _call_gpt(self, text: str, turn_context: dict = None) -> Optional[tuple]:
        ctx = turn_context or {}
        try:
            user_message = self._format_user_message(
                child_text=text,
                leo_previous=ctx.get("leo_previous"),
                topic=ctx.get("topic"),
                relevant_fields=ctx.get("relevant_fields"),
                response_mode=ctx.get("response_mode"),
                expected_correction_field=ctx.get("expected_correction_field"),
                correction_question=ctx.get("correction_question"),
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.few_shot_messages,
                {"role": "user", "content": user_message},
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
