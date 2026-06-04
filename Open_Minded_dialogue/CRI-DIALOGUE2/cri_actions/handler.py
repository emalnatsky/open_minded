"""
ActionHandler — intent-to-action routing for the CRI dialogue.

Constructed once in CRI_ScriptedDialogue.__init__:

    self.actions = ActionHandler(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.action_handler, self.confirm_topic_change, self.classify_with_repeat,
self.llm_response, ...) stay identical.

What lives here:
  - classify_with_repeat: classify, ask child to repeat on low confidence
  - llm_response: L3 GPT runtime fallback
  - extract_json_object: parse model-emitted JSON safely
  - Confirmation text generation: clean_confirmation_question, confirmation_text
  - Turn helpers: turn_memory_context, allowed_change_fields, action_result
  - Change detection: change_from_intent_result, is_rejection_without_value,
    is_confirmation_yes, is_confirmation_no, confirmation_decision_from_intent
  - The two routing brains: action_handler + follow_up_action_handler
  - The confirmation loop: confirm_topic_change

State coupling (intentional): this module talks to many other modules
via self.d — speech, classifier, UM client, nudge, memory access, logger.
The dialogue is the meeting point.
"""

import json
import time
import logging
import re
import inspect

from sic_framework.services.llm import GPTRequest

from .l3_runtime import L3Runtime
from cri_classifier import REPEAT_SENTINEL
from tablet_state import FIELD_TO_CATEGORY

logger = logging.getLogger(__name__)


class ActionHandler:
    """Routes classified intents into Leo's next action."""

    def __init__(self, dialogue):
        self.d = dialogue
        self.l3 = L3Runtime(dialogue)

    # ── classifier helpers ───────────────────────────────────────────────────

    def classifier_context(self, turn_context: dict = None) -> dict:
        """Build optional context for GPT classifiers; stub classifiers ignore it."""
        turn = turn_context or getattr(self.d, "current_turn_context", {}) or {}
        relevant_fields = []

        for field in (turn.get("used_fields") or {}).keys():
            if field not in relevant_fields:
                relevant_fields.append(field)

        mistake_field = turn.get("mistake_field")
        if mistake_field and mistake_field not in relevant_fields:
            relevant_fields.append(mistake_field)

        correction_field = turn.get("memory_correction_field")
        if not correction_field and turn.get("memory_correction_requested"):
            correction_field = mistake_field
        if correction_field and correction_field not in relevant_fields:
            relevant_fields.append(correction_field)

        for field in turn.get("memory_review_fields") or []:
            if field not in relevant_fields:
                relevant_fields.append(field)

        topic = turn.get("topic")
        if isinstance(topic, dict):
            for field in (topic.get("current_values") or {}).keys():
                if field not in relevant_fields:
                    relevant_fields.append(field)
            topic = topic.get("kind") or topic.get("label") or topic.get("name")

        context = {
            "leo_previous": getattr(self.d, "last_leo_utterance", "") or self.d.turn_text(turn),
            "topic": topic,
            "script_phase": self.d.turn_phase(turn),
            "relevant_fields": relevant_fields,
            "response_mode": turn.get("response_mode"),
            "expected_correction_field": correction_field,
            "correction_question": turn.get("last_correction_question"),
        }
        return {key: value for key, value in context.items() if value}

    def call_classifier(self, method_name: str, transcript: str, turn_context: dict = None):
        method = getattr(self.d.clf, method_name)
        context = self.classifier_context(turn_context)
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = None

        if parameters is not None:
            if len(parameters) >= 2:
                return method(transcript, context)
            return method(transcript)

        try:
            return method(transcript, context)
        except TypeError:
            return method(transcript)

    def classify_with_repeat(self, transcript: str, turn_context: dict = None):
        """Classify once, ask for repetition on low confidence, then retry."""
        result = self.call_classifier("classify", transcript, turn_context)
        if result.intent == REPEAT_SENTINEL:
            self.d.logger.info("Low confidence - asking to repeat.")
            self.d.speech.say("Kun je dat nog een keer zeggen?")
            time.sleep(0.8)
            transcript = self.d.speech.listen_with_review()
            result = self.call_classifier("classify_retry", transcript, turn_context)
        self.d.logger.info("Intent: %s", result.to_dict())
        self.d.log_intent_classifier_result(transcript, result)
        return result

    def llm_response(self, child_input: str, turn: dict = None) -> str:
        """L3: GPT generates a personalised Dutch follow-up."""
        if turn and self.l3.is_enabled(turn):
            return self.l3.generate(child_input, turn)

        if not child_input:
            return self.d.LLM_FALLBACK
        prompt = (
            f"Het kind zei: \"{child_input}\". "
            f"Reageer warm en enthousiast in een korte zin in het Nederlands."
        )
        try:
            if self.d.gpt is not None:
                reply = self.d.gpt.request(GPTRequest(prompt=prompt, stream=False))
                response = reply.response.strip() if reply and reply.response else ""
            else:
                reply = self.d.openai_client.chat.completions.create(
                    model=self.d.TOPIC_CHANGE_MODEL,
                    messages=[
                        {"role": "system", "content": self.d.LLM_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=80,
                    temperature=0.7,
                )
                response = reply.choices[0].message.content.strip()
            response = self.d.speech.strip_non_bmp(response)
            return (response.split(".")[0].strip() + ".") if response else self.d.LLM_FALLBACK
        except Exception as e:
            self.d.logger.error("LLM error: %s", e)
            return self.d.LLM_FALLBACK

    def extract_json_object(self, raw: str) -> dict:
        """Parse a JSON object even if the model accidentally adds light wrapping."""
        if not raw:
            return {}
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}

    # ── confirmation text ────────────────────────────────────────────────────

    def clean_confirmation_question(self, question: str, change: dict) -> str:
        """Keep LLM-generated confirmation wording, but enforce style constraints."""
        if not self.d.is_known(question):
            return ""

        clean = str(question).strip()
        clean = clean.replace("Zeg ja of nee.", "").replace("Zeg ja of nee", "")
        clean = clean.replace("zeg ja of nee.", "").replace("zeg ja of nee", "")
        clean = " ".join(clean.split())

        old_value = change.get("old_value")
        if self.d.is_known(old_value) and str(old_value).strip().lower() in clean.lower():
            return ""

        return clean

    def confirmation_text(self, change: dict) -> str:
        llm_question = self.clean_confirmation_question(change.get("confirmation_question"), change)
        if llm_question:
            return llm_question

        if change["action"] == "delete":
            return f"Wil je dat ik {change['field_label']} vergeet?"

        if change["action"] == "multi_update":
            return "Wil je dat ik dit zo verander?"

        new_value = change.get("new_value")
        return f"Wil je dat ik {change['field_label']} verander naar {new_value}?"

    # ── turn helpers / action_result builder ─────────────────────────────────

    def turn_memory_context(self, turn: dict) -> dict:
        topic = turn.get("topic") or turn.get("mistake_topic") or {}
        fields = list(topic.get("fields", []) or [])
        field_labels = dict(topic.get("field_labels", {}) or {})
        current_values = dict(topic.get("current_values", {}) or {})
        stored_values = dict(topic.get("stored_values", {}) or {})
        visible_mistakes = dict(topic.get("visible_mistakes", {}) or {})

        for field in self.mentioned_um_fields(turn):
            if field not in fields:
                fields.insert(0, field)
            field_labels.setdefault(field, self.d.field_label(field))
            current_values[field] = (turn.get("used_fields") or {}).get(field)
            stored_values[field] = self.d.memory_value(field)

        for field in turn.get("memory_review_fields") or []:
            if field not in fields:
                fields.append(field)
            field_labels.setdefault(field, self.d.field_label(field))
            current_values.setdefault(field, self.d.memory_value(field))
            stored_values[field] = self.d.memory_value(field)

        return {
            "topic": topic,
            "fields": fields,
            "field_labels": field_labels,
            "current_values": current_values,
            "stored_values": stored_values,
            "visible_mistakes": visible_mistakes,
        }

    def mentioned_um_fields(self, turn: dict) -> list:
        used_fields = turn.get("used_fields") or {}
        if not isinstance(used_fields, dict):
            return []
        fields = []
        for field, value in used_fields.items():
            if field in self.d.UM_FIELDS and self.d.is_known(value) and field not in fields:
                fields.append(field)
        return fields

    def allowed_change_fields(self, turn: dict) -> list:
        context = self.turn_memory_context(turn)
        fields = list(context.get("fields") or [])
        if turn.get("mistake_field") and turn["mistake_field"] not in fields:
            fields.append(turn["mistake_field"])
        return fields or list(self.d.UM_FIELDS)

    def preferred_memory_correction_field(self, result, turn: dict) -> str:
        topic_field = self.topic_correction_field(result, turn)
        if topic_field:
            return topic_field
        mentioned = self.mentioned_um_fields(turn)
        if result and result.field in mentioned:
            return result.field
        if len(mentioned) == 1:
            return mentioned[0]
        allowed = self.allowed_change_fields(turn)
        if result and result.field in allowed:
            return result.field
        if len(allowed) == 1:
            return allowed[0]
        return ""

    def normalized_field_value(self, field: str, value: str) -> str:
        clean = str(value or "").strip().lower()
        clean = re.sub(r"\s+", " ", clean)
        clean = clean.strip(" .,!?;:")
        if field == "aspiration":
            clean = re.sub(r"^(ik wil later|later wil ik|ik wil|wil ik later|later)\s+", "", clean)
            clean = re.sub(r"\s+worden$", "", clean)
            clean = clean.strip(" .,!?;:")
        return clean

    def field_values_match(self, field: str, left: str, right: str) -> bool:
        return self.normalized_field_value(field, left) == self.normalized_field_value(field, right)

    def is_explicit_memory_rejection(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        if text in {"nee", "nee hoor", "nope"}:
            return False
        return any(
            phrase in text
            for phrase in (
                "klopt niet",
                "niet klopt",
                "niet waar",
                "verkeerd",
                "fout",
                "het is geen",
                "dat is geen",
                "is geen",
                "heet niet",
            )
        )

    def has_explicit_memory_change_cue(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        if self.is_plain_memory_acknowledgement(text):
            return False
        if self.is_explicit_memory_rejection(text):
            return True
        instruction_patterns = (
            r"\bonthoud\b",
            r"\bmoet(?: je| jij)? onthouden\b",
            r"\bwil(?: je| jij)? onthouden\b",
            r"\bkun(?: je| jij)? onthouden\b",
            r"\bkan(?: je| jij)? onthouden\b",
        )
        if any(re.search(pattern, text) for pattern in instruction_patterns):
            return True
        return any(
            phrase in text
            for phrase in (
                "verander",
                "pas aan",
                "aanpassen",
                "vergeet",
                "mijn huisdier",
                "mijn favoriete",
                "mijn lievelings",
                "het is een",
                "dat is een",
                "heet",
                "ik doe",
                "ik speel",
            )
        )

    def is_plain_memory_acknowledgement(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        negative_markers = (
            "niet goed onthouden",
            "niet goed herinnerd",
            "klopt niet",
            "niet waar",
            "verkeerd",
            "fout",
        )
        if (
            ("goed onthouden" in text or "goed herinnerd" in text)
            and not any(marker in text for marker in negative_markers)
        ):
            return True
        return text in {
            "ja",
            "ja hoor",
            "ja dat klopt",
            "ja klopt",
            "dat klopt",
            "klopt",
            "jawel",
            "zeker",
            "ja zeker",
            "inderdaad",
            "ja inderdaad",
            "ok",
            "oke",
            "oké",
            "prima",
            "goed",
            "is goed",
            "dat is goed",
        }

    def turn_has_memory_correction_available(self, turn: dict) -> bool:
        if turn.get("memory_correction_available"):
            return True
        if turn.get("response_mode") in ("mistake_interpretation", "topic_interpretation"):
            return True
        if self.mentioned_um_fields(turn):
            return True
        return bool(turn.get("allow_memory_change"))

    def has_inline_memory_correction_cue(self, result, transcript: str, turn: dict) -> bool:
        if turn.get("memory_correction_requested"):
            return False
        if not self.turn_has_memory_correction_available(turn):
            return False
        if not self.mentioned_um_fields(turn) and not turn.get("memory_correction_field"):
            return False
        if self.is_plain_memory_acknowledgement(transcript):
            return False

        allowed = self.allowed_change_fields(turn)
        correction_value_available = (
            turn.get("memory_correction_available")
            or turn.get("m3_school_difficulty_resolution")
        )
        if (
            correction_value_available
            and result.intent in ("um_add", "um_update", "dialogue_update")
            and result.field in allowed
            and self.meaningful_classifier_value(result.value)
        ):
            return True

        if self.is_explicit_memory_rejection(transcript):
            return True

        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        inline_prefixes = (
            "nee dat is ",
            "nee het is ",
            "nee eigenlijk ",
            "nee, dat is ",
            "nee, het is ",
            "dat is eigenlijk ",
            "het is eigenlijk ",
        )
        if any(text.startswith(prefix) for prefix in inline_prefixes):
            return True
        return self.has_explicit_memory_change_cue(text)

    def response_mode_allows_direct_memory_update(self, mode: str) -> bool:
        return mode in {
            "memory_access_change",
            "memory_review_group",
            "memory_review_add_final",
            "change_confirmation",
            "value_completion",
            "value_limit_clarification",
        }

    def listen_only_allows_topic_memory_correction(self, result, transcript: str, turn: dict) -> bool:
        if turn.get("response_mode") != "listen_only":
            return False
        if not self.is_topic_turn(turn):
            return False
        if not self.mentioned_um_fields(turn):
            return False
        if self.is_explicit_memory_rejection(transcript):
            return True
        if result.intent not in ("um_add", "um_update", "dialogue_update"):
            return False
        if result.field not in self.allowed_change_fields(turn):
            return False
        if not self.d.is_known(result.value):
            return False
        return self.has_explicit_memory_change_cue(transcript)

    def turn_allows_memory_change(self, turn: dict) -> bool:
        if turn.get("allow_memory_change"):
            return True
        if turn.get("memory_correction_requested"):
            return True
        if self.mentioned_um_fields(turn):
            return True
        mode = turn.get("response_mode")
        if mode in (
            "mistake_interpretation",
            "topic_interpretation",
            "memory_access_change",
            "memory_review_group",
            "memory_review_add_final",
        ):
            return True
        if mode == "nudge_interpretation" and turn.get("nudge_correction_requested"):
            return True
        return False

    def is_topic_turn(self, turn: dict) -> bool:
        return bool(turn.get("topic")) and not bool(turn.get("mistake_id"))

    def should_continue_phase_after_change(self, turn: dict, change: dict) -> bool:
        if change.get("topic_correction"):
            return True
        changes = change.get("changes") if change.get("action") == "multi_update" else [change]
        changed_fields = {
            single_change.get("field")
            for single_change in changes or []
            if single_change.get("field")
        }
        phase_id = str(turn.get("phase_id") or "")
        if turn.get("phase") == 17 or phase_id in {"3.4", "3.5", "3.4/5"}:
            return "aspiration" in changed_fields
        if turn.get("phase_id") == "3.3" or turn.get("phase") == 16:
            return "role_model" in changed_fields
        return False

    def topic_correction_field(self, result, turn: dict) -> str:
        if not self.is_topic_turn(turn):
            return ""
        topic = turn.get("topic") or {}
        domain = topic.get("domain")
        allowed = self.allowed_change_fields(turn)
        mentioned = self.mentioned_um_fields(turn)
        preferred_by_domain = {
            "sport": ("sports_fav_play",),
            "boeken": ("books_fav_title",),
            "muziek": ("music_enjoys",),
            "huisdier": ("pet_name", "pet_type", "animal_fav", "has_pet"),
            "hobby": ("hobby_fav", "hobbies", "freetime_fav"),
        }
        if (
            domain == "huisdier"
            and result
            and result.field in allowed
            and self.meaningful_classifier_value(result.value)
        ):
            return result.field
        for field in preferred_by_domain.get(domain, ()):
            if field in mentioned:
                return field
        if turn.get("memory_correction_field") in allowed:
            return turn["memory_correction_field"]
        if result and result.field in mentioned:
            return result.field
        for field in preferred_by_domain.get(domain, ()):
            if field in allowed:
                return field
        return ""

    def clean_pet_correction_value(self, value: str, *, name: bool = False) -> str:
        clean = str(value or "").strip()
        clean = re.sub(r"\s+", " ", clean)
        clean = clean.strip(" .,!?;:")
        clean = re.sub(r"^(een|de|het)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\s+(?:en|maar|die|hij|zij|het|is|heet)\b.*$",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip(" .,!?;:")
        if not clean:
            return ""
        if name:
            return clean[:1].upper() + clean[1:]
        return clean.lower()

    def pet_corrections_from_transcript(self, result, transcript: str) -> dict:
        text = str(transcript or "").strip()
        corrections = {}

        if result and result.field in ("pet_type", "pet_name") and self.meaningful_classifier_value(result.value):
            corrections[result.field] = self.clean_pet_correction_value(
                result.value,
                name=result.field == "pet_name",
            )

        type_patterns = (
            r"\b(?:het|hij|zij|ze|die|dat|mijn huisdier)\s+is\s+(?:een\s+)?([^,.!?]+)",
            r"\b(?:dat\s+)?(?:het|mijn huisdier)\s+(?:een\s+)?([^,.!?]+?)\s+is\b",
            r"\b(?:ik heb|we hebben)\s+(?:een\s+)?([^,.!?]+)",
        )
        for pattern in type_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            pet_type = self.clean_pet_correction_value(match.group(1))
            if pet_type:
                corrections.setdefault("pet_type", pet_type)
                break

        name_patterns = (
            r"\b(?:hij|zij|ze|die|mijn huisdier)\s+(?!(?:is|een|de|het)\b)([^,.!?]+?)\s+heet\b",
            r"\b(?:hij|zij|ze|die|het|mijn huisdier)\s+heet\s+([^,.!?]+)",
            r"\b(?:heet|noemt)\s+([^,.!?]+)",
            r"\b(?:de naam is|zijn naam is|haar naam is)\s+([^,.!?]+)",
        )
        for pattern in name_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            pet_name = self.clean_pet_correction_value(match.group(1), name=True)
            if pet_name:
                corrections.setdefault("pet_name", pet_name)
                break

        return {field: value for field, value in corrections.items() if self.d.is_known(value)}

    def pet_multi_topic_change_from_answer(
        self,
        result,
        turn: dict,
        transcript: str,
        context: dict,
        inline_memory_correction: bool,
    ) -> dict:
        topic = (turn or {}).get("topic") or {}
        if topic.get("domain") != "huisdier":
            return {}
        if not (turn.get("memory_correction_requested") or inline_memory_correction):
            return {}

        allowed = self.allowed_change_fields(turn)
        corrections = self.pet_corrections_from_transcript(result, transcript)
        changes = []
        for field in ("pet_type", "pet_name"):
            value = corrections.get(field)
            if field not in allowed or not self.d.is_known(value):
                continue

            old_value = (
                context.get("current_values", {}).get(field)
                or self.d.last_um_preview.get(field)
                or self.d.UNKNOWN_VALUE
            )
            if self.d.is_known(old_value) and self.field_values_match(field, old_value, value):
                continue

            field_label = context.get("field_labels", {}).get(field) or self.d.field_label(field)
            changes.append({
                "action": "update",
                "field": field,
                "field_label": field_label,
                "old_value": str(old_value),
                "new_value": str(value),
                "confidence": result.confidence,
                "reason": "Child corrected multiple pet memory fields in one answer.",
                "source_text": transcript,
                "replace_field": True,
                "topic_correction": True,
            })

        if len(changes) < 2:
            return {}

        pet_type = next((change["new_value"] for change in changes if change["field"] == "pet_type"), "")
        pet_name = next((change["new_value"] for change in changes if change["field"] == "pet_name"), "")
        question = "Wil je dat ik dit over je huisdier zo onthoud?"
        if pet_type and pet_name:
            question = f"Wil je dat ik onthoud dat je huisdier een {pet_type} is en {pet_name} heet?"

        return {
            "action": "multi_update",
            "field_label": "je huisdier",
            "changes": changes,
            "confidence": result.confidence,
            "reason": "Child corrected pet type and pet name in one answer.",
            "source_text": transcript,
            "confirmation_question": question,
            "replace_field": True,
            "topic_correction": True,
        }

    def extract_pet_value_for_field(self, result, transcript: str, field: str) -> str:
        corrections = self.pet_corrections_from_transcript(result, transcript)
        if corrections.get(field):
            return corrections[field]

        result_value = (
            self.meaningful_classifier_value(result.value)
            if result and result.field == field
            else None
        )
        candidate = result_value or self.short_correction_candidate(transcript)
        if not candidate:
            return ""
        return self.clean_pet_correction_value(candidate, name=field == "pet_name")

    def pet_pair_completion_question(self, known_field: str, known_value: str) -> str:
        if known_field == "pet_type":
            article = "een " if not str(known_value).strip().lower().startswith(("een ", "de ", "het ")) else ""
            return f"Oké, {article}{known_value}. Hoe heet je huisdier?"
        if known_field == "pet_name":
            return f"Oké, {known_value}. Wat voor huisdier is {known_value}?"
        return "Wat moet ik daar nog bij onthouden?"

    def pet_pair_change_from_single_change(
        self,
        change: dict,
        missing_field: str,
        missing_value: str,
        turn: dict,
        transcript: str,
    ) -> dict:
        corrections = {
            change.get("field"): change.get("new_value"),
            missing_field: missing_value,
        }
        context = self.turn_memory_context(turn)
        changes = []
        for field in ("pet_type", "pet_name"):
            value = corrections.get(field)
            if not self.d.is_known(value):
                continue
            old_value = (
                context.get("current_values", {}).get(field)
                or self.d.last_um_preview.get(field)
                or self.d.UNKNOWN_VALUE
            )
            field_label = context.get("field_labels", {}).get(field) or self.d.field_label(field)
            changes.append({
                "action": "update",
                "field": field,
                "field_label": field_label,
                "old_value": str(old_value),
                "new_value": str(value),
                "confidence": change.get("confidence", 0.0),
                "reason": "Child completed paired pet memory correction.",
                "source_text": " / ".join(
                    part for part in (change.get("source_text"), transcript) if self.d.is_known(part)
                ),
                "replace_field": True,
                "topic_correction": True,
            })

        if len(changes) < 2:
            return change

        pet_type = next((item["new_value"] for item in changes if item["field"] == "pet_type"), "")
        pet_name = next((item["new_value"] for item in changes if item["field"] == "pet_name"), "")
        return {
            "action": "multi_update",
            "field_label": "je huisdier",
            "changes": changes,
            "confidence": change.get("confidence", 0.0),
            "reason": "Child completed paired pet memory correction.",
            "source_text": " / ".join(
                part for part in (change.get("source_text"), transcript) if self.d.is_known(part)
            ),
            "confirmation_question": f"Wil je dat ik onthoud dat je huisdier een {pet_type} is en {pet_name} heet?",
            "replace_field": True,
            "topic_correction": True,
        }

    def complete_pet_topic_pair(self, change: dict, turn: dict) -> dict:
        if not change or change.get("action") != "update":
            return change
        topic = (turn or {}).get("topic") or {}
        if topic.get("domain") != "huisdier" or not change.get("topic_correction"):
            return change
        field = change.get("field")
        if field not in ("pet_type", "pet_name"):
            return change

        missing_field = "pet_name" if field == "pet_type" else "pet_type"
        question = self.pet_pair_completion_question(field, change.get("new_value"))
        self.d.speech.say(question)

        completion_turn = dict(turn or {})
        self.mark_waiting_for_memory_correction(completion_turn, missing_field, question)
        completion_turn["response_mode"] = "value_completion"
        completion_turn["allow_memory_change"] = True
        completion_turn["used_fields"] = {field: change.get("new_value")}

        time.sleep(0.5)
        transcript = self.d.speech.listen_with_review()
        time.sleep(0.8)
        result = self.classify_with_repeat(transcript, completion_turn)
        missing_value = self.extract_pet_value_for_field(result, transcript, missing_field)
        if not self.d.is_known(missing_value):
            return change

        return self.pet_pair_change_from_single_change(change, missing_field, missing_value, turn, transcript)

    def expected_memory_value_count(self, turn: dict, field: str) -> int:
        explicit_count = self.explicit_expected_memory_value_count(turn, field)
        if explicit_count:
            return explicit_count
        if field == "fav_subject":
            topic = (turn or {}).get("topic") or (turn or {}).get("mistake_topic") or {}
            current_values = topic.get("current_values") if isinstance(topic, dict) else {}
            current = (current_values or {}).get(field) or getattr(self.d, "last_um_preview", {}).get(field)
            return max(1, min(len(self.d.split_memory_values(current)), 2))
        return 1

    def explicit_expected_memory_value_count(self, turn: dict, field: str) -> int:
        if field == "aspiration":
            return 1

        topic = (turn or {}).get("topic") or (turn or {}).get("mistake_topic") or {}
        counts = topic.get("expected_value_count") if isinstance(topic, dict) else {}
        if isinstance(counts, dict):
            try:
                count = int(counts.get(field) or 0)
                if count > 0:
                    return count
            except (TypeError, ValueError):
                pass
        return 0

    def is_no_topic_value_answer(self, result, transcript: str, turn: dict) -> bool:
        if not turn.get("memory_correction_requested"):
            return False
        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        if not text:
            return True
        no_value_exact = {
            "weet ik niet",
            "ik weet het niet",
            "geen idee",
            "ik heb geen idee",
            "geen",
            "niks",
            "niets",
            "maakt niet uit",
            "laat maar",
        }
        if text in no_value_exact:
            return True
        if "maar" in text or "het is" in text or "dat is" in text:
            return False
        no_value_phrases = (
            "weet ik niet",
            "geen idee",
            "niks",
            "niets",
            "geen sport",
            "geen boek",
            "geen boeken",
            "geen muziek",
            "geen huisdier",
        )
        return any(phrase in text for phrase in no_value_phrases)

    def neutral_topic_correction_response(self, turn: dict) -> str:
        topic = turn.get("topic") or {}
        domain = topic.get("domain")
        label = {
            "sport": "sport",
            "boeken": "boeken",
            "muziek": "muziek",
            "huisdier": "dieren",
            "hobby": "je hobby's",
        }.get(domain, "dat")
        return f"Geen probleem, dan verander ik dat nu niet. Dan praat ik even wat algemener over {label}."

    def classifier_value_is_in_transcript(self, field: str, value: str, transcript: str) -> bool:
        value_norm = self.normalized_field_value(field, value)
        transcript_norm = self.normalized_field_value(field, transcript)
        if not value_norm:
            return False
        return value_norm in transcript_norm

    def is_placeholder_classifier_value(self, value: str) -> bool:
        clean = str(value or "").strip().lower()
        clean = re.sub(r"\s+", "_", clean)
        return clean in {
            "unclear",
            "unknown",
            "none",
            "null",
            "n/a",
            "na",
            "not_applicable",
            "no_idea",
            "geen_idee",
            "weet_ik_niet",
            "ik_weet_het_niet",
            "correct_listening",
            "wrong_guess",
            "plays_along_yes",
            "rejects_joke_no",
            "likes_school",
            "dislikes_school",
            "mixed_or_depends",
        }

    def meaningful_classifier_value(self, value: str):
        if self.d.is_known(value) and not self.is_placeholder_classifier_value(value):
            return value
        return None

    def repair_value_from_short_answer(self, result, transcript: str, turn: dict, mistake_state: dict) -> str:
        """Extract a correction value from terse answers like 'pannenkoeken' or 'nee, voetbal'."""
        result_value = self.meaningful_classifier_value(result.value)
        if result.intent != "dialogue_answer" or turn.get("response_mode") != "mistake_interpretation":
            return result_value
        if not turn.get("mistake_field"):
            return result_value
        if result_value is not None:
            return result_value

        raw = str(transcript or "").strip()
        if not raw:
            return result_value

        candidate = " ".join(raw.replace("!", " ").replace("?", " ").replace(".", " ").split())
        lower = candidate.lower().strip(" ,")

        rejection_prefixes = (
            "nee,",
            "nee ",
            "nee hoor,",
            "nee hoor ",
            "dat klopt niet,",
            "dat klopt niet ",
            "klopt niet,",
            "klopt niet ",
            "niet waar,",
            "niet waar ",
        )
        had_inline_rejection = False
        for prefix in rejection_prefixes:
            if lower.startswith(prefix):
                candidate = candidate[len(prefix):].strip(" ,")
                lower = candidate.lower().strip(" ,")
                had_inline_rejection = True
                break

        empty_rejections = {
            "nee",
            "nee hoor",
            "nee dat is niet zo",
            "nee dat is niet",
            "nee zo is het niet",
            "klopt niet",
            "dat klopt niet",
            "dat is niet zo",
            "dat is niet",
            "zo is het niet",
            "niet waar",
            "verkeerd",
            "dat is verkeerd",
            "dat is fout",
            "fout",
            "helemaal niet",
            "ja",
            "jawel",
            "klopt",
            "dat klopt",
        }
        value_less_rejection_phrases = (
            "klopt niet",
            "niet waar",
            "niet zo",
            "is niet",
            "verkeerd",
            "fout",
            "helemaal niet",
        )
        if not candidate or lower in empty_rejections:
            return result_value
        if any(phrase in lower for phrase in value_less_rejection_phrases) and not any(
            marker in lower for marker in ("maar", "het is", "dat is eigenlijk", "namelijk")
        ):
            return result_value
        if not mistake_state.get("wrong_value_rejected") and not had_inline_rejection:
            return result_value
        if len(candidate.split()) > 8:
            return result_value
        return candidate

    def repair_value_from_memory_correction_answer(
        self,
        result,
        transcript: str,
        turn: dict,
        allow_inline_rejection: bool = False,
    ) -> str:
        """Extract a correction value after Leo asks what was wrong about a UM mention."""
        result_value = self.meaningful_classifier_value(result.value)
        if not turn.get("memory_correction_requested") and not allow_inline_rejection:
            return result_value
        if result.intent not in (
            "dialogue_answer",
            "dialogue_none",
            "dialogue_social",
            "um_add",
            "um_update",
            "dialogue_update",
        ):
            return result_value
        if result_value is not None:
            return result_value

        raw = str(transcript or "").strip()
        if not raw:
            return result_value

        candidate = " ".join(raw.replace("!", " ").replace("?", " ").replace(".", " ").split())
        lower = candidate.lower().strip(" ,")
        prefixes = (
            "nee,",
            "nee ",
            "dat klopt niet,",
            "dat klopt niet ",
            "niet waar,",
            "niet waar ",
            "het is ",
            "dat is ",
            "dat is eigenlijk ",
            "eigenlijk ",
            "maar ",
        )
        stripped_prefix = True
        while stripped_prefix:
            stripped_prefix = False
            for prefix in prefixes:
                if lower.startswith(prefix):
                    candidate = candidate[len(prefix):].strip(" ,")
                    lower = candidate.lower().strip(" ,")
                    stripped_prefix = True
                    break

        empty_answers = {
            "nee",
            "nee hoor",
            "nee dat is niet zo",
            "klopt niet",
            "dat klopt niet",
            "niet waar",
            "niet zo",
            "dat is niet zo",
            "is niet zo",
            "verkeerd",
            "fout",
            "weet ik niet",
            "ik weet het niet",
            "ja",
            "klopt",
            "dat klopt",
        }
        if not candidate or lower in empty_answers:
            return result_value
        if len(candidate.split()) > 8:
            return result_value
        return candidate

    def correction_question_for_field(self, field: str, field_label: str = "", turn: dict = None) -> str:
        if field == "hobby_fav":
            return "Oeps, wat is dan je favoriete hobby?"
        if field == "sports_fav_play":
            return "Oeps, welke sport moet ik dan onthouden?"
        if field == "sports_enjoys":
            return "Oeps, sport je dan niet, of moet ik iets anders onthouden over sport?"
        if field == "fav_food":
            return "Oeps, wat is dan je lievelingseten?"
        if field == "fav_subject":
            if self.expected_memory_value_count(turn or {}, field) >= 2:
                return "Oeps, vertel mij maximaal twee lievelingsvakken die ik moet onthouden."
            return "Oeps, wat is dan je lievelingsvak?"
        if field == "school_strength":
            return "Oeps, waar ben jij dan vooral goed in op school? Noem een ding."
        if field == "aspiration":
            return "Oeps, wat wil jij dan later worden?"
        if field == "role_model":
            return "Oeps, wie is dan iemand naar wie je opkijkt?"
        label = field_label or self.d.field_label(field)
        if label.startswith("je "):
            return f"Oeps, wat is dan {label}?"
        return f"Oeps, wat moet ik dan onthouden over {label}?"

    def role_model_candidate_from_response(self, result, transcript: str, allow_inline_correction: bool = False) -> str:
        if self.is_plain_memory_acknowledgement(transcript):
            return ""

        value = self.meaningful_classifier_value(getattr(result, "value", None))
        raw = str(transcript or "").strip()
        text = " ".join(raw.replace("!", " ").replace("?", " ").replace(".", " ").split())
        lower = text.lower().strip(" ,")

        no_person_phrases = {
            "",
            "nee",
            "nee hoor",
            "ja",
            "ja hoor",
            "dat klopt",
            "klopt",
            "weet ik niet",
            "ik weet het niet",
            "geen idee",
            "niemand",
            "geen",
            "misschien",
            "dat weet ik niet",
        }
        if lower in no_person_phrases:
            return ""

        if allow_inline_correction:
            prefixes = (
                "nee dat klopt niet,",
                "nee dat klopt niet ",
                "dat klopt niet,",
                "dat klopt niet ",
                "nee dat is ",
                "nee het is ",
                "nee, dat is ",
                "nee, het is ",
                "dat is ",
                "het is ",
            )
            stripped = True
            while stripped:
                stripped = False
                for prefix in prefixes:
                    if lower.startswith(prefix):
                        text = text[len(prefix):].strip(" ,")
                        lower = text.lower().strip(" ,")
                        stripped = True
                        break

        if value and str(value).strip().lower() not in no_person_phrases:
            candidate = str(value).strip()
        else:
            candidate = text.strip(" ,")

        lower_candidate = candidate.lower().strip(" ,")
        if lower_candidate in no_person_phrases:
            return ""

        person_markers = (
            "mijn ",
            "je ",
            "juf",
            "meester",
            "trainer",
            "coach",
            "vader",
            "moeder",
            "papa",
            "mama",
            "opa",
            "oma",
            "broer",
            "zus",
            "tante",
            "oom",
            "vriend",
            "vriendin",
        )
        known_person_names = (
            "superman",
            "ronaldo",
            "messi",
            "beyonce",
            "roald dahl",
        )
        non_person_words = (
            "voetbal",
            "hockey",
            "school",
            "gym",
            "rekenen",
            "lief",
            "aardig",
            "sterk",
            "goed zijn",
            "sport",
        )

        has_marker = any(marker in lower_candidate for marker in person_markers)
        is_known_name = any(name in lower_candidate for name in known_person_names)
        looks_like_name = len(candidate.split()) <= 3 and candidate[:1].isupper()
        if any(word == lower_candidate for word in non_person_words) and not (has_marker or is_known_name):
            return ""
        if not (has_marker or is_known_name or looks_like_name):
            return ""
        if len(candidate.split()) > 6:
            return ""
        return candidate

    def role_model_change_from_candidate(self, candidate: str, transcript: str, turn: dict) -> dict:
        return {
            "action": "update",
            "field": "role_model",
            "field_label": self.d.field_label("role_model"),
            "old_value": str((turn.get("used_fields") or {}).get("role_model") or self.d.UNKNOWN_VALUE),
            "new_value": candidate,
            "confidence": 0.95,
            "reason": "Child named a role model in the no-role-model branch.",
            "source_text": transcript,
            "confirmation_question": f"Wil je dat ik onthoud dat {candidate} iemand is naar wie je opkijkt?",
            "replace_field": True,
        }

    def mistake_correction_question(self, turn: dict) -> str:
        context = self.turn_memory_context(turn)
        field = turn.get("mistake_field")
        field_label = context.get("field_labels", {}).get(field) or self.d.field_label(field)
        return self.correction_question_for_field(field, field_label, turn)

    def topic_correction_question(self, topic: dict) -> str:
        domain = topic.get("domain")
        domain_questions = {
            "sport": "Oeps, welke sport moet ik dan onthouden?",
            "boeken": "Oeps, welk boek moet ik dan onthouden?",
            "muziek": "Oeps, wat moet ik dan onthouden over muziek?",
            "huisdier": "Oeps, wat moet ik dan onthouden over je huisdier?",
            "hobby": "Oeps, welke hobby moet ik dan onthouden?",
        }
        if domain in domain_questions:
            return domain_questions[domain]
        fields = topic.get("fields") or []
        field = fields[0] if len(fields) == 1 else ""
        field_label = (topic.get("field_labels") or {}).get(field) or topic.get("label") or "mijn geheugen"
        if field:
            return self.correction_question_for_field(field, field_label, {"topic": topic})
        return "Oeps, wat klopt er dan niet?"

    def memory_mention_correction_question(self, result, turn: dict) -> tuple[str, str]:
        field = self.preferred_memory_correction_field(result, turn)
        if field:
            return self.correction_question_for_field(field, turn=turn), field
        return self.topic_correction_question(turn.get("topic", {})), ""

    def mark_waiting_for_memory_correction(self, turn: dict, field: str, question: str) -> None:
        turn["memory_correction_requested"] = True
        if field:
            turn["memory_correction_field"] = field
        if question:
            turn["last_correction_question"] = question

    def clear_pending_mistake_correction(self, turn: dict) -> None:
        if not isinstance(turn, dict):
            return
        turn.pop("memory_correction_requested", None)
        turn.pop("memory_correction_field", None)
        turn.pop("last_correction_question", None)
        mistake_id = turn.get("mistake_id")
        if mistake_id:
            state = self.d.mistake_states.get(mistake_id)
            if isinstance(state, dict):
                state.pop("wrong_value_rejected", None)

    def value_limit_question(self, field: str, existing_values: list = None, retry: bool = False) -> str:
        existing_values = existing_values or []
        if field == "hobby_fav":
            prefix = "Dat zijn er nog te veel. " if retry else ""
            return f"{prefix}Ik kan hier één favoriete hobby onthouden. Wat is jouw allerliefste hobby? Noem één ding."
        if field == "fav_subject" and existing_values:
            existing = self.d.format_dutch_list(existing_values)
            prefix = "Dat zijn er nog te veel. " if retry else ""
            return f"{prefix}Ik heb al {existing}. Vertel mij maximaal één ander lievelingsvak."
        if field == "fav_subject":
            prefix = "Dat zijn er nog te veel. " if retry else ""
            return f"{prefix}Ik kan er maximaal twee onthouden. Vertel mij maximaal twee lievelingsvakken die ik moet bewaren."
        if field == "school_strength":
            prefix = "Dat zijn er nog te veel. " if retry else ""
            return f"{prefix}Ik kan hier een ding onthouden. Waar ben jij vooral goed in op school? Noem een ding."
        if field == "aspiration":
            prefix = "Dat zijn er nog te veel. " if retry else ""
            return f"{prefix}Ik kan hier een beroep onthouden. Wat wil jij later worden? Noem een ding."
        return "Dat zijn er te veel. Welke moet ik onthouden?"

    def aspiration_values_from_text(self, text: str) -> list:
        clean = str(text or "").strip()
        if not clean:
            return []

        clean = re.sub(r"[.!?]+$", "", clean).strip(" ,")
        prefixes = (
            "nee,",
            "nee ",
            "nee hoor,",
            "nee hoor ",
            "dat klopt niet,",
            "dat klopt niet ",
            "klopt niet,",
            "klopt niet ",
            "niet waar,",
            "niet waar ",
        )
        stripped = True
        while stripped:
            stripped = False
            lower = clean.lower()
            for prefix in prefixes:
                if lower.startswith(prefix):
                    clean = clean[len(prefix):].strip(" ,")
                    stripped = True
                    break

        clean = re.sub(r"^(?:ik\s+)?wil\s+(?:later\s+)?", "", clean, flags=re.IGNORECASE).strip(" ,")
        clean = re.sub(r"^later\s+wil\s+ik\s+", "", clean, flags=re.IGNORECASE).strip(" ,")
        clean = re.sub(r"^wil\s+ik\s+later\s+", "", clean, flags=re.IGNORECASE).strip(" ,")
        clean = re.sub(r"^later\s+", "", clean, flags=re.IGNORECASE).strip(" ,")
        clean = re.sub(r"\s+worden$", "", clean, flags=re.IGNORECASE).strip(" ,")
        if not clean:
            return []

        parts = re.split(r"\s+(?:en|of)\s+|,", clean, flags=re.IGNORECASE)
        values = []
        for part in parts:
            value = re.sub(r"\s+worden$", "", part.strip(" ,"), flags=re.IGNORECASE).strip(" ,")
            if value:
                values.append(value)
        return values

    def limited_memory_values(self, field: str, value: str, transcript: str = "") -> list:
        values = self.d.split_memory_values(value)
        if field != "aspiration":
            return values

        value_values = self.aspiration_values_from_text(value)
        transcript_values = self.aspiration_values_from_text(transcript)
        candidates = [candidate for candidate in (transcript_values, value_values, values) if candidate]
        if not candidates:
            return values
        return max(candidates, key=len)

    def is_unknown_aspiration_answer(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        text = re.sub(r"\s+", " ", text).strip(" .,!?;:")
        if not text:
            return False

        prefixes = (
            "nee,",
            "nee ",
            "nee hoor,",
            "nee hoor ",
            "dat klopt niet,",
            "dat klopt niet ",
            "klopt niet,",
            "klopt niet ",
            "niet waar,",
            "niet waar ",
        )
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip(" .,!?;:")
                    stripped = True
                    break

        no_aspiration_exact = {
            "weet ik niet",
            "ik weet het niet",
            "ik weet het nog niet",
            "ik weet nog niet",
            "dat weet ik niet",
            "dat weet ik nog niet",
            "geen idee",
            "ik heb geen idee",
            "nog geen idee",
            "nog niet",
            "niks",
            "niets",
            "geen",
            "ik wil niks worden",
            "ik wil niets worden",
            "ik wil nog niks worden",
            "ik wil nog niets worden",
        }
        if text in no_aspiration_exact:
            return True
        return any(
            phrase in text
            for phrase in (
                "weet ik nog niet",
                "weet ik niet",
                "geen idee",
                "nog geen idee",
                "wil niks worden",
                "wil niets worden",
            )
        )

    def unknown_aspiration_change(self, field_label: str, old_value: str, transcript: str, confidence: float) -> dict:
        return {
            "action": "update",
            "field": "aspiration",
            "field_label": field_label,
            "old_value": str(old_value),
            "new_value": self.d.UNKNOWN_VALUE,
            "confidence": confidence,
            "reason": "Child said they do not know yet what they want to become.",
            "source_text": transcript,
            "confirmation_question": "Wil je dat ik onthoud dat je dat nog niet weet?",
            "replace_field": True,
            "sets_unknown_value": True,
        }

    def ask_for_limited_memory_values(
        self,
        change: dict,
        turn: dict,
        expected_count: int,
        existing_values: list = None,
    ) -> dict:
        field = change.get("field")
        existing_values = list(existing_values or [])
        remaining = max(expected_count - len(existing_values), 1)

        for attempt in range(2):
            question = self.value_limit_question(field, existing_values, retry=attempt > 0)
            self.d.speech.say(question)

            clarification_turn = dict(turn or {})
            self.mark_waiting_for_memory_correction(clarification_turn, field, question)
            clarification_turn["response_mode"] = "value_limit_clarification"
            clarification_turn["allow_memory_change"] = True
            clarification_turn["used_fields"] = {field: self.d.format_dutch_list(existing_values)}

            time.sleep(0.5)
            transcript = self.d.speech.listen_with_review()
            time.sleep(0.8)
            result = self.classify_with_repeat(transcript, clarification_turn)
            clarified_value = self.repair_value_from_memory_correction_answer(
                result,
                transcript,
                clarification_turn,
            )
            clarified_values = self.limited_memory_values(field, clarified_value, transcript)
            if not clarified_values:
                continue
            if len(clarified_values) > remaining:
                continue

            combined_values = self.d.unique_values(existing_values + clarified_values)
            if len(combined_values) <= len(existing_values):
                continue

            updated = dict(change)
            updated["new_value"] = self.d.format_dutch_list(combined_values)
            updated["reason"] = change.get("reason", "") + " Clarified limited multi-value correction."
            updated["source_text"] = " / ".join(
                part for part in (change.get("source_text"), transcript) if self.d.is_known(part)
            )
            updated["confirmation_question"] = (
                f"Wil je dat ik {updated['field_label']} verander naar {updated['new_value']}?"
            )
            return updated

        unresolved = dict(change)
        unresolved["value_limit_unresolved"] = True
        return unresolved

    def complete_expected_memory_values(self, change: dict, turn: dict) -> dict:
        if not change or change.get("action") != "update":
            return change
        field = change.get("field")
        expected_count = self.expected_memory_value_count(turn, field)
        explicit_count = self.explicit_expected_memory_value_count(turn, field)

        values = self.limited_memory_values(field, change.get("new_value"), change.get("source_text"))
        if explicit_count and len(values) > expected_count:
            change = self.ask_for_limited_memory_values(change, turn, expected_count)
            if change.get("value_limit_unresolved"):
                return change
            values = self.limited_memory_values(field, change.get("new_value"), change.get("source_text"))

        if expected_count <= 1:
            return change

        if len(values) >= expected_count:
            return change

        if field == "fav_subject" and expected_count == 2:
            first_value = self.d.format_dutch_list(values)
            question = f"Oké, {first_value}. Vertel mij maximaal één ander lievelingsvak."
        else:
            question = f"En wat moet ik daar nog bij onthouden?"

        self.d.speech.say(question)
        completion_turn = dict(turn or {})
        self.mark_waiting_for_memory_correction(completion_turn, field, question)
        completion_turn["response_mode"] = "value_completion"
        completion_turn["allow_memory_change"] = True
        completion_turn["used_fields"] = {field: change.get("new_value")}

        time.sleep(0.5)
        transcript = self.d.speech.listen_with_review()
        time.sleep(0.8)
        result = self.classify_with_repeat(transcript, completion_turn)
        extra_value = self.repair_value_from_memory_correction_answer(result, transcript, completion_turn)
        if not self.d.is_known(extra_value):
            return change

        extra_values = self.d.split_memory_values(extra_value)
        if field == "fav_subject" and len(values) + len(extra_values) > expected_count:
            return self.ask_for_limited_memory_values(change, turn, expected_count, existing_values=values)

        combined_values = self.d.unique_values(values + extra_values)
        if len(combined_values) <= len(values):
            return change

        updated = dict(change)
        updated["new_value"] = self.d.format_dutch_list(combined_values)
        updated["reason"] = change.get("reason", "") + " Completed expected multi-value correction."
        updated["source_text"] = " / ".join(
            part for part in (change.get("source_text"), transcript) if self.d.is_known(part)
        )
        updated["confirmation_question"] = (
            f"Wil je dat ik {updated['field_label']} verander naar {updated['new_value']}?"
        )
        return updated

    def action_result(
        self,
        action: str,
        handled: bool,
        reason: str = "",
        change: dict = None,
        leo_response: str = None,
        follow_up_needed: bool = False,
        **extra,
    ) -> dict:
        result = {
            "action": action,
            "handled": handled,
            "reason": reason,
            "change": change or {},
            "leo_response": leo_response,
            "follow_up_needed": follow_up_needed,
        }
        result.update(extra)
        return result

    # ── change detection from classifier output ──────────────────────────────

    def remember_confirmed_change_locally(self, change: dict) -> None:
        field = change.get("field")
        if not field:
            return
        if change.get("action") == "delete":
            self.d.last_um_preview[field] = self.d.UNKNOWN_VALUE
        elif change.get("sets_unknown_value"):
            self.d.last_um_preview[field] = self.d.UNKNOWN_VALUE
        elif self.d.is_known(change.get("new_value")):
            self.d.last_um_preview[field] = change["new_value"]

    def refresh_tablet_state_after_change(self, turn: dict = None) -> None:
        tablet_state = getattr(self.d, "tablet_state", None)
        refresh = getattr(tablet_state, "refresh", None)
        if not callable(refresh):
            return
        phase = self.d.turn_phase(turn or getattr(self.d, "current_turn_context", {}) or {})
        refresh(phase=phase)

    def uses_tablet_condition(self) -> bool:
        try:
            condition = self.d.tutorial_condition(getattr(self.d, "last_um_preview", {}) or {})
        except Exception:
            condition = getattr(self.d, "local_condition", "")
        return condition == getattr(self.d, "CONDITION_EXPERIMENT", "E")

    def should_gate_tablet_reveal(self, change: dict) -> bool:
        return bool((change.get("replace_field") or change.get("topic_correction")) and self.uses_tablet_condition())

    def successful_change_acknowledgement(self, change: dict) -> str:
        text = "Dankjewel, ik heb dat aangepast."
        if self.should_gate_tablet_reveal(change):
            return text + " Kijk maar op de tablet, daar zie je het veranderen."
        return text

    def reveal_tablet_change_after_operator(self, change: dict, turn: dict = None) -> None:
        if not self.should_gate_tablet_reveal(change):
            self.refresh_tablet_state_after_change(turn)
            return

        if getattr(self.d, "WAIT_FOR_OPERATOR_TABLET_REVEAL", True):
            print()
            input("Press Enter when the child is looking at the tablet to reveal the change...")

        self.reveal_tablet_change(change, turn)
        time.sleep(getattr(self.d, "TABLET_REVEAL_WAIT_SECONDS", 5.0))

    def tablet_reveal_payload(self, change: dict, turn: dict = None) -> dict:
        field = change.get("field")
        new_value = change.get("new_value")
        old_value = change.get("old_value")
        if turn and turn.get("mistake_field") == field and self.d.is_known(turn.get("mistake_wrong")):
            old_value = turn.get("mistake_wrong")
        phase = self.d.turn_phase(turn or getattr(self.d, "current_turn_context", {}) or {})
        return {
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "phase": phase,
        }

    def prepare_tablet_change_reveal(self, change: dict, turn: dict = None) -> None:
        if not self.should_gate_tablet_reveal(change):
            return
        tablet_state = getattr(self.d, "tablet_state", None)
        prepare_reveal_change = getattr(tablet_state, "prepare_reveal_change", None)
        if not callable(prepare_reveal_change):
            return
        payload = self.tablet_reveal_payload(change, turn)
        prepare_reveal_change(**payload)
        self.d.log_conversation_event(
            "tablet_event",
            tablet_event_type="mistake_changed" if (change.get("mistake_id") or change.get("visible_mistake_id") or (turn or {}).get("mistake_id")) else "memory_updated",
            field=payload.get("field"),
            old=payload.get("old_value"),
            new=payload.get("new_value"),
            phase=payload.get("phase"),
        )

    def clear_pending_tablet_reveal(self, turn: dict = None) -> None:
        tablet_state = getattr(self.d, "tablet_state", None)
        clear_pending_reveal = getattr(tablet_state, "clear_pending_reveal", None)
        if not callable(clear_pending_reveal):
            return
        phase = self.d.turn_phase(turn or getattr(self.d, "current_turn_context", {}) or {})
        clear_pending_reveal(phase=phase)

    def reveal_tablet_change(self, change: dict, turn: dict = None) -> None:
        tablet_state = getattr(self.d, "tablet_state", None)
        reveal_change = getattr(tablet_state, "reveal_change", None)
        if not callable(reveal_change):
            self.refresh_tablet_state_after_change(turn)
            return

        payload = self.tablet_reveal_payload(change, turn)
        reveal_change(**payload)
        self.d.log_conversation_event(
            "tablet_event",
            tablet_event_type="tablet_display_change",
            field=payload.get("field"),
            changed_to=payload.get("new_value"),
            phase=payload.get("phase"),
        )

    def change_from_intent_result(self, result, turn: dict, transcript: str) -> dict:
        intent = result.intent
        mistake_state = self.d.mistake_states.get(turn.get("mistake_id"), {})
        if self.is_plain_memory_acknowledgement(transcript) and not turn.get("memory_correction_requested"):
            return {}
        repaired_answer_value = self.repair_value_from_short_answer(result, transcript, turn, mistake_state)
        inline_memory_correction = self.has_inline_memory_correction_cue(result, transcript, turn)
        clear_mistake_update = (
            turn.get("response_mode") == "mistake_interpretation"
            and bool(turn.get("mistake_field"))
            and intent in ("um_add", "um_update", "dialogue_update")
            and (not result.field or result.field == turn.get("mistake_field"))
            and self.meaningful_classifier_value(result.value)
        )
        memory_correction_value = self.repair_value_from_memory_correction_answer(
            result,
            transcript,
            turn,
            allow_inline_rejection=inline_memory_correction,
        )
        unknown_aspiration_correction = (
            self.is_unknown_aspiration_answer(transcript)
            and (
                turn.get("memory_correction_field") == "aspiration"
                or (
                    turn.get("response_mode") == "mistake_interpretation"
                    and turn.get("mistake_field") == "aspiration"
                )
            )
        )
        if (
            intent in ("um_add", "um_update", "dialogue_update", "um_delete")
            and not turn.get("memory_correction_requested")
            and not inline_memory_correction
            and not clear_mistake_update
            and not self.response_mode_allows_direct_memory_update(turn.get("response_mode"))
        ):
            return {}
        dialogue_answer_corrects_mistake = (
            intent == "dialogue_answer"
            and turn.get("response_mode") == "mistake_interpretation"
            and turn.get("mistake_field")
            and self.d.is_known(repaired_answer_value)
            and (
                mistake_state.get("wrong_value_rejected")
                or str(transcript or "").strip().lower().startswith("nee")
                or "klopt niet" in str(transcript or "").strip().lower()
            )
        )
        dialogue_answer_corrects_memory = (
            intent in ("dialogue_answer", "dialogue_none", "dialogue_social", "um_add", "um_update", "dialogue_update")
            and (bool(turn.get("memory_correction_requested")) or inline_memory_correction)
            and self.d.is_known(memory_correction_value)
        )
        if (
            intent not in ("um_add", "um_update", "dialogue_update", "um_delete")
            and not dialogue_answer_corrects_mistake
            and not dialogue_answer_corrects_memory
            and not unknown_aspiration_correction
        ):
            return {}
        if not self.turn_allows_memory_change(turn):
            return {}

        allowed = self.allowed_change_fields(turn)
        context = self.turn_memory_context(turn)
        field = "aspiration" if unknown_aspiration_correction else result.field
        value = (
            self.d.UNKNOWN_VALUE
            if unknown_aspiration_correction
            else memory_correction_value if dialogue_answer_corrects_memory else repaired_answer_value
        )
        if (
            self.is_topic_turn(turn)
            and (turn.get("memory_correction_requested") or inline_memory_correction)
            and self.is_no_topic_value_answer(result, transcript, turn)
        ):
            return {}

        pet_multi_change = self.pet_multi_topic_change_from_answer(
            result,
            turn,
            transcript,
            context,
            inline_memory_correction,
        )
        if pet_multi_change:
            return pet_multi_change

        if not field and turn.get("mistake_field") and (
            intent in ("um_add", "um_update", "dialogue_update") or dialogue_answer_corrects_mistake
        ):
            field = turn.get("mistake_field")
        if not field and dialogue_answer_corrects_memory:
            field = turn.get("memory_correction_field") or self.preferred_memory_correction_field(result, turn)
        if self.is_topic_turn(turn) and (turn.get("memory_correction_requested") or inline_memory_correction):
            topic_field = self.topic_correction_field(result, turn)
            if topic_field:
                field = topic_field
        if not field and len(allowed) == 1:
            field = allowed[0]
        if field not in allowed:
            return {}

        field_label = context["field_labels"].get(field) or self.d.field_label(field)
        old_value = (
            context["current_values"].get(field)
            or self.d.last_um_preview.get(field)
            or self.d.UNKNOWN_VALUE
        )
        stored_value = (
            context.get("stored_values", {}).get(field)
            or self.d.last_um_preview.get(field)
            or self.d.UNKNOWN_VALUE
        )
        visible_mistake = (context.get("visible_mistakes") or {}).get(field) or {}
        replaces_mistake_field = (
            turn.get("response_mode") == "mistake_interpretation"
            and bool(turn.get("mistake_id"))
            and turn.get("mistake_field") == field
        )
        replaces_spoken_field = field in self.mentioned_um_fields(turn) or bool(turn.get("memory_correction_requested"))
        is_topic_correction = self.is_topic_turn(turn) and replaces_spoken_field

        if field == "aspiration" and self.is_unknown_aspiration_answer(transcript):
            return self.unknown_aspiration_change(field_label, old_value, transcript, result.confidence)

        if intent == "um_delete":
            return {
                "action": "delete",
                "field": field,
                "field_label": field_label,
                "old_value": str(old_value),
                "new_value": None,
                "confidence": result.confidence,
                "reason": f"Intent classifier returned {intent}.",
                "source_text": transcript,
                "confirmation_question": f"Wil je dat ik {field_label} vergeet?",
            }

        if not self.d.is_known(value):
            return {}
        if not self.classifier_value_is_in_transcript(field, value, transcript):
            return {}

        mistake_actual = turn.get("mistake_actual")
        if (
            turn.get("response_mode") == "mistake_interpretation"
            and turn.get("mistake_id")
            and turn.get("mistake_field") == field
            and self.d.is_known(mistake_actual)
            and self.field_values_match(field, mistake_actual, value)
        ):
            if replaces_mistake_field and self.uses_tablet_condition():
                return {
                    "action": "update",
                    "field": field,
                    "field_label": field_label,
                    "old_value": str(old_value),
                    "new_value": str(value),
                    "confidence": result.confidence,
                    "reason": "Child corrected Leo with the already-stored value; tablet reveal still needs confirmation.",
                    "source_text": transcript,
                    "confirmation_question": f"Wil je dat ik {field_label} verander naar {value}?",
                    "replace_field": True,
                }
            return {
                "action": "already_correct",
                "field": field,
                "field_label": field_label,
                "old_value": str(mistake_actual),
                "new_value": str(value),
                "confidence": result.confidence,
                "reason": "Child corrected Leo by giving the scripted actual value for this deliberate mistake.",
                "source_text": transcript,
            }

        if self.d.is_known(old_value) and self.field_values_match(field, old_value, value):
            if turn.get("mistake_id") and (
                result.intent in ("um_add", "um_update", "dialogue_update") or dialogue_answer_corrects_mistake
            ):
                if replaces_mistake_field and self.uses_tablet_condition():
                    return {
                        "action": "update",
                        "field": field,
                        "field_label": field_label,
                        "old_value": str(old_value),
                        "new_value": str(value),
                        "confidence": result.confidence,
                        "reason": "Child corrected Leo with the already-stored value; tablet reveal still needs confirmation.",
                        "source_text": transcript,
                        "confirmation_question": f"Wil je dat ik {field_label} verander naar {value}?",
                        "replace_field": True,
                    }
                return {
                    "action": "already_correct",
                    "field": field,
                    "field_label": field_label,
                    "old_value": str(old_value),
                    "new_value": str(value),
                    "confidence": result.confidence,
                    "reason": "Child corrected Leo by restating the already-correct UM value.",
                    "source_text": transcript,
                }
            if turn.get("response_mode") == "memory_access_change":
                return {
                    "action": "already_stored",
                    "field": field,
                    "field_label": field_label,
                    "old_value": str(old_value),
                    "new_value": str(value),
                    "confidence": result.confidence,
                    "reason": "Child requested a memory change, but the requested value was already stored.",
                    "source_text": transcript,
                }
            return {}

        change = {
            "action": "update",
            "field": field,
            "field_label": field_label,
            "old_value": str(old_value),
            "new_value": str(value),
            "confidence": result.confidence,
            "reason": f"Intent classifier returned {intent}.",
            "source_text": transcript,
            "confirmation_question": f"Wil je dat ik {field_label} verander naar {value}?",
        }
        if replaces_mistake_field or replaces_spoken_field:
            change["replace_field"] = True
        if visible_mistake:
            change["visible_mistake_id"] = visible_mistake.get("id")
            change["visible_mistake_field"] = visible_mistake.get("field")
            change["visible_mistake_wrong"] = visible_mistake.get("wrong")
            change["visible_mistake_actual"] = visible_mistake.get("actual")
            change["replace_field"] = True
        if (
            turn.get("response_mode") == "memory_access_change"
            and visible_mistake
            and self.d.is_known(stored_value)
            and self.field_values_match(field, stored_value, value)
        ):
            change["reason"] = "Child corrected the child-visible memory value back to the stored UM value."
        if is_topic_correction:
            change["topic_correction"] = True
        return change

    # ── confirmation intent predicates ───────────────────────────────────────

    def is_rejection_without_value(self, result, transcript: str) -> bool:
        text = str(transcript or "").lower()
        rejection_words = ("klopt niet", "niet waar", "niet zo", "verkeerd", "helemaal niet", "nee")
        if result.intent in ("um_update", "dialogue_update") and not self.meaningful_classifier_value(result.value):
            return True
        return any(word in text for word in rejection_words) and result.intent in (
            "dialogue_answer",
            "dialogue_social",
            "dialogue_none",
            "um_update",
        )

    def contains_response_phrase(self, text: str, phrase: str) -> bool:
        if " " in phrase:
            return phrase in text
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    def is_confirmation_yes(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        yes_phrases = (
            "ja", "jawel", "zeker", "klopt", "dat klopt", "goed",
            "is goed", "dat is goed", "doe dat maar", "alsjeblieft",
            "prima", "ok", "oke", "oké", "mag",
        )
        return any(self.contains_response_phrase(text, phrase) for phrase in yes_phrases)

    def is_confirmation_no(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        no_phrases = ("nee", "niet", "klopt niet", "laat maar", "verander niets")
        return any(self.contains_response_phrase(text, phrase) for phrase in no_phrases)

    def confirmation_decision_from_intent(self, result, transcript: str, change: dict) -> dict:
        refined_change = self.change_from_intent_result(
            result,
            {
                "topic": {
                    "fields": [change.get("field")],
                    "field_labels": {change.get("field"): change.get("field_label")},
                    "current_values": {change.get("field"): change.get("old_value")},
                },
                "allow_memory_change": True,
            },
            transcript,
        )
        if refined_change and refined_change.get("new_value") != change.get("new_value"):
            return self.action_result(
                "refine_confirmation_change",
                True,
                "Child refined the value during confirmation.",
                change=refined_change,
            )
        if self.is_confirmation_no(result, transcript):
            return self.action_result(
                "reject_change",
                True,
                "Child rejected the proposed change.",
                change=change,
            )
        if self.is_confirmation_yes(result, transcript):
            return self.action_result(
                "confirm_change",
                True,
                "Child confirmed the proposed change.",
                change=change,
            )
        return self.action_result(
            "ask_confirmation_again",
            True,
            "Confirmation intent was unclear.",
            change=change,
            leo_response=self.confirmation_text(change),
        )

    # ── nudge interpretation ─────────────────────────────────────────────────

    def nudge_memory_offer_text(self) -> str:
        condition = self.d.tutorial_condition(self.d.last_um_preview)
        if condition == self.d.CONDITION_EXPERIMENT:
            return "We kunnen ook samen op de tablet kijken wat ik over jou onthoud, als je wilt."
        return "We kunnen ook samen kijken wat ik over jou onthoud, als je wilt."

    def nudge_memory_access_action(self, result, turn: dict) -> dict:
        condition = self.d.tutorial_condition(self.d.last_um_preview)
        if condition == self.d.CONDITION_EXPERIMENT:
            response = "Je kunt mijn geheugenboek op de tablet bekijken."
            self.d.speech.say(response)
            return self.action_result(
                "memory_access_tablet",
                True,
                "Child accepted the nudge memory-book offer in the experiment condition.",
                leo_response=response,
                tutorial_condition=condition,
                requested_field=result.field,
            )

        response, memory_scope, returned_fields = self.d.memory_access_response(result, turn)
        self.d.speech.say(response)
        return self.action_result(
            "memory_access",
            True,
            "Child accepted the nudge memory-access offer.",
            leo_response=response,
            memory_scope=memory_scope,
            returned_fields=returned_fields,
            requested_field=result.field,
            tutorial_condition=condition,
        )

    def short_correction_candidate(self, transcript: str) -> str:
        candidate = " ".join(str(transcript or "").replace("!", " ").replace("?", " ").replace(".", " ").split())
        lower = candidate.lower().strip(" ,")
        empty_answers = {
            "",
            "nee",
            "nee hoor",
            "klopt niet",
            "dat klopt niet",
            "niet waar",
            "verkeerd",
            "helemaal niet",
            "ja",
            "jawel",
            "klopt",
            "dat klopt",
        }
        if lower in empty_answers or len(candidate.split()) > 8:
            return ""
        return candidate.strip(" ,")

    def nudge_correction_detail_action(self, result, transcript: str, turn: dict) -> dict:
        field = result.field
        value = result.value
        if not self.d.is_known(value) and result.intent == "dialogue_answer":
            value = self.short_correction_candidate(transcript)
        if not self.d.is_known(value):
            return {}

        matched_state = {}
        value_norm = str(value).strip().lower()
        for state in self.d.mistake_states.values():
            if not state.get("mentioned") or state.get("corrected"):
                continue
            actual_norm = str(state.get("actual") or "").strip().lower()
            field_matches = field and state.get("field") == field
            value_matches = actual_norm and actual_norm == value_norm
            if field_matches or value_matches:
                matched_state = state
                break

        if matched_state:
            self.d.mark_mistake_state_corrected(matched_state, turn)
            self.d.corrections_seen += 1
            self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))

        response = "Dankjewel, dan let ik daar beter op."
        self.d.speech.say(response)
        return self.action_result(
            "nudge_correction_detail",
            True,
            "Child provided a correction detail after the nudge.",
            leo_response=response,
            mistake_id=matched_state.get("id"),
            mistake_field=matched_state.get("field") or field,
            corrected_value=value,
        )

    def nudge_interpretation_action(self, result, transcript: str, turn: dict) -> dict:
        if turn.get("nudge_memory_offer_made"):
            if self.is_confirmation_yes(result, transcript):
                return self.nudge_memory_access_action(result, turn)
            if self.is_confirmation_no(result, transcript):
                response = "Oké, dan hoeft dat nu niet."
                self.d.speech.say(response)
                return self.action_result(
                    "nudge_memory_offer_declined",
                    True,
                    "Child declined the optional memory-access offer.",
                    leo_response=response,
                )

        if turn.get("nudge_correction_requested"):
            correction_action = self.nudge_correction_detail_action(result, transcript, turn)
            if correction_action:
                return correction_action
            response = "Oké, als je het zo weet, mag je het gewoon zeggen."
            self.d.speech.say(response)
            return self.action_result(
                "nudge_correction_detail_missing",
                True,
                "Child did not provide a usable correction detail after the nudge.",
                leo_response=response,
            )

        if self.is_confirmation_no(result, transcript) or self.is_rejection_without_value(result, transcript):
            response = "Oeps. Wil je zeggen wat er niet klopte?"
            turn["nudge_correction_requested"] = True
            self.d.speech.say(response)
            return self.action_result(
                "nudge_ask_correction_detail",
                True,
                "Child indicated something Leo said was wrong, but did not provide the correction yet.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_confirmation_yes(result, transcript):
            response = self.nudge_memory_offer_text()
            turn["nudge_memory_offer_made"] = True
            self.d.speech.say(response)
            return self.action_result(
                "nudge_memory_offer",
                True,
                "Child indicated everything was fine; Leo offers optional memory access.",
                leo_response=response,
                follow_up_needed=True,
            )

        response = "Als er iets niet klopte, mag je het gewoon zeggen."
        self.d.speech.say(response)
        return self.action_result(
            "nudge_unclear",
            True,
            "Nudge response was unclear.",
            leo_response=response,
        )

    # ── action handler (the routing brain) ───────────────────────────────────

    SCHOOL_JOKE_TRANSITION_RESPONSES = {
        "plays_along_yes": (
            "Haha, dan ben jij al verder dan ik. Ik maakte vooral een grapje, "
            "maar mooi dat school ook leuk kan zijn."
        ),
        "rejects_joke_no": (
            "Haha, nee, dat dacht ik al. School is meestal niet iemands hobby. "
            "Ik maakte maar een grapje hoor."
        ),
        "likes_school": (
            "Dat vind ik eigenlijk wel leuk om te horen. Ik maakte een grapje, "
            "maar school kan natuurlijk ook echt leuk zijn."
        ),
        "dislikes_school": (
            "Dat snap ik ook. Ik maakte een grapje; school hoeft echt niet altijd leuk te zijn."
        ),
        "mixed_or_depends": (
            "Ja, dat klinkt eerlijk. School kan soms leuk zijn en soms ook gewoon school zijn."
        ),
        "unclear": (
            "Haha, ik bedoelde het vooral als grapje. We gaan het er gewoon rustig over hebben."
        ),
    }

    def school_joke_transition_category(self, result, transcript: str) -> str:
        """Use the GPT-provided category, with a small offline fallback for tests."""
        value = str(getattr(result, "value", "") or "").strip().lower().replace("-", "_")
        aliases = {
            "yes": "plays_along_yes",
            "ja": "plays_along_yes",
            "no": "rejects_joke_no",
            "nee": "rejects_joke_no",
            "likes": "likes_school",
            "dislikes": "dislikes_school",
            "mixed": "mixed_or_depends",
            "depends": "mixed_or_depends",
            "unknown": "unclear",
        }
        category = aliases.get(value, value)
        if category in self.SCHOOL_JOKE_TRANSITION_RESPONSES:
            return category

        text = str(transcript or "").strip().lower()
        if not text:
            return "unclear"

        mixed_words = ("soms", "hangt ervan af", "ligt eraan", "beetje", "allebei", "verschilt")
        dislike_words = ("stom", "saai", "haat", "niet leuk", "vervelend", "vreselijk", "irritant")
        like_words = ("leuk", "fijn", "gezellig", "hou van school", "houd van school", "best leuk")
        yes_words = ("ja", "jawel", "klopt", "zeker", "inderdaad", "tuurlijk", "natuurlijk")
        no_words = ("nee", "nooit", "echt niet", "helemaal niet", "geen hobby")

        if any(word in text for word in mixed_words):
            return "mixed_or_depends"
        if any(word in text for word in dislike_words):
            return "dislikes_school"
        if any(word in text for word in like_words):
            return "likes_school"
        if any(word in text for word in no_words):
            return "rejects_joke_no"
        if any(word in text for word in yes_words):
            return "plays_along_yes"
        return "unclear"

    def school_joke_transition_action(self, result, transcript: str, turn: dict) -> dict:
        category = self.school_joke_transition_category(result, transcript)
        response = self.SCHOOL_JOKE_TRANSITION_RESPONSES[category]
        self.d.log_conversation_event(
            "school_joke_transition",
            transcript=transcript,
            school_joke_category=category,
            phase=turn.get("phase"),
            phase_id=turn.get("phase_id"),
        )
        self.d.speech.say(response)
        return self.action_result(
            "school_joke_transition",
            True,
            "Child response to the school joke was categorized and answered.",
            leo_response=response,
            school_joke_category=category,
        )

    ROBOT_SCHOOL_GUESS_RESPONSES = {
        "correct_listening": "Haha, goed geraden. Luisteren ging best oké. De rest was soms wat lastiger.",
        "wrong_guess": "Haha, bijna. Ik was eigenlijk best goed in luisteren. De rest was soms wat lastiger.",
        "unclear": "Ik verklap het: luisteren ging best oké. De rest was soms wat lastiger.",
    }

    def robot_school_guess_category(self, result, transcript: str) -> str:
        """Use the GPT-provided guess category, with a small offline fallback."""
        value = str(getattr(result, "value", "") or "").strip().lower().replace("-", "_")
        aliases = {
            "correct": "correct_listening",
            "listening": "correct_listening",
            "luisteren": "correct_listening",
            "wrong": "wrong_guess",
            "incorrect": "wrong_guess",
            "unknown": "unclear",
            "no_idea": "unclear",
        }
        category = aliases.get(value, value)
        if category in self.ROBOT_SCHOOL_GUESS_RESPONSES:
            return category

        text = str(transcript or "").strip().lower()
        if not text:
            return "unclear"
        if any(phrase in text for phrase in ("weet ik niet", "geen idee", "ik weet het niet", "geen gok")):
            return "unclear"
        if any(word in text for word in ("luister", "luisteren", "horen", "hoorde", "opletten", "aandacht")):
            return "correct_listening"
        return "wrong_guess"

    def robot_school_guess_action(self, result, transcript: str, turn: dict) -> dict:
        category = self.robot_school_guess_category(result, transcript)
        response = self.ROBOT_SCHOOL_GUESS_RESPONSES[category]
        self.d.log_conversation_event(
            "robot_school_guess",
            transcript=transcript,
            robot_school_guess_category=category,
            phase=turn.get("phase"),
            phase_id=turn.get("phase_id"),
        )
        self.d.speech.say(response)
        return self.action_result(
            "robot_school_guess",
            True,
            "Child's robot-school guess was categorized and answered.",
            leo_response=response,
            robot_school_guess_category=category,
        )

    def middle_school_feeling(self, transcript: str) -> str:
        text = str(transcript or "").strip().lower()
        if not text:
            return "unknown"

        not_thought_words = (
            "niet over nagedacht",
            "nog niet",
            "weet ik niet",
            "geen idee",
            "niet echt",
        )
        calm_words = (
            "niet spannend",
            "helemaal niet spannend",
            "niet echt spannend",
            "geen spanning",
            "niet zenuwachtig",
            "niet nerveus",
        )
        nervous_words = ("spannend", "zenuw", "eng", "bang", "nerveus", "lastig")
        looking_forward_words = ("zin", "leuk", "blij", "gezellig", "kijk ernaar uit")
        mixed_words = ("allebei", "beetje van allebei", "ook spannend", "maar ook")

        has_not_thought = any(word in text for word in not_thought_words)
        has_calm = any(word in text for word in calm_words)
        has_nervous = any(word in text for word in nervous_words)
        has_looking_forward = any(word in text for word in looking_forward_words)
        has_mixed = any(word in text for word in mixed_words)

        if has_mixed or (has_nervous and has_looking_forward):
            return "mixed"
        if has_not_thought:
            return "not_thought_about_it"
        if has_calm:
            return "calm"
        if has_nervous:
            return "nervous"
        if has_looking_forward:
            return "looking_forward"
        return "unknown"

    def middle_school_feeling_fallback(self, feeling: str) -> str:
        return {
            "calm": "Fijn, dan kijk je er best rustig naar.",
            "nervous": "Dat snap ik. Zo'n nieuwe stap kan best spannend voelen.",
            "looking_forward": "Leuk, dan kijk je daar echt met zin naar uit.",
            "mixed": "Dat snap ik. Je kunt er zin in hebben en het tegelijk spannend vinden.",
            "not_thought_about_it": "Dat is ook logisch. Je hoeft dat nog niet precies te weten.",
            "unknown": "Dat snap ik wel. Voor iedereen voelt zo'n stap weer anders.",
        }.get(feeling, "Dat snap ik wel. Voor iedereen voelt zo'n stap weer anders.")

    def middle_school_feeling_response(self, transcript: str, turn: dict, feeling: str) -> str:
        fallback = self.middle_school_feeling_fallback(feeling)
        if not self.l3.is_enabled(turn or {}):
            return fallback
        l3_turn = dict(turn or {})
        l3 = dict(l3_turn.get("l3") or {})
        l3["fallback"] = fallback
        l3_turn["l3"] = l3
        return self.llm_response(transcript, l3_turn)

    def middle_school_feeling_action(self, transcript: str, turn: dict) -> dict:
        feeling = self.middle_school_feeling(transcript)
        response = self.middle_school_feeling_response(transcript, turn, feeling)
        self.d.log_conversation_event(
            "middle_school_feeling",
            transcript=transcript,
            middle_school_feeling=feeling,
            phase=turn.get("phase"),
            phase_id=turn.get("phase_id"),
            leo_response=response,
        )
        self.d.speech.say(response)
        return self.action_result(
            "middle_school_feeling",
            True,
            "Child response about middle school was categorized and answered with an L3 wrap-up.",
            middle_school_feeling=feeling,
            leo_response=response,
        )

    def is_repeat_request(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        repeat_words = (
            "nog een keer",
            "opnieuw",
            "herhaal",
            "herhalen",
            "niet gehoord",
            "niet verstaan",
            "wat zei je",
        )
        return any(word in text for word in repeat_words)

    def is_memory_review_fine(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        fine_phrases = (
            "alles klopt",
            "dat klopt",
            "klopt allemaal",
            "het klopt",
            "is goed",
            "prima",
            "ok",
            "oke",
            "oké",
            "nee hoor",
        )
        return any(phrase in text for phrase in fine_phrases) or text == "nee"

    def is_memory_review_not_sure(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        not_sure_phrases = (
            "weet ik niet",
            "ik weet het niet",
            "niet zeker",
            "geen idee",
            "misschien",
            "twijfel",
        )
        return any(phrase in text for phrase in not_sure_phrases)

    def is_vague_memory_correction(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        vague_phrases = (
            "er klopt iets niet",
            "iets klopt niet",
            "dat klopt niet",
            "niet alles klopt",
            "er is iets fout",
            "iets is fout",
        )
        return any(phrase in text for phrase in vague_phrases)

    def is_memory_inspection_decline(self, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        decline_phrases = (
            "nee",
            "nee hoor",
            "liever niet",
            "ik wil niet",
            "hoeft niet",
            "niet kijken",
        )
        return text in decline_phrases

    def activate_tablet_memory_access(self, fields: list, turn: dict):
        tablet_state = getattr(self.d, "tablet_state", None)
        if hasattr(tablet_state, "activate_memory_access"):
            tablet_state.activate_memory_access(fields, phase=self.d.turn_phase(turn))
        categories = []
        for field in fields or []:
            category = FIELD_TO_CATEGORY.get(field)
            if category and category not in categories:
                categories.append(category)
                self.d.log_conversation_event(
                    "tablet_event",
                    tablet_event_type="category_unlocked",
                    category=category,
                    field=field,
                    phase=self.d.turn_phase(turn),
                )
        if categories:
            self.d.log_conversation_event(
                "tablet_event",
                tablet_event_type="shown",
                memory_item=", ".join(categories),
                phase=self.d.turn_phase(turn),
            )

    def speak_memory_review_lines(self, lines: list):
        for line in lines:
            self.d.speech.say(line)

    def explicit_memory_inspection_initial_action(self, transcript: str, turn: dict) -> dict:
        if self.is_memory_inspection_decline(transcript):
            self.d.memory_review_requested = False
            response = "Dat is ook goed. Dan gaan we gewoon nog even verder."
            self.d.speech.say(response)
            return self.action_result(
                "explicit_memory_inspection_declined",
                True,
                "Child declined the explicit memory inspection offer.",
                leo_response=response,
            )

        if not self.is_confirmation_yes(None, transcript):
            self.d.memory_review_requested = False
            response = "Dat is goed. Je hoeft niet te kijken. Dan gaan we gewoon verder."
            self.d.speech.say(response)
            return self.action_result(
                "explicit_memory_inspection_unclear",
                True,
                "Child did not clearly accept the explicit memory inspection offer.",
                leo_response=response,
            )

        fields = self.d.memory_access_scope(turn)
        condition = self.d.tutorial_condition(self.d.last_um_preview)
        self.d.memory_review_requested = True
        turn["explicit_memory_inspection_fields"] = list(fields)
        return self.action_result(
            "explicit_memory_inspection_accepted",
            True,
            "Child accepted explicit memory inspection; the current phase will perform the review.",
            tutorial_condition=condition,
            memory_scope=list(fields),
        )

    def explicit_memory_inspection_followup_action(self, transcript: str, turn: dict) -> dict:
        if self.is_repeat_request(transcript):
            if turn.get("tablet_memory_instruction_lines"):
                lines = [
                    "Natuurlijk. Ik zeg het nog een keer.",
                    *turn["tablet_memory_instruction_lines"],
                ]
            else:
                lines = [
                    "Natuurlijk. Ik vertel het nog een keer.",
                    *(turn.get("memory_review_lines") or []),
                ]
            self.speak_memory_review_lines(lines)
            return self.action_result(
                "explicit_memory_inspection_repeat",
                True,
                "Child asked to hear the memory inspection again.",
                leo_response=" ".join(lines),
                follow_up_needed=True,
            )

        if self.is_vague_memory_correction(transcript):
            response = "Oeps, wil je zeggen wat er niet klopt?"
            self.d.speech.say(response)
            return self.action_result(
                "explicit_memory_inspection_ask_correction_detail",
                True,
                "Child said something was wrong but did not give the correction yet.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_memory_review_fine(transcript):
            response = "Fijn. Dan gaan we gewoon weer verder."
            self.d.speech.say(response)
            return self.action_result(
                "explicit_memory_inspection_done",
                True,
                "Child indicated the memory inspection was fine.",
                leo_response=response,
            )

        response = "Oké, dan gaan we gewoon verder."
        self.d.speech.say(response)
        return self.action_result(
            "explicit_memory_inspection_done_unclear",
            True,
            "Child did not ask for repeat or provide a correction after memory inspection.",
            leo_response=response,
        )

    def explicit_memory_inspection_action(self, transcript: str, turn: dict) -> dict:
        if turn.get("explicit_memory_inspection_active"):
            return self.explicit_memory_inspection_followup_action(transcript, turn)
        return self.explicit_memory_inspection_initial_action(transcript, turn)

    def role_model_absence_check_action(self, result, transcript: str, turn: dict) -> dict:
        if turn.get("memory_correction_requested"):
            return self.role_model_discovery_action(result, transcript, turn)

        candidate = self.role_model_candidate_from_response(result, transcript, allow_inline_correction=True)
        if candidate:
            change = self.role_model_change_from_candidate(candidate, transcript, turn)
            accepted = self.confirm_topic_change(change)
            return self.action_result(
                "confirm_role_model_discovery",
                True,
                "Child corrected the no-role-model assumption with a role model.",
                change=change,
                change_confirmed=accepted,
                stop_phase_after_change=False,
                continue_phase_after_change=bool(accepted),
            )
        if self.is_explicit_memory_rejection(transcript) or self.is_rejection_without_value(result, transcript):
            response = "Oeps, wie is dan iemand naar wie je opkijkt?"
            self.d.speech.say(response)
            self.mark_waiting_for_memory_correction(turn, "role_model", response)
            return self.action_result(
                "role_model_absence_ask_detail",
                True,
                "Child rejected the no-role-model assumption but did not name a person yet.",
                leo_response=response,
                follow_up_needed=True,
            )
        return self.action_result(
            "role_model_absence_continue",
            True,
            "Child did not provide a role model yet; continue to the discovery prompt.",
        )

    def role_model_discovery_action(self, result, transcript: str, turn: dict) -> dict:
        candidate = self.role_model_candidate_from_response(result, transcript)
        if not candidate:
            response = (
                (turn.get("l3") or {}).get("fallback")
                or "Dat snap ik wel. Soms weet je dat ook niet meteen."
            )
            self.d.speech.say(response)
            return self.action_result(
                "role_model_discovery_no_person",
                True,
                "Child did not name a clear role model in the no-role-model branch.",
                leo_response=response,
            )
        change = self.role_model_change_from_candidate(candidate, transcript, turn)
        accepted = self.confirm_topic_change(change)
        return self.action_result(
            "confirm_role_model_discovery",
            True,
            "Child named a role model in the no-role-model branch.",
            change=change,
            change_confirmed=accepted,
            stop_phase_after_change=False,
            continue_phase_after_change=bool(accepted),
        )

    def memory_review_group_action(self, result, transcript: str, turn: dict) -> dict:
        if self.is_repeat_request(transcript):
            response = self.d.turn_text(turn)
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_group_repeat",
                True,
                "Child asked to hear the current memory review group again.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_vague_memory_correction(transcript) or self.is_confirmation_no(result, transcript):
            response = "Oeps, wat moet ik dan anders onthouden?"
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_group_ask_correction_detail",
                True,
                "Child said a memory review group was wrong but did not provide the correction yet.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_memory_review_not_sure(transcript):
            response = "Dat is ook oke. Dan laat ik het nu even zo staan."
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_group_not_sure",
                True,
                "Child was unsure about the reviewed memory group.",
                leo_response=response,
            )

        if self.is_memory_review_fine(transcript) or self.is_confirmation_yes(result, transcript):
            response = "Fijn, dan laat ik dat zo staan."
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_group_confirmed",
                True,
                "Child confirmed the reviewed memory group.",
                leo_response=response,
            )

        response = "Oke, dan gaan we door naar het volgende stukje."
        self.d.speech.say(response)
        return self.action_result(
            "memory_review_group_unclear",
            True,
            "Child response to memory review group was unclear.",
            leo_response=response,
        )

    def memory_review_final_action(self, result, transcript: str, turn: dict) -> dict:
        if self.is_repeat_request(transcript):
            response = self.d.turn_text(turn)
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_final_repeat",
                True,
                "Child asked to hear the co-construction question again.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_confirmation_no(result, transcript) or self.is_memory_review_not_sure(transcript):
            response = "Dat is ook goed. Dan laat ik het zo."
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_final_no_extra",
                True,
                "Child did not add anything extra to Leo's memory.",
                leo_response=response,
            )

        if self.is_vague_memory_correction(transcript):
            response = "Oeps, wat wil je dat ik nog onthoud?"
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_final_ask_detail",
                True,
                "Child wanted to add something but did not provide enough detail yet.",
                leo_response=response,
                follow_up_needed=True,
            )

        if self.is_confirmation_yes(result, transcript):
            response = "Wat wil je dat ik nog onthoud?"
            self.d.speech.say(response)
            return self.action_result(
                "memory_review_final_ask_detail",
                True,
                "Child said there is something extra but did not give it yet.",
                leo_response=response,
                follow_up_needed=True,
            )

        response = "Dat is leuk om te weten. Ik heb het in dit gesprek gehoord."
        self.d.speech.say(response)
        self.d.log_conversation_event(
            "memory_review_extra_unmapped",
            transcript=transcript,
            phase=turn.get("phase"),
            phase_id=turn.get("phase_id"),
        )
        return self.action_result(
            "memory_review_final_extra_unmapped",
            True,
            "Child added something extra, but no UM field/value pair was extracted.",
            leo_response=response,
        )

    def action_handler(self, result, transcript: str, turn: dict) -> dict:
        intent = result.intent
        mode = turn.get("response_mode")
        listen_only_topic_memory_correction = self.listen_only_allows_topic_memory_correction(
            result,
            transcript,
            turn,
        )
        listen_only_locks_memory = (
            mode == "listen_only"
            and not turn.get("allow_memory_change")
            and not turn.get("memory_correction_requested")
            and not listen_only_topic_memory_correction
        )

        if mode == "role_model_absence_check":
            return self.role_model_absence_check_action(result, transcript, turn)

        if mode == "role_model_discovery":
            return self.role_model_discovery_action(result, transcript, turn)

        change = {} if listen_only_locks_memory else self.change_from_intent_result(result, turn, transcript)

        if change:
            change = self.complete_expected_memory_values(change, turn)
            if not change.get("direct_mistake_correction"):
                change = self.complete_pet_topic_pair(change, turn)

            if change.get("value_limit_unresolved"):
                response = "Dan verander ik het nu nog niet, want ik weet niet zeker welke ik moet bewaren."
                self.d.speech.say(response)
                return self.action_result(
                    "change_value_limit_unresolved",
                    True,
                    "Child gave too many values for a limited UM correction and did not clarify.",
                    change=change,
                    leo_response=response,
                )

            if change.get("action") == "already_correct":
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))
                self.refresh_tablet_state_after_change(turn)
                response = None if turn.get("defer_corrected_response") else "O ja, dankjewel."
                if response:
                    self.d.speech.say(response)
                return self.action_result(
                    "mistake_corrected_no_um_change",
                    True,
                    "Child corrected deliberate mistake by restating the already-correct UM value.",
                    change=change,
                    leo_response=response,
                )

            if change.get("action") == "already_stored":
                response = (
                    f"Dat staat al zo in mijn geheugen. Ik heb al onthouden dat "
                    f"{change['field_label']} {change['new_value']} is."
                )
                self.d.speech.say(response)
                return self.action_result(
                    "memory_access_change_already_stored",
                    True,
                    "Child requested a visible memory change to the value already stored.",
                    change=change,
                    leo_response=response,
                )

            if change.get("direct_mistake_correction"):
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))
                self.remember_confirmed_change_locally(change)
                written = self.d.write_um_change(change)
                if written:
                    self.refresh_tablet_state_after_change(turn)
                response = None if turn.get("defer_corrected_response") else (
                    "Oeps, dan had ik dat verkeerd. Dan pas ik het aan."
                    if written
                    else "Oeps, dan had ik dat verkeerd."
                )
                if response:
                    self.d.speech.say(response)
                return self.action_result(
                    "mistake_corrected_update",
                    True,
                    "Child answered Leo's explicit correction question; updated without a second confirmation.",
                    change=change,
                    leo_response=response,
                    write_success=written,
                )

            accepted = self.confirm_topic_change(change)
            continue_phase = self.should_continue_phase_after_change(turn, change)
            repeat_current_segment = (
                not accepted
                and turn.get("response_mode") == "mistake_interpretation"
                and bool(turn.get("mistake_id"))
            )
            if repeat_current_segment:
                self.clear_pending_mistake_correction(turn)
            return self.action_result(
                f"confirm_{change['action']}",
                True,
                "UM-changing intent from classifier.",
                change=change,
                change_confirmed=accepted,
                stop_phase_after_change=bool(accepted and not turn.get("mistake_id") and not continue_phase),
                continue_phase_after_change=bool(accepted and continue_phase),
                repeat_current_segment_after_rejected_correction=repeat_current_segment,
            )

        if self.is_topic_turn(turn) and self.is_no_topic_value_answer(result, transcript, turn):
            response = self.neutral_topic_correction_response(turn)
            turn["topic_neutral_after_correction"] = True
            self.d.speech.say(response)
            return self.action_result(
                "topic_correction_no_value",
                True,
                "Child rejected the topic value but gave no replacement value; continue with neutral topic talk.",
                leo_response=response,
                continue_phase_after_change=True,
                topic_neutral_after_correction=True,
            )

        if intent == "um_inspect":
            condition = self.d.tutorial_condition(self.d.last_um_preview)
            memory_scope = self.d.memory_access_scope(turn)
            if condition == self.d.CONDITION_EXPERIMENT:
                self.activate_tablet_memory_access(memory_scope, turn)
                response = (
                    "Tik op mijn geheugenboek om te zien wat erin staat. "
                    "Als er iets niet klopt, zeg het dan tegen mij."
                )
                self.d.speech.say(response)
                return self.action_result(
                    "memory_access_tablet",
                    True,
                    "Experiment condition memory access: tablet stays on the book cover and unlocks mentioned chapters.",
                    leo_response=response,
                    tutorial_condition=condition,
                    requested_field=result.field,
                    visible_fields=list(memory_scope),
                    memory_scope=memory_scope,
                )

            response, memory_scope, returned_fields = self.d.memory_access_response(result, turn)
            response = f"Welkom in mijn geheugen! {response}"
            self.d.speech.say(response)
            return self.action_result(
                "memory_access",
                True,
                "Control condition memory access: returned spoken chapter-scoped memory.",
                leo_response=response,
                memory_scope=memory_scope,
                returned_fields=returned_fields,
                visible_fields=list(memory_scope),
                requested_field=result.field,
                tutorial_condition=condition,
            )

        if mode == "school_joke_transition":
            return self.school_joke_transition_action(result, transcript, turn)

        if mode == "robot_school_guess":
            return self.robot_school_guess_action(result, transcript, turn)

        listen_only_rejected_topic_memory = (
            mode == "listen_only"
            and self.is_topic_turn(turn)
            and bool(self.mentioned_um_fields(turn))
            and self.is_explicit_memory_rejection(transcript)
        )
        can_ask_memory_correction = (
            (
                mode != "listen_only"
                and (
                    mode in ("mistake_interpretation", "topic_interpretation")
                    or bool(turn.get("memory_correction_requested"))
                    or self.has_inline_memory_correction_cue(result, transcript, turn)
                )
            )
            or listen_only_rejected_topic_memory
        )
        if can_ask_memory_correction and self.is_rejection_without_value(result, transcript):
            if mode == "mistake_interpretation":
                response = self.mistake_correction_question(turn)
                correction_field = turn.get("mistake_field", "")
            elif listen_only_rejected_topic_memory:
                response = self.topic_correction_question(turn.get("topic", {}))
                correction_field = ""
            else:
                response, correction_field = self.memory_mention_correction_question(result, turn)
            self.mark_waiting_for_memory_correction(turn, correction_field, response)
            if mode == "mistake_interpretation" and turn.get("mistake_id"):
                state = self.d.mistake_states.setdefault(turn["mistake_id"], {"id": turn["mistake_id"]})
                state["wrong_value_rejected"] = True
            self.d.speech.say(response)
            return self.action_result(
                "ask_correction_detail",
                True,
                "Child rejected remembered information without giving a new value.",
                leo_response=response,
                follow_up_needed=True,
            )

        if mode == "nudge_interpretation":
            return self.nudge_interpretation_action(result, transcript, turn)

        if mode == "middle_school_feeling":
            return self.middle_school_feeling_action(transcript, turn)

        if mode == "explicit_memory_inspection_offer":
            return self.explicit_memory_inspection_action(transcript, turn)

        if mode == "memory_review_group":
            return self.memory_review_group_action(result, transcript, turn)

        if mode == "memory_review_add_final":
            return self.memory_review_final_action(result, transcript, turn)

        if intent == "dialogue_question":
            response = self.llm_response(transcript)
            self.d.speech.say(response)
            return self.action_result(
                "answer_dialogue_question",
                True,
                "Child asked a dialogue question.",
                leo_response=response,
            )

        if mode == "listen_only":
            return self.action_result(
                "listen_only",
                True,
                "Child response heard; no Leo response planned for this phase segment.",
            )

        if mode == "acknowledge":
            self.d.mark_m3_corrected_by_school_difficulty_ack(turn)
            response = self.llm_response(transcript, turn) if turn.get("llm_turn") and transcript else turn.get("follow_up")
            self.d.speech.say(response or self.d.LLM_FALLBACK)
            return self.action_result(
                "acknowledge",
                True,
                "Acknowledgement phase.",
                leo_response=response,
            )

        if mode == "mistake_interpretation":
            return self.action_result(
                "continue_wrong_value_followup",
                True,
                "No correction detected after deliberate mistake; continue with the scripted follow-up about the wrong value.",
            )

        if mode == "topic_interpretation":
            return self.action_result(
                "no_memory_change",
                True,
                "No UM-changing intent detected for topic value statement; continue with scripted topic flow.",
            )

        if intent == "dialogue_social":
            response = "Haha ja! Oké, verder!"
            self.d.speech.say(response)
            return self.action_result("dialogue_social", True, "Social response.", leo_response=response)

        if intent == "dialogue_answer":
            response = turn.get("follow_up") or "Oké, dankjewel."
            self.d.speech.say(response)
            return self.action_result("dialogue_answer", True, "Direct answer.", leo_response=response)

        return self.action_result("unhandled", False, f"No route for intent {intent}.")

    def follow_up_action_handler(self, turn: dict, max_rounds: int = 3):
        """Listen after action-handler prompts such as correction questions or nudges."""
        action = {}
        for _ in range(max_rounds):
            time.sleep(0.5)
            transcript = self.d.speech.listen_with_review()
            time.sleep(0.8)
            result = self.classify_with_repeat(transcript, turn)
            action = self.action_handler(result, transcript, turn)
            self.d.log_action_handler_result(action)
            if not action.get("follow_up_needed"):
                return action
        return action

    # ── confirmation loop ────────────────────────────────────────────────────

    def primary_reveal_change(self, change: dict) -> dict:
        changes = change.get("changes") or []
        if not changes:
            return change
        for single_change in changes:
            if single_change.get("field") == "pet_name":
                return single_change
        return changes[0]

    def is_pet_pair_change(self, change: dict) -> bool:
        fields = {
            single_change.get("field")
            for single_change in (change.get("changes") or [])
            if single_change.get("field")
        }
        return "pet_type" in fields and "pet_name" in fields

    def confirm_multi_topic_change(self, change: dict) -> bool:
        self.d.pending_change = change
        changes = list(change.get("changes") or [])

        while True:
            self.d.speech.say(self.confirmation_text(change))
            time.sleep(0.5)

            confirmation = self.d.speech.listen_with_review()
            time.sleep(0.8)

            confirmation_context = dict(getattr(self.d, "current_turn_context", {}) or {})
            confirmation_context["response_mode"] = "change_confirmation"
            confirmation_context["used_fields"] = {
                single_change.get("field"): single_change.get("new_value")
                for single_change in changes
                if single_change.get("field")
            }
            result = self.classify_with_repeat(confirmation, confirmation_context)

            if self.is_confirmation_no(result, confirmation):
                decision = self.action_result(
                    "reject_change",
                    True,
                    "Child rejected the proposed multi-field change.",
                    change=change,
                )
                self.d.log_action_handler_result(decision)
                self.d.speech.say("Oké, dan verander ik niets.")
                self.d.pending_change = None
                return False

            if self.is_confirmation_yes(result, confirmation):
                decision = self.action_result(
                    "confirm_change",
                    True,
                    "Child confirmed the proposed multi-field change.",
                    change=change,
                )
                self.d.log_action_handler_result(decision)
                self.d.corrections_seen += 1
                reveal_change = self.primary_reveal_change(change)
                for single_change in changes:
                    self.d.handle_confirmed_mistake_related_change(single_change, self.d.current_turn_context)
                    self.remember_confirmed_change_locally(single_change)
                self.prepare_tablet_change_reveal(reveal_change, self.d.current_turn_context)
                if self.is_pet_pair_change(change):
                    written = [self.d.write_pet_pair_change(change)]
                else:
                    written = [self.d.write_um_change(single_change) for single_change in changes]
                if self.d.current_turn_context:
                    self.d.phases_with_confirmed_change.add(self.d.current_turn_context.get("phase"))
                if all(written):
                    self.d.speech.say(self.successful_change_acknowledgement(change))
                    self.reveal_tablet_change_after_operator(reveal_change, self.d.current_turn_context)
                else:
                    self.clear_pending_tablet_reveal(self.d.current_turn_context)
                    self.d.speech.say("Dankjewel, ik heb dat genoteerd, maar opslaan lukte nu niet.")
                self.d.pending_change = None
                return True

            self.d.log_action_handler_result(self.action_result(
                "ask_confirmation_again",
                True,
                "Multi-field confirmation intent was unclear.",
                change=change,
                leo_response=self.confirmation_text(change),
            ))
            self.d.speech.say(self.confirmation_text(change))

    def confirm_topic_change(self, change: dict) -> bool:
        if change.get("action") == "multi_update":
            return self.confirm_multi_topic_change(change)

        self.d.pending_change = change

        while True:
            self.d.speech.say(self.confirmation_text(change))
            time.sleep(0.5)

            confirmation = self.d.speech.listen_with_review()
            time.sleep(0.8)

            confirmation_context = dict(getattr(self.d, "current_turn_context", {}) or {})
            confirmation_context["response_mode"] = "change_confirmation"
            if change.get("field"):
                confirmation_context["used_fields"] = {change["field"]: change.get("new_value")}
            result = self.classify_with_repeat(confirmation, confirmation_context)
            decision = self.confirmation_decision_from_intent(result, confirmation, change)
            self.d.log_action_handler_result(decision)

            if decision["action"] == "refine_confirmation_change":
                change = decision["change"]
                self.d.pending_change = change
                continue

            if decision["action"] == "confirm_change":
                self.d.corrections_seen += 1
                self.d.handle_confirmed_mistake_related_change(change, self.d.current_turn_context)
                self.remember_confirmed_change_locally(change)
                self.prepare_tablet_change_reveal(change, self.d.current_turn_context)
                written = self.d.write_um_change(change)
                if self.d.current_turn_context:
                    self.d.phases_with_confirmed_change.add(self.d.current_turn_context.get("phase"))
                if written:
                    self.d.speech.say(self.successful_change_acknowledgement(change))
                    self.reveal_tablet_change_after_operator(change, self.d.current_turn_context)
                else:
                    self.clear_pending_tablet_reveal(self.d.current_turn_context)
                    self.d.speech.say("Dankjewel, ik heb dat genoteerd, maar opslaan lukte nu niet.")
                self.d.pending_change = None
                return True

            if decision["action"] == "reject_change":
                self.d.speech.say("Oké, dan verander ik niets.")
                self.d.pending_change = None
                return False

            self.d.speech.say(decision.get("leo_response") or self.confirmation_text(change))
