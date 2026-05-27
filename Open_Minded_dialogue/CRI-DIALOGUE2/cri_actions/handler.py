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

from sic_framework.services.llm import GPTRequest

from .l3_runtime import L3Runtime
from cri_classifier import REPEAT_SENTINEL

logger = logging.getLogger(__name__)


class ActionHandler:
    """Routes classified intents into Leo's next action."""

    def __init__(self, dialogue):
        self.d = dialogue
        self.l3 = L3Runtime(dialogue)

    # ── classifier helpers ───────────────────────────────────────────────────

    def classify_with_repeat(self, transcript: str):
        """Classify once, ask for repetition on low confidence, then retry."""
        result = self.d.clf.classify(transcript)
        if result.intent == REPEAT_SENTINEL:
            self.d.logger.info("Low confidence - asking to repeat.")
            self.d.speech.say("Kun je dat nog een keer zeggen?")
            time.sleep(0.8)
            transcript = self.d.speech.listen_with_review()
            result = self.d.clf.classify_retry(transcript)
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

        new_value = change.get("new_value")
        return f"Wil je dat ik {change['field_label']} verander naar {new_value}?"

    # ── turn helpers / action_result builder ─────────────────────────────────

    def turn_memory_context(self, turn: dict) -> dict:
        topic = turn.get("topic") or turn.get("mistake_topic") or {}
        return {
            "topic": topic,
            "fields": topic.get("fields", []) or [],
            "field_labels": topic.get("field_labels", {}) or {},
            "current_values": topic.get("current_values", {}) or {},
        }

    def allowed_change_fields(self, turn: dict) -> list:
        context = self.turn_memory_context(turn)
        fields = list(context.get("fields") or [])
        if turn.get("mistake_field") and turn["mistake_field"] not in fields:
            fields.append(turn["mistake_field"])
        return fields or list(self.d.UM_FIELDS)

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

    def turn_allows_memory_change(self, turn: dict) -> bool:
        if turn.get("allow_memory_change"):
            return True
        mode = turn.get("response_mode")
        if mode in ("mistake_interpretation", "memory_review_group", "memory_review_add_final"):
            return True
        if mode == "nudge_interpretation" and turn.get("nudge_correction_requested"):
            return True
        return False

    def classifier_value_is_in_transcript(self, field: str, value: str, transcript: str) -> bool:
        value_norm = self.normalized_field_value(field, value)
        transcript_norm = self.normalized_field_value(field, transcript)
        if not value_norm:
            return False
        return value_norm in transcript_norm

    def repair_value_from_short_answer(self, result, transcript: str, turn: dict, mistake_state: dict) -> str:
        """Extract a correction value from terse answers like 'pannenkoeken' or 'nee, voetbal'."""
        if result.intent != "dialogue_answer" or turn.get("response_mode") != "mistake_interpretation":
            return result.value
        if not turn.get("mistake_field"):
            return result.value
        if self.d.is_known(result.value):
            return result.value

        raw = str(transcript or "").strip()
        if not raw:
            return result.value

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
            return result.value
        if any(phrase in lower for phrase in value_less_rejection_phrases) and not any(
            marker in lower for marker in ("maar", "het is", "dat is eigenlijk", "namelijk")
        ):
            return result.value
        if not mistake_state.get("wrong_value_rejected") and not had_inline_rejection:
            return result.value
        if len(candidate.split()) > 8:
            return result.value
        return candidate

    def correction_question_for_field(self, field: str, field_label: str = "") -> str:
        if field == "hobby_fav":
            return "Oeps, wat is dan je favoriete hobby?"
        if field == "fav_food":
            return "Oeps, wat is dan je lievelingseten?"
        if field == "school_strength":
            return "Oeps, waar ben jij dan vooral goed in op school?"
        if field == "aspiration":
            return "Oeps, wat wil jij dan later worden?"
        label = field_label or self.d.field_label(field)
        if label.startswith("je "):
            return f"Oeps, wat is dan {label}?"
        return f"Oeps, wat moet ik dan onthouden over {label}?"

    def mistake_correction_question(self, turn: dict) -> str:
        context = self.turn_memory_context(turn)
        field = turn.get("mistake_field")
        field_label = context.get("field_labels", {}).get(field) or self.d.field_label(field)
        return self.correction_question_for_field(field, field_label)

    def topic_correction_question(self, topic: dict) -> str:
        fields = topic.get("fields") or []
        field = fields[0] if len(fields) == 1 else ""
        field_label = (topic.get("field_labels") or {}).get(field) or topic.get("label") or "mijn geheugen"
        if field:
            return self.correction_question_for_field(field, field_label)
        return "Oeps, wat klopt er dan niet?"

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

    def change_from_intent_result(self, result, turn: dict, transcript: str) -> dict:
        intent = result.intent
        mistake_state = self.d.mistake_states.get(turn.get("mistake_id"), {})
        repaired_answer_value = self.repair_value_from_short_answer(result, transcript, turn, mistake_state)
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
        if intent not in ("um_add", "um_update", "dialogue_update", "um_delete") and not dialogue_answer_corrects_mistake:
            return {}
        if not self.turn_allows_memory_change(turn):
            return {}

        allowed = self.allowed_change_fields(turn)
        context = self.turn_memory_context(turn)
        field = result.field
        value = repaired_answer_value

        if not field and turn.get("mistake_field") and (
            intent in ("um_add", "um_update", "dialogue_update") or dialogue_answer_corrects_mistake
        ):
            field = turn.get("mistake_field")
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

        if (
            turn.get("response_mode") == "mistake_interpretation"
            and turn.get("mistake_id")
            and turn.get("mistake_field") == field
            and mistake_state.get("wrong_value_rejected")
            and not self.field_values_match(field, turn.get("mistake_wrong"), value)
            and not (self.d.is_known(old_value) and self.field_values_match(field, old_value, value))
            and len(str(transcript or "").split()) <= 8
        ):
            return {
                "action": "update",
                "field": field,
                "field_label": field_label,
                "old_value": str(old_value),
                "new_value": str(value),
                "confidence": result.confidence,
                "reason": "Child answered Leo's explicit deliberate-mistake correction question.",
                "source_text": transcript,
                "direct_mistake_correction": True,
            }

        if self.d.is_known(old_value) and self.field_values_match(field, old_value, value):
            if turn.get("mistake_id") and (
                result.intent in ("um_add", "um_update", "dialogue_update") or dialogue_answer_corrects_mistake
            ):
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
            return {}

        return {
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

    # ── confirmation intent predicates ───────────────────────────────────────

    def is_rejection_without_value(self, result, transcript: str) -> bool:
        text = str(transcript or "").lower()
        rejection_words = ("klopt niet", "niet waar", "verkeerd", "helemaal niet", "nee")
        if result.intent in ("um_update", "dialogue_update") and not self.d.is_known(result.value):
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
            matched_state["corrected"] = True
            matched_state["corrected_at_phase"] = turn.get("phase")
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
        nervous_words = ("spannend", "zenuw", "eng", "bang", "nerveus", "lastig")
        looking_forward_words = ("zin", "leuk", "blij", "gezellig", "kijk ernaar uit")
        mixed_words = ("allebei", "beetje van allebei", "ook spannend", "maar ook")

        has_not_thought = any(word in text for word in not_thought_words)
        has_nervous = any(word in text for word in nervous_words)
        has_looking_forward = any(word in text for word in looking_forward_words)
        has_mixed = any(word in text for word in mixed_words)

        if has_mixed or (has_nervous and has_looking_forward):
            return "mixed"
        if has_not_thought:
            return "not_thought_about_it"
        if has_nervous:
            return "nervous"
        if has_looking_forward:
            return "looking_forward"
        return "unknown"

    def middle_school_feeling_action(self, transcript: str, turn: dict) -> dict:
        feeling = self.middle_school_feeling(transcript)
        self.d.log_conversation_event(
            "middle_school_feeling",
            transcript=transcript,
            middle_school_feeling=feeling,
            phase=turn.get("phase"),
            phase_id=turn.get("phase_id"),
        )
        return self.action_result(
            "middle_school_feeling",
            True,
            "Child response about middle school was categorized for session logging only.",
            middle_school_feeling=feeling,
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
        response = (
            "Goed. Dan kunnen we zo samen op de tablet naar mijn geheugenboek kijken."
            if condition == self.d.CONDITION_EXPERIMENT
            else "Goed. Dan gaan we zo rustig samen door mijn geheugen heen."
        )
        self.d.speech.say(response)
        return self.action_result(
            "explicit_memory_inspection_accepted",
            True,
            "Child accepted explicit memory inspection; phase 3.7 will perform the review.",
            leo_response=response,
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
        change = self.change_from_intent_result(result, turn, transcript)

        if change:
            if change.get("action") == "already_correct":
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))
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

            if change.get("direct_mistake_correction"):
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))
                written = self.d.write_um_change(change)
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

            self.confirm_topic_change(change)
            return self.action_result(
                f"confirm_{change['action']}",
                True,
                "UM-changing intent from classifier.",
                change=change,
            )

        if intent == "um_inspect":
            condition = self.d.tutorial_condition(self.d.last_um_preview)
            if condition == self.d.CONDITION_EXPERIMENT:
                fields = self.d.memory_access_scope(turn)
                self.activate_tablet_memory_access(fields, turn)
                response = "Je kunt mijn geheugenboek op de tablet bekijken."
                self.d.speech.say(response)
                return self.action_result(
                    "memory_access_tablet",
                    True,
                    "Experiment condition memory access: child is redirected to the tablet memory book.",
                    leo_response=response,
                    tutorial_condition=condition,
                    requested_field=result.field,
                    visible_fields=list(fields),
                )

            response, memory_scope, returned_fields = self.d.memory_access_response(result, turn)
            self.d.speech.say(response)
            return self.action_result(
                "memory_access",
                True,
                "Child requested memory access; returned scoped memory only.",
                leo_response=response,
                memory_scope=memory_scope,
                returned_fields=returned_fields,
                requested_field=result.field,
                tutorial_condition=condition,
            )

        if mode in ("mistake_interpretation", "topic_interpretation") and self.is_rejection_without_value(result, transcript):
            response = (
                self.mistake_correction_question(turn)
                if mode == "mistake_interpretation"
                else self.topic_correction_question(turn.get("topic", {}))
            )
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
            response = self.d.topic_no_update_response(turn.get("topic", {}))
            self.d.speech.say(response)
            return self.action_result(
                "no_memory_change",
                True,
                "No UM-changing intent detected for topic phase.",
                leo_response=response,
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
            result = self.classify_with_repeat(transcript)
            action = self.action_handler(result, transcript, turn)
            self.d.log_action_handler_result(action)
            if not action.get("follow_up_needed"):
                return action
        return action

    # ── confirmation loop ────────────────────────────────────────────────────

    def confirm_topic_change(self, change: dict) -> bool:
        self.d.pending_change = change

        while True:
            self.d.speech.say(self.confirmation_text(change))
            time.sleep(0.5)

            confirmation = self.d.speech.listen_with_review()
            time.sleep(0.8)

            result = self.classify_with_repeat(confirmation)
            decision = self.confirmation_decision_from_intent(result, confirmation, change)
            self.d.log_action_handler_result(decision)

            if decision["action"] == "refine_confirmation_change":
                change = decision["change"]
                self.d.pending_change = change
                continue

            if decision["action"] == "confirm_change":
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                written = self.d.write_um_change(change)
                if self.d.current_turn_context:
                    self.d.phases_with_confirmed_change.add(self.d.current_turn_context.get("phase"))
                if written:
                    self.d.speech.say("Dankjewel, ik heb dat aangepast.")
                else:
                    self.d.speech.say("Dankjewel, ik heb dat genoteerd, maar opslaan lukte nu niet.")
                self.d.pending_change = None
                return True

            if decision["action"] == "reject_change":
                self.d.speech.say("Oké, dan verander ik niets.")
                self.d.pending_change = None
                return False

            self.d.speech.say(decision.get("leo_response") or self.confirmation_text(change))
