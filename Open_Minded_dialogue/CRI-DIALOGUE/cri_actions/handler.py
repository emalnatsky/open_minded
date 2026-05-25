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

from sic_framework.services.llm import GPTRequest

from cri_classifier import REPEAT_SENTINEL

logger = logging.getLogger(__name__)


class ActionHandler:
    """Routes classified intents into Leo's next action."""

    def __init__(self, dialogue):
        self.d = dialogue

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

    def llm_response(self, child_input: str) -> str:
        """L3: GPT generates a personalised Dutch follow-up."""
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
        dialogue_answer_corrects_mistake = (
            intent == "dialogue_answer"
            and turn.get("response_mode") == "mistake_interpretation"
            and turn.get("mistake_field")
            and mistake_state.get("wrong_value_rejected")
            and self.d.is_known(result.value)
        )
        if intent not in ("um_add", "um_update", "dialogue_update", "um_delete") and not dialogue_answer_corrects_mistake:
            return {}

        allowed = self.allowed_change_fields(turn)
        context = self.turn_memory_context(turn)
        field = result.field
        value = result.value

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
        if self.d.is_known(old_value) and str(old_value).strip().lower() == str(value).strip().lower():
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

    def is_confirmation_yes(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        yes_phrases = (
            "ja", "jawel", "zeker", "klopt", "dat klopt", "goed",
            "is goed", "dat is goed", "doe dat maar", "alsjeblieft",
            "prima", "ok", "oke", "oké", "mag",
        )
        return any(phrase in text for phrase in yes_phrases)

    def is_confirmation_no(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        no_phrases = ("nee", "niet", "klopt niet", "laat maar", "verander niets")
        return any(phrase in text for phrase in no_phrases)

    def confirmation_decision_from_intent(self, result, transcript: str, change: dict) -> dict:
        refined_change = self.change_from_intent_result(
            result,
            {
                "topic": {
                    "fields": [change.get("field")],
                    "field_labels": {change.get("field"): change.get("field_label")},
                    "current_values": {change.get("field"): change.get("old_value")},
                }
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

    # ── action handler (the routing brain) ───────────────────────────────────

    def action_handler(self, result, transcript: str, turn: dict) -> dict:
        intent = result.intent
        mode = turn.get("response_mode")
        change = self.change_from_intent_result(result, turn, transcript)

        if change:
            if change.get("action") == "already_correct":
                self.d.corrections_seen += 1
                self.d.mark_current_mistake_corrected()
                self.d.phases_with_confirmed_change.add(self.d.turn_phase(turn))
                response = "O ja, dankjewel."
                self.d.speech.say(response)
                return self.action_result(
                    "mistake_corrected_no_um_change",
                    True,
                    "Child corrected deliberate mistake by restating the already-correct UM value.",
                    change=change,
                    leo_response=response,
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
            if condition == "C2":
                response = "Je kunt mijn geheugenboek op de tablet bekijken."
                self.d.speech.say(response)
                return self.action_result(
                    "memory_access_tablet",
                    True,
                    "C2 memory access: child is redirected to the tablet memory book.",
                    leo_response=response,
                    tutorial_condition=condition,
                    requested_field=result.field,
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
                self.d.mistake_correction_question(turn)
                if mode == "mistake_interpretation"
                else self.d.topic_correction_question(turn.get("topic", {}))
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
            response = self.llm_response(transcript) if turn.get("llm_turn") and transcript else turn.get("follow_up")
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
