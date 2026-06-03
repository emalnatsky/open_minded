"""
NudgeManager — tracks deliberate-mistake state and produces nudge utterances.

Constructed once in CRI_ScriptedDialogue.__init__:

    self.nudge = NudgeManager(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.register_mistake_phase, self.mistake_nudge_action, ...) stay identical.

What lives here:
  - State machine: register_mistake_phase, mark_current_mistake_corrected
  - Lookups: mistake_field_label, first_uncorrected_mistake_state,
    nudge_state_for_turn
  - Utterances: gentle_mistake_nudge_text, explicit_mistake_nudge_text
  - The action: mistake_nudge_action — chooses gentle vs explicit, calls
    speech.say, returns an action_result dict

This module READS:
  - Dialogue state: mistake_states, current_turn_context
  - Methods on self.d: field_label, action_result, speech.say

It WRITES self.d.mistake_states (the state machine) on each call.

Cross-package: this module is intentionally side-effecting (it speaks
through self.d.speech and mutates self.d.mistake_states). The action
handler in handler.py calls self.d.mistake_nudge_action(...) when no
correction or memory-access intent was detected.
"""

import logging

logger = logging.getLogger(__name__)


class NudgeManager:
    """Mistake state + nudge text + nudge action."""

    def __init__(self, dialogue):
        self.d = dialogue

    # ── state machine ────────────────────────────────────────────────────────

    def register_mistake_phase(self, turn: dict):
        """Remember that a deliberate mistake has been stated in this phase."""
        mistake_id = turn.get("mistake_id")
        if not mistake_id:
            return
        state = self.d.mistake_states.setdefault(mistake_id, {})
        state.update({
            "id": mistake_id,
            "mentioned": True,
            "corrected": state.get("corrected", False),
            "nudge_count": state.get("nudge_count", 0),
            "phase": turn.get("phase"),
            "field": turn.get("mistake_field"),
            "field_label": self.mistake_field_label(turn),
            "actual": turn.get("mistake_actual"),
            "wrong": turn.get("mistake_wrong"),
            "type": turn.get("mistake_type"),
            "m3_requires_school_difficulty_resolution": bool(
                turn.get("m3_requires_school_difficulty_resolution")
            ),
        })

    def mark_current_mistake_corrected(self):
        """Mark the current deliberate mistake corrected after a confirmed UM change."""
        context = self.d.current_turn_context or {}
        mistake_id = context.get("mistake_id")
        if not mistake_id:
            return
        state = self.d.mistake_states.setdefault(mistake_id, {"id": mistake_id})
        state["corrected"] = True
        state["corrected_at_phase"] = context.get("phase")

    # ── lookups ──────────────────────────────────────────────────────────────

    def mistake_field_label(self, turn: dict) -> str:
        field = turn.get("mistake_field")
        topic = turn.get("mistake_topic") or {}
        return (topic.get("field_labels") or {}).get(field) or self.d.field_label(field)

    def first_uncorrected_mistake_state(self) -> dict:
        for mistake_id in sorted(self.d.mistake_states):
            state = self.d.mistake_states[mistake_id]
            if state.get("mentioned") and not state.get("corrected"):
                return state
        return {}

    def nudge_state_for_turn(self, turn: dict) -> dict:
        mistake_id = turn.get("mistake_id")
        if mistake_id:
            self.register_mistake_phase(turn)
            return self.d.mistake_states.get(mistake_id, {})
        return self.first_uncorrected_mistake_state()

    # ── utterances ───────────────────────────────────────────────────────────

    def gentle_mistake_nudge_text(self, state: dict) -> str:
        field_label = state.get("field_label") or "iets uit mijn geheugen"
        return (
            f"Ik zei net iets uit mijn geheugen over {field_label}. "
            "Als daar iets niet helemaal klopt, mag je mij gewoon verbeteren. "
            "Wat moet ik hierover onthouden?"
        )

    def explicit_mistake_nudge_text(self, state: dict) -> str:
        field_label = state.get("field_label") or "iets uit mijn geheugen"
        wrong = state.get("wrong") or "dat"
        return (
            f"Ik zei net dat {wrong} jouw {field_label} is. "
            "Dat was misschien niet goed. Wat moet ik daarover onthouden?"
        )

    # ── the action ───────────────────────────────────────────────────────────

    def mistake_nudge_action(self, turn: dict, force_explicit: bool = False) -> dict:
        """ActionHandler-owned nudge decision after no correction/memory access was detected."""
        state = self.nudge_state_for_turn(turn)
        if not state:
            return {}

        if force_explicit:
            level = 2
        else:
            level = int(state.get("nudge_count", 0)) + 1
            if level > 2:
                return {}

        state["nudge_count"] = max(level, int(state.get("nudge_count", 0)))
        response = (
            self.explicit_mistake_nudge_text(state)
            if level >= 2
            else self.gentle_mistake_nudge_text(state)
        )
        self.d.speech.say(response)
        return self.d.action_result(
            "explicit_mistake_nudge" if level >= 2 else "gentle_mistake_nudge",
            True,
            "Action handler triggered nudge because no correction or memory access intent was detected.",
            leo_response=response,
            follow_up_needed=True,
            nudge_level=level,
            mistake_id=state.get("id"),
            mistake_field=state.get("field"),
            mistake_wrong=state.get("wrong"),
        )
