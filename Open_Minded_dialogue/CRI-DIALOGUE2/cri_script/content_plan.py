"""
ContentPlan — layered utterance machinery for the CRI dialogue.

What this module does:
    - Builds typed content-plan dicts (L1 scripted text, L2-slot UM templates,
      L2-pregen stored utterances, sequences of these).
    - Renders a content plan into one final Dutch utterance Leo speaks.
    - Derives metadata about a turn (phase, dialogue-case label, whether the
      turn requires runtime LLM, a JSON-safe log shape).

What this module does NOT do:
    - It does not build the script (see builder.py).
    - It does not pull UM values from the API (it reads the already-pulled
      snapshot from self.d.last_um_preview).

Pattern:
    Constructed once in CRI_ScriptedDialogue.__init__ as self.cp.
    Pass-through wrappers on the dialogue keep self.l1, self.l2_slot,
    self.sequence, self.turn_text, self.render_content_plan, etc. working
    at every call site without change.
"""

import logging

logger = logging.getLogger(__name__)


class ContentPlan:
    """Layered utterance machinery (L1/L2-slot/L2-pregen + sequence + render)."""

    def __init__(self, dialogue):
        self.d = dialogue

    # ── content-plan constructors ────────────────────────────────────────────

    def l1(self, text: str) -> dict:
        return {"type": self.d.CASE_FULLY_SCRIPTED, "layer": "L1", "text": text}

    def l2_slot(self, template: str, values: dict = None, wrong: bool = False) -> dict:
        return {
            "type": self.d.CASE_UM_TEMPLATE,
            "layer": "L2-slot WRONG" if wrong else "L2-slot",
            "template": template,
            "values": values or {},
            "wrong_slot": bool(wrong),
        }

    def l2_pregen(
        self,
        key: str,
        fallback: str,
        input_fields: list = None,
        require_input_values: bool = False,
    ) -> dict:
        return {
            "type": self.d.CASE_LLM_PREGENERATED,
            "layer": "L2-pregen",
            "key": key,
            "input_fields": input_fields or [],
            "fallback": fallback,
            "require_input_values": bool(require_input_values),
        }

    def sequence(self, *parts) -> dict:
        return {"type": self.d.CONTENT_PLAN_SEQUENCE, "parts": [part for part in parts if part]}

    # ── rendering ────────────────────────────────────────────────────────────

    def render_template_text(self, template: str, values: dict = None) -> str:
        """Render an L2-slot template with UM values."""
        values = values or {}
        merged = dict(self.d.last_um_preview or {})
        merged.update(values)
        safe_values = {
            field: (value if self.d.is_known(value) else self.d.UNKNOWN_VALUE)
            for field, value in merged.items()
        }
        try:
            return template.format(**safe_values)
        except KeyError as e:
            missing = str(e).strip("'")
            safe_values[missing] = self.d.UNKNOWN_VALUE
            return template.format(**safe_values)

    def text_mentions_input_values(self, text: str, input_fields: list) -> bool:
        """True when a pregenerated utterance still matches the current UM values."""
        text_norm = str(text or "").lower()
        for field in input_fields or []:
            value = self.d.last_um_preview.get(field)
            if not self.d.is_known(value):
                continue
            value_norm = str(value).strip().lower()
            if value_norm and value_norm not in text_norm:
                return False
        return True

    def pregenerated_utterance(
        self,
        key: str,
        fallback: str = "",
        branch: str = "default",
        input_fields: list = None,
        require_input_values: bool = False,
    ) -> str:
        """Read an L2-pregen utterance from the CRI scenario or UM profile."""
        scenario_text = self.d.scenario_utterance(key, branch=branch, fallback="")
        if self.d.is_known(scenario_text) and (
            not require_input_values
            or self.text_mentions_input_values(scenario_text, input_fields or [])
        ):
            return scenario_text

        for prefix in self.d.PREGENERATED_UTTERANCE_PREFIXES:
            field = f"{prefix}{key}"
            value = self.d.last_um_preview.get(field)
            if self.d.is_known(value) and (
                not require_input_values
                or self.text_mentions_input_values(value, input_fields or [])
            ):
                return str(value)
        return self.render_template_text(fallback, {})

    def render_content_plan(self, plan, turn: dict = None) -> str:
        """Render explicit L1/L2/L2-pregen content plans into one Leo utterance."""
        turn = turn or {}
        if isinstance(plan, list):
            return " ".join(
                rendered
                for rendered in (self.render_content_plan(part, turn).strip() for part in plan)
                if rendered
            )

        plan_type = plan.get("type")

        if plan_type == self.d.CONTENT_PLAN_SEQUENCE:
            return " ".join(
                rendered
                for rendered in (self.render_content_plan(part, turn).strip() for part in plan.get("parts", []))
                if rendered
            )

        if plan_type == self.d.CASE_FULLY_SCRIPTED:
            return plan.get("text", "")

        if plan_type == self.d.CASE_UM_TEMPLATE:
            return self.render_template_text(plan.get("template", ""), plan.get("values", {}))

        if plan_type == self.d.CASE_LLM_PREGENERATED:
            return self.pregenerated_utterance(
                plan.get("key", ""),
                plan.get("fallback", ""),
                plan.get("branch", "default"),
                plan.get("input_fields", []),
                plan.get("require_input_values", False),
            )

        if plan_type == self.d.CASE_PREAUTHORED_POOL:
            return plan.get("text", turn.get("leo_text", ""))

        return turn.get("leo_text", "")

    # ── turn metadata ────────────────────────────────────────────────────────

    def turn_text(self, turn: dict) -> str:
        """Single source of truth for the scripted text Leo says for a phase/segment."""
        if turn.get("content_plan"):
            return self.render_content_plan(turn["content_plan"], turn)
        if turn.get("segments"):
            return " ".join(
                self.turn_text({**turn, **segment, "segments": []}).strip()
                for segment in turn["segments"]
                if self.turn_text({**turn, **segment, "segments": []}).strip()
            )
        return turn.get("leo_text", "")

    def turn_phase(self, turn: dict):
        """Return the dialogue phase number."""
        return turn.get("phase")

    def dialogue_case(self, turn: dict) -> str:
        """Explicit dialogue structure case from the content-layer reference."""
        if turn.get("dialogue_case"):
            return turn["dialogue_case"]
        if turn.get("segments"):
            return self.d.CASE_MIXED_SEQUENCE
        plan = turn.get("content_plan") or {}
        if isinstance(plan, list):
            return self.d.CASE_MIXED_SEQUENCE
        if isinstance(plan, dict) and plan.get("type"):
            plan_type = plan["type"]
            return self.d.CASE_MIXED_SEQUENCE if plan_type == self.d.CONTENT_PLAN_SEQUENCE else plan_type
        if turn.get("llm_turn"):
            return self.d.CASE_RUNTIME_LLM_BRANCH
        return self.d.CASE_FULLY_SCRIPTED

    def requires_runtime_llm(self, turn: dict) -> bool:
        """True when this phase has an L3 runtime branch after child input."""
        if turn.get("runtime_llm"):
            return True
        if turn.get("llm_turn"):
            return True
        if self.dialogue_case(turn) == self.d.CASE_RUNTIME_LLM_BRANCH:
            return True
        return any(segment.get("runtime_llm") or segment.get("llm_turn") for segment in turn.get("segments", []))

    def content_plan_log(self, plan):
        """Compact JSON-safe description of L1/L2-slot/L2-pregen/L3 structure."""
        if not plan:
            return None
        if isinstance(plan, list):
            return [self.content_plan_log(part) for part in plan]
        if not isinstance(plan, dict):
            return plan
        logged = {
            key: value
            for key, value in plan.items()
            if key not in ("text", "template", "fallback")
        }
        for text_key in ("text", "template", "fallback"):
            if text_key in plan:
                logged[text_key] = plan[text_key]
        return logged
