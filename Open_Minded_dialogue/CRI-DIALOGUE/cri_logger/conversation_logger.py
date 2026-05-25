"""
ConversationLogger — owns the conversation-log lifecycle for one session.

Constructed once in CRI_ScriptedDialogue.__init__ with a back-reference to
the dialogue instance:

    self.conv_log = ConversationLogger(self)

The dialogue keeps thin pass-through wrappers (self.log_conversation_event,
self.start_conversation_log, ...) so existing call sites don't change.

This class READS many fields from the dialogue (CHILD_ID, last_um_preview,
mistake_states, current_turn_context, etc.) and WRITES the log-bookkeeping
fields back to it (conversation_log, current_turn_log, the monotonic offsets).

That two-way binding is intentional: those fields are mutated from many
other dialogue methods (action_handler, run_phase, write_um_change), so
ownership has to stay on the dialogue.  The logger is just a wrapper.
"""

import os
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Manages the JSON + text conversation log for a single session."""

    def __init__(self, dialogue):
        # Back-reference. `self.d` is shorter than `self.dialogue` and reads
        # well at every call site below (`self.d.CHILD_ID`, `self.d.last_um_preview`).
        self.d = dialogue

    # ── timestamps & filenames ───────────────────────────────────────────────

    def log_timestamp(self) -> float:
        start = getattr(self.d, "conversation_log_started_monotonic", None)
        offset = float(getattr(self.d, "conversation_log_time_offset", 0.0) or 0.0)
        if start is None:
            return round(offset, 3)
        return round(offset + max(0.0, time.monotonic() - start), 3)

    def format_log_timestamp(self, timestamp) -> str:
        try:
            total_seconds = max(0, int(float(timestamp)))
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes:02d}:{seconds:02d}"
        except (TypeError, ValueError):
            return "00:00"

    def safe_filename_part(self, value: str) -> str:
        clean = str(value or "").strip()
        safe = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in clean
        )
        return safe.strip("_") or "child"

    def conversation_child_name(self) -> str:
        local_name = str(getattr(self.d, "local_child_name", "") or "").strip()
        if local_name:
            return local_name
        return (
            self.d.known(self.d.last_um_preview, "name")
            or self.d.known(self.d.last_um_preview, "child_name")
            or self.d.CHILD_ID
        )

    def conversation_session_id(self, child_name: str, started: datetime) -> str:
        child_part = self.safe_filename_part(child_name)
        child_id_part = self.safe_filename_part(self.d.CHILD_ID)
        readable_time = started.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{child_part}_{child_id_part}_{readable_time}"

    # ── turn snapshots & runtime state ───────────────────────────────────────

    def planned_turn_log(self, turn: dict) -> dict:
        entry = {
            "phase": self.d.turn_phase(turn),
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "dialogue_case": self.d.dialogue_case(turn),
            "runtime_llm": self.d.requires_runtime_llm(turn),
            "content_plan": self.d.content_plan_log(turn.get("content_plan")),
            "leo_text": self.d.turn_text(turn),
            "expects_response": turn.get("expects_response", True),
            "response_mode": turn.get("response_mode"),
            "condition": turn.get("condition"),
            "condition_label": turn.get("condition_label"),
            "tutorial_condition": turn.get("tutorial_condition"),
            "used_fields": turn.get("used_fields", {}),
        }
        if turn.get("segments"):
            entry["segments"] = [
                {
                    "index": index + 1,
                    "layer": segment.get("layer", turn.get("layer")),
                    "dialogue_case": self.d.dialogue_case({**turn, **segment}),
                    "runtime_llm": self.d.requires_runtime_llm({**turn, **segment}),
                    "content_plan": self.d.content_plan_log(segment.get("content_plan")),
                    "leo_text": self.d.turn_text({**turn, **segment}),
                    "expects_response": segment.get("expects_response", True),
                    "response_mode": segment.get("response_mode", turn.get("response_mode")),
                }
                for index, segment in enumerate(turn["segments"])
            ]
        if turn.get("topic"):
            entry["topic"] = turn["topic"]
        if turn.get("mistake_topic"):
            entry["mistake"] = {
                "id": turn.get("mistake_id"),
                "type": turn.get("mistake_type"),
                "topic": turn.get("mistake_topic"),
                "field": turn.get("mistake_field"),
                "actual": turn.get("mistake_actual"),
                "wrong": turn.get("mistake_wrong"),
            }
        return entry

    def runtime_state_snapshot(self) -> dict:
        return {
            "mistakes_mentioned": self.d.mistakes_mentioned,
            "corrections_seen": self.d.corrections_seen,
            "mistake_states": dict(self.d.mistake_states),
            "phases_with_confirmed_change": sorted(self.d.phases_with_confirmed_change),
            "memory_fields_mentioned_so_far": sorted(self.d.memory_fields_mentioned_so_far),
        }

    def sync_runtime_state_to_log(self):
        if not getattr(self.d, "conversation_log", None):
            return
        self.d.conversation_log.update(self.runtime_state_snapshot())

    def max_conversation_timestamp(self, log: dict) -> float:
        """Find the last timestamp from a previous log so resumed logs can continue counting upward."""
        timestamps = []
        for key in ("started_at", "ended_at"):
            value = log.get(key)
            if isinstance(value, (int, float)):
                timestamps.append(float(value))

        for event in log.get("events") or []:
            value = event.get("timestamp") if isinstance(event, dict) else None
            if isinstance(value, (int, float)):
                timestamps.append(float(value))

        for turn in log.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            for key in ("started_at", "ended_at"):
                value = turn.get(key)
                if isinstance(value, (int, float)):
                    timestamps.append(float(value))

        return max(timestamps) if timestamps else 0.0

    # ── conversation lifecycle ───────────────────────────────────────────────

    def start_conversation_log(self, script: list):
        if not self.d.CONVERSATION_LOG_ENABLED:
            return
        started = datetime.now().astimezone()
        self.d.conversation_log_started_monotonic = time.monotonic()
        previous_log = self.d.resume_source_log if self.d.resume_from_log_path else {}
        previous_events = (previous_log.get("events") if isinstance(previous_log, dict) else None) or []
        previous_turns = (previous_log.get("turns") if isinstance(previous_log, dict) else None) or []
        previous_last_timestamp = self.max_conversation_timestamp(previous_log) if isinstance(previous_log, dict) else 0.0
        self.d.conversation_log_time_offset = previous_last_timestamp + 0.001 if previous_events or previous_turns else 0.0

        import copy
        events_seed = copy.deepcopy(previous_events)
        turns_seed = copy.deepcopy(previous_turns)

        child_name = self.conversation_child_name()
        session_id = self.conversation_session_id(child_name, started)
        self.d.conversation_log = {
            "session_id": session_id,
            "child_id": self.d.CHILD_ID,
            "child_name": child_name,
            "researcher_name": self.d.researcher_name,
            "tutorial_condition": self.d.tutorial_condition(self.d.last_um_preview),
            "session_config": dict(self.d.session_config or {}),
            "script_version": self.d.SCRIPT_VERSION,
            "started_at": started.isoformat(timespec="seconds"),
            "input_mode": self.d.child_input_mode,
            "resume_from_log": self.d.resume_from_log_path,
            "start_phase_index": self.d.start_phase_index,
            "total_phases": self.d.TOTAL_SCRIPT_PHASES,
            "script_plan": [self.planned_turn_log(turn) for turn in script],
            "events": events_seed,
            "turns": turns_seed,
            "ended_at": None,
            "last_completed_phase": previous_log.get("last_completed_phase", 0) if isinstance(previous_log, dict) else 0,
            "session_status": "in_progress",
            "previous_log_included": bool(previous_events or previous_turns),
        }
        os.makedirs(self.d.CONVERSATION_LOG_ROOT, exist_ok=True)
        json_path = os.path.join(self.d.CONVERSATION_LOG_ROOT, f"{session_id}.json")
        text_path = os.path.join(self.d.CONVERSATION_LOG_ROOT, f"{session_id}.txt")
        self.d.conversation_log["json_path"] = json_path
        self.d.conversation_log["text_path"] = text_path
        self.d.conversation_log.update(self.runtime_state_snapshot())
        self.d.current_turn_log = None
        if self.d.resume_from_log_path:
            self.log_conversation_event(
                "session_resumed",
                from_log=self.d.resume_from_log_path,
                resume_phase=self.d.start_phase_index + 1,
            )
        self.write_conversation_logs()

    def finish_conversation_log(self):
        if not getattr(self.d, "conversation_log", None):
            return
        self.d.conversation_log["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.d.conversation_log["session_status"] = "completed"
        self.sync_runtime_state_to_log()
        self.write_conversation_logs()

    # ── per-turn ─────────────────────────────────────────────────────────────

    def start_turn_log(self, turn: dict):
        if not getattr(self.d, "conversation_log", None):
            return
        phase = self.d.turn_phase(turn)
        self.d.current_turn_log = {
            "phase": phase,
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "dialogue_case": self.d.dialogue_case(turn),
            "leo_text": self.d.turn_text(turn),
            "started_at": self.log_timestamp(),
            "ended_at": None,
            "utterances": [],
            "events": [],
            "tutorial_condition": turn.get("tutorial_condition"),
        }
        self.d.conversation_log.setdefault("turns", []).append(self.d.current_turn_log)
        self.log_conversation_event(
            "phase_start",
            phase=phase,
            name=turn.get("name"),
            layer=turn.get("layer"),
            dialogue_case=self.d.dialogue_case(turn),
        )

    def finish_turn_log(self):
        if not getattr(self.d, "conversation_log", None) or not self.d.current_turn_log:
            return
        phase = self.d.current_turn_log.get("phase")
        self.d.current_turn_log["ended_at"] = self.log_timestamp()
        self.log_conversation_event("phase_end", phase=phase)
        if phase:
            self.d.conversation_log["last_completed_phase"] = max(
                int(self.d.conversation_log.get("last_completed_phase") or 0),
                int(phase),
            )
        self.sync_runtime_state_to_log()
        self.write_conversation_logs()
        self.d.current_turn_log = None

    # ── event recorder (the hot path) ────────────────────────────────────────

    def log_conversation_event(self, event_type: str, **data):
        if not getattr(self.d, "conversation_log", None):
            return
        event = {"type": event_type, "timestamp": self.log_timestamp()}
        for key, value in data.items():
            if value is not None:
                event[key] = value
        self.d.conversation_log["events"].append(event)
        if self.d.current_turn_log is not None:
            if event_type == "utterance":
                self.d.current_turn_log["utterances"].append({
                    "speaker": event.get("speaker"),
                    "text": event.get("text"),
                    "timestamp": event["timestamp"],
                    **{
                        key: value
                        for key, value in event.items()
                        if key not in {"type", "speaker", "text", "timestamp"}
                    },
                })
            else:
                self.d.current_turn_log["events"].append(event)

    # ── rendering & disk writes ──────────────────────────────────────────────

    def render_conversation_text(self) -> str:
        if not getattr(self.d, "conversation_log", None):
            return ""
        log = self.d.conversation_log
        lines = []
        lines.append(f"Session: {log.get('session_id')}")
        lines.append(f"Started: {log.get('started_at')}")
        lines.append(f"Child:   {log.get('child_name')} ({log.get('child_id')})")
        lines.append(f"Researcher: {log.get('researcher_name')}")
        lines.append(f"Condition:  {log.get('tutorial_condition')}")
        lines.append(f"Input mode: {log.get('input_mode')}")
        lines.append("")
        lines.append("=" * 72)
        lines.append("SCRIPT PLAN")
        lines.append("=" * 72)
        for plan in log.get("script_plan") or []:
            phase = plan.get("phase")
            name = plan.get("name")
            case = plan.get("dialogue_case")
            runtime = plan.get("runtime_llm")
            leo = plan.get("leo_text") or "(content plan only)"
            content_plan = plan.get("content_plan")
            lines.append(f"Phase {phase}: {name} [{case}] runtime_llm={runtime}")
            lines.append(f"  Leo: {leo}")
            if content_plan:
                if isinstance(content_plan, dict):
                    case_label = content_plan.get("case")
                    extras = []
                    if "sources" in content_plan:
                        extras.append("sources=" + ", ".join(content_plan.get("sources") or []))
                    if "template" in content_plan and not content_plan.get("sources"):
                        extras.append("template=" + str(content_plan.get("template")))
                    if "branch" in content_plan:
                        extras.append("branch=" + str(content_plan.get("branch")))
                    detail = " | ".join(extras)
                    lines.append(f"  Plan: {case_label}{(' | ' + detail) if detail else ''}")
                    sequence = content_plan.get("sequence") or []
                    for step in sequence:
                        if not isinstance(step, dict):
                            continue
                        kind = step.get("kind", "?")
                        if kind == "text":
                            preview = (step.get("text") or "").strip().replace("\n", " ")
                            if len(preview) > 80:
                                preview = preview[:77] + "..."
                            lines.append(f"    - text: {preview}")
                        elif kind == "template":
                            extras = []
                            if step.get("template"):
                                extras.append("template=" + str(step["template"]))
                            if step.get("fields"):
                                extras.append("fields=" + ", ".join(step["fields"]))
                            lines.append(f"    - template: {' | '.join(extras) if extras else '(unspecified)'}")
                        elif kind == "preauthored":
                            extras = []
                            if step.get("pool"):
                                extras.append("pool=" + str(step["pool"]))
                            lines.append(f"    - preauthored: {' | '.join(extras) if extras else '(unspecified)'}")
                        elif kind == "llm_pregenerated":
                            extras = []
                            if step.get("source"):
                                extras.append("source=" + str(step["source"]))
                            lines.append(f"    - llm_pregenerated: {' | '.join(extras) if extras else '(unspecified)'}")
                        elif kind == "llm_runtime":
                            extras = []
                            if step.get("branch"):
                                extras.append("branch=" + str(step["branch"]))
                            lines.append(f"    - llm_runtime: {' | '.join(extras) if extras else '(unspecified)'}")
                        else:
                            lines.append(f"    - {kind}")
                else:
                    lines.append(f"  Plan: {content_plan}")
            tutorial = plan.get("tutorial_condition")
            if tutorial:
                lines.append(f"  Tutorial: {tutorial}")
            topic = plan.get("topic")
            if isinstance(topic, dict):
                lines.append(
                    f"  Topic: domain={topic.get('domain')} primary_field={topic.get('primary_field')}"
                )
            mistake = plan.get("mistake")
            if mistake:
                lines.append(
                    f"  Mistake: {mistake.get('field')} -> wrong={mistake.get('wrong')} actual={mistake.get('actual')}"
                )
            for segment in plan.get("segments") or []:
                lines.append(
                    f"    Seg {segment.get('index')}: [{segment.get('dialogue_case')}] runtime_llm={segment.get('runtime_llm')}"
                )
                seg_text = segment.get("leo_text") or "(content plan only)"
                lines.append(f"      Leo: {seg_text}")
                seg_plan = segment.get("content_plan")
                if isinstance(seg_plan, dict):
                    case_label = seg_plan.get("case")
                    extras = []
                    if "sources" in seg_plan:
                        extras.append("sources=" + ", ".join(seg_plan.get("sources") or []))
                    if "template" in seg_plan and not seg_plan.get("sources"):
                        extras.append("template=" + str(seg_plan.get("template")))
                    if "branch" in seg_plan:
                        extras.append("branch=" + str(seg_plan.get("branch")))
                    detail = " | ".join(extras)
                    lines.append(f"      Plan: {case_label}{(' | ' + detail) if detail else ''}")
            lines.append("")

        lines.append("=" * 72)
        lines.append("CONVERSATION")
        lines.append("=" * 72)
        for turn in log.get("turns") or []:
            phase = turn.get("phase")
            name = turn.get("name")
            case = turn.get("dialogue_case")
            lines.append("")
            lines.append(f"Phase {phase}: {name} [{case}]")
            for utterance in turn.get("utterances") or []:
                speaker = utterance.get("speaker", "").upper() or "?"
                text = utterance.get("text", "")
                lines.append(f"  {speaker}: {text}")
            for event in turn.get("events") or []:
                event_type = event.get("type")
                if event_type == "stt_error":
                    lines.append(f"  [stt_error: {event.get('error')}]")
                elif event_type == "transcript_review":
                    lines.append(f"  [review:{event.get('action')} {event.get('transcript')}]")
                elif event_type == "researcher_choice":
                    lines.append(f"  [research:{event.get('choice')}]")
                elif event_type == "tutorial_chosen":
                    lines.append(f"  [tutorial:{event.get('condition')}]")
                elif event_type == "llm_decision":
                    lines.append(
                        f"  [llm:{event.get('mode')} -> {event.get('intent')}/{event.get('field')}]"
                    )
        if log.get("ended_at"):
            lines.append("")
            lines.append(f"Ended at: {log['ended_at']}")
        return "\n".join(lines)

    def write_conversation_logs(self):
        if not getattr(self.d, "conversation_log", None):
            return
        log = self.d.conversation_log
        json_path = log.get("json_path")
        text_path = log.get("text_path")
        if not json_path or not text_path:
            return
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(log, json_file, ensure_ascii=False, indent=2)
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(self.render_conversation_text())

    # ── per-turn structured event helpers ────────────────────────────────────

    def log_llm_decision(self, mode: str, transcript: str, result: dict, context: dict = None):
        payload = {
            "mode": mode,
            "transcript": transcript or "(nothing)",
        }
        if isinstance(result, dict):
            payload["intent"] = result.get("intent")
            payload["field"] = result.get("field")
            payload["value"] = result.get("value")
            payload["confidence"] = result.get("confidence")
        if context:
            payload["context"] = context
        self.log_conversation_event("llm_decision", **payload)

    def intent_result_to_dict(self, result) -> dict:
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {
            "intent": getattr(result, "intent", None),
            "field": getattr(result, "field", None),
            "value": getattr(result, "value", None),
            "confidence": getattr(result, "confidence", None),
        }

    def log_intent_classifier_result(self, transcript: str, result):
        self.log_conversation_event(
            "intent_classifier",
            transcript=transcript or "(nothing)",
            result=self.intent_result_to_dict(result),
            phase=(self.d.current_turn_context or {}).get("phase"),
            name=(self.d.current_turn_context or {}).get("name"),
        )

    def log_action_handler_result(self, action_result: dict):
        change = action_result.get("change") if isinstance(action_result.get("change"), dict) else {}
        if change:
            logger.info(
                "ActionHandler: %s -> %s[%s] %s -> %s",
                action_result.get("action"),
                change.get("action"),
                change.get("field"),
                change.get("old_value"),
                change.get("new_value"),
            )
        else:
            logger.info("ActionHandler: %s", action_result.get("action"))
        self.log_conversation_event(
            "action_handler",
            phase=(self.d.current_turn_context or {}).get("phase"),
            name=(self.d.current_turn_context or {}).get("name"),
            **action_result,
        )
