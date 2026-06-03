"""
ConversationLogger â€” owns the conversation-log lifecycle for one session.

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
import tempfile
import copy
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Manages the JSON + text conversation log for a single session."""

    def __init__(self, dialogue):
        # Back-reference. `self.d` is shorter than `self.dialogue` and reads
        # well at every call site below (`self.d.CHILD_ID`, `self.d.last_um_preview`).
        self.d = dialogue

    # â”€â”€ timestamps & filenames â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    def omr_timestamp(self, timestamp) -> str:
        return self.format_log_timestamp(timestamp)

    def omr_yes_no(self, value) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "ja", "true", "1"}:
                return "yes"
            if normalized in {"no", "nee", "false", "0"}:
                return "no"
        return "yes" if bool(value) else "no"

    def safe_filename_part(self, value: str) -> str:
        clean = str(value or "").strip()
        safe = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in clean
            if char.isprintable()
        )
        safe = safe.strip(" ._")
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }
        if not safe or safe.upper() in reserved:
            return "child"
        return safe[:80]

    def conversation_child_name(self) -> str:
        local_name = str(getattr(self.d, "local_child_name", "") or "").strip()
        if local_name:
            return local_name
        return (
            self.d.known(self.d.last_um_preview, "name")
            or self.d.known(self.d.last_um_preview, "child_name")
            or getattr(self.d, "CHILD_ID", "unknown")
        )

    def child_initial(self) -> str:
        name = self.conversation_child_name()
        for char in str(name or ""):
            if char.isalpha() or char.isdigit():
                return char.upper()
        return "X"

    def child_log_label(self) -> str:
        child_id = self.safe_filename_part(getattr(self.d, "CHILD_ID", "unknown"))
        return f"{self.child_initial()}_{child_id}"

    def private_child_names(self) -> list:
        candidates = [
            getattr(self.d, "local_child_name", ""),
            getattr(self.d, "local_child_name_cri", ""),
            getattr(self.d, "local_child_name_tablet", ""),
            self.d.known(self.d.last_um_preview, "name"),
            self.d.known(self.d.last_um_preview, "child_name"),
        ]
        session_config = getattr(self.d, "session_config", {}) or {}
        if isinstance(session_config, dict):
            candidates.extend([
                session_config.get("child_name"),
                session_config.get("first_name_cri"),
                session_config.get("first_name_tablet"),
                session_config.get("cri_name"),
                session_config.get("tablet_name"),
            ])
        names = []
        for candidate in candidates:
            clean = str(candidate or "").strip()
            if clean and clean not in names and clean != getattr(self.d, "UNKNOWN_VALUE", ""):
                names.append(clean)
        return sorted(names, key=len, reverse=True)

    def scrub_text(self, value):
        if not isinstance(value, str):
            return value
        scrubbed = value
        replacement = self.child_log_label()
        for name in self.private_child_names():
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
            scrubbed = re.sub(
                pattern,
                replacement,
                scrubbed,
                flags=re.IGNORECASE,
            )
        return scrubbed

    def scrub_private_values(self, value):
        if isinstance(value, dict):
            return {
                key: self.scrub_private_values(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.scrub_private_values(item) for item in value]
        if isinstance(value, str):
            return self.scrub_text(value)
        return value

    def sanitized_log_for_disk(self, log: dict) -> dict:
        sanitized = self.scrub_private_values(copy.deepcopy(log))
        sanitized.pop("child_name", None)
        sanitized["child_label"] = self.child_log_label()
        sanitized["child_initial"] = self.child_initial()
        return sanitized

    def child_speaker_label(self) -> str:
        return self.child_log_label()

    def speaker_label(self, speaker: str) -> str:
        normalized = str(speaker or "").strip().upper()
        if normalized == "LEO":
            return "Leo"
        if normalized == "CHILD":
            return self.child_speaker_label()
        return normalized.title() or "System"

    def conversation_session_id(self, child_name: str, started: datetime) -> str:
        child_part = self.safe_filename_part(self.child_log_label())
        child_id_part = self.safe_filename_part(self.d.CHILD_ID)
        readable_time = started.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{child_part}_{readable_time}" if child_part.endswith(child_id_part) else f"{child_part}_{child_id_part}_{readable_time}"

    # â”€â”€ turn snapshots & runtime state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def planned_turn_log(self, turn: dict) -> dict:
        entry = {
            "phase": self.d.turn_phase(turn),
            "part": turn.get("part"),
            "phase_id": turn.get("phase_id"),
            "script_phase": turn.get("script_phase"),
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
                    "l3": segment.get("l3"),
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
            "memory_review_requested": bool(getattr(self.d, "memory_review_requested", False)),
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

    # â”€â”€ conversation lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        session_dir = os.path.join(self.d.CONVERSATION_LOG_ROOT, session_id)
        counter = 2
        while os.path.exists(session_dir):
            session_dir = os.path.join(self.d.CONVERSATION_LOG_ROOT, f"{session_id}_{counter}")
            counter += 1

        file_base = self.safe_filename_part(self.child_log_label())
        compact_started = started.strftime("%Y%m%dT%H%M%S")
        os.makedirs(session_dir, exist_ok=True)
        json_path = os.path.join(session_dir, f"{file_base}_debug.json")
        text_path = os.path.join(session_dir, f"{file_base}_conversation_debug.txt")
        omr_log_path = os.path.join(session_dir, f"{file_base}_{compact_started}.log")

        planned_phases = [self.planned_turn_log(turn) for turn in script]
        self.d.conversation_log = {
            "session_id": os.path.basename(session_dir),
            "script_version": self.d.SCRIPT_VERSION,
            "child_id": self.d.CHILD_ID,
            "child_label": self.child_log_label(),
            "child_initial": self.child_initial(),
            "child_name": child_name,
            "child_input_mode": self.d.child_input_mode,
            "input_mode": self.d.child_input_mode,
            "researcher_name": self.d.researcher_name,
            "session_config": dict(self.d.session_config or {}),
            "resume_from_log": self.d.resume_from_log_path,
            "start_phase": self.d.start_phase_index + 1,
            "start_phase_index": self.d.start_phase_index,
            "current_phase": None,
            "last_completed_phase": previous_log.get("last_completed_phase", 0) if isinstance(previous_log, dict) else 0,
            "resume_phase": self.d.start_phase_index + 1,
            "tutorial_condition": self.d.tutorial_condition(self.d.last_um_preview),
            "started_at": self.log_timestamp(),
            "started_wall_time": started.isoformat(timespec="seconds"),
            "ended_at": None,
            "timestamp_unit": "seconds_from_interaction_start",
            "folder": session_dir,
            "txt_path": text_path,
            "text_path": text_path,
            "conversation_debug_path": text_path,
            "json_path": json_path,
            "omr_log_path": omr_log_path,
            "um_snapshot_start": dict(self.d.last_um_preview),
            "planned_phases": planned_phases,
            "script_plan": planned_phases,
            "previous_log_included": bool(previous_events or previous_turns),
            "previous_session_id": previous_log.get("session_id") if isinstance(previous_log, dict) else None,
            "events": events_seed,
            "turns": turns_seed,
            "session_status": "in_progress",
            "total_phases": self.d.TOTAL_SCRIPT_PHASES,
        }
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
        if self.d.current_turn_log:
            self.finish_turn_log()
        self.d.conversation_log["ended_at"] = self.log_timestamp()
        self.d.conversation_log["ended_wall_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.d.conversation_log["session_status"] = "completed"
        self.d.conversation_log["um_snapshot_end"] = dict(getattr(self.d, "last_um_preview", {}) or {})
        self.sync_runtime_state_to_log()
        self.write_conversation_logs()

    # â”€â”€ per-turn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def start_turn_log(self, turn: dict):
        if not getattr(self.d, "conversation_log", None):
            return
        phase = self.d.turn_phase(turn)
        self.d.conversation_log["current_phase"] = phase
        self.d.conversation_log["resume_phase"] = phase
        self.sync_runtime_state_to_log()
        self.d.current_turn_log = {
            "phase": phase,
            "part": turn.get("part"),
            "phase_id": turn.get("phase_id"),
            "script_phase": turn.get("script_phase"),
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
            part=turn.get("part"),
            phase_id=turn.get("phase_id"),
            name=turn.get("name"),
            layer=turn.get("layer"),
            dialogue_case=self.d.dialogue_case(turn),
        )

    def finish_turn_log(self):
        if not getattr(self.d, "conversation_log", None) or not self.d.current_turn_log:
            return
        phase = self.d.current_turn_log.get("phase")
        name = self.d.current_turn_log.get("name")
        self.d.current_turn_log["ended_at"] = self.log_timestamp()
        if phase:
            self.d.conversation_log["last_completed_phase"] = int(phase)
            self.d.conversation_log["current_phase"] = None
            self.d.conversation_log["resume_phase"] = min(
                int(phase) + 1,
                self.d.TOTAL_SCRIPT_PHASES,
            )
        self.sync_runtime_state_to_log()
        self.log_conversation_event("phase_end", phase=phase, name=name)
        self.write_conversation_logs()
        self.d.current_turn_log = None

    # â”€â”€ event recorder (the hot path) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        self.sync_runtime_state_to_log()
        self.write_conversation_logs()

    # â”€â”€ rendering & disk writes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def render_conversation_text(self, log: dict = None) -> str:
        if log is None and not getattr(self.d, "conversation_log", None):
            return ""
        log = log or self.sanitized_log_for_disk(self.d.conversation_log)
        lines = []
        lines.append(f"Session: {log.get('session_id')}")
        lines.append(f"Started: {self.format_log_timestamp(log.get('started_at', 0.0))}")
        lines.append(f"Child:   {log.get('child_label')} ({log.get('child_id')})")
        lines.append(f"Researcher: {log.get('researcher_name')}")
        lines.append(f"Condition:  {log.get('tutorial_condition')}")
        lines.append(f"Input mode: {log.get('input_mode')}")
        lines.append("")
        lines.append("=" * 72)
        lines.append("CONVERSATION DEBUG TRANSCRIPT")
        lines.append("=" * 72)
        for turn in log.get("turns") or []:
            name = turn.get("name")
            lines.append("")
            lines.append(self.debug_phase_label(turn))
            for utterance in turn.get("utterances") or []:
                speaker = self.speaker_label(utterance.get("speaker", ""))
                text = utterance.get("text", "")
                timestamp = self.format_log_timestamp(utterance.get("timestamp", 0.0))
                lines.append(f"  [{timestamp}] [{speaker}] {text}")
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
            lines.append(f"Ended at: {self.format_log_timestamp(log['ended_at'])}")
        return "\n".join(lines)

    def debug_phase_label(self, turn: dict) -> str:
        part = turn.get("part")
        phase_id = str(turn.get("phase_id") or "").strip()
        name = turn.get("name") or ""
        if part and phase_id:
            display_phase = phase_id.split(".", 1)[1] if "." in phase_id else phase_id
            return f"Part {part} phase {display_phase}: {name}"
        if phase_id:
            if "." in phase_id:
                part_label, display_phase = phase_id.split(".", 1)
                return f"Part {part_label} phase {display_phase}: {name}"
            return f"Phase {phase_id}: {name}"
        return f"Phase {turn.get('phase')}: {name}"

    def previous_child_utterance(self, events: list, index: int) -> str:
        for event in reversed(events[:index]):
            if event.get("type") == "utterance" and str(event.get("speaker", "")).upper() == "CHILD":
                return event.get("text", "")
        return ""

    def omr_turn_number(self, event: dict) -> int:
        phase = event.get("phase") or event.get("step")
        try:
            return int(phase)
        except (TypeError, ValueError):
            return 0

    def omr_memory_act_from_action(self, event: dict, index: int, events: list) -> dict:
        action = event.get("action")
        change = event.get("change") if isinstance(event.get("change"), dict) else {}
        if action in ("memory_access", "memory_access_tablet", "explicit_memory_inspection_accepted"):
            return {
                "type": "inspection_request",
                "target_field": None,
                "new_value": None,
                "spontaneous_or_triggered": "triggered" if action == "explicit_memory_inspection_accepted" else "spontaneous",
                "turn_number": self.omr_turn_number(event),
                "timestamp": self.omr_timestamp(event.get("timestamp")),
                "child_response_transcript": self.previous_child_utterance(events, index),
            }
        if not change:
            return {}
        confirmed_actions = {
            "mistake_corrected_no_um_change",
            "mistake_corrected_update",
            "confirm_add",
            "confirm_update",
            "confirm_replace",
            "confirm_delete",
            "confirm_multi_update",
            "confirm_role_model_discovery",
        }
        if action not in confirmed_actions:
            return {}
        if action.startswith("confirm_") and not event.get("change_confirmed"):
            return {}
        if action == "mistake_corrected_update" and event.get("write_success") is False:
            return {}
        change_action = change.get("action") or event.get("action")
        memory_type = {
            "add": "addition",
            "update": "correction",
            "replace": "correction",
            "multi_update": "correction",
            "delete": "deletion",
        }.get(change_action, "correction")
        return {
            "type": memory_type,
            "target_field": change.get("field"),
            "new_value": change.get("new_value"),
            "spontaneous_or_triggered": "triggered" if change.get("mistake_id") or change.get("visible_mistake_id") else "spontaneous",
            "turn_number": self.omr_turn_number(event),
            "timestamp": self.omr_timestamp(event.get("timestamp")),
        }

    def dedupe_omr_memory_acts(self, memory_acts: list) -> list:
        deduped = []
        seen = set()
        for act in memory_acts:
            key = (
                act.get("type"),
                act.get("target_field"),
                json.dumps(act.get("new_value"), ensure_ascii=False, sort_keys=True),
                act.get("turn_number"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(act)
        return deduped

    def tablet_events_log_path(self, log: dict) -> str:
        configured = getattr(self.d, "TABLET_EVENTS_LOG_PATH", "")
        if configured:
            return os.path.abspath(os.fspath(configured))
        folder = log.get("folder")
        if folder:
            local_root = os.path.dirname(os.path.dirname(os.path.abspath(os.fspath(folder))))
            return os.path.join(local_root, "tablet_events.jsonl")
        root = getattr(self.d, "CONVERSATION_LOG_ROOT", "")
        if root:
            return os.path.join(os.path.dirname(os.path.abspath(os.fspath(root))), "tablet_events.jsonl")
        return ""

    def session_start_epoch(self, log: dict):
        started_wall = log.get("started_wall_time")
        if not started_wall:
            return None
        try:
            return datetime.fromisoformat(started_wall).timestamp()
        except (TypeError, ValueError):
            return None

    def omr_timestamp_from_wall_epoch(self, wall_epoch, log: dict) -> str:
        start_epoch = self.session_start_epoch(log)
        if start_epoch is None:
            return self.omr_timestamp(0)
        try:
            return self.omr_timestamp(max(0.0, float(wall_epoch) - start_epoch))
        except (TypeError, ValueError):
            return self.omr_timestamp(0)

    def load_external_tablet_events(self, log: dict) -> list:
        path = self.tablet_events_log_path(log)
        if not path or not os.path.exists(path):
            return []
        session_id = log.get("session_id")
        child_id = str(log.get("child_id") or "")
        events = []
        try:
            with open(path, "r", encoding="utf-8") as event_file:
                for line in event_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if session_id and event.get("session_id") and event.get("session_id") != session_id:
                        continue
                    if child_id and event.get("child_id") and str(event.get("child_id")) != child_id:
                        continue
                    tablet_type = event.get("type")
                    if not tablet_type:
                        continue
                    payload = {
                        "type": tablet_type,
                        "timestamp": self.omr_timestamp_from_wall_epoch(
                            event.get("server_wall_time") or event.get("client_wall_time"),
                            log,
                        ),
                    }
                    for key in ("field", "memory_item", "category", "screen", "phase", "old_value", "new_value", "locked"):
                        if event.get(key) is not None:
                            payload[key] = event.get(key)
                    events.append(payload)
        except OSError as exc:
            logger.debug("Could not load tablet events from %s: %s", path, exc)
        return events

    def build_omr_log(self, log: dict) -> dict:
        events = log.get("events") or []
        condition = log.get("tutorial_condition") or log.get("condition")
        mistakes = []
        offered_inspections = []
        memory_acts = []
        um_updates = []
        tablet_events = []

        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "mistake_outcome":
                mistakes.append({
                    "mistake_id": event.get("mistake_id"),
                    "field": event.get("field"),
                    "wrong_value": event.get("wrong_value") or event.get("mistake_value"),
                    "true_value": event.get("real_value") or event.get("actual_value"),
                    "corrected": bool(event.get("corrected")),
                    "child_initiated": self.omr_yes_no(event.get("child_initiated", event.get("corrected"))),
                    "trigger": event.get("trigger") or "mistake-triggered",
                    "spt_layer": event.get("spt_layer") or event.get("layer"),
                    "mistake_type": event.get("mistake_type"),
                    "latency_seconds": event.get("latency_seconds"),
                    "correction_turn_number": event.get("correction_turn_number") or self.omr_turn_number(event),
                    "outcome": event.get("outcome"),
                    "leo_memory_key": event.get("leo_memory_key"),
                    "leo_memory_value": event.get("leo_memory_value"),
                    "timestamp": self.omr_timestamp(event.get("timestamp")),
                })
                if condition == getattr(self.d, "CONDITION_EXPERIMENT", "E"):
                    tablet_events.append({
                        "type": "mistake_changed",
                        "mistake_id": event.get("mistake_id"),
                        "field": event.get("field"),
                        "old_value": event.get("real_value") or event.get("actual_value"),
                        "new_value": event.get("wrong_value") or event.get("mistake_value"),
                        "outcome": event.get("outcome"),
                        "timestamp": self.omr_timestamp(event.get("timestamp")),
                    })
            elif event_type == "action_handler":
                action = event.get("action")
                if action in (
                    "explicit_memory_inspection_accepted",
                    "explicit_memory_inspection_declined",
                    "explicit_memory_inspection_unclear",
                ):
                    offered_inspections.append({
                        "offered": True,
                        "accepted": action == "explicit_memory_inspection_accepted",
                        "child_response_transcript": self.previous_child_utterance(events, index),
                        "timestamp": self.omr_timestamp(event.get("timestamp")),
                    })
                memory_act = self.omr_memory_act_from_action(event, index, events)
                if memory_act:
                    memory_acts.append(memory_act)
            elif event_type == "um_write":
                update = {
                    "field": event.get("field"),
                    "old_value": event.get("old_value"),
                    "new_value": event.get("new_value"),
                    "cause": event.get("cause") or event.get("action"),
                    "timestamp": self.omr_timestamp(event.get("timestamp")),
                    "success": event.get("success"),
                }
                um_updates.append(update)
                if condition == getattr(self.d, "CONDITION_EXPERIMENT", "E") and event.get("success"):
                    tablet_events.append({
                        "type": "memory_updated",
                        "field": event.get("field"),
                        "old_value": event.get("old_value"),
                        "new_value": event.get("new_value"),
                        "timestamp": self.omr_timestamp(event.get("timestamp")),
                    })
            elif event_type == "tablet_event":
                tablet_payload = {
                    "type": event.get("tablet_event_type"),
                    "timestamp": self.omr_timestamp(event.get("timestamp")),
                }
                tablet_payload.update({
                    key: value
                    for key, value in event.items()
                    if key not in {"type", "tablet_event_type", "timestamp"}
                })
                tablet_events.append(tablet_payload)

        tablet_events.extend(self.load_external_tablet_events(log))
        if condition == getattr(self.d, "CONDITION_EXPERIMENT", "E"):
            if not any(event.get("type") == "session_start" for event in tablet_events):
                tablet_events.insert(0, {
                    "type": "session_start",
                    "timestamp": self.omr_timestamp(log.get("started_at", 0.0)),
                })
            if log.get("ended_at") is not None and not any(event.get("type") == "session_end" for event in tablet_events):
                tablet_events.append({
                    "type": "session_end",
                    "timestamp": self.omr_timestamp(log.get("ended_at")),
                })
        tablet_events.sort(key=lambda event: str(event.get("timestamp") or "99:99"))

        return {
            "session_metadata": {
                "child_id": log.get("child_id"),
                "child_label": log.get("child_label"),
                "condition": condition,
                "start_time": log.get("started_wall_time"),
                "end_time": log.get("ended_wall_time"),
                "um_snapshot_begin": log.get("um_snapshot_start") or {},
                "um_snapshot_end": log.get("um_snapshot_end") or {},
            },
            "mistakes_and_corrections": mistakes,
            "offered_inspection": offered_inspections,
            "memory_acts": self.dedupe_omr_memory_acts(memory_acts),
            "um_updates": um_updates,
            "tablet_events": tablet_events,
            "audio_files": [],
        }

    def render_omr_log(self, log: dict) -> str:
        return json.dumps(self.build_omr_log(log), ensure_ascii=False, indent=2)

    def write_conversation_logs(self):
        if not getattr(self.d, "conversation_log", None):
            return
        log = self.d.conversation_log
        json_path = log.get("json_path")
        text_path = log.get("text_path")
        omr_log_path = log.get("omr_log_path")
        if not json_path or not text_path:
            return
        try:
            json_path = os.path.abspath(os.fspath(json_path))
            text_path = os.path.abspath(os.fspath(text_path))
            omr_log_path = os.path.abspath(os.fspath(omr_log_path)) if omr_log_path else None
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            if omr_log_path:
                os.makedirs(os.path.dirname(omr_log_path), exist_ok=True)
            log["json_path"] = json_path
            log["txt_path"] = text_path
            log["text_path"] = text_path
            log["conversation_debug_path"] = text_path
            if omr_log_path:
                log["omr_log_path"] = omr_log_path

            disk_log = self.sanitized_log_for_disk(log)
            json_payload = json.dumps(disk_log, ensure_ascii=False, indent=2)
            text_payload = self.render_conversation_text(disk_log)
            self.atomic_write_text(json_path, json_payload)
            self.atomic_write_text(text_path, text_payload)
            if omr_log_path:
                self.atomic_write_text(omr_log_path, self.render_omr_log(disk_log))
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
            log["last_log_write_error"] = error
            logger.debug("Could not write conversation log to %s / %s: %s", json_path, text_path, error)

            fallback_dir = os.path.join(tempfile.gettempdir(), "cri_dialogue2_failed_logs")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_base = self.safe_filename_part(log.get("session_id") or "conversation")
                fallback_path = os.path.join(fallback_dir, f"{fallback_base}.json")
                self.atomic_write_text(
                    fallback_path,
                    json.dumps(self.sanitized_log_for_disk(log), ensure_ascii=False, indent=2),
                )
                log["last_log_write_fallback_path"] = fallback_path
                logger.debug("Conversation log backup written to %s", fallback_path)
            except OSError as fallback_exc:
                log["last_log_write_fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"

    def atomic_write_text(self, path: str, payload: str):
        """Write via a temporary file so Windows never sees a half-written log."""
        directory = os.path.dirname(path)
        prefix = f".{self.safe_filename_part(os.path.basename(path))}."
        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(payload)
            os.replace(temp_path, path)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

    # â”€â”€ per-turn structured event helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
