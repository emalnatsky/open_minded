"""
ResumeHelper — load a previous conversation log and restore runtime state.

Constructed once in CRI_ScriptedDialogue.__init__ alongside ConversationLogger:

    self.resume = ResumeHelper(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.clean_pasted_path, self.run_resume_session_interface, ...) stay
identical.

This module READS class constants (TOTAL_SCRIPT_PHASES, CHILD_ID),
class methods (child_facing_memory_fields, format_log_timestamp,
condition_display, ask_session_value, apply_session_config), and
WRITES many runtime state fields (mistakes_mentioned, mistake_states,
local_child_name, researcher_name, start_phase_index, etc.) back onto
the dialogue.

Two-way binding is intentional: this is a resume hook that needs to
splice prior session state back into the live dialogue.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResumeHelper:
    """Reads previous session logs and restores them to the live dialogue."""

    def __init__(self, dialogue):
        # Back-reference. Matches the pattern used by ConversationLogger.
        self.d = dialogue

    # ── path / file helpers ──────────────────────────────────────────────────

    def clean_pasted_path(self, path: str) -> str:
        clean = str(path or "").strip()
        quote_chars = "\"'“”‘’"
        for _ in range(3):
            clean = clean.strip()
            if len(clean) >= 2 and clean[0] in quote_chars and clean[-1] in quote_chars:
                clean = clean[1:-1]
                continue
            if len(clean) >= 2 and clean[0] == "<" and clean[-1] == ">":
                clean = clean[1:-1]
                continue
            break
        return os.path.expandvars(os.path.expanduser(clean.strip()))

    def load_conversation_log_file(self, path: str) -> dict:
        clean_path = self.clean_pasted_path(path)
        with open(clean_path, "r", encoding="utf-8") as log_file:
            return json.load(log_file)

    # ── derive resume phase from prior log ───────────────────────────────────

    def compute_resume_phase_from_log(self, log: dict) -> int:
        explicit = log.get("resume_phase")
        if explicit:
            return max(1, min(int(explicit), self.d.TOTAL_SCRIPT_PHASES))

        events = log.get("events") or []
        last_start_index = None
        last_start_phase = None
        for index, event in enumerate(events):
            if event.get("type") == "phase_start" and event.get("phase"):
                last_start_index = index
                last_start_phase = int(event["phase"])

        if last_start_phase is None:
            return max(1, min(int(log.get("last_completed_phase", 0) or 0) + 1, self.d.TOTAL_SCRIPT_PHASES))

        ended_after_start = any(
            event.get("type") == "phase_end" and int(event.get("phase") or 0) == last_start_phase
            for event in events[last_start_index + 1:]
        )
        if ended_after_start:
            return min(last_start_phase + 1, self.d.TOTAL_SCRIPT_PHASES)
        return last_start_phase

    def session_config_from_resume_log(self, log: dict, resume_path: str) -> dict:
        config = dict(log.get("session_config") or {})
        config.setdefault("child_id", log.get("child_id", self.d.CHILD_ID))
        config.setdefault("child_name", log.get("child_name", ""))
        config.setdefault("researcher_name", log.get("researcher_name", ""))
        config.setdefault("condition", log.get("tutorial_condition", "C1"))
        resume_phase = self.compute_resume_phase_from_log(log)
        config["mode"] = "resume"
        config["resume_from_log"] = self.clean_pasted_path(resume_path)
        config["resume_phase"] = resume_phase
        config["start_phase_index"] = resume_phase - 1
        config["resumed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return config

    # ── restore runtime state from prior log ─────────────────────────────────

    def restore_runtime_state_from_log(self, log: dict):
        self.d.mistakes_mentioned = int(log.get("mistakes_mentioned", 0) or 0)
        self.d.corrections_seen = int(log.get("corrections_seen", 0) or 0)
        self.d.mistake_states = dict(log.get("mistake_states") or {})
        self.d.phases_with_confirmed_change = set(log.get("phases_with_confirmed_change") or [])
        self.d.memory_fields_mentioned_so_far = set(
            self.d.child_facing_memory_fields(log.get("memory_fields_mentioned_so_far") or [])
        )

    # ── console replay of prior conversation ─────────────────────────────────

    def resume_console_transcript_lines(self, log: dict) -> list:
        """Human-readable replay of a previous conversation for the CMD window."""
        lines = []
        for event in log.get("events") or []:
            if not isinstance(event, dict):
                continue
            timestamp = self.d.format_log_timestamp(event.get("timestamp", 0.0))
            event_type = event.get("type")
            if event_type == "phase_start":
                lines.append("")
                lines.append(f"[{timestamp}] Phase {event.get('phase')}: {event.get('name')}")
            elif event_type == "utterance":
                speaker = event.get("speaker", "").upper() or "UNKNOWN"
                lines.append(f"[{timestamp}] {speaker}: {event.get('text')}")
            elif event_type == "phase_end":
                lines.append(f"[{timestamp}] Phase {event.get('phase')} finished")
        return lines

    def print_resume_console_transcript(self, log: dict):
        lines = self.resume_console_transcript_lines(log)
        print("\n" + "-" * 72)
        print("PREVIOUS CONVERSATION FROM LOG")
        if lines:
            for line in lines:
                print(line)
        else:
            print("(No previous utterances found in this log.)")
        print("-" * 72)

    # ── interactive resume entry point ───────────────────────────────────────

    def run_resume_session_interface(self, resume_path: str = ""):
        print("\n" + "=" * 72)
        print("CRI SESSION RESUME")
        path = resume_path or self.d.ask_session_value("Paste previous conversation JSON path", "")
        log = self.load_conversation_log_file(path)
        config = self.session_config_from_resume_log(log, path)
        self.d.resume_from_log_path = self.clean_pasted_path(path)
        self.d.resume_source_log = log
        self.d.apply_session_config(config)
        self.restore_runtime_state_from_log(log)

        print("\nLoaded previous conversation log.")
        print(f"  Child ID:       {self.d.CHILD_ID}")
        print(f"  Child name:     {self.d.local_child_name or '(not set)'}")
        print(f"  Researcher:     {self.d.researcher_name or '(not set)'}")
        print(f"  Condition:      {self.d.condition_display(self.d.local_condition)}")
        print(f"  Resume phase:   {self.d.start_phase_index + 1}")
        print(f"  Mentioned UM fields restored: {len(self.d.memory_fields_mentioned_so_far)}")
        input("\nPress Enter to continue from this phase...")
