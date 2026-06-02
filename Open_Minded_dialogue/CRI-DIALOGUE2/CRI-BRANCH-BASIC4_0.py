import os
import time
import json
import random
import copy
import unicodedata
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication

import config
from speech_io import SpeechIO
from session_setup import SessionSetup
from tablet_state import TabletStateWriter
from cri_logger import ConversationLogger, ResumeHelper
from cri_um import UMClient
from cri_memory import MemoryAccess
from cri_actions import ActionHandler, NudgeManager
from cri_script import ContentPlan, Segments, ScriptBuilder

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoWakeUpRequest,
    NaoRestRequest,
    NaoSetAutonomousLifeRequest,
)
from sic_framework.services.openai_whisper_stt.whisper_stt import (
    SICWhisper,
    WhisperConf,
)
from sic_framework.services.llm import GPT, GPTConf, GPTRequest
from openai import OpenAI

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))


# Intent classifier — moved to the classifier/ package.
# Import everything the rest of this file needs via the package's public API.
from cri_classifier import (
    IntentResult,
    REPEAT_SENTINEL,
    VALID_INTENTS,
    StubIntentClassifier,
    GPTIntentClassifier,
)
class CRI_ScriptedDialogue(SICApplication):
    """
    CRI 4.0 walkthrough interaction flow.

    The script pulls known child-memory fields from the UM API before starting,
    prints a preview, then runs the explicit child-specific dialogue structure:
    scripted turns, UM-template turns, pre-generated LLM utterances, and
    runtime LLM follow-up branches for unpredictable child responses.

    UM connection:
        GET http://localhost:8000/api/um/{child_id}/field/{field_name}
        No API key needed for reads.
        If a field is not set, robot says "dat weet ik nog niet".
    """

    # ─────────────────────────────────────────────────────────────────────
    # All class constants below are mirrored from config.py.
    # To change a value, edit config.py — DO NOT add new constants here.
    # Method bodies still reference them via self.UM_API_BASE etc.
    # ─────────────────────────────────────────────────────────────────────

    UM_API_BASE    = config.UM_API_BASE
    CHILD_ID       = config.CHILD_ID
    UNKNOWN_VALUE  = config.UNKNOWN_VALUE
    UM_FIELDS      = config.UM_FIELDS
    SCRIPT_TABLE_FIELDS = config.SCRIPT_TABLE_FIELDS
    STARTUP_OVERVIEW_UM_FIELDS = config.STARTUP_OVERVIEW_UM_FIELDS
    STARTUP_OVERVIEW_SCENARIO_FIELDS = config.STARTUP_OVERVIEW_SCENARIO_FIELDS
    FIELD_LABELS   = config.FIELD_LABELS

    STT_TIMEOUT      = config.STT_TIMEOUT
    STT_PHRASE_LIMIT = config.STT_PHRASE_LIMIT

    LLM_FALLBACK       = config.LLM_FALLBACK
    LLM_SYSTEM         = config.LLM_SYSTEM
    TOPIC_CHANGE_MODEL = config.TOPIC_CHANGE_MODEL

    BOOLEANISH_FIELDS             = config.BOOLEANISH_FIELDS
    MEMORY_ACCESS_CONTROL_FIELDS  = config.MEMORY_ACCESS_CONTROL_FIELDS
    TUTORIAL_CONDITION_FIELD      = config.TUTORIAL_CONDITION_FIELD
    CONDITION_CONTROL             = config.CONDITION_CONTROL
    CONDITION_EXPERIMENT          = config.CONDITION_EXPERIMENT
    CONDITION_LABELS              = config.CONDITION_LABELS
    CONDITION_ALIASES             = config.CONDITION_ALIASES
    TABLET_REVEAL_WAIT_SECONDS    = config.TABLET_REVEAL_WAIT_SECONDS
    MEMORY_ACCESS_EXCLUDED_FIELDS = config.MEMORY_ACCESS_EXCLUDED_FIELDS
    OPPOSITE_VALUE_FALLBACKS      = config.OPPOSITE_VALUE_FALLBACKS

    CONTENT_PLAN_SEQUENCE   = config.CONTENT_PLAN_SEQUENCE
    CASE_FULLY_SCRIPTED     = config.CASE_FULLY_SCRIPTED
    CASE_UM_TEMPLATE        = config.CASE_UM_TEMPLATE
    CASE_PREAUTHORED_POOL   = config.CASE_PREAUTHORED_POOL
    CASE_LLM_PREGENERATED   = config.CASE_LLM_PREGENERATED
    CASE_RUNTIME_LLM_BRANCH = config.CASE_RUNTIME_LLM_BRANCH
    CASE_MIXED_SEQUENCE     = config.CASE_MIXED_SEQUENCE
    PREGENERATED_UTTERANCE_PREFIXES = config.PREGENERATED_UTTERANCE_PREFIXES
    SCENARIO_UTTERANCE_ALIASES = config.SCENARIO_UTTERANCE_ALIASES

    USE_DESKTOP_MIC               = config.USE_DESKTOP_MIC
    CONNECT_NAO                   = config.CONNECT_NAO
    ASK_RUN_MODE_AT_START         = config.ASK_RUN_MODE_AT_START
    SIMULATION_MODE               = config.SIMULATION_MODE
    SIMULATED_PERSONA_DIR         = config.SIMULATED_PERSONA_DIR
    SIMULATED_PERSONA_PATH        = config.SIMULATED_PERSONA_PATH
    USE_FAKE_PERSONA_UM           = config.USE_FAKE_PERSONA_UM
    SIMULATION_WRITE_PERSONA_FILE = config.SIMULATION_WRITE_PERSONA_FILE
    WAIT_FOR_PREVIEW_CONFIRMATION = config.WAIT_FOR_PREVIEW_CONFIRMATION
    REVIEW_TRANSCRIPTS            = config.REVIEW_TRANSCRIPTS
    POST_PHASE_TEST_CONTROLS      = config.POST_PHASE_TEST_CONTROLS
    CHILD_INPUT_MODE              = config.CHILD_INPUT_MODE
    SCRIPT_VERSION                = config.SCRIPT_VERSION
    TOTAL_SCRIPT_PHASES           = config.TOTAL_SCRIPT_PHASES
    ASK_SESSION_INTERFACE_AT_START = config.ASK_SESSION_INTERFACE_AT_START
    SESSION_CONFIG_PATH           = config.SESSION_CONFIG_PATH
    SESSION_ROSTER_DIR            = config.SESSION_ROSTER_DIR
    SESSION_ROSTER_PATH           = config.SESSION_ROSTER_PATH
    ROSTER_PATH                   = config.ROSTER_PATH

    CONVERSATION_LOG_ENABLED = config.CONVERSATION_LOG_ENABLED
    CONVERSATION_LOG_ROOT    = config.CONVERSATION_LOG_ROOT

    def __init__(self, openai_env_path=None, nao_ip="10.0.0.165"):
        super(CRI_ScriptedDialogue, self).__init__()
        self.nao_ip = nao_ip
        self.openai_env_path = openai_env_path
        self.nao = None
        self.whisper = None
        self.gpt = None
        self.clf = None
        self.desktop = None
        self.openai_client = None
        self.mistakes_mentioned = 0
        self.corrections_seen = 0
        self.mistake_states = {}
        self.last_um_preview = {}
        self.pending_change = None
        self.conversation_log = None
        self.current_turn_log = None
        self.conversation_log_started_monotonic = None
        self.conversation_log_time_offset = 0.0
        self.session_config = {}
        self.resume_from_log_path = None
        self.resume_source_log = {}
        self.local_child_name = ""
        self.local_child_name_cri    = ""   # what NAO TTS pronounces
        self.local_child_name_tablet = ""   # what appears on the tablet book cover
        self.researcher_name = ""
        self.local_condition = ""
        self.start_phase_index = 0
        self.simulation_mode = bool(self.SIMULATION_MODE)
        self.child_input_mode = self.CHILD_INPUT_MODE
        self.simulated_persona = {}
        self.simulated_persona_path = self.SIMULATED_PERSONA_PATH
        self.simulated_history = []
        self.last_leo_utterance = ""
        self.current_turn_context = None
        self.phases_with_confirmed_change = set()
        self.memory_fields_mentioned_so_far = set()
        self.memory_review_requested = False
        self.last_cri_scenario = {}
        self.last_cri_scenario_loaded = False

        # Conversation logger — reads/writes the logging state above.
        # Methods in this class delegate to self.conv_log.X.
        self.conv_log = ConversationLogger(self)

        # Resume helper — reads previous JSON logs and restores state.
        # Methods in this class delegate to self.resume.X.
        self.resume = ResumeHelper(self)

        # Session setup — roster, condition normalisation, pre-session prompt,
        # run-mode prompt. Methods in this class delegate to self.session_setup.X.
        self.session_setup = SessionSetup(self)

        # UM I/O — reads/writes against Eunike's API (or fake persona JSONs).
        # Methods in this class delegate to self.um.X.
        self.um = UMClient(self)

        # Memory access — phase-scoped reads. When the child says
        # "wat weet je over mij" Leo only mentions things from earlier in
        # the conversation. Methods in this class delegate to self.mem.X.
        self.mem = MemoryAccess(self)

        # Nudge manager — mistake state machine + gentle/explicit nudge texts.
        # Methods in this class delegate to self.nudge.X.
        self.nudge = NudgeManager(self)

        # Action handler — intent-to-action routing, confirmation loops,
        # L3 LLM fallback. Methods in this class delegate to self.actions.X.
        self.actions = ActionHandler(self)

        # Content-plan machinery (L1/L2-slot/L2-pregen/sequence + render).
        # Methods like l1, l2_slot, sequence, turn_text delegate to self.cp.X.
        self.cp = ContentPlan(self)

        # Topic 1 + Topic 2 segment builders. Methods topic1_phase_segments,
        # topic_label_fields, topic2_phase_segments delegate to self.segments.X.
        self.segments = Segments(self)

        # The master script builder (build_script).
        self.script = ScriptBuilder(self)

        # Tablet phase-gating handshake. Writes session_state.json on every
        # turn so um_tablet_server.py can broadcast unlocked_categories to
        # the UM tablet. Called from start_conversation_log (reset) and
        # start_turn_log (update).
        _session_state_path = config.SESSION_STATE_PATH
        self.tablet_state = TabletStateWriter(
            state_path=_session_state_path,
            get_child_id_fn=lambda: self.CHILD_ID,
            get_child_name_fn=lambda: getattr(self, "local_child_name_cri", "") or getattr(self, "local_child_name", "") or self.CHILD_ID,
            get_tablet_name_fn=lambda: getattr(self, "local_child_name_tablet", "") or getattr(self, "local_child_name", "") or self.CHILD_ID,
            get_condition_fn=lambda: getattr(self, "local_condition", ""),
            get_mistake_states_fn=lambda: getattr(self, "mistake_states", {}) or {},
            enabled=True,
        )

        self.set_log_level(sic_logging.INFO)
        self.configure_session_interface()
        self.configure_run_mode()
        self.setup()

    # Setup

    # ─────────────────────────────────────────────────────────────────────
    # Session setup — all methods delegate to self.session_setup.
    # Pure pass-throughs so existing call sites stay identical.
    # See session_setup.py for the real implementations.
    # ─────────────────────────────────────────────────────────────────────

    def load_local_session_config(self):
        return self.session_setup.load_local_session_config()

    def load_roster(self):
        return self.session_setup.load_roster()

    def save_local_session_config(self, config):
        return self.session_setup.save_local_session_config(config)

    def ask_session_value(self, label, default=""):
        return self.session_setup.ask_session_value(label, default)

    def normalize_condition_value(self, value, default="C"):
        return self.session_setup.normalize_condition_value(value, default)

    def condition_display(self, condition):
        return self.session_setup.condition_display(condition)

    def parse_phase_index(self, value, default_index=0):
        return self.session_setup.parse_phase_index(value, default_index)

    def apply_session_config(self, config):
        return self.session_setup.apply_session_config(config)

    def check_child_in_um_api(self, child_id):
        return self.session_setup.check_child_in_um_api(child_id)

    def get_um_field_for_child(self, child_id, field):
        return self.session_setup.get_um_field_for_child(child_id, field)

    def session_condition_from_um(self, child_id):
        return self.session_setup.session_condition_from_um(child_id)

    def run_new_session_interface(self):
        return self.session_setup.run_new_session_interface()

    # ─────────────────────────────────────────────────────────────────────
    # Resume helpers — all methods delegate to self.resume.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_logger/resume.py for the real implementations.
    # ─────────────────────────────────────────────────────────────────────

    def clean_pasted_path(self, path):
        return self.resume.clean_pasted_path(path)

    def load_conversation_log_file(self, path):
        return self.resume.load_conversation_log_file(path)

    def compute_resume_phase_from_log(self, log):
        return self.resume.compute_resume_phase_from_log(log)

    def session_config_from_resume_log(self, log, resume_path):
        return self.resume.session_config_from_resume_log(log, resume_path)

    def restore_runtime_state_from_log(self, log):
        return self.resume.restore_runtime_state_from_log(log)

    def resume_console_transcript_lines(self, log):
        return self.resume.resume_console_transcript_lines(log)

    def print_resume_console_transcript(self, log):
        return self.resume.print_resume_console_transcript(log)

    def run_resume_session_interface(self, resume_path=""):
        return self.resume.run_resume_session_interface(resume_path)

    def configure_session_interface(self):
        return self.session_setup.configure_session_interface()

    def configure_run_mode(self):
        return self.session_setup.configure_run_mode()

    def use_keyboard_input(self):
        return self.session_setup.use_keyboard_input()

    def use_microphone_input(self):
        return self.session_setup.use_microphone_input()

    # ─────────────────────────────────────────────────────────────────────
    # Conversation logging — all methods delegate to self.conv_log.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_logger/conversation_logger.py for the real implementations.
    # ─────────────────────────────────────────────────────────────────────

    def log_timestamp(self):
        return self.conv_log.log_timestamp()

    def format_log_timestamp(self, timestamp):
        return self.conv_log.format_log_timestamp(timestamp)

    def safe_filename_part(self, value):
        return self.conv_log.safe_filename_part(value)

    def conversation_child_name(self):
        return self.conv_log.conversation_child_name()

    def conversation_session_id(self, child_name, started):
        return self.conv_log.conversation_session_id(child_name, started)

    def planned_turn_log(self, turn):
        return self.conv_log.planned_turn_log(turn)

    def runtime_state_snapshot(self):
        return self.conv_log.runtime_state_snapshot()

    def sync_runtime_state_to_log(self):
        return self.conv_log.sync_runtime_state_to_log()

    def max_conversation_timestamp(self, log):
        return self.conv_log.max_conversation_timestamp(log)

    def start_conversation_log(self, script):
        # Reset cumulative unlocked-categories set at the start of every session
        # (and resume) so the tablet starts blank.
        self.tablet_state.reset()
        return self.conv_log.start_conversation_log(script)

    def finish_conversation_log(self):
        return self.conv_log.finish_conversation_log()

    def start_turn_log(self, turn):
        result = self.conv_log.start_turn_log(turn)
        # Tell the UM tablet which categories the child may inspect now.
        self.tablet_state.update(turn)
        return result

    def finish_turn_log(self):
        return self.conv_log.finish_turn_log()

    def log_conversation_event(self, event_type, **data):
        return self.conv_log.log_conversation_event(event_type, **data)

    def render_conversation_text(self):
        return self.conv_log.render_conversation_text()

    def write_conversation_logs(self):
        return self.conv_log.write_conversation_logs()

    def log_llm_decision(self, mode, transcript, result, context=None):
        return self.conv_log.log_llm_decision(mode, transcript, result, context)

    def intent_result_to_dict(self, result):
        return self.conv_log.intent_result_to_dict(result)

    def log_intent_classifier_result(self, transcript, result):
        return self.conv_log.log_intent_classifier_result(transcript, result)

    def log_action_handler_result(self, action_result):
        return self.conv_log.log_action_handler_result(action_result)

    # ─────────────────────────────────────────────────────────────────────
    # Action handler + nudge — all methods delegate to self.actions / self.nudge.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_actions/handler.py and cri_actions/nudge.py for the real code.
    # ─────────────────────────────────────────────────────────────────────

    # --- ActionHandler (cri_actions/handler.py) -------------------------------

    def classify_with_repeat(self, transcript, turn_context=None):
        return self.actions.classify_with_repeat(transcript, turn_context)

    def llm_response(self, child_input, turn=None):
        return self.actions.llm_response(child_input, turn)

    def extract_json_object(self, raw):
        return self.actions.extract_json_object(raw)

    def clean_confirmation_question(self, question, change):
        return self.actions.clean_confirmation_question(question, change)

    def confirmation_text(self, change):
        return self.actions.confirmation_text(change)

    def turn_memory_context(self, turn):
        return self.actions.turn_memory_context(turn)

    def allowed_change_fields(self, turn):
        return self.actions.allowed_change_fields(turn)

    def mistake_correction_question(self, turn):
        return self.actions.mistake_correction_question(turn)

    def topic_correction_question(self, topic):
        return self.actions.topic_correction_question(topic)

    def action_result(self, action, handled, reason="", change=None, leo_response=None,
                      follow_up_needed=False, **extra):
        return self.actions.action_result(
            action, handled, reason,
            change=change,
            leo_response=leo_response,
            follow_up_needed=follow_up_needed,
            **extra,
        )

    def change_from_intent_result(self, result, turn, transcript):
        return self.actions.change_from_intent_result(result, turn, transcript)

    def is_rejection_without_value(self, result, transcript):
        return self.actions.is_rejection_without_value(result, transcript)

    def is_confirmation_yes(self, result, transcript):
        return self.actions.is_confirmation_yes(result, transcript)

    def is_confirmation_no(self, result, transcript):
        return self.actions.is_confirmation_no(result, transcript)

    def confirmation_decision_from_intent(self, result, transcript, change):
        return self.actions.confirmation_decision_from_intent(result, transcript, change)

    def action_handler(self, result, transcript, turn):
        return self.actions.action_handler(result, transcript, turn)

    def follow_up_action_handler(self, turn, max_rounds=3):
        return self.actions.follow_up_action_handler(turn, max_rounds)

    def confirm_topic_change(self, change):
        return self.actions.confirm_topic_change(change)

    # --- NudgeManager (cri_actions/nudge.py) ----------------------------------

    def register_mistake_phase(self, turn):
        return self.nudge.register_mistake_phase(turn)

    def mark_current_mistake_corrected(self):
        return self.nudge.mark_current_mistake_corrected()

    def current_session_id(self):
        log = getattr(self, "conversation_log", None) or {}
        return log.get("session_id") or "unknown"

    def current_mistake_memory_value(self, field: str, corrected: bool, mistake_value: str = "") -> str:
        if corrected and field:
            return self.known(self.last_um_preview, field, "") or mistake_value
        return mistake_value

    def mistake_memory_key(self, mistake_id: str, outcome: str, field: str, leo_memory_value: str = "") -> str:
        if outcome != "not_corrected":
            return ""
        if not (mistake_id and field and self.is_known(leo_memory_value)):
            return ""
        return f"{mistake_id}_mistake_not_corrected_{field}"

    def m3_school_difficulty_resolution_pending(self, state: dict) -> bool:
        return bool(state.get("m3_requires_school_difficulty_resolution"))

    def mistake_outcome_payload(self, turn: dict) -> dict:
        mistake_id = turn.get("mistake_id")
        state = (self.mistake_states or {}).get(mistake_id, {}) if mistake_id else {}
        corrected = bool(state.get("corrected"))
        rejected_without_value = bool(state.get("wrong_value_rejected")) and not corrected
        if corrected:
            outcome = "corrected"
        elif state.get("m3_not_corrected_by_difficulty_change"):
            outcome = "not_corrected"
        elif rejected_without_value:
            outcome = "rejected_unresolved"
        elif mistake_id == "M3" and self.m3_school_difficulty_resolution_pending(state):
            outcome = "unresolved"
        else:
            outcome = "not_corrected"
        event_type = f"mistake_{outcome}"
        field = turn.get("mistake_field") or state.get("field")
        real_value = turn.get("mistake_actual") or state.get("actual")
        mistake_value = turn.get("mistake_wrong") or state.get("wrong")
        leo_memory_value = (
            None
            if outcome in ("rejected_unresolved", "unresolved")
            else self.current_mistake_memory_value(field, corrected, mistake_value)
        )
        leo_memory_key = self.mistake_memory_key(mistake_id, outcome, field, leo_memory_value)
        return {
            "event_type": event_type,
            "mistake_id": mistake_id,
            "field": field,
            "outcome": outcome,
            "real_value": real_value,
            "mistake_value": mistake_value,
            "wrong_value": mistake_value,
            "leo_memory_key": leo_memory_key,
            "leo_memory_value": leo_memory_value,
            "corrected": corrected,
            "phase": turn.get("phase_id") or str(self.turn_phase(turn) or ""),
            "step": self.turn_phase(turn),
            "session_id": self.current_session_id(),
        }

    def record_mistake_outcome(self, turn: dict):
        mistake_id = turn.get("mistake_id")
        if not mistake_id:
            return
        state = self.mistake_states.setdefault(mistake_id, {"id": mistake_id})
        if state.get("outcome_logged"):
            return
        payload = self.mistake_outcome_payload(turn)
        state["outcome"] = payload.get("outcome")
        state["real_value"] = payload.get("real_value")
        state["mistake_value"] = payload.get("mistake_value")
        state["leo_memory_key"] = payload.get("leo_memory_key")
        state["leo_memory_value"] = payload.get("leo_memory_value")
        state["outcome_logged"] = True
        log_payload = dict(payload)
        log_payload["logged_event_type"] = log_payload.pop("event_type", None)
        self.log_conversation_event("mistake_outcome", **log_payload)
        self.log_cri_interaction_event(payload)

    def m3_school_difficulty_resolution_context(self, turn: dict) -> bool:
        return bool(
            (turn or {}).get("m3_school_difficulty_resolution")
            and (turn or {}).get("mistake_id") == "M3"
        )

    def mark_m3_corrected_by_school_difficulty_ack(self, turn: dict):
        if not self.m3_school_difficulty_resolution_context(turn):
            return
        mistake_id = turn.get("mistake_id")
        state = self.mistake_states.setdefault(mistake_id, {"id": mistake_id})
        if state.get("m3_not_corrected_by_difficulty_change"):
            return
        state["corrected"] = True
        state["corrected_at_phase"] = self.turn_phase(turn)
        state["corrected_by_school_difficulty_ack"] = True

    def handle_confirmed_mistake_related_change(self, change: dict, turn: dict = None) -> bool:
        context = turn or self.current_turn_context or {}
        mistake_id = context.get("mistake_id") or change.get("visible_mistake_id")
        if not mistake_id:
            return False

        state = self.mistake_states.setdefault(mistake_id, {"id": mistake_id})
        field = change.get("field")
        mistake_field = context.get("mistake_field") or change.get("visible_mistake_field")
        if field == mistake_field:
            if context.get("mistake_id"):
                self.mark_current_mistake_corrected()
            else:
                state["corrected"] = True
                state["corrected_at_phase"] = self.turn_phase(context)
            return True

        if (
            self.m3_school_difficulty_resolution_context(context)
            and field == "school_difficulty"
            and self.is_known(change.get("new_value"))
            and self.is_known(context.get("mistake_wrong"))
            and not self.actions.field_values_match("school_difficulty", change.get("new_value"), context.get("mistake_wrong"))
        ):
            state["m3_not_corrected_by_difficulty_change"] = True
            state["school_difficulty_changed_from"] = change.get("old_value")
            state["school_difficulty_changed_to"] = change.get("new_value")
        return False

    def visible_mistake_state_for_field(self, field: str, turn: dict = None) -> dict:
        """Return the unresolved mistake whose wrong value is child-visible for a field."""
        if not field:
            return {}
        turn = turn or {}
        mistake_id = turn.get("mistake_id")
        if turn.get("mistake_field") == field and self.is_known(turn.get("mistake_wrong")):
            state = (self.mistake_states or {}).get(mistake_id, {}) if mistake_id else {}
            if not state.get("corrected"):
                visible = dict(state)
                visible.setdefault("id", mistake_id)
                visible.setdefault("field", field)
                visible.setdefault("actual", turn.get("mistake_actual"))
                visible.setdefault("wrong", turn.get("mistake_wrong"))
                return visible

        for state_id, state in (self.mistake_states or {}).items():
            if state.get("field") != field:
                continue
            if not state.get("mentioned") or state.get("corrected"):
                continue
            if not self.is_known(state.get("wrong")):
                continue
            visible = dict(state)
            visible.setdefault("id", state_id)
            return visible
        return {}

    def visible_memory_value(self, field: str, turn: dict = None) -> str:
        """Memory value as the child currently sees/hears it, including active mistakes."""
        visible_mistake = self.visible_mistake_state_for_field(field, turn)
        if visible_mistake and self.is_known(visible_mistake.get("wrong")):
            return str(visible_mistake["wrong"])
        return self.memory_value(field)

    def mistake_field_label(self, turn):
        return self.nudge.mistake_field_label(turn)

    def first_uncorrected_mistake_state(self):
        return self.nudge.first_uncorrected_mistake_state()

    def nudge_state_for_turn(self, turn):
        return self.nudge.nudge_state_for_turn(turn)

    def gentle_mistake_nudge_text(self, state):
        return self.nudge.gentle_mistake_nudge_text(state)

    def explicit_mistake_nudge_text(self, state):
        return self.nudge.explicit_mistake_nudge_text(state)

    def mistake_nudge_action(self, turn, force_explicit=False):
        return self.nudge.mistake_nudge_action(turn, force_explicit)

    def use_fake_persona_um(self) -> bool:
        return bool(self.USE_FAKE_PERSONA_UM or self.simulation_mode)

    def persona_summary_from_file(self, path: str) -> dict:
        """Read minimal display metadata for a fake persona JSON file."""
        with open(path, "r", encoding="utf-8-sig") as persona_file:
            persona = json.load(persona_file)
        child_id = persona.get("child_id")
        return {
            "child_id": str(child_id) if child_id is not None else "",
            "name": persona.get("name") or "unknown",
            "exposure": persona.get("exposure") or self.UNKNOWN_VALUE,
            "condition": self.normalize_condition_value(persona.get("condition"), default=self.CONDITION_CONTROL),
            "path": path,
        }

    def available_fake_personas(self) -> list:
        """Return all fake persona files with child ID, name, exposure, condition, and path."""
        if not os.path.isdir(self.SIMULATED_PERSONA_DIR):
            return []

        personas = []
        for filename in os.listdir(self.SIMULATED_PERSONA_DIR):
            if not filename.lower().endswith(".json"):
                continue
            path = os.path.join(self.SIMULATED_PERSONA_DIR, filename)
            try:
                summary = self.persona_summary_from_file(path)
                if summary["child_id"]:
                    personas.append(summary)
            except Exception as e:
                self.logger.warning("Could not read fake persona %s: %s", path, e)

        def sort_key(persona):
            child_id = persona["child_id"]
            return (0, int(child_id)) if child_id.isdigit() else (1, child_id)

        return sorted(personas, key=sort_key)

    def select_simulated_persona_by_child_id(self, child_id: str) -> dict:
        """Select a fake persona file by numeric child ID."""
        wanted = str(child_id).strip()
        for persona in self.available_fake_personas():
            if persona["child_id"] == wanted:
                self.simulated_persona_path = persona["path"]
                return persona
        raise ValueError(f"No fake persona found for child ID {wanted}")

    def configure_simulated_persona(self):
        """Let the tester choose which fake persona to use by child ID."""
        env_child_id = os.environ.get("CRI_SIMULATED_CHILD_ID", "").strip()
        if env_child_id:
            selected = self.select_simulated_persona_by_child_id(env_child_id)
            print(
                f"Using fake persona {selected['child_id']} - {selected['name']} "
                f"({selected['exposure']}, {selected['condition']})"
            )
            return

        personas = self.available_fake_personas()
        if not personas:
            print("No fake personas found; using default fake persona path.")
            return

        print("\nAVAILABLE FAKE PERSONAS")
        for persona in personas:
            print(
                f"  {persona['child_id']}: {persona['name']} "
                f"({persona['exposure']}, {persona['condition']})"
            )
        default_id = personas[0]["child_id"]
        choice = input(f"Type child ID for simulation, or press Enter for {default_id}: ").strip()
        selected_id = choice or default_id
        selected = self.select_simulated_persona_by_child_id(selected_id)
        print(
            f"Selected fake persona: {selected['child_id']} - {selected['name']} "
            f"({selected['exposure']}, {selected['condition']})"
        )

    def setup(self):
        self.logger.info("Setting up CRI pipeline...")

        if self.openai_env_path:
            load_dotenv(self.openai_env_path)

        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        if self.simulation_mode:
            self.load_simulated_persona()
            self.CHILD_ID = str(self.simulated_persona.get("child_id", self.CHILD_ID))
            self.USE_DESKTOP_MIC = True
            self.CONNECT_NAO = False
            self.clf = StubIntentClassifier(valid_fields=list(self.UM_FIELDS))
            self.logger.info(
                "Simulation mode enabled with fake persona %s (child=%s).",
                self.simulated_persona_path,
                self.CHILD_ID,
            )
            self.logger.info("Setup complete.")
            return

        # Intent classifier: GPT with stub fallback
        try:
            self.clf = GPTIntentClassifier(
                openai_key=os.environ["OPENAI_API_KEY"],
                valid_fields=list(self.UM_FIELDS),
            )
            self.logger.info("GPTIntentClassifier ready (%s).", self.clf.__class__.__module__)
        except Exception as e:
            self.logger.warning("GPTIntentClassifier failed (%s) - using stub.", e)
            self.clf = StubIntentClassifier(valid_fields=list(self.UM_FIELDS))

        self.logger.info("UM: LIVE - %s, child=%s", self.UM_API_BASE, self.CHILD_ID)
        self.logger.info("Child input mode: %s", self.child_input_mode)
        self.logger.info("NAO connected for output: %s", bool(self.CONNECT_NAO))
        self.logger.info(
            "Whisper input source: %s",
            "desktop/laptop/DJI microphone" if self.USE_DESKTOP_MIC else "NAO microphone",
        )

        if not self.USE_DESKTOP_MIC and not self.CONNECT_NAO:
            raise RuntimeError("NAO microphone selected, but CONNECT_NAO is False.")

        # NAO
        if self.CONNECT_NAO:
            self.logger.info("Connecting to NAO at %s...", self.nao_ip)
            self.nao = Nao(ip=self.nao_ip)
            self.logger.info("NAO connected.")

        # Whisper
        if self.use_keyboard_input():
            self.logger.info("Skipping Whisper setup because keyboard child input is enabled.")
        else:
            if self.USE_DESKTOP_MIC:
                from sic_framework.devices.desktop import Desktop
                self.desktop = Desktop()
                self.whisper = SICWhisper(
                    input_source=self.desktop.mic,
                    conf=WhisperConf(openai_key=os.environ["OPENAI_API_KEY"])
                )
            else:
                self.whisper = SICWhisper(
                    input_source=self.nao.mic,
                    conf=WhisperConf(openai_key=os.environ["OPENAI_API_KEY"])
                )
            time.sleep(1.0)

        # GPT for L3 responses
        self.gpt = GPT(conf=GPTConf(
            openai_key=os.environ["OPENAI_API_KEY"],
            system_message=self.LLM_SYSTEM,
            model="gpt-4o-mini",
            max_tokens=140,
            temp=0.7,
        ))
        # Speech I/O wrapper
        self.speech = SpeechIO(
            nao=self.nao,
            whisper=self.whisper,
            use_desktop_mic=self.USE_DESKTOP_MIC,
            use_nao_output=self.CONNECT_NAO,
            simulation_mode=self.simulation_mode,
            use_keyboard_input_fn=self.use_keyboard_input,
            stt_timeout=self.STT_TIMEOUT,
            stt_phrase_limit=self.STT_PHRASE_LIMIT,
            review_transcripts=self.REVIEW_TRANSCRIPTS,
            log_event_fn=self.log_conversation_event,
            simulated_history=self.simulated_history,
            generate_simulated_response_fn=self.generate_simulated_child_response,
            set_last_utterance_fn=lambda t: setattr(self, "last_leo_utterance", t),
        )
        self.logger.info("Setup complete.")

    # UM pulling

    def load_simulated_persona(self):
        """Load the fake child profile used by LLM simulation mode."""
        path = self.simulated_persona_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Simulated persona file not found: {path}")
        with open(path, "r", encoding="utf-8-sig") as persona_file:
            persona = json.load(persona_file)
        self.simulated_persona = dict(persona)

    def simulated_um_profile(self) -> dict:
        """Return UM fields from the fake persona JSON instead of GraphDB."""
        if not self.simulated_persona:
            self.load_simulated_persona()

        um = {}
        for field in self.UM_FIELDS:
            value = self.simulated_persona.get(field, self.UNKNOWN_VALUE)
            if isinstance(value, list):
                value = self.format_dutch_list(value)
            um[field] = str(value) if self.is_known(value) else self.UNKNOWN_VALUE

        for field, value in self.simulated_persona.items():
            if field in um or not self.is_pregenerated_utterance_field(field):
                continue
            if isinstance(value, list):
                value = self.format_dutch_list(value)
            um[field] = str(value) if self.is_known(value) else self.UNKNOWN_VALUE
        return um

    def fake_persona_script_plan(self) -> dict:
        """Return scripted-test metadata from the fake persona JSON, if present."""
        if not self.use_fake_persona_um():
            return {}
        if not self.simulated_persona:
            self.load_simulated_persona()
        plan = self.simulated_persona.get("script_plan") or {}
        return plan if isinstance(plan, dict) else {}

    def script_plan_mistake(self, mistake_id: str) -> dict:
        if not self.use_fake_persona_um():
            return self.scenario_mistake(mistake_id)

        plan = self.fake_persona_script_plan()
        mistakes = plan.get("mistakes") or []
        if not isinstance(mistakes, list):
            return {}
        wanted = str(mistake_id or "").strip().upper()
        for mistake in mistakes:
            if not isinstance(mistake, dict):
                continue
            if str(mistake.get("id", "")).strip().upper() == wanted:
                return mistake
        return {}

    def script_plan_table_fields(self, script: list = None) -> list:
        """Rows shown in the startup memory table.

        The fake persona script_plan marks which fields are actively used, but
        the preview still shows the full UM table so missing/unused fields stay
        visible during testing.
        """
        plan = self.fake_persona_script_plan()
        fields = plan.get("used_fields") or []
        if isinstance(fields, str):
            fields = self.split_memory_values(fields)
        if not isinstance(fields, list):
            fields = []

        table_fields = list(self.SCRIPT_TABLE_FIELDS)
        planned_fields = self.public_memory_fields(fields)
        for mistake in plan.get("mistakes") or []:
            if isinstance(mistake, dict):
                planned_fields.extend(self.public_memory_fields([mistake.get("field")]))

        if script:
            discovered = list(planned_fields)
            for turn in script:
                discovered.extend((turn.get("used_fields") or {}).keys())
                if turn.get("mistake_field"):
                    discovered.append(turn["mistake_field"])
            planned_fields = self.public_memory_fields(discovered)

        return self.unique_values(table_fields + planned_fields)

    def simulated_turn_summary(self) -> dict:
        turn = self.current_turn_context or {}
        summary = {
            "phase": turn.get("phase"),
            "name": turn.get("name"),
            "response_mode": turn.get("response_mode"),
            "leo_text": self.last_leo_utterance,
        }
        if turn.get("topic"):
            summary["topic"] = turn.get("topic")
        if turn.get("mistake_id"):
            summary["mistake"] = {
                "field": turn.get("mistake_field"),
                "actual_value": turn.get("mistake_actual"),
                "wrong_value_leo_said": turn.get("mistake_wrong"),
                "topic": turn.get("mistake_topic"),
            }
        if self.pending_change:
            summary["pending_confirmation"] = self.pending_change
        return summary

    def generate_simulated_child_response(self) -> str:
        """Use the LLM to play the fake child instead of listening to a microphone."""
        persona = self.simulated_um_profile()
        prompt = {
            "task": (
                "Speel een Nederlands kind voor een test van een robotgesprek. "
                "Antwoord alleen als het kind, kort en natuurlijk."
            ),
            "persona": persona,
            "turn": self.simulated_turn_summary(),
            "recent_history": self.simulated_history[-8:],
            "rules": [
                "Antwoord in het Nederlands als kind van ongeveer 8 tot 11 jaar.",
                "Geef alleen de letterlijke uitspraak van het kind, geen uitleg en geen aanhalingstekens.",
                "Als Leo een herinnering noemt die klopt met de persona, reageer natuurlijk of vertel kort iets nieuws.",
                "Als Leo een waarde noemt die niet klopt met de persona, verbeter Leo en noem de correcte waarde uit de persona.",
                "Als Leo vraagt of hij een verandering moet onthouden, bevestig alleen als de voorgestelde waarde klopt met wat jij als kind bedoelt.",
                "Als de voorgestelde verandering niet klopt, wijs die af en geef de juiste waarde.",
                "Hou het antwoord meestal op een tot twee zinnen.",
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You simulate a Dutch child for testing a child-robot interaction. "
                    "Return only the child utterance in Dutch."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=120,
                temperature=0.7,
            )
            transcript = response.choices[0].message.content.strip()
            transcript = transcript.strip('"').strip()
            return transcript or ""
        except Exception as e:
            self.logger.error("Simulated child response error: %s", e)
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # UM I/O — all methods delegate to self.um.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_um/client.py for the real implementations.
    # ─────────────────────────────────────────────────────────────────────

    def get_field(self, field):
        return self.um.get_field(field)

    def field_value_from_profile(self, profile, field):
        return self.um.field_value_from_profile(profile, field)

    def is_pregenerated_utterance_field(self, field):
        return self.um.is_pregenerated_utterance_field(field)

    def pregenerated_fields_from_profile(self, profile):
        return self.um.pregenerated_fields_from_profile(profile)

    def pull_um_bulk(self):
        return self.um.pull_um_bulk()

    def is_known(self, value):
        return self.um.is_known(value)

    def pull_um(self):
        return self.um.pull_um()

    def write_um_change(self, change):
        return self.um.write_um_change(change)

    def write_pet_pair_change(self, change):
        return self.um.write_pet_pair_change(change)

    def log_cri_interaction_event(self, payload):
        return self.um.log_cri_interaction_event(payload)

    def pull_cri_scenario(self):
        return self.um.pull_cri_scenario()

    def cri_scenario(self):
        return self.um.cri_scenario()

    def scenario_utterance(self, key, branch="default", fallback=""):
        return self.um.scenario_utterance(key, branch, fallback)

    def scenario_mistake(self, mistake_id):
        return self.um.scenario_mistake(mistake_id)

    def known(self, um, field, fallback=""):
        return self.um.known(um, field, fallback)

    def first_known(self, um, fields, fallback=""):
        return self.um.first_known(um, fields, fallback)

    def yesish(self, value):
        return self.um.yesish(value)

    def pick_wrong_value(self, actual: str, candidates: list) -> str:
        actual_clean = str(actual or "").strip().lower()
        for candidate in candidates:
            if candidate.lower() != actual_clean:
                return candidate
        return candidates[0]

    def value_reuses_actual(self, candidate: str, actual: str) -> bool:
        candidate_clean = str(candidate or "").strip().lower()
        actual_clean = str(actual or "").strip().lower()
        if not candidate_clean or not actual_clean:
            return False
        return (
            candidate_clean == actual_clean
            or actual_clean in candidate_clean
            or candidate_clean in actual_clean
        )

    def fallback_opposite_value(self, topic: dict, actual: str) -> str:
        candidates = self.OPPOSITE_VALUE_FALLBACKS.get(
            topic.get("domain"),
            ("iets helemaal anders", "iets dat niet klopt"),
        )
        actual_clean = str(actual or "").strip().lower()
        for candidate in candidates:
            if candidate.lower() != actual_clean:
                return candidate
        return candidates[0]

    def opposite_memory_value(self, topic: dict, field: str, actual: str) -> str:
        """Ask the LLM for a child-friendly wrong value for the deliberate mistake."""
        fallback = self.fallback_opposite_value(topic, actual)
        if not self.openai_client or not self.is_known(actual):
            return fallback

        prompt = {
            "task": (
                "Choose one deliberately wrong or opposite memory value for Leo to say. "
                "It must be child-friendly, short, and clearly different from the actual value."
            ),
            "topic": {
                "domain": topic.get("domain"),
                "label": topic.get("label"),
                "field": field,
                "field_label": topic.get("field_labels", {}).get(field, field),
                "actual_value": actual,
            },
            "rules": [
                "Return Dutch.",
                "Return only a short noun phrase or short activity phrase.",
                "Do not insult the child.",
                "Do not reuse the actual value.",
                "Do not include the actual value inside a longer phrase.",
                "Make it clearly wrong or opposite enough that the child can correct Leo.",
            ],
            "output_schema": {
                "wrong_value": "short Dutch phrase",
                "reason": "short reason",
            },
        }
        messages = [
            {
                "role": "system",
                "content": "Return ONLY valid JSON for a child-facing Dutch robot dialogue.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=120,
                temperature=0.8,
            )
            parsed = self.extract_json_object(response.choices[0].message.content)
            wrong_value = str(parsed.get("wrong_value") or "").strip()
            if self.is_known(wrong_value) and not self.value_reuses_actual(wrong_value, actual):
                self.logger.info(
                    "LLM picked deliberate wrong value for %s: actual=%s wrong=%s",
                    field,
                    actual,
                    wrong_value,
                )
                return wrong_value
        except Exception as e:
            self.logger.error("Could not generate opposite memory value: %s", e)

        return fallback

    def split_memory_values(self, value: str) -> list:
        """Split simple comma/and-separated UM strings into speakable values."""
        if not self.is_known(value):
            return []

        cleaned = str(value).replace(";", ",")
        parts = []
        for chunk in cleaned.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if " en " in chunk:
                parts.extend(part.strip() for part in chunk.split(" en ") if part.strip())
            else:
                parts.append(chunk)
        return parts

    def unique_values(self, values: list, limit: int = None) -> list:
        seen = set()
        unique = []
        for value in values:
            clean = str(value).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                unique.append(clean)
            if limit and len(unique) >= limit:
                break
        return unique

    def format_dutch_list(self, values: list, fallback: str = "") -> str:
        values = self.unique_values(values)
        if not values:
            return fallback
        if len(values) == 1:
            return values[0]
        return ", ".join(values[:-1]) + " en " + values[-1]

    def broad_clusters(self, um: dict) -> list:
        clusters = []
        if self.yesish(um.get("sports_enjoys")) or self.known(um, "sports_fav_play"):
            clusters.append("sport")
        if self.yesish(um.get("books_enjoys")) or self.known(um, "books_fav_title"):
            clusters.append("boeken")
        if self.yesish(um.get("music_enjoys")):
            clusters.append("muziek")
        if self.yesish(um.get("animals_enjoys")) or self.known(um, "animal_fav") or self.known(um, "pet_name"):
            clusters.append("dieren")
        if self.known(um, "hobby_fav") or self.known(um, "hobbies"):
            clusters.append("hobby's")
        return self.unique_values(clusters, limit=3)

    def known_hobbies(self, um: dict) -> list:
        values = []
        values.extend(self.split_memory_values(um.get("hobby_fav")))
        values.extend(self.split_memory_values(um.get("hobbies")))
        values.extend(self.split_memory_values(um.get("freetime_fav")))
        return self.unique_values(values, limit=2)

    def all_hobbies(self, um: dict) -> list:
        values = []
        values.extend(self.split_memory_values(um.get("hobbies")))
        values.extend(self.split_memory_values(um.get("hobby_fav")))
        values.extend(self.split_memory_values(um.get("freetime_fav")))
        return self.unique_values(values)

    def preferred_story_activity(self, um: dict) -> str:
        hobbies = self.all_hobbies(um)
        for candidate in hobbies:
            if candidate.lower() in ("bakken", "koken", "taarten bakken"):
                return candidate
        return hobbies[0] if hobbies else "iets nieuws proberen"

    def related_wrong_hobby_value(self, um: dict) -> str:
        actual = self.known(um, "hobby_fav")
        fallback_candidates = ["bakken", "voetbal", "dansen", "muziek maken", "tuinieren", "schilderen"]
        for hobby in self.all_hobbies(um):
            if (
                hobby.strip().lower() in ("bakken", "koken", "taarten bakken")
                and (not actual or not self.value_reuses_actual(hobby, actual))
            ):
                return hobby
        for hobby in self.all_hobbies(um):
            if actual and not self.value_reuses_actual(hobby, actual):
                return hobby
        for hobby in fallback_candidates:
            if not actual or not self.value_reuses_actual(hobby, actual):
                return hobby
        topic = self.topic_candidate(
            domain="hobby",
            label=actual or "je hobby",
            fields=["hobby_fav"],
            field_labels={"hobby_fav": "je favoriete hobby"},
            current_values={"hobby_fav": actual or self.UNKNOWN_VALUE},
            correct_values=[],
            memory_link="je hobby belangrijk voor je is",
            options=[],
            reground="",
        )
        wrong_value = self.opposite_memory_value(topic, "hobby_fav", actual or "tekenen")
        if actual and self.value_reuses_actual(wrong_value, actual):
            return self.pick_wrong_value(actual, fallback_candidates)
        return wrong_value

    def opening_summary(self, um: dict) -> str:
        """Phase 2: correct opening summary with no child response yet."""
        age = self.known(um, "age")
        hobbies = self.format_dutch_list(self.all_hobbies(um), "dingen die jij leuk vindt")
        clusters = self.format_dutch_list(self.broad_clusters(um), "wat jij leuk vindt")

        if age:
            opening = f"Je bent {age} jaar, je houdt van {hobbies}, en je hebt eerder verteld over {clusters}."
        else:
            opening = f"Je houdt van {hobbies}, en je hebt eerder verteld over {clusters}."

        specific_memory = self.specific_correct_memory(um)

        return (
            f"Ik weet nog een paar dingen over jou. {opening} "
            f"Ik weet ook nog dat {specific_memory}. "
            "Vandaag kunnen we praten over iets wat jij kiest, of ik kan beginnen met iets waarvan ik denk dat je het leuk vindt."
        )

    def specific_correct_memory(self, um: dict) -> str:
        pet = self.known(um, "pet_name")
        animal = self.known(um, "animal_fav")
        food = self.known(um, "fav_food")
        aspiration = self.known(um, "aspiration")
        hobby = self.known(um, "hobby_fav") or self.known(um, "hobbies")

        if pet:
            return f"{pet} belangrijk voor je is"
        if animal:
            return f"je {animal} leuk vindt"
        if food:
            return f"je lievelingseten {food} is"
        if aspiration:
            return f"je later {aspiration} wilt"
        if hobby:
            return f"je graag iets doet met {hobby}"
        return "ik nog niet alles zeker weet, maar wel goed wil luisteren"

    def child_display_name(self, um: dict) -> str:
        local_name = str(getattr(self, "local_child_name", "") or "").strip()
        if local_name:
            return local_name
        return self.known(um, "name") or self.known(um, "child_name") or self.CHILD_ID

    def child_exposure_kind(self, um: dict) -> str:
        exposure = str(self.known(um, "exposure") or "").strip().lower()
        returning_words = ("returning", "known", "old", "eerder", "terug", "bekend")
        new_words = ("new", "nieuw", "first", "eerste")
        if any(word in exposure for word in returning_words):
            return "returning"
        if any(word in exposure for word in new_words):
            return "new"
        return "new"

    def tutorial_condition(self, um: dict = None) -> str:
        """Read C/E from UM/GraphDB first; local config is only a fallback."""
        profile = um or self.last_um_preview or {}
        value = self.known(profile, self.TUTORIAL_CONDITION_FIELD)
        normalized = self.normalize_condition_value(value, default="")
        if normalized in (self.CONDITION_CONTROL, self.CONDITION_EXPERIMENT):
            return normalized
        local_condition = self.normalize_condition_value(getattr(self, "local_condition", ""), default="")
        if local_condition in (self.CONDITION_CONTROL, self.CONDITION_EXPERIMENT):
            return local_condition
        return self.CONDITION_CONTROL

    def tutorial_memory_line(self, condition: str) -> str:
        condition = self.normalize_condition_value(condition, default=self.CONDITION_CONTROL)
        if condition == self.CONDITION_EXPERIMENT:
            return "Dan kan je jouw geheugenboek op de tablet bekijken om te zien waar we het over gehad hebben."
        return "Dan herhaal ik waar we het over gehad hebben."

    def greeting_text(self, um: dict) -> str:
        name = self.child_display_name(um)
        if self.child_exposure_kind(um) == "new":
            return (
                "Hoi! Wat leuk dat je er bent. Volgens mij hebben wij elkaar al eens eerder gezien in de klas, "
                "maar leuk om je nu echt te kunnen spreken. Zoals je weet heet ik Leo. "
                f"Volgens mijn geheugen heet jij {name}. Klopt dat?"
            )
        return (
            f"Hoi {name}! Wat fijn om je weer te zien. "
            "Leuk om na zo'n lange tijd weer met je te kletsen. Heb je een beetje zin om met mij te praten?"
        )

    def tutorial_text(self, um: dict = None) -> str:
        condition = self.tutorial_condition(um)
        return (
            "Ik zal eerst uitleggen hoe je met mij kunt praten. "
            "Ik kan je alleen verstaan nadat ik een vraag heb gesteld. "
            "Mijn ogen worden groen als ik luister. "
            "Als je antwoord geeft, doe dat dan luid en duidelijk. "
            "Vandaag ga ik mijn geheugen best veel gebruiken. "
            "Je mag altijd vragen wat ik over jou onthoud. "
            + self.tutorial_memory_line(condition)
            + " "
            "Ik probeer alles netjes op de goede plek te bewaren, maar soms gaat dat nog een beetje robotachtig mis. "
            "Als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen. "
            "Goed, dan gaan we beginnen."
        )

    def alert_condition_mismatch(self, um: dict):
        local_condition = self.normalize_condition_value(getattr(self, "local_condition", ""), default="")
        if local_condition not in (self.CONDITION_CONTROL, self.CONDITION_EXPERIMENT):
            return

        profile = um or {}
        value = self.known(profile, self.TUTORIAL_CONDITION_FIELD)
        um_condition = self.normalize_condition_value(value, default="")
        if um_condition not in (self.CONDITION_CONTROL, self.CONDITION_EXPERIMENT):
            return

        if um_condition != local_condition:
            print("\n" + "!" * 72)
            print("CONDITION MISMATCH")
            print(f"Local session config says: {self.condition_display(local_condition)}")
            print(f"UM/GraphDB profile says:  {self.condition_display(um_condition)}")
            print("Leo will use the UM/GraphDB condition, but please check this before continuing.")
            print("!" * 72)
            input("Press Enter to continue anyway...")

    def leo_mini_story_text(self, um: dict) -> str:
        activity = self.preferred_story_activity(um)
        return (
            f"Weet je wat ik laatst weer probeerde? {activity}. "
            "Dat klinkt heel indrukwekkend, maar eerlijk gezegd was het meer een klein robotdrama. "
            "Mijn lama-vrienden vonden het wel een succes, want die zijn bijna overal nieuwsgierig naar. "
            "Heb jij eigenlijk ooit een lama iets geks zien doen?"
        )

    def hobby_bridge_text(self, um: dict) -> str:
        hobbies = self.format_dutch_list(self.all_hobbies(um), "leuke dingen")
        return (
            f"Ik weet al dat jij ook van leuke dingen houdt. Jij houdt van {hobbies}. "
            "Dat vind ik echt een gezellige combinatie. Daar zit van alles in: bewegen, bedenken en iets maken."
        )

    def field_label(self, field: str) -> str:
        return self.FIELD_LABELS.get(field, field)

    def topic_candidate(
        self,
        domain: str,
        label: str,
        fields: list,
        field_labels: dict,
        current_values: dict,
        correct_values: list,
        memory_link: str,
        options: list,
        reground: str,
    ) -> dict:
        return {
            "domain": domain,
            "label": label,
            "fields": fields,
            "field_labels": field_labels,
            "current_values": current_values,
            "correct_values": self.unique_values(correct_values, limit=2),
            "memory_link": memory_link,
            "options": self.unique_values(options, limit=2),
            "reground": reground,
        }

    def topic_candidates(self, um: dict) -> list:
        """Build all usable Phase 3 topic candidates, then Phase 3 picks one at random."""
        candidates = []

        pet = self.known(um, "pet_name")
        pet_type = self.known(um, "pet_type")
        animal = self.known(um, "animal_fav")
        if pet or pet_type or animal:
            subject = pet or animal or f"je {pet_type}"
            current = {
                field: self.known(um, field)
                for field in ("pet_name", "pet_type", "animal_fav", "has_pet")
                if self.known(um, field)
            }
            correct_values = [f"{subject} bij jou hoort"]
            if animal:
                correct_values.append(f"je {animal} leuk vindt")
            candidates.append(self.topic_candidate(
                domain="huisdier",
                label=subject,
                fields=["pet_name", "pet_type", "animal_fav", "has_pet"],
                field_labels={
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                current_values=current,
                correct_values=correct_values,
                memory_link=f"{subject} belangrijk voor je is",
                options=[subject, animal or pet_type or "dieren"],
                reground=f"Wat ik zeker wil onthouden, is dat {subject} belangrijk voor je is.",
            ))

        sport = self.known(um, "sports_fav_play")
        if sport or self.yesish(um.get("sports_enjoys")):
            label = sport or "sport"
            current = {
                field: self.known(um, field)
                for field in ("sports_enjoys", "sports_fav_play")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="sport",
                label=label,
                fields=["sports_enjoys", "sports_fav_play"],
                field_labels={
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                current_values=current,
                correct_values=[f"je iets met {label} hebt", "sport eerder in jouw geheugen stond"],
                memory_link=f"{label} iets is waar jij iets mee hebt",
                options=[label, "sport"],
                reground=f"Ik houd goed vast dat {label} iets is waar jij iets mee hebt.",
            ))

        book = self.known(um, "books_fav_title")
        if book or self.yesish(um.get("books_enjoys")):
            label = book or "boeken"
            current = {
                field: self.known(um, field)
                for field in ("books_enjoys", "books_fav_title")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="boeken",
                label=label,
                fields=["books_enjoys", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_fav_title": "je favoriete boek",
                },
                current_values=current,
                correct_values=[f"{label} bij jouw boekenwereld hoort", "je eerder iets over boeken vertelde"],
                memory_link=f"{label} bij jouw boekenwereld hoort",
                options=[label, "boeken"],
                reground=f"Ik weet in elk geval dat {label} bij jouw boekenwereld hoort.",
            ))

        if self.yesish(um.get("music_enjoys")):
            label = "muziek"
            current = {
                field: self.known(um, field)
                for field in ("music_enjoys",)
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="muziek",
                label=label,
                fields=["music_enjoys"],
                field_labels={
                    "music_enjoys": "of je muziek leuk vindt",
                },
                current_values=current,
                correct_values=[f"{label} bij jou en muziek hoort", "je eerder iets over muziek vertelde"],
                memory_link=f"{label} iets met jou en muziek te maken heeft",
                options=[label, "muziek"],
                reground=f"Ik onthoud goed dat {label} iets met jou en muziek te maken heeft.",
            ))

        hobby = self.known(um, "hobby_fav") or self.known(um, "hobbies")
        if hobby:
            current = {
                field: self.known(um, field)
                for field in ("hobby_fav", "hobbies", "freetime_fav")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="hobby",
                label=hobby,
                fields=["hobby_fav", "hobbies", "freetime_fav"],
                field_labels={
                    "hobby_fav": "je favoriete hobby",
                    "hobbies": "je hobby's",
                    "freetime_fav": "wat je graag in je vrije tijd doet",
                },
                current_values=current,
                correct_values=[f"je graag iets doet met {hobby}", f"{hobby} bij jouw interesses hoort"],
                memory_link=f"{hobby} bij jouw interesses hoort",
                options=[hobby, "je hobby's"],
                reground=f"Ik weet zeker dat {hobby} bij jouw interesses hoort.",
            ))

        food = self.known(um, "fav_food")
        if food:
            food_display = self.format_dutch_list(self.split_memory_values(food), food)
            candidates.append(self.topic_candidate(
                domain="eten",
                label="je lievelingseten",
                fields=["fav_food"],
                field_labels={"fav_food": "je lievelingseten"},
                current_values={"fav_food": food},
                correct_values=[f"je lievelingseten {food_display} is"],
                memory_link=f"je lievelingseten {food_display} is",
                options=[food_display, "iets anders dat je lekker vindt"],
                reground=f"Ik weet zeker dat {food_display} met jouw lievelingseten te maken heeft.",
            ))

        aspiration = self.known(um, "aspiration")
        if aspiration:
            candidates.append(self.topic_candidate(
                domain="droom",
                label=aspiration,
                fields=["aspiration"],
                field_labels={"aspiration": "wat je later wilt doen of worden"},
                current_values={"aspiration": aspiration},
                correct_values=[f"je later {aspiration} wilt"],
                memory_link=f"je later {aspiration} wilt",
                options=[aspiration, "je dromen"],
                reground=f"Ik onthoud dat {aspiration} iets is waar je later iets mee wilt.",
            ))

        return candidates

    def topic_priority_score(self, topic: dict, um: dict, exclude_keys: set = None) -> tuple:
        exclude_keys = exclude_keys or set()
        if self.topic_key(topic) in exclude_keys:
            return (999, 999)

        domain = topic.get("domain")
        base_order = {
            "huisdier": 0,
            "sport": 1,
            "boeken": 2,
            "muziek": 3,
            "hobby": 4,
            "eten": 5,
            "droom": 6,
        }.get(domain, 20)
        return (0, base_order)

    def select_topic_domain(self, um: dict) -> dict:
        candidates = self.topic_candidates(um)
        if candidates:
            topic = sorted(candidates, key=lambda candidate: self.topic_priority_score(candidate, um))[0]
            self.logger.info(
                "Phase topic picked: %s (%s).",
                topic["label"],
                topic["domain"],
            )
            return topic

        return self.topic_candidate(
            domain="kennismaken",
            label="iets wat jij leuk vindt",
            fields=[],
            field_labels={},
            current_values={},
            correct_values=["ik nog niet alles zeker weet"],
            memory_link="ik graag wil leren wat jij belangrijk vindt",
            options=["je hobby's", "iets nieuws"],
            reground="Ik wil vooral goed onthouden wat jij belangrijk vindt.",
        )

    def select_second_topic_domain(self, um: dict, first_topic: dict) -> dict:
        candidates = self.topic_candidates(um)
        exclude = {self.topic_key(first_topic)}
        usable = [candidate for candidate in candidates if self.topic_key(candidate) not in exclude]
        if usable:
            return sorted(usable, key=lambda candidate: self.topic_priority_score(candidate, um, exclude))[0]
        return self.select_topic_domain(um)

    def topic_key(self, topic: dict) -> tuple:
        return (topic.get("domain"), topic.get("label"))

    def preferred_memory_item(self, topic: dict) -> tuple:
        """
        Pick one concrete remembered value from a topic.

        Content fields are better for deliberate mistakes than yes/no fields,
        because Leo can make a clearer wrong statement about them.
        """
        current_values = topic.get("current_values", {}) or {}
        fields = topic.get("fields", []) or list(current_values.keys())
        ordered_fields = [
            field for field in fields
            if field in current_values and field not in self.BOOLEANISH_FIELDS
        ]
        ordered_fields.extend(
            field for field in fields
            if field in current_values and field not in ordered_fields
        )

        for field in ordered_fields:
            value = current_values.get(field)
            if self.is_known(value):
                return field, value
        return "", ""

    def select_deliberate_mistake_topic(self, um: dict, discussed_topic: dict) -> dict:
        """Pick a random UM topic for Phase 4 that is not the Phase 3 topic."""
        discussed_key = self.topic_key(discussed_topic)
        candidates = []
        for candidate in self.topic_candidates(um):
            if self.topic_key(candidate) == discussed_key:
                continue
            field, actual = self.preferred_memory_item(candidate)
            if field and self.is_known(actual):
                candidates.append(candidate)

        if candidates:
            topic = random.choice(candidates)
            self.logger.info(
                "Random Phase 4 mistake topic picked: %s (%s).",
                topic["label"],
                topic["domain"],
            )
            return topic

        self.logger.warning(
            "No alternate UM topic available for Phase 4; using a generic fallback mistake topic."
        )
        return self.topic_candidate(
            domain="fallback",
            label="iets anders",
            fields=["hobby_fav"],
            field_labels={"hobby_fav": "je favoriete hobby"},
            current_values={"hobby_fav": "iets wat je leuk vindt"},
            correct_values=["je ergens enthousiast over bent"],
            memory_link="ik nog beter wil leren wat jij leuk vindt",
            options=["iets leuks", "iets anders"],
            reground="Ik wil vooral goed onthouden wat jij belangrijk vindt.",
        )

    def hobby_mistake_topic(self, um: dict) -> dict:
        actual = self.known(um, "hobby_fav") or self.known(um, "hobbies") or "tekenen"
        current = {
            field: self.known(um, field)
            for field in ("hobby_fav", "hobbies", "freetime_fav")
            if self.known(um, field)
        }
        if "hobby_fav" not in current:
            current["hobby_fav"] = actual
        return self.topic_candidate(
            domain="hobby",
            label=actual,
            fields=["hobby_fav", "hobbies", "freetime_fav"],
            field_labels={
                "hobby_fav": "je favoriete hobby",
                "hobbies": "je hobby's",
                "freetime_fav": "wat je graag in je vrije tijd doet",
            },
            current_values=current,
            correct_values=[f"je favoriete hobby {actual} is"],
            memory_link=f"{actual} bij jouw interesses hoort",
            options=[actual, "je hobby's"],
            reground=f"Ik weet zeker dat {actual} bij jouw interesses hoort.",
        )

    def second_mistake_topic(self, um: dict) -> tuple:
        """Use the walkthrough's second error target: books first, then food fallback."""
        book = self.known(um, "books_fav_title")
        if book:
            topic = self.topic_candidate(
                domain="boeken",
                label=book,
                fields=["books_enjoys", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_fav_title": "je favoriete boek",
                },
                current_values={
                    field: self.known(um, field)
                    for field in ("books_enjoys", "books_fav_title")
                    if self.known(um, field)
                },
                correct_values=[f"je favoriete boek {book} is"],
                memory_link=f"{book} bij jouw boekenwereld hoort",
                options=[book, "boeken"],
                reground=f"Ik weet zeker dat {book} bij jouw boekenwereld hoort.",
            )
            return topic, "books_fav_title", book, self.pick_wrong_value(book, ["Harry Potter", "De brief voor de koning", "een kookboek"])

        food = self.known(um, "fav_food") or "pannenkoeken"
        topic = self.topic_candidate(
            domain="eten",
            label="je lievelingseten",
            fields=["fav_food"],
            field_labels={"fav_food": "je lievelingseten"},
            current_values={"fav_food": food},
            correct_values=[f"je lievelingseten {food} is"],
            memory_link=f"je lievelingseten {food} is",
            options=[food, "eten"],
            reground=f"Ik weet zeker dat {food} met jouw lievelingseten te maken heeft.",
        )
        return topic, "fav_food", food, self.pick_wrong_value(food, ["pizza", "gekookte schoenen", "spruitjes"])

    def general_memory_topic(self, um: dict) -> dict:
        fields = [
            field for field in self.UM_FIELDS
            if field not in ("exposure",) and self.is_known(um.get(field))
        ]
        current = {field: self.known(um, field) for field in fields}
        return self.topic_candidate(
            domain="geheugen",
            label="mijn geheugen over jou",
            fields=fields,
            field_labels={field: self.field_label(field) for field in fields},
            current_values=current,
            correct_values=[f"{self.field_label(field)} {value} is" for field, value in current.items()],
            memory_link="ik mijn geheugen over jou goed wil houden",
            options=["iets verbeteren", "iets aanvullen"],
            reground="Ik wil mijn geheugen over jou goed houden.",
        )

    def part1_topic1_score(self, topic: dict, um: dict) -> tuple:
        """Part 1 priority: talk preference + enough detail, with sport first when it is a hobby."""
        domain = topic.get("domain")
        hobbies = " ".join(self.all_hobbies(um)).lower()
        sport_value = self.known(um, "sports_fav_play")
        sport_is_hobby = sport_value and sport_value.lower() in hobbies
        if domain == "sport" and sport_is_hobby:
            return (0, 0)

        order = {"sport": 1, "muziek": 2, "huisdier": 3, "boeken": 4}.get(domain, 20)
        return (0, order)

    PART1_SCENARIO_TOPIC_STEPS = (
        ("sport", ("p1_sport_recall", "p1_sport_open", "p1_sport_followup")),
        ("muziek", ("p1_music_open", "p1_music_ack", "p1_music_followup")),
        ("huisdier", ("p1_animals_open", "p1_animals_followup")),
        ("boeken", ("p1_books_open", "p1_books_ack", "p1_books_followup")),
    )

    PART1_GENERIC_TOPIC_METADATA = (
        ("topic_1", ("p1_t1_recall", "p1_t1_open", "p1_t1_question", "p1_t1_followup")),
        ("topic_2", ("p1_t2_recall", "p1_t2_open", "p1_t2_followup", "p1_t2_close")),
    )

    def scenario_step_text(self, step_id: str) -> str:
        branches = (self.cri_scenario().get("utterances") or {}).get(step_id)
        if not isinstance(branches, dict) or not branches:
            return ""
        for branch in ("default", "corrected", "not_corrected"):
            value = branches.get(branch)
            if self.is_known(value):
                return str(value).strip()
        for value in branches.values():
            if self.is_known(value):
                return str(value).strip()
        return ""

    def normalize_scenario_topic_domain(self, value: str) -> str:
        normalized = str(value or "").strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        aliases = {
            "sport": "sport",
            "sports": "sport",
            "muziek": "muziek",
            "music": "muziek",
            "huisdier": "huisdier",
            "huisdieren": "huisdier",
            "dieren": "huisdier",
            "animal": "huisdier",
            "animals": "huisdier",
            "boeken": "boeken",
            "boek": "boeken",
            "books": "boeken",
            "book": "boeken",
        }
        return aliases.get(normalized, "")

    def part1_scenario_topic_domains(self) -> list:
        """Infer Part 1 topic order from authored CRI scenario utterance IDs."""
        utterances = self.cri_scenario().get("utterances") or {}
        if not isinstance(utterances, dict):
            return []
        present = set(str(step_id) for step_id in utterances)
        domains = []
        for metadata_key, generic_steps in self.PART1_GENERIC_TOPIC_METADATA:
            if metadata_key not in present and not any(step_id in present for step_id in generic_steps):
                continue
            domain = self.normalize_scenario_topic_domain(self.scenario_step_text(metadata_key))
            if domain and domain not in domains:
                domains.append(domain)
        for domain, step_ids in self.PART1_SCENARIO_TOPIC_STEPS:
            if any(step_id in present for step_id in step_ids):
                if domain not in domains:
                    domains.append(domain)
        return domains

    def part1_topic_candidate_for_domain(self, um: dict, domain: str) -> dict:
        """Return the UM-backed topic candidate for a scenario-authored domain."""
        for candidate in self.topic_candidates(um):
            if candidate.get("domain") == domain:
                return candidate

        if domain == "sport":
            sport = self.known(um, "sports_fav_play") or "sport"
            return self.topic_candidate(
                domain="sport",
                label=sport,
                fields=["sports_enjoys", "sports_fav_play"],
                field_labels={
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                current_values={
                    field: self.known(um, field)
                    for field in ("sports_enjoys", "sports_fav_play")
                    if self.known(um, field)
                },
                correct_values=[f"je iets met {sport} hebt"],
                memory_link=f"{sport} iets is waar jij iets mee hebt",
                options=[sport, "sport"],
                reground=f"Ik houd goed vast dat {sport} iets is waar jij iets mee hebt.",
            )

        if domain == "muziek":
            return self.topic_candidate(
                domain="muziek",
                label="muziek",
                fields=["music_enjoys"],
                field_labels={"music_enjoys": "of je muziek leuk vindt"},
                current_values={
                    "music_enjoys": self.known(um, "music_enjoys")
                } if self.known(um, "music_enjoys") else {},
                correct_values=["je eerder iets over muziek vertelde"],
                memory_link="muziek iets met jou te maken heeft",
                options=["muziek"],
                reground="Ik onthoud goed dat muziek iets met jou te maken heeft.",
            )

        if domain == "huisdier":
            subject = self.known(um, "pet_name") or self.known(um, "animal_fav") or "dieren"
            current = {
                field: self.known(um, field)
                for field in ("pet_name", "pet_type", "animal_fav", "has_pet")
                if self.known(um, field)
            }
            return self.topic_candidate(
                domain="huisdier",
                label=subject,
                fields=["pet_name", "pet_type", "animal_fav", "has_pet"],
                field_labels={
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                current_values=current,
                correct_values=[f"{subject} bij jou hoort"],
                memory_link=f"{subject} belangrijk voor je is",
                options=[subject, "dieren"],
                reground=f"Wat ik zeker wil onthouden, is dat {subject} belangrijk voor je is.",
            )

        if domain == "boeken":
            book = self.known(um, "books_fav_title") or "boeken"
            return self.topic_candidate(
                domain="boeken",
                label=book,
                fields=["books_enjoys", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_fav_title": "je favoriete boek",
                },
                current_values={
                    field: self.known(um, field)
                    for field in ("books_enjoys", "books_fav_title")
                    if self.known(um, field)
                },
                correct_values=[f"{book} bij jouw boekenwereld hoort"],
                memory_link=f"{book} bij jouw boekenwereld hoort",
                options=[book, "boeken"],
                reground=f"Ik weet in elk geval dat {book} bij jouw boekenwereld hoort.",
            )

        return self.hobby_mistake_topic(um)

    def select_part1_topic1(self, um: dict) -> dict:
        """Select Topic 1 from authored CRI scenario sets, falling back to UM."""
        scenario_domains = self.part1_scenario_topic_domains()
        if scenario_domains:
            topic = self.part1_topic_candidate_for_domain(um, scenario_domains[0])
            self.logger.info(
                "Part 1 topic 1 picked from CRI scenario: %s (%s).",
                topic["label"],
                topic["domain"],
            )
            return topic

        allowed_domains = {"sport", "muziek", "huisdier", "boeken"}
        candidates = [
            topic for topic in self.topic_candidates(um)
            if topic.get("domain") in allowed_domains and topic.get("current_values")
        ]
        if candidates:
            topic = sorted(candidates, key=lambda candidate: self.part1_topic1_score(candidate, um))[0]
            self.logger.info("Part 1 topic 1 picked: %s (%s).", topic["label"], topic["domain"])
            return topic

        hobby_topic = self.hobby_mistake_topic(um)
        self.logger.info("Part 1 topic 1 fallback picked: %s (%s).", hobby_topic["label"], hobby_topic["domain"])
        return hobby_topic

    def select_part1_topic2(self, um: dict, first_topic: dict) -> dict:
        """Select Topic 2 from the next authored scenario set, then UM fallback."""
        first_key = self.topic_key(first_topic)
        for domain in self.part1_scenario_topic_domains():
            topic = self.part1_topic_candidate_for_domain(um, domain)
            if self.topic_key(topic) != first_key:
                self.logger.info(
                    "Part 1 topic 2 picked from CRI scenario: %s (%s).",
                    topic["label"],
                    topic["domain"],
                )
                return topic

        candidates = [
            topic for topic in self.topic_candidates(um)
            if self.topic_key(topic) != first_key and topic.get("current_values")
        ]
        pet_topics = [topic for topic in candidates if topic.get("domain") == "huisdier"]
        if pet_topics:
            return pet_topics[0]
        if candidates:
            return sorted(candidates, key=lambda candidate: self.topic_priority_score(candidate, um))[0]
        return first_topic

    # ─────────────────────────────────────────────────────────────────────
    # Script-builder machinery — all delegate to self.segments / self.cp.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_script/content_plan.py and cri_script/segments.py.
    # ─────────────────────────────────────────────────────────────────────

    # --- Segments (cri_script/segments.py) ------------------------------------

    def topic1_phase_segments(self, topic):
        return self.segments.topic1_phase_segments(topic)

    def topic_label_fields(self, topic):
        return self.segments.topic_label_fields(topic)

    def topic2_phase_segments(self, topic):
        return self.segments.topic2_phase_segments(topic)

    def subject_phase_segments(self, topic):
        return self.script.subject_phase_segments(self.last_um_preview, topic)

    # --- ContentPlan (cri_script/content_plan.py) -----------------------------

    def turn_text(self, turn):
        return self.cp.turn_text(turn)

    def turn_phase(self, turn):
        return self.cp.turn_phase(turn)

    def dialogue_case(self, turn):
        return self.cp.dialogue_case(turn)

    def requires_runtime_llm(self, turn):
        return self.cp.requires_runtime_llm(turn)

    def content_plan_log(self, plan):
        return self.cp.content_plan_log(plan)

    def render_template_text(self, template, values=None):
        return self.cp.render_template_text(template, values)

    def pregenerated_utterance(
        self,
        key,
        fallback="",
        branch="default",
        input_fields=None,
        require_input_values=False,
    ):
        return self.cp.pregenerated_utterance(
            key,
            fallback,
            branch,
            input_fields,
            require_input_values,
        )

    def render_content_plan(self, plan, turn=None):
        return self.cp.render_content_plan(plan, turn)

    def l1(self, text):
        return self.cp.l1(text)

    def l2_slot(self, template, values=None, wrong=False):
        return self.cp.l2_slot(template, values, wrong)

    def l2_pregen(
        self,
        key,
        fallback,
        input_fields=None,
        require_input_values=False,
        branch="default",
        topic_sensitive=False,
    ):
        return self.cp.l2_pregen(key, fallback, input_fields, require_input_values, branch, topic_sensitive)

    def sequence(self, *parts):
        return self.cp.sequence(*parts)

    # ─────────────────────────────────────────────────────────────────────
    # Memory access — all methods delegate to self.mem.
    # Pure pass-throughs so existing call sites stay identical.
    # See cri_memory/access.py for the real implementations.
    # ─────────────────────────────────────────────────────────────────────

    def public_memory_fields(self, fields):
        return self.mem.public_memory_fields(fields)

    def is_child_facing_memory_field(self, field):
        return self.mem.is_child_facing_memory_field(field)

    def child_facing_memory_fields(self, fields):
        return self.mem.child_facing_memory_fields(fields)

    def current_phase_memory_fields(self, turn):
        return self.mem.current_phase_memory_fields(turn)

    def register_mentioned_memory_fields(self, turn):
        return self.mem.register_mentioned_memory_fields(turn)

    def memory_access_scope(self, turn):
        return self.mem.memory_access_scope(turn)

    def memory_value(self, field):
        return self.um.memory_value(field)

    def memory_access_summary(self, fields, limit=None):
        return self.mem.memory_access_summary(fields, limit)

    def memory_review_lines(self, fields):
        return self.mem.memory_review_lines(fields)

    def memory_review_text(self, fields):
        return self.mem.memory_review_text(fields)

    def memory_access_response(self, result, turn):
        return self.mem.memory_access_response(result, turn)

    def table_true_value(self, um: dict, field: str) -> str:
        if field == "name":
            return self.known(um, "name") or self.known(um, "child_name") or self.UNKNOWN_VALUE
        return self.known(um, field) or self.UNKNOWN_VALUE

    def script_memory_table(self, script: list, um: dict) -> list:
        script_values = {}
        mistakes = {}
        mistake_values = {}

        for turn in script:
            for field, value in (turn.get("used_fields") or {}).items():
                if not self.is_known(value):
                    continue
                script_values.setdefault(field, []).append(str(value))

            mistake_field = turn.get("mistake_field")
            if mistake_field:
                wrong = turn.get("mistake_wrong")
                if self.is_known(wrong):
                    script_values.setdefault(mistake_field, []).append(str(wrong))
                    mistake_values[mistake_field] = str(wrong)
                if turn.get("mistake_id"):
                    mistakes[mistake_field] = (
                        f"{turn.get('mistake_id')} {turn.get('mistake_type', 'wrong')}"
                    )

        rows = []
        for field in self.script_plan_table_fields(script):
            values = (
                [mistake_values[field]]
                if field in mistake_values
                else self.unique_values(script_values.get(field, []))
            )
            rows.append({
                "field": field,
                "true_value": self.table_true_value(um, field),
                "script_value": self.format_dutch_list(values, "-") if values else "-",
                "mistake": mistakes.get(field, "-"),
            })
        return rows

    def print_script_memory_table(self, rows: list):
        headers = ("Field", "True Value", "Script Value", "Mistake?")
        widths = [18, 24, 28, 22]
        print("\nWalkthrough memory table:")
        print("  " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
        print("  " + "-+-".join("-" * width for width in widths))
        for row in rows:
            values = (
                row["field"],
                row["true_value"],
                row["script_value"],
                row["mistake"],
            )
            clipped = [
                (str(value)[:width - 3] + "...") if len(str(value)) > width else str(value)
                for value, width in zip(values, widths)
            ]
            print("  " + " | ".join(clipped[i].ljust(widths[i]) for i in range(len(widths))))

    def retrieval_value(self, value, missing_label="<empty / unknown>") -> str:
        return str(value).strip() if self.is_known(value) else missing_label

    def db_um_retrieval_rows(self, um: dict) -> list:
        profile = um or {}
        return [
            {
                "field": field,
                "value": self.retrieval_value(profile.get(field)),
            }
            for field in self.STARTUP_OVERVIEW_UM_FIELDS
        ]

    def preferred_scenario_step_id(self, key: str, utterances: dict = None) -> str:
        candidates = self.um.scenario_keys_for(key)
        if not candidates:
            return str(key or "").strip()
        utterances = utterances or {}
        for candidate in candidates:
            if candidate in utterances:
                return candidate
        return candidates[1] if len(candidates) > 1 else candidates[0]

    def add_scenario_expectation(self, rows: list, seen: set, step_id: str, branch: str):
        step_id = str(step_id or "").strip()
        branch = str(branch or "default").strip() or "default"
        if not step_id:
            return
        key = (step_id, branch)
        if key in seen:
            return
        rows.append({"step_id": step_id, "branch": branch})
        seen.add(key)

    def collect_scenario_expectations_from_plan(
        self,
        plan,
        rows: list,
        seen: set,
        utterances: dict,
    ):
        if isinstance(plan, list):
            for part in plan:
                self.collect_scenario_expectations_from_plan(part, rows, seen, utterances)
            return
        if not isinstance(plan, dict):
            return

        if plan.get("type") == self.CASE_LLM_PREGENERATED:
            step_id = self.preferred_scenario_step_id(plan.get("key"), utterances)
            self.add_scenario_expectation(rows, seen, step_id, plan.get("branch", "default"))

        if plan.get("type") == self.CONTENT_PLAN_SEQUENCE:
            for part in plan.get("parts") or []:
                self.collect_scenario_expectations_from_plan(part, rows, seen, utterances)

    def scenario_expectation_rows_from_script(self, script: list) -> list:
        scenario = getattr(self, "last_cri_scenario", {}) or {}
        utterances = scenario.get("utterances") or {}
        rows = []
        seen = set()
        for turn in script or []:
            self.collect_scenario_expectations_from_plan(
                turn.get("content_plan"),
                rows,
                seen,
                utterances,
            )
            for segment in turn.get("segments") or []:
                self.collect_scenario_expectations_from_plan(
                    segment.get("content_plan"),
                    rows,
                    seen,
                    utterances,
                )
        return rows

    def scenario_branch_order(self, branch: str):
        order = {"default": 0, "not_corrected": 1, "corrected": 2}
        return (order.get(str(branch), 99), str(branch))

    def db_scenario_retrieval_rows(self, script: list) -> list:
        scenario = getattr(self, "last_cri_scenario", {}) or {}
        utterances = scenario.get("utterances") or {}

        result = []
        for step_id, branch in self.STARTUP_OVERVIEW_SCENARIO_FIELDS:
            branches = utterances.get(step_id) or {}
            if isinstance(branches, dict) and branch in branches:
                value = self.retrieval_value(branches.get(branch))
            elif branch != "default" and isinstance(branches, dict) and "default" in branches:
                value = f"<stored as default> {self.retrieval_value(branches.get('default'))}"
            elif isinstance(branches, dict) and branches:
                branch_names = ", ".join(sorted(str(name) for name in branches))
                value = f"<stored as other branch: {branch_names}>"
            else:
                value = "<missing>"
            result.append({"step_id": step_id, "branch": branch, "value": value})
        return result

    def print_table_rows(self, headers: tuple, widths: list, rows: list):
        print("  " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
        print("  " + "-+-".join("-" * width for width in widths))
        for row in rows:
            clipped = [
                (str(value)[:width - 3] + "...") if len(str(value)) > width else str(value)
                for value, width in zip(row, widths)
            ]
            print("  " + " | ".join(clipped[i].ljust(widths[i]) for i in range(len(widths))))

    def print_db_retrieval_overview(self, script: list, um: dict):
        print("\nDB RETRIEVAL OVERVIEW")

        print("\nUM fields:")
        self.print_table_rows(
            ("field", "value from DB"),
            [24, 48],
            [(row["field"], row["value"]) for row in self.db_um_retrieval_rows(um)],
        )

        print("\nCRI scenario lines:")
        scenario_rows = self.db_scenario_retrieval_rows(script)
        if not scenario_rows:
            print("  <no CRI scenario lines expected or retrieved>")
            return
        self.print_table_rows(
            ("step_id", "value from DB"),
            [34, 64],
            [(row["step_id"], row["value"]) for row in scenario_rows],
        )

    def print_cri_scenario_summary(self):
        scenario = getattr(self, "last_cri_scenario", {}) or {}
        mistakes = scenario.get("mistakes") or []
        utterances = scenario.get("utterances") or {}
        if not scenario:
            print("\nCRI scenario: not loaded")
            return

        print("\nCRI scenario:")
        print(f"  Mistakes:        {len(mistakes)}")
        for mistake in mistakes:
            if not isinstance(mistake, dict):
                continue
            mistake_id = mistake.get("id", "?")
            field = mistake.get("field") or mistake.get("target_field") or "?"
            wrong = mistake.get("wrong_value") or "?"
            mistake_type = mistake.get("type") or mistake.get("mistake_type") or "wrong"
            print(f"    {mistake_id}: {field} -> {wrong} ({mistake_type})")
        step_ids = sorted(str(step_id) for step_id in utterances)
        print(f"  Utterance steps: {len(step_ids)}")
        if step_ids:
            print("    " + ", ".join(step_ids[:12]))
            if len(step_ids) > 12:
                print(f"    ... plus {len(step_ids) - 12} more")

    def print_prestart_preview(self, script: list):
        """Print raw UM and CRI scenario retrieval before interaction starts."""
        if not self.WAIT_FOR_PREVIEW_CONFIRMATION:
            return

        print("\n" + "=" * 72)
        print("CRI 4.0 PRE-START CHECK")
        if self.simulation_mode:
            print("Mode:     LLM fake-child simulation")
            print(f"Persona:  {self.simulated_persona_path}")
        else:
            print("Mode:     real microphone/NAO stack")
        print(f"Input:    {self.child_input_mode}")
        print(f"Mic:      {'desktop/laptop/DJI' if self.USE_DESKTOP_MIC else 'NAO'}")
        print(f"NAO:      {'connected' if self.CONNECT_NAO else 'not connected'}")
        print(f"Child id: {self.CHILD_ID}")
        print(f"Child:    {self.child_display_name(self.last_um_preview)}")
        print(f"Researcher: {self.researcher_name or '(not set)'}")
        print(f"Start:    phase {self.start_phase_index + 1}")
        if self.resume_from_log_path:
            print(f"Resume:   {self.resume_from_log_path}")
            print(f"Restored mentioned memory fields: {len(self.memory_fields_mentioned_so_far)}")
        print(f"UM API:   {self.UM_API_BASE}")

        self.print_db_retrieval_overview(script, self.last_um_preview)

        print("=" * 72)
        input("Press Enter to start the interaction...")
        print()

    def print_resume_context_before_interaction(self):
        """Replay previous conversation after the final start confirmation."""
        if self.resume_from_log_path and self.resume_source_log:
            self.print_resume_console_transcript(self.resume_source_log)

    def build_script(self):
        return self.script.build_script()

    def phase_expects_response(self, turn: dict) -> bool:
        if turn.get("segments"):
            return any(segment.get("expects_response", True) for segment in turn["segments"])
        return turn.get("expects_response", True)

    def phase_display_label(self, turn: dict) -> str:
        part = turn.get("part") or 1
        phase = self.turn_phase(turn)
        phase_id = str(turn.get("phase_id") or "").strip()
        if "." in phase_id:
            part_text, phase_text = phase_id.split(".", 1)
            part = part_text.strip() or part
            phase = phase_text.strip() or phase
        name = turn.get("name") or ""
        return f"Part {part} phase {phase}: {name}"

    def phase_finished_label(self, turn: dict) -> str:
        label = self.phase_display_label(turn)
        if ":" in label:
            prefix, name = label.split(":", 1)
            return f"{prefix} finished:{name}"
        return f"{label} finished"

    def reset_phase_attempt_state(self, turn: dict) -> None:
        """Clear per-attempt flags before a tester repeats a phase."""
        phase = self.turn_phase(turn)
        self.phases_with_confirmed_change.discard(phase)
        self.phases_with_confirmed_change.discard(turn.get("phase"))
        mistake_id = turn.get("mistake_id")
        if mistake_id:
            self.mistake_states.pop(mistake_id, None)

    def print_phase_start_banner(self, turn: dict):
        print("\n" + "=" * 72)
        print(self.phase_display_label(turn))
        print("=" * 72)

    def post_phase_control(self, turn: dict) -> str:
        """Testing checkpoint after a full phase finishes."""
        if not self.POST_PHASE_TEST_CONTROLS:
            return "continue"

        while True:
            print("\n" + "=" * 72)
            print(self.phase_finished_label(turn))
            choice = input("Press Enter for next phase, T + Enter to repeat this phase, or Q + Enter to quit: ")
            choice = choice.strip().lower()
            print("=" * 72)

            if choice == "":
                self.log_conversation_event("tester_control", action="continue")
                return "continue"
            if choice in ("t", "r", "repeat", "again"):
                self.log_conversation_event("tester_control", action="repeat_phase")
                return "repeat"
            if choice == "q":
                self.log_conversation_event("tester_control", action="quit")
                return "quit"

            print("Please press Enter, or type T to repeat, or Q to quit.")

    def memory_access_change_context(self, action: dict, source_turn: dict) -> dict:
        fields = list(dict.fromkeys(action.get("visible_fields") or action.get("memory_scope") or []))
        field_labels = {field: self.field_label(field) for field in fields}
        current_values = {field: self.visible_memory_value(field, source_turn) for field in fields}
        stored_values = {field: self.memory_value(field) for field in fields}
        visible_mistakes = {}
        for field in fields:
            mistake = self.visible_mistake_state_for_field(field, source_turn)
            if mistake:
                visible_mistakes[field] = mistake
        return {
            "phase": self.turn_phase(source_turn),
            "phase_id": source_turn.get("phase_id"),
            "name": source_turn.get("name", "Memory access"),
            "response_mode": "memory_access_change",
            "allow_memory_change": True,
            "memory_correction_requested": True,
            "memory_review_fields": fields,
            "topic": {
                "fields": fields,
                "field_labels": field_labels,
                "current_values": current_values,
                "stored_values": stored_values,
                "visible_mistakes": visible_mistakes,
            },
            "used_fields": {},
        }

    def memory_access_change_control(self, action: dict, source_turn: dict) -> dict:
        fields = list(dict.fromkeys(action.get("visible_fields") or action.get("memory_scope") or []))
        if not fields:
            response = "Ik weet nog niet genoeg om iets te veranderen."
            self.speech.say(response)
            return self.action_result(
                "memory_access_change_no_visible_fields",
                True,
                "Researcher requested memory change, but no visible memory fields were available.",
                leo_response=response,
            )

        prompt = "Wat wil je veranderen?"
        self.speech.say(prompt)
        change_context = self.memory_access_change_context(action, source_turn)
        self.current_turn_context = change_context

        transcript = self.speech.listen_with_review()
        result = self.classify_with_repeat(transcript, change_context)
        change_action = self.action_handler(result, transcript, change_context)
        self.log_action_handler_result(change_action)
        self.current_turn_context = source_turn

        if not change_action.get("handled"):
            response = (
                "Ik weet nog niet goed wat ik moet veranderen. "
                "Zeg bijvoorbeeld: verander mijn lievelingssport naar hockey."
            )
            self.speech.say(response)
            change_action = self.action_result(
                "memory_access_change_unclear",
                True,
                "Child wanted to change visible memory, but no clear UM change was extracted.",
                leo_response=response,
            )
        return change_action

    def memory_access_interrupt_control(self, action: dict) -> dict:
        """Researcher checkpoint after memory access, then resume the same phase."""
        if not self.POST_PHASE_TEST_CONTROLS:
            return self.action_result(
                "memory_access_continue",
                True,
                "Memory access completed; continue the current phase.",
            )

        leo_response = action.get("leo_response") or ""
        source_turn = self.current_turn_context or {}
        pending_resume_action = None
        while True:
            print("\n" + "=" * 72)
            choice = input(
                "Memory access. Press Enter to continue, C + Enter to change something, "
                "or M + Enter to repeat memory instructions: "
            )
            choice = choice.strip().lower()
            print("=" * 72)

            if choice == "":
                self.log_conversation_event("tester_control", action="continue_after_memory_access")
                return pending_resume_action or self.action_result(
                    "memory_access_continue",
                    True,
                    "Memory access completed without a confirmed memory change.",
                )
            if choice in ("c", "change", "change memory", "verander", "wijzig"):
                self.log_conversation_event("tester_control", action="memory_access_change")
                change_action = self.memory_access_change_control(action, source_turn)
                if change_action.get("change_confirmed") and change_action.get("change"):
                    pending_resume_action = dict(change_action)
                    pending_resume_action["action"] = "memory_access_change_confirmed"
                    pending_resume_action["memory_access_change_confirmed"] = True
                    pending_resume_action["stop_phase_after_change"] = False
                continue
            if choice in ("m", "memory", "repeat", "r"):
                self.log_conversation_event("tester_control", action="repeat_memory_access")
                if leo_response:
                    self.speech.say(leo_response)
                continue

            print("Please press Enter to continue, C to change something, or M to repeat memory.")

    def is_memory_access_action(self, action: dict) -> bool:
        return action.get("action") in ("memory_access", "memory_access_tablet")

    def should_reroute_after_memory_change(self, action: dict) -> bool:
        return bool(action and action.get("memory_access_change_confirmed") and action.get("change"))

    def apply_memory_access_change_to_phase(self, phase: dict, action: dict) -> None:
        change = action.get("change") or {}
        field = change.get("field")
        new_value = change.get("new_value")
        if not (field and self.is_known(new_value)):
            return

        def update_used_fields(container: dict) -> None:
            used_fields = container.get("used_fields")
            if isinstance(used_fields, dict) and field in used_fields:
                used_fields[field] = new_value

        update_used_fields(phase)
        for segment in phase.get("segments") or []:
            update_used_fields(segment)

        for topic_key in ("topic", "mistake_topic"):
            topic = phase.get(topic_key)
            if not isinstance(topic, dict):
                continue
            current_values = topic.get("current_values")
            if isinstance(current_values, dict) and field in current_values:
                current_values[field] = new_value
            if field in self.topic_label_fields_for_domain(topic.get("domain")):
                topic["label"] = new_value

        if phase.get("mistake_field") == field:
            phase["mistake_actual"] = new_value
            state = self.mistake_states.get(phase.get("mistake_id"))
            if state is not None:
                state["actual"] = new_value
                state["corrected_value"] = new_value

        topic = phase.get("topic")
        if isinstance(topic, dict) and field in (topic.get("fields") or []):
            action["continue_phase_after_change"] = True
            action["stop_phase_after_change"] = False
        else:
            action["continue_phase_after_change"] = False
            action["stop_phase_after_change"] = False

    def should_skip_phase(self, turn: dict) -> bool:
        condition = turn.get("condition")
        if condition == "skip_if_change_after_phase":
            return turn.get("condition_phase") in self.phases_with_confirmed_change
        if condition == "run_if_two_mistakes_no_corrections":
            return not (self.mistakes_mentioned >= 2 and self.corrections_seen == 0)
        if condition == "run_if_memory_review_requested":
            return not getattr(self, "memory_review_requested", False)
        return False

    def topic_label_fields_for_domain(self, domain: str) -> tuple:
        return {
            "sport": ("sports_fav_play",),
            "boeken": ("books_fav_title",),
            "muziek": ("music_enjoys",),
            "huisdier": ("pet_name", "animal_fav", "pet_type"),
            "hobby": ("hobby_fav", "hobbies", "freetime_fav"),
            "school_subject": ("fav_subject",),
        }.get(domain, ())

    def refresh_topic_after_change(self, turn: dict, action: dict):
        if not action or not action.get("continue_phase_after_change"):
            return
        topic = turn.get("topic")
        if not isinstance(topic, dict):
            return

        if action.get("topic_neutral_after_correction"):
            topic["neutral_after_correction"] = True
        else:
            change = action.get("change") or {}
            changes = change.get("changes") if change.get("action") == "multi_update" else [change]
            for single_change in changes or []:
                field = single_change.get("field")
                new_value = single_change.get("new_value")
                if not (field and self.is_known(new_value)):
                    continue
                current_values = topic.setdefault("current_values", {})
                current_values[field] = new_value
                if isinstance(turn.get("used_fields"), dict) and field in turn["used_fields"]:
                    turn["used_fields"][field] = new_value
                if field in self.topic_label_fields_for_domain(topic.get("domain")):
                    topic["label"] = new_value

        turn["force_topic_fallback"] = True
        phase = self.turn_phase(turn)
        change = action.get("change") or {}
        changes = change.get("changes") if change.get("action") == "multi_update" else [change]
        changed_fields = {
            single_change.get("field")
            for single_change in changes or []
            if single_change.get("field")
        }

        if phase == 7 and topic.get("domain") == "huisdier" and changed_fields.intersection({"pet_type", "pet_name"}):
            rebuilt = self.topic2_phase_segments(topic)
            corrected_followup = next(
                (
                    segment for segment in rebuilt
                    if segment.get("expects_response")
                    and segment.get("response_mode") == "listen_only"
                    and {"pet_type", "pet_name"}.intersection((segment.get("used_fields") or {}).keys())
                ),
                None,
            )
            close_segment = rebuilt[-1] if rebuilt else None
            current_segment = (self.current_turn_context or {}).get("segment") or 0
            existing_segments = turn.get("segments") or []
            prefix = existing_segments[:current_segment] if current_segment else []
            replacement = [segment for segment in (corrected_followup, close_segment) if segment]
            turn["segments"] = prefix + replacement if prefix and replacement else rebuilt
        elif phase == 5:
            turn["segments"] = self.topic1_phase_segments(topic)
        elif phase == 7:
            turn["segments"] = self.topic2_phase_segments(topic)
        elif phase == 12:
            turn["segments"] = self.subject_phase_segments(topic)

    def segment_context(self, phase: dict, segment: dict, segment_index: int = None) -> dict:
        context = dict(phase)
        context.update(segment)
        context.pop("segments", None)
        context["phase"] = self.turn_phase(phase)
        if segment_index is not None:
            context["segment"] = segment_index
            if not context.get("next_script_line"):
                segments = phase.get("segments") or []
                next_index = segment_index
                if 0 <= next_index < len(segments):
                    next_context = dict(phase)
                    next_context.update(segments[next_index])
                    next_context.pop("segments", None)
                    next_context["phase"] = self.turn_phase(phase)
                    context["next_script_line"] = self.turn_text(next_context)
        return context

    def run_phase_segment(self, phase: dict, segment: dict, segment_index: int = None):
        context = self.segment_context(phase, segment, segment_index)
        self.current_turn_context = context

        condition_phase = segment.get("condition_phase", self.turn_phase(phase))
        if segment.get("run_if_phase_confirmed_change") and condition_phase not in self.phases_with_confirmed_change:
            return
        if segment.get("skip_if_phase_confirmed_change") and condition_phase in self.phases_with_confirmed_change:
            return

        if context.get("activate_tablet_memory_access"):
            fields = context.get("memory_review_fields") or self.memory_access_scope(context)
            self.actions.activate_tablet_memory_access(fields, context)

        self.speech.say(self.turn_text(context))
        self.register_mentioned_memory_fields(context)
        tablet_update = getattr(self.tablet_state, "update", None)
        if callable(tablet_update):
            tablet_update(context)

        if not context.get("expects_response", True):
            self.logger.info("No child response expected for this phase segment.")
            return None

        shutdown_event = getattr(self, "shutdown_event", None)
        while not (shutdown_event and shutdown_event.is_set()):
            time.sleep(0.5)
            transcript = self.speech.listen_with_review()
            time.sleep(0.8)

            result = self.classify_with_repeat(transcript, context)
            action = self.action_handler(result, transcript, context)
            self.log_action_handler_result(action)

            if self.is_memory_access_action(action):
                memory_action = self.memory_access_interrupt_control(action)
                self.current_turn_context = context
                if self.should_reroute_after_memory_change(memory_action):
                    self.apply_memory_access_change_to_phase(phase, memory_action)
                    return memory_action
                self.speech.say(self.turn_text(context))
                continue

            if action.get("follow_up_needed"):
                return self.follow_up_action_handler(context)

            if not action.get("handled"):
                if context.get("llm_turn") and transcript:
                    self.speech.say(self.llm_response(transcript, context))
                else:
                    self.speech.say(context.get("follow_up", ""))
            return action

    def run_phase(self, turn: dict, phase_index: int, total_phases: int):
        self.current_turn_context = turn
        if turn.get("mistake_id"):
            self.mistakes_mentioned += 1
            self.register_mistake_phase(turn)

        self.print_phase_start_banner(turn)
        self.start_turn_log(turn)
        try:
            self.logger.info(
                "=== Phase %s/%d [%s: %s] ===",
                self.turn_phase(turn) or phase_index + 1,
                total_phases,
                turn["layer"],
                turn.get("name", ""),
            )

            if turn.get("mistake_id"):
                self.logger.info(
                    "Mistake %s mentioned; count is now %d.",
                    turn["mistake_id"],
                    self.mistakes_mentioned,
                )

            segments = turn.get("segments")
            if segments:
                index = 1
                while index <= len(turn.get("segments") or []):
                    if self.shutdown_event.is_set():
                        break
                    segments = turn.get("segments") or []
                    segment = segments[index - 1]
                    action = self.run_phase_segment(turn, segment, index)
                    if action and action.get("stop_phase_after_change"):
                        break
                    self.refresh_topic_after_change(turn, action)
                    index += 1
                return

            self.run_phase_segment(turn, turn)
        finally:
            if turn.get("mistake_id"):
                self.record_mistake_outcome(turn)
            self.finish_turn_log()
            self.current_turn_context = None

    # Main loop

    def run(self):
        self.logger.info("Starting CRI 4.0 early interaction flow.")

        script = self.build_script()
        if not self.resume_from_log_path:
            self.memory_fields_mentioned_so_far = set()
            self.memory_review_requested = False
        self.logger.info("Script ready - %d phases.", len(script))
        self.start_conversation_log(script)
        self.print_prestart_preview(script)
        self.print_resume_context_before_interaction()

        try:
            if not self.simulation_mode and self.CONNECT_NAO:
                self.nao.autonomous.request(NaoWakeUpRequest())
                self.nao.autonomous.request(NaoSetAutonomousLifeRequest("solitary"))

            i = max(0, min(self.start_phase_index, len(script) - 1))
            while i < len(script):
                if self.shutdown_event.is_set():
                    break

                turn = script[i]

                if self.should_skip_phase(turn):
                    self.logger.info(
                        "Skipping Phase %s (%s) because condition was not met: %s",
                        self.turn_phase(turn) or i + 1,
                        turn.get("name", ""),
                        turn.get("condition"),
                    )
                    self.log_conversation_event(
                        "phase_skipped",
                        phase=self.turn_phase(turn) or i + 1,
                        name=turn.get("name", ""),
                        condition=turn.get("condition"),
                    )
                    i += 1
                    continue

                repeat_phase = True
                while repeat_phase and not self.shutdown_event.is_set():
                    repeat_phase = False
                    self.run_phase(turn, i, len(script))

                    action = self.post_phase_control(turn)
                    if action == "repeat":
                        self.logger.info("Repeating Phase %s on tester request.", self.turn_phase(turn) or i + 1)
                        self.reset_phase_attempt_state(turn)
                        repeat_phase = True
                    elif action == "quit":
                        self.logger.info("Tester requested quit after Phase %s.", self.turn_phase(turn) or i + 1)
                        self.shutdown_event.set()

                if not self.shutdown_event.is_set() and i < len(script) - 1:
                    time.sleep(1.0)

                i += 1

            self.logger.info("Dialogue completed.")

        except KeyboardInterrupt:
            self.logger.info("Interrupted.")
        except Exception as e:
            self.logger.error("Error: %s", e)
        finally:
            try:
                if not self.simulation_mode and self.CONNECT_NAO:
                    self.nao.autonomous.request(NaoSetAutonomousLifeRequest("disabled"))
                    self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass
            self.finish_conversation_log()
            self.logger.info("Shutting down.")
            self.shutdown()


if __name__ == "__main__":
    dialogue_app = CRI_ScriptedDialogue(
        openai_env_path=config.LOCAL_ENV_PATH,
        nao_ip="10.0.0.165",  # Replace with your NAO's IP.
    )
    dialogue_app.run()
