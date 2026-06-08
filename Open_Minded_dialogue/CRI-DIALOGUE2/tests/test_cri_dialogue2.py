import importlib.util
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUTER_DIR = PACKAGE_DIR.parent
REPO_ROOT = OUTER_DIR.parent
LOCAL_DIR = OUTER_DIR / "_local"
SCRIPT_PATH = PACKAGE_DIR / "CRI-BRANCH-BASIC4_0.py"
LAUNCHER_PATH = LOCAL_DIR / "launchers" / "run_cri_dialogue2.py"
SHARED_LAUNCHER_PATH = OUTER_DIR / "launchers" / "run_cri_dialogue2.py"


def load_cri_module():
    module_names = (
        "config",
        "speech_io",
        "session_setup",
        "tablet_state",
        "cri_logger",
        "cri_um",
        "cri_memory",
        "cri_actions",
        "cri_script",
        "cri_classifier",
    )
    for name in module_names:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(PACKAGE_DIR))
    spec = importlib.util.spec_from_file_location("cri_dialogue2_for_tests", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_xlsx_scenario_converter():
    script_path = REPO_ROOT / "OM_Ontology_Database" / "scripts" / "xlsx_to_json_scenarios.py"
    spec = importlib.util.spec_from_file_location("xlsx_to_json_scenarios_for_tests", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cri_module = load_cri_module()
CRI = cri_module.CRI_ScriptedDialogue
IntentResult = cri_module.IntentResult


class DummyLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeOpenAIClient:
    def __init__(self, payloads=None):
        self.payloads = list(payloads or [])
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = self.payloads.pop(0) if self.payloads else "Prima."
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


class FakeSpeech:
    def __init__(self, heard=None):
        self.heard = list(heard or [])
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)

    def listen_with_review(self):
        return self.heard.pop(0) if self.heard else ""

    @staticmethod
    def strip_non_bmp(text):
        return cri_module.SpeechIO.strip_non_bmp(text)


class FakeHalo:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeSTTRecorder:
    def __init__(self, text="nee", sleep_seconds=0):
        self.text_value = text
        self.sleep_seconds = sleep_seconds
        self.halo = FakeHalo()
        self.spinner = True
        self.microphone_states = []
        self.queue_cleared = False
        self.aborted = False
        self.shutdown_called = False
        self.audio_queue = queue.Queue()
        self.recorded_audio_queue = queue.Queue()
        self.frames = []
        self.last_frames = []
        self.audio_buffer = []
        self.audio_buffer_metadata = []
        self.last_words_buffer = []
        self.text_storage = []
        self.audio = None
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.last_transcription_metadata = None
        self.last_preroll_selection = None
        self._pending_preroll_selection = None
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.continuous_listening = False
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.is_recording = False
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.wakeword_detected = False
        self.listen_start = 0
        self.recording_start_time = 0
        self.recording_start_monotonic = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.speech_end_silence_start = 0
        self.speech_end_silence_candidate_start = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.interrupt_stop_event = threading.Event()

    def set_microphone(self, enabled):
        self.microphone_states.append(enabled)

    def clear_audio_queue(self):
        self.queue_cleared = True

    def text(self):
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return self.text_value

    def abort(self):
        self.aborted = True

    def shutdown(self):
        self.shutdown_called = True


class FakeCudaBackend:
    def __init__(self, available=False, name="Fake CUDA GPU", fail_if_checked=False):
        self.available = available
        self.name = name
        self.fail_if_checked = fail_if_checked
        self.available_checks = 0

    def is_available(self):
        self.available_checks += 1
        if self.fail_if_checked:
            raise AssertionError("CUDA should not be checked for this case")
        return self.available

    def get_device_name(self, _index):
        return self.name


class FakeMpsBackend:
    def __init__(self, available=False):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorchModule:
    def __init__(self, cuda_available=False, cuda_name="Fake CUDA GPU", mps_available=False, fail_cuda_check=False):
        self.cuda = FakeCudaBackend(cuda_available, cuda_name, fail_cuda_check)
        self.backends = SimpleNamespace(mps=FakeMpsBackend(mps_available))
        self.__version__ = "test"
        self.version = SimpleNamespace(cuda="test")


class FakeCTranslate2Module:
    def __init__(self, cuda_count=0, cpu_types=None, fail_cuda_count=False):
        self.cuda_count = cuda_count
        self.cpu_types = set(cpu_types or {"int8", "float32"})
        self.fail_cuda_count = fail_cuda_count
        self.cuda_count_checks = 0

    def get_cuda_device_count(self):
        self.cuda_count_checks += 1
        if self.fail_cuda_count:
            raise AssertionError("CUDA device count should not be checked for this case")
        return self.cuda_count

    def get_supported_compute_types(self, device):
        if device != "cpu":
            return set()
        return set(self.cpu_types)


class FakeNaoChannel:
    def __init__(self):
        self.requests = []

    def request(self, request):
        self.requests.append(request)


class FakeNao:
    def __init__(self):
        self.tts = FakeNaoChannel()
        self.leds = FakeNaoChannel()
        self.autonomous = FakeNaoChannel()
        self.motion = FakeNaoChannel()


def make_app(openai_payloads=None):
    app = object.__new__(CRI)
    app.logger = DummyLogger()
    app.openai_client = FakeOpenAIClient(openai_payloads)
    app.nao = None
    app.whisper = None
    app.gpt = None
    app.clf = cri_module.StubIntentClassifier(valid_fields=list(CRI.UM_FIELDS))
    app.desktop = None
    app.speech = FakeSpeech()
    app.tablet_state = SimpleNamespace(
        reset=lambda: None,
        update=lambda turn: None,
        refresh=lambda phase=None: None,
        reveal_change=lambda **kwargs: None,
    )

    app.mistakes_mentioned = 0
    app.corrections_seen = 0
    app.mistake_states = {}
    app.last_um_preview = {}
    app.pending_change = None
    app.conversation_log = None
    app.current_turn_log = None
    app.conversation_log_started_monotonic = None
    app.conversation_log_time_offset = 0.0
    app.session_config = {}
    app.resume_from_log_path = None
    app.resume_source_log = {}
    app.local_child_name = ""
    app.researcher_name = ""
    app.local_condition = ""
    app.start_phase_index = 0
    app.simulation_mode = False
    app.child_input_mode = "keyboard"
    app.simulated_persona = {}
    app.simulated_persona_path = str(PACKAGE_DIR / "fake_personas" / "noor_1001.json")
    app.simulated_history = []
    app.last_leo_utterance = ""
    app.current_turn_context = None
    app.phases_with_confirmed_change = set()
    app.memory_fields_mentioned_so_far = set()
    app.memory_review_requested = False
    app.last_cri_scenario = {}
    app.last_cri_scenario_loaded = False
    app.pregenerated_rewrite_cache = {}

    app.SIMULATED_PERSONA_DIR = str(PACKAGE_DIR / "fake_personas")
    app.SIMULATION_WRITE_PERSONA_FILE = False
    app.USE_FAKE_PERSONA_UM = True
    app.CONVERSATION_LOG_ENABLED = True
    app.CONVERSATION_LOG_ROOT = str(LOCAL_DIR / "conversations")
    app.SESSION_CONFIG_PATH = str(LOCAL_DIR / "session_config.local.json")
    app.SESSION_STATE_PATH = str(LOCAL_DIR / "session_state.json")
    app.SESSION_ROSTER_DIR = str(LOCAL_DIR / "session_rosters")
    app.SESSION_ROSTER_PATH = str(LOCAL_DIR / "session_rosters" / "active.json")
    app.ROSTER_PATH = str(LOCAL_DIR / "test_config.txt")
    app.WAIT_FOR_OPERATOR_TABLET_REVEAL = False
    app.TABLET_REVEAL_WAIT_SECONDS = 0

    app.conv_log = cri_module.ConversationLogger(app)
    app.resume = cri_module.ResumeHelper(app)
    app.session_setup = cri_module.SessionSetup(app)
    app.um = cri_module.UMClient(app)
    app.mem = cri_module.MemoryAccess(app)
    app.nudge = cri_module.NudgeManager(app)
    app.actions = cri_module.ActionHandler(app)
    app.cp = cri_module.ContentPlan(app)
    app.segments = cri_module.Segments(app)
    app.script = cri_module.ScriptBuilder(app)
    return app


def sample_um():
    unknown = CRI.UNKNOWN_VALUE
    um = {field: unknown for field in CRI.UM_FIELDS}
    um.update({
        "name": "Noor",
        "exposure": "returning",
        "condition": "C",
        "age": "9",
        "hobbies": "tekenen, tuinieren, lego bouwen",
        "hobby_fav": "tekenen",
        "sports_enjoys": "ja",
        "sports_fav_play": "zwemmen",
        "books_enjoys": "ja",
        "books_fav_title": "De Gorgels",
        "music_enjoys": "ja",
        "animals_enjoys": "ja",
        "animal_fav": "dolfijn",
        "has_pet": "ja",
        "pet_type": "kat",
        "pet_name": "Momo",
        "freetime_fav": "stripfiguren tekenen",
        "fav_food": "pannenkoeken",
        "fav_subject": "natuur",
        "school_strength": "taal",
        "school_difficulty": "rekenen",
        "aspiration": "dierenarts worden",
        "role_model": "haar moeder",
        "interest": "dieren en natuur",
        "has_best_friend": "ja",
    })
    return um


class CRIDialogue2Tests(unittest.TestCase):
    def capture_pet_pair_writes(self, app, writes):
        def fake_write_pet_pair(change):
            writes.extend(dict(single_change) for single_change in change.get("changes", []))
            return True

        app.write_pet_pair_change = fake_write_pet_pair
        app.write_um_change = lambda change: self.fail(
            "Pet type/name corrections should use write_pet_pair_change."
        )

    def test_xlsx_scenario_converter_stores_all_utterances_as_default_branch(self):
        converter = load_xlsx_scenario_converter()

        scenario = converter.row_to_scenario({
            "child_id": "701",
            "M1_target_field": "hobby_fav",
            "M1_wrong_value": "padel",
            "p1_m1_followup_wrong_hobby": "Wat vind je het leukste aan padel?",
            "p1_m2_postcorrection_true_food": "Boerenkool klinkt gezellig.",
            "p1_hobby_bridge_comment": "Dat is een gezellige combinatie.",
        })

        branches = {
            utterance["step_id"]: utterance["branch"]
            for utterance in scenario["utterances"]
        }
        self.assertEqual(branches["p1_m1_followup_wrong_hobby"], "default")
        self.assertEqual(branches["p1_m2_postcorrection_true_food"], "default")
        self.assertEqual(branches["p1_hobby_bridge_comment"], "default")

    def test_pull_um_missing_child_skips_per_field_fallback(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.CHILD_ID = "000"
        app.local_child_name_cri = "Unika"

        response = SimpleNamespace(status_code=404)
        with patch("cri_um.client.requests.get", return_value=response) as requests_get:
            with patch.object(app.um, "get_field", side_effect=AssertionError("per-field fallback should be skipped")) as get_field:
                um = app.um.pull_um()

        requests_get.assert_called_once_with("http://localhost:8000/api/um/000", timeout=8)
        get_field.assert_not_called()
        self.assertEqual(um["name"], "Unika")
        self.assertEqual(app.last_um_preview, um)
        for field in CRI.UM_FIELDS:
            if field != "name":
                self.assertEqual(um[field], CRI.UNKNOWN_VALUE)

    def test_shareable_paths_point_to_inner_package_and_local_folder(self):
        config = cri_module.config

        self.assertEqual(Path(config.PACKAGE_ROOT), PACKAGE_DIR)
        self.assertEqual(Path(config.OUTER_ROOT), OUTER_DIR)
        self.assertEqual(Path(config.LOCAL_ROOT), LOCAL_DIR)
        self.assertEqual(Path(config.SIMULATED_PERSONA_DIR), PACKAGE_DIR / "fake_personas")
        self.assertEqual(Path(config.CONVERSATION_LOG_ROOT), LOCAL_DIR / "conversations")
        self.assertEqual(Path(config.SESSION_CONFIG_PATH), LOCAL_DIR / "session_config.local.json")
        self.assertEqual(Path(config.SESSION_STATE_PATH), LOCAL_DIR / "session_state.json")
        self.assertEqual(Path(config.LOCAL_ENV_PATH), LOCAL_DIR / ".env")
        self.assertEqual(Path(config.SESSION_ROSTER_DIR), LOCAL_DIR / "session_rosters")
        self.assertEqual(Path(config.SESSION_ROSTER_PATH), LOCAL_DIR / "session_rosters" / "active.json")
        self.assertTrue((PACKAGE_DIR / ".example_env").exists())
        self.assertTrue((PACKAGE_DIR / "_local_template" / "session_rosters" / "example_day.json").exists())
        self.assertTrue((OUTER_DIR / "launchers" / "start_cri_dialogue2_stack.bat").exists())
        self.assertTrue((OUTER_DIR / "launchers" / "start_cri_dialogue2.bat").exists())
        self.assertEqual(config.CHILD_INPUT_MODE, "keyboard")
        self.assertFalse((PACKAGE_DIR / ".env").exists())
        self.assertFalse((PACKAGE_DIR / "conversations").exists())

    def test_fake_personas_are_bundled_balanced_and_complete(self):
        app = make_app()
        personas = app.available_fake_personas()

        self.assertEqual([persona["child_id"] for persona in personas], ["1001", "1002", "1003", "1004"])
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "C"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "E"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["exposure"] == "new"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["exposure"] == "returning"), 2)

        for persona in personas:
            app.simulated_persona_path = persona["path"]
            app.simulated_persona = {}
            app.load_simulated_persona()
            profile = app.simulated_um_profile()
            self.assertEqual(set(app.UM_FIELDS), set(profile.keys()))
            self.assertTrue(all(profile[field] for field in app.UM_FIELDS))
            self.assertIn("script_plan", app.simulated_persona)
            self.assertEqual(
                {mistake["id"] for mistake in app.simulated_persona["script_plan"]["mistakes"]},
                {"M1", "M2"},
            )

    def test_session_setup_selects_persona_by_child_id_and_reads_condition(self):
        app = make_app()
        selected = app.select_simulated_persona_by_child_id("1002")
        app.load_simulated_persona()

        self.assertEqual(selected["name"], "Mila")
        self.assertEqual(selected["exposure"], "new")
        self.assertEqual(app.session_condition_from_um("1002"), "E")

    def test_roster_session_config_sets_child_and_input_mode(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_roster_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.SESSION_ROSTER_DIR = temp_dir
        roster_path = Path(temp_dir) / "study_day.json"
        roster_path.write_text(
            json.dumps({
                "researcher_name": "Sander",
                "default_input_mode": "keyboard",
                "children": [
                    {
                        "child_id": "test_boy_1",
                        "child_name": "Sam",
                        "condition": "E",
                        "start_phase_index": 0,
                    }
                ],
            }),
            encoding="utf-8",
        )

        rosters = app.session_setup.list_session_rosters()
        loaded = app.session_setup.load_session_roster(rosters[0])
        config = app.session_setup.session_config_from_roster_child(
            loaded,
            loaded["children"][0],
            rosters[0],
        )
        app.apply_session_config(config)

        self.assertEqual(config["mode"], "roster")
        self.assertEqual(app.CHILD_ID, "test_boy_1")
        self.assertEqual(app.local_child_name, "Sam")
        self.assertEqual(app.researcher_name, "Sander")
        self.assertEqual(app.local_condition, "E")
        self.assertEqual(app.child_input_mode, "keyboard")

    def test_roster_session_defaults_to_keyboard_input(self):
        app = make_app()
        loaded = {
            "researcher_name": "Sander",
            "children": [
                {
                    "child_id": "test_boy_1",
                    "child_name": "Sam",
                    "condition": "E",
                }
            ],
        }

        config = app.session_setup.session_config_from_roster_child(
            loaded,
            loaded["children"][0],
            "example_day.json",
        )
        app.apply_session_config(config)

        self.assertEqual(config["child_input_mode"], "keyboard")
        self.assertEqual(app.child_input_mode, "keyboard")

    def test_run_mode_defaults_to_keyboard_without_prompt(self):
        app = make_app()
        app.ASK_RUN_MODE_AT_START = True
        app.session_config = {}
        app.child_input_mode = "keyboard"

        with patch("builtins.input", side_effect=AssertionError("input prompt should not be shown")):
            app.configure_run_mode()

        self.assertEqual(app.child_input_mode, "keyboard")

    def test_single_roster_file_is_selected_without_prompt(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_roster_auto_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.SESSION_ROSTER_DIR = temp_dir
        roster_path = Path(temp_dir) / "example_day.json"
        roster_path.write_text(
            json.dumps({"children": [{"child_id": "test_boy_1"}]}),
            encoding="utf-8",
        )

        selected = app.session_setup.select_session_roster_path()

        self.assertEqual(Path(selected), roster_path)

    def test_greeting_and_tutorial_route_by_exposure_and_condition(self):
        app = make_app()
        returning_c = {"name": "Noor", "exposure": "returning", "condition": "C"}
        new_e = {"name": "Mila", "exposure": "new", "condition": "E"}

        self.assertIn("weer te zien", app.greeting_text(returning_c))
        self.assertIn("Volgens mijn geheugen heet jij Mila", app.greeting_text(new_e))
        self.assertIn("Dan herhaal ik waar we het over gehad hebben", app.tutorial_text(returning_c))
        self.assertIn("geheugenboek op de tablet bekijken", app.tutorial_text(new_e))
        self.assertNotIn("categorieen tikken", app.tutorial_text(returning_c))
        self.assertIn("categorieen tikken", app.tutorial_text(new_e))

    def test_tutorial_condition_can_fall_back_to_local_config(self):
        app = make_app()
        app.local_condition = "E"
        um = sample_um()
        um["condition"] = ""

        self.assertEqual(app.tutorial_condition(um), "E")
        self.assertIn("geheugenboek op de tablet bekijken", app.tutorial_text(um))
        self.assertIn("categorieen tikken", app.tutorial_text(um))

    def test_condition_display_uses_control_and_experiment_labels(self):
        app = make_app()

        self.assertEqual(app.normalize_condition_value("C1"), "C")
        self.assertEqual(app.normalize_condition_value("C2"), "E")
        self.assertEqual(app.normalize_condition_value("condition_1"), "C")
        self.assertEqual(app.normalize_condition_value("condition_2"), "E")
        self.assertEqual(
            app.condition_display("C"),
            "C (Control: conversational-only memory access)",
        )
        self.assertEqual(
            app.condition_display("E"),
            "E (Experiment: transmedial metaphor-supported memory access)",
        )

    def test_phase_finished_label_uses_part_phase_numbering(self):
        app = make_app()
        turn = {
            "phase": 10,
            "part": 2,
            "phase_id": "2.1",
            "name": "School joke transition",
        }

        self.assertEqual(
            app.phase_finished_label(turn),
            "Part 2 phase 1 finished: School joke transition",
        )

    def test_start_phase_parser_accepts_global_phase_numbers_and_script_ids(self):
        app = make_app()

        self.assertEqual(app.TOTAL_SCRIPT_PHASES, 19)
        self.assertEqual(app.parse_phase_index("1"), 0)
        self.assertEqual(app.parse_phase_index("10"), 9)
        self.assertEqual(app.parse_phase_index("19"), 18)
        self.assertEqual(app.parse_phase_index("2.1"), 9)
        self.assertEqual(app.parse_phase_index("2.4"), 12)
        self.assertEqual(app.parse_phase_index("3.1"), 13)
        self.assertEqual(app.parse_phase_index("3.4"), 16)
        self.assertEqual(app.parse_phase_index("3.5"), 16)
        self.assertEqual(app.parse_phase_index("3.4/5"), 16)
        self.assertEqual(app.parse_phase_index("3.6"), 17)
        self.assertEqual(app.parse_phase_index("3.7"), 17)
        self.assertEqual(app.parse_phase_index("3.6/7"), 17)
        self.assertEqual(app.parse_phase_index("3.8"), 18)
        self.assertEqual(app.parse_phase_index("phase 2.1"), 9)
        self.assertEqual(app.parse_phase_index("2,1"), 9)
        self.assertEqual(app.parse_phase_index("2.5", default_index=4), 4)

    def test_start_phase_display_shows_global_and_script_phase_id(self):
        app = make_app()

        self.assertEqual(app.session_setup.start_phase_display(0), "1 (1.1)")
        self.assertEqual(app.session_setup.start_phase_display(9), "10 (2.1)")
        self.assertEqual(app.session_setup.start_phase_display(16), "17 (3.4/5)")
        self.assertEqual(app.session_setup.start_phase_display(17), "18 (3.6/7)")
        self.assertEqual(app.session_setup.start_phase_display(18), "19 (3.8)")

    def test_graphdb_pet_nodes_are_exposed_as_dialogue_aliases(self):
        app = make_app()
        profile = {
            "scalars": {
                "child_name": {"value": "Sam"},
            },
            "nodes": {
                "pets": [
                    {"value": "Hond", "extra_props": {"petName": "Buddy"}},
                    {"value": "Konijn", "extra_props": {"petName": "Stamper"}},
                ],
            },
        }

        self.assertEqual(app.um.field_value_from_profile(profile, "name"), "Sam")
        self.assertEqual(app.um.field_value_from_profile(profile, "pet_type"), "Hond en Konijn")
        self.assertEqual(app.um.field_value_from_profile(profile, "pet_name"), "Buddy en Stamper")
        self.assertEqual(
            app.um.field_value_from_profile(profile, "pets"),
            "Buddy (Hond) en Stamper (Konijn)",
        )

    def test_bulk_profile_reads_current_and_legacy_node_keys(self):
        app = make_app()
        profile = {
            "scalars": {
                "child_name": {"value": "Sam"},
                "condition": {"value": "E"},
            },
            "nodes": {
                "LIKES_HOBBY": [
                    {"value": "voetbal"},
                    {"value": "gamen"},
                ],
                "hasFavouriteHobby": [
                    {"value": "voetbal"},
                ],
                "PLAYS_SPORT": [
                    {"value": "zwemmen"},
                ],
                "hasFavouriteFood": [
                    {"value": "shoarma"},
                    {"value": "frietjes"},
                ],
                "hasPetInstance": [
                    {"value": "Hond", "extra_props": {"petName": "Buddy"}},
                ],
            },
        }

        self.assertEqual(app.um.field_value_from_profile(profile, "hobbies"), "voetbal en gamen")
        self.assertEqual(app.um.field_value_from_profile(profile, "hobby_fav"), "voetbal")
        self.assertEqual(app.um.field_value_from_profile(profile, "sports_fav_play"), "zwemmen")
        self.assertEqual(app.um.field_value_from_profile(profile, "fav_food"), "shoarma en frietjes")
        self.assertEqual(app.um.field_value_from_profile(profile, "pets"), "Buddy (Hond)")

    def test_cri_scenario_utterances_and_mistakes_can_feed_dialogue(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M1",
                    "target_field": "hobby_fav",
                    "wrong_value": "gamen",
                    "mistake_type": "related-but-wrong",
                    "spt_level": "orientation",
                    "step": 4,
                }
            ],
            "utterances": {
                "p1_leo_ministory_opening": {
                    "default": "[STUB] Graph opening."
                },
                "p1_hobby_bridge_comment": {
                    "default": "[STUB] Dat is echt een actieve mix."
                },
                "p1_m1_followup_wrong_hobby": {
                    "default": "[STUB] Wat is jouw allerliefste game?"
                },
            },
        }

        self.assertEqual(
            app.pregenerated_utterance("hobbies_bridge", "fallback"),
            "Dat is echt een actieve mix.",
        )
        self.assertEqual(
            app.pregenerated_utterance("leo_ministory_opening", "fallback"),
            "Graph opening.",
        )
        self.assertEqual(
            app.pregenerated_utterance("m1_wrong_followup", "fallback", branch="not_corrected"),
            "Wat is jouw allerliefste game?",
        )

        mistake = app.script_plan_mistake("M1")
        self.assertEqual(mistake["field"], "hobby_fav")
        self.assertEqual(mistake["type"], "related-but-wrong")
        self.assertEqual(mistake["wrong_value"], "gamen")
        self.assertEqual(mistake["spt_layer"], "orientation")

    def test_build_script_carries_db_spt_level_into_mistake_phase(self):
        app = make_app()
        app.local_child_name = "Sam"
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {},
            "mistakes": [
                {
                    "id": "M1",
                    "target_field": "hobby_fav",
                    "wrong_value": "gamen",
                    "mistake_type": "related-but-wrong",
                    "spt_level": "orientation",
                    "step": 1,
                },
            ],
        }

        script = app.build_script()
        phase16 = next(turn for turn in script if turn.get("mistake_id") == "M1")

        self.assertEqual(phase16["spt_layer"], "orientation")

    def test_build_script_uses_part1_phase_sequence_and_mistakes(self):
        app = make_app()
        app.local_child_name = "Sam"
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_hobby_bridge_comment": {
                    "default": "[STUB] Daar zit van alles in: creatief bezig zijn."
                }
            },
            "mistakes": [],
        }

        script = app.build_script()

        self.assertEqual(
            [turn["name"] for turn in script],
            [
                "Greeting",
                "Tutorial",
                "Leo mini-story",
                "Correct hobby bridge",
                "Topic 1",
                "Mistake 1 - hobby_fav",
                "Topic 2",
                "Mistake 2 - fav_food",
                "Nudge",
                "School joke transition",
                "Robot school self-disclosure",
                "Correct fav_subject + connection to interests",
                "Mistake 3 - school_strength",
                "Bridge from school to future",
                "Leo self-disclosure",
                "role_model rapport",
                "Mistake 4 - aspiration + reflection",
                "Explicit memory inspection",
                "Closing",
            ],
        )
        self.assertEqual([turn["phase"] for turn in script], list(range(1, 20)))
        self.assertEqual(
            [turn["phase_id"] for turn in script],
            [f"1.{phase}" for phase in range(1, 10)]
            + ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.4/5", "3.6/7", "3.8"],
        )
        self.assertEqual(script[5]["part"], 1)
        self.assertEqual(script[5]["phase_id"], "1.6")
        self.assertEqual(script[5]["script_phase"], "part1_mistake1")
        phase16 = script[5]
        self.assertEqual(phase16["segments"][0]["response_mode"], "mistake_interpretation")
        self.assertTrue(phase16["segments"][0]["memory_correction_available"])
        self.assertEqual(phase16["segments"][0]["memory_correction_field"], "hobby_fav")
        self.assertNotIn("memory_correction_available", phase16["segments"][1])
        phase18 = script[7]
        self.assertEqual(phase18["phase_id"], "1.8")
        self.assertEqual(phase18["mistake_field"], "fav_food")
        self.assertEqual(phase18["segments"][0]["response_mode"], "mistake_interpretation")
        self.assertTrue(phase18["segments"][0]["memory_correction_available"])
        self.assertEqual(phase18["segments"][0]["memory_correction_field"], "fav_food")
        self.assertNotIn("memory_correction_available", phase18["segments"][1])
        self.assertEqual(script[9]["part"], 2)
        self.assertEqual(script[9]["phase_id"], "2.1")
        self.assertEqual(script[9]["script_phase"], "part2_school_joke_transition")
        self.assertEqual(script[9]["segments"][0]["response_mode"], "school_joke_transition")
        self.assertEqual(len(script[9]["segments"]), 1)
        self.assertEqual(script[9]["layer"], "L1")
        self.assertIn(
            "Zeg... als we het toch over jouw hobby's en lievelingsdingen hebben",
            app.turn_text(script[9]["segments"][0]),
        )
        phase22 = script[10]
        self.assertEqual(phase22["part"], 2)
        self.assertEqual(phase22["phase_id"], "2.2")
        self.assertEqual(phase22["script_phase"], "part2_robot_school_self_disclosure")
        self.assertEqual(phase22["segments"][0]["response_mode"], "robot_school_guess")
        self.assertEqual(phase22["segments"][0]["l3"]["script_phase"], "part2_school")
        self.assertEqual(phase22["segments"][0]["l3"]["topic"], "school")
        self.assertEqual(phase22["segments"][0]["l3"]["response_function"], "wrap_up")
        self.assertFalse(phase22["segments"][0]["l3"]["question_allowed"])
        self.assertEqual(phase22["segments"][0]["l3"]["relevant_um_fields"], [])
        self.assertIn("robotschool", app.turn_text(phase22["segments"][0]))
        phase23 = script[11]
        self.assertEqual(phase23["part"], 2)
        self.assertEqual(phase23["phase_id"], "2.3")
        self.assertEqual(phase23["script_phase"], "part2_correct_fav_subject_connection")
        self.assertEqual(len(phase23["segments"]), 5)
        self.assertIn("Ik weet ook nog dat natuur jouw lievelingsvak is.", app.turn_text(phase23["segments"][0]))
        self.assertEqual(phase23["segments"][0]["response_mode"], "topic_interpretation")
        self.assertTrue(phase23["segments"][0]["memory_correction_available"])
        self.assertNotIn("memory_correction_requested", phase23["segments"][0])
        self.assertEqual(phase23["segments"][0]["memory_correction_field"], "fav_subject")
        self.assertTrue(phase23["segments"][0]["expects_response"])
        self.assertFalse(phase23["segments"][1]["expects_response"])
        self.assertFalse(phase23["segments"][2]["expects_response"])
        self.assertEqual(phase23["segments"][3]["response_mode"], "listen_only")
        self.assertEqual(phase23["segments"][3]["used_fields"], {})
        self.assertFalse(phase23["segments"][4]["expects_response"])
        phase24 = script[12]
        self.assertEqual(phase24["part"], 2)
        self.assertEqual(phase24["phase_id"], "2.4")
        self.assertEqual(phase24["script_phase"], "part2_mistake3_school_strength")
        self.assertEqual(phase24["mistake_id"], "M3")
        self.assertEqual(phase24["mistake_field"], "school_strength")
        self.assertEqual(len(phase24["segments"]), 4)
        self.assertEqual(phase24["segments"][0]["response_mode"], "mistake_interpretation")
        self.assertTrue(phase24["segments"][0]["memory_correction_available"])
        self.assertEqual(phase24["segments"][0]["memory_correction_field"], "school_strength")
        self.assertTrue(phase24["segments"][1]["run_if_phase_confirmed_change"])
        self.assertTrue(phase24["segments"][2]["skip_if_phase_confirmed_change"])
        self.assertEqual(phase24["segments"][2]["l3"]["response_function"], "acknowledge")
        self.assertEqual(phase24["segments"][3]["l3"]["response_function"], "bridge")
        phase31 = script[13]
        self.assertEqual(phase31["part"], 3)
        self.assertEqual(phase31["phase_id"], "3.1")
        self.assertEqual(phase31["script_phase"], "part3_future_bridge")
        self.assertEqual(phase31["segments"][0]["response_mode"], "acknowledge")
        self.assertTrue(phase31["segments"][0]["llm_turn"])
        self.assertEqual(phase31["segments"][0]["l3"]["script_phase"], "part3_aspiration")
        self.assertEqual(phase31["segments"][0]["l3"]["topic"], "future")
        self.assertEqual(phase31["segments"][0]["l3"]["response_function"], "bridge")
        self.assertFalse(phase31["segments"][0]["l3"]["question_allowed"])
        self.assertEqual(phase31["segments"][0]["l3"]["relevant_um_fields"], [])
        self.assertIn("Denk jij daar wel eens over na over later?", app.turn_text(phase31["segments"][0]))
        phase32 = script[14]
        self.assertEqual(phase32["part"], 3)
        self.assertEqual(phase32["phase_id"], "3.2")
        self.assertEqual(phase32["script_phase"], "part3_leo_self_disclosure")
        self.assertEqual(phase32["layer"], "L1")
        self.assertEqual(phase32["segments"][0]["response_mode"], "listen_only")
        self.assertFalse(phase32["segments"][1]["expects_response"])
        self.assertIn("Mijn droom is om een hele goede en behulpzame schoolrobot te worden", app.turn_text(phase32["segments"][0]))
        self.assertIn("Daarom kijk ik daar goed naar.", app.turn_text(phase32["segments"][1]))
        phase33 = script[15]
        self.assertEqual(phase33["part"], 3)
        self.assertEqual(phase33["phase_id"], "3.3")
        self.assertEqual(phase33["script_phase"], "part3_rolemodel_rapport")
        self.assertEqual(phase33["layer"], "L1 + L2-pregen + L3")
        self.assertEqual(len(phase33["segments"]), 1)
        self.assertEqual(phase33["segments"][0]["response_mode"], "acknowledge")
        self.assertTrue(phase33["segments"][0]["llm_turn"])
        self.assertEqual(phase33["segments"][0]["used_fields"], {"role_model": "haar moeder"})
        self.assertEqual(phase33["segments"][0]["l3"]["script_phase"], "part3_rolemodel")
        self.assertEqual(phase33["segments"][0]["l3"]["topic"], "rolemodel")
        self.assertEqual(phase33["segments"][0]["l3"]["response_function"], "wrap_up")
        self.assertFalse(phase33["segments"][0]["l3"]["question_allowed"])
        self.assertEqual(phase33["segments"][0]["l3"]["relevant_um_fields"], ["role_model"])
        self.assertIn("Ik weet nog dat haar moeder", app.turn_text(phase33["segments"][0]))
        self.assertIn("Wat maakt die persoon voor jou zo bijzonder?", app.turn_text(phase33["segments"][0]))
        phase34 = script[16]
        self.assertEqual(phase34["part"], 3)
        self.assertEqual(phase34["phase_id"], "3.4/5")
        self.assertEqual(phase34["phase_aliases"], ["3.4", "3.5"])
        self.assertEqual(phase34["script_phase"], "part3_mistake4_aspiration_reflection")
        self.assertEqual(phase34["mistake_id"], "M4")
        self.assertEqual(phase34["mistake_field"], "aspiration")
        self.assertEqual(phase34["mistake_actual"], "dierenarts worden")
        self.assertEqual(len(phase34["segments"]), 7)
        self.assertEqual(phase34["segments"][0]["response_mode"], "listen_only")
        self.assertFalse(phase34["segments"][1]["expects_response"])
        self.assertEqual(phase34["segments"][2]["response_mode"], "mistake_interpretation")
        self.assertTrue(phase34["segments"][2]["memory_correction_available"])
        self.assertEqual(phase34["segments"][2]["memory_correction_field"], "aspiration")
        self.assertTrue(phase34["segments"][3]["run_if_phase_confirmed_change"])
        self.assertEqual(phase34["segments"][3]["condition_phase"], 17)
        self.assertTrue(phase34["segments"][4]["skip_if_phase_confirmed_change"])
        self.assertEqual(phase34["segments"][6]["response_mode"], "middle_school_feeling")
        self.assertTrue(phase34["segments"][6]["llm_turn"])
        self.assertIn("Snap jij een beetje wat ik bedoel?", app.turn_text(phase34["segments"][0]))
        self.assertIn("En volgens mij wil jij later", app.turn_text(phase34["segments"][2]))
        self.assertIn("Wat lijkt jou het mooiste aan", app.turn_text(phase34["segments"][3]))
        self.assertIn("middelbare school", app.turn_text(phase34["segments"][5]))
        phase36 = script[17]
        self.assertEqual(phase36["part"], 3)
        self.assertEqual(phase36["phase_id"], "3.6/7")
        self.assertEqual(phase36["phase_aliases"], ["3.6", "3.7"])
        self.assertEqual(phase36["script_phase"], "part3_explicit_memory_inspection_review")
        self.assertEqual(phase36["segments"][0]["response_mode"], "explicit_memory_inspection_offer")
        self.assertEqual(phase36["segments"][0]["used_fields"], {})
        self.assertEqual(
            app.turn_text(phase36["segments"][0]),
            "Wil je misschien zien wat ik allemaal over jou onthoud?",
        )
        self.assertEqual(phase36["segments"][1]["condition"], "run_if_memory_review_requested")
        self.assertEqual(phase36["segments"][1]["expects_response"], False)
        self.assertTrue(phase36["segments"][1]["memory_review_from_access_scope"])
        self.assertTrue(phase36["segments"][1]["speak_memory_review_from_access_scope"])
        self.assertFalse(phase36["segments"][1]["activate_tablet_memory_access"])
        self.assertEqual(phase36["segments"][2]["response_mode"], "memory_access_change")
        self.assertIn("Wil je iets aan mijn geheugen veranderen?", app.turn_text(phase36["segments"][2]))
        self.assertEqual(phase36["segments"][3]["response_mode"], "memory_review_add_final")
        self.assertIn("wat ik nog niet heb onthouden", app.turn_text(phase36["segments"][3]))
        self.assertIn("Dat is ook goed.", app.turn_text(phase36["segments"][-1]))
        phase38 = script[18]
        self.assertEqual(phase38["part"], 3)
        self.assertEqual(phase38["phase_id"], "3.8")
        self.assertEqual(phase38["script_phase"], "part3_closing")
        self.assertEqual(phase38["layer"], "L1 + L2-slot: first_name")
        self.assertEqual(len(phase38["segments"]), 3)
        self.assertFalse(phase38["segments"][0]["expects_response"])
        self.assertFalse(phase38["segments"][1]["expects_response"])
        self.assertFalse(phase38["segments"][2]["expects_response"])
        self.assertIn("Dank je wel dat je met mij hebt gepraat.", app.turn_text(phase38["segments"][0]))
        self.assertIn("maakte ik soms een foutje", app.turn_text(phase38["segments"][1]))
        self.assertEqual(app.turn_text(phase38["segments"][2]), "Tot de volgende keer, Sam.")
        self.assertEqual(phase38["used_fields"], {"name": "Sam"})
        self.assertEqual(script[0]["phase"], 1)
        self.assertEqual(script[0]["name"], "Greeting")
        self.assertEqual(script[0]["content_plan"]["values"]["first_name"], "Sam")
        self.assertEqual(
            app.turn_text(script[0]),
            "Hoi Sam! Wat fijn om je weer te zien. Heb je een beetje zin om met mij te kletsen?",
        )
        phase4_text = app.turn_text(script[3])
        self.assertIn("Dat vind ik echt een gezellige combinatie.", phase4_text)
        self.assertIn("Daar zit van alles in: creatief bezig zijn.", phase4_text)
        self.assertTrue(script[3]["expects_response"])
        self.assertEqual(script[3]["response_mode"], "listen_only")
        self.assertEqual(script[5]["mistake_id"], "M1")
        self.assertEqual(script[5]["mistake_field"], "hobby_fav")
        self.assertEqual(script[7]["mistake_id"], "M2")
        self.assertEqual(script[7]["mistake_field"], "fav_food")
        self.assertEqual(script[8]["condition"], "run_if_two_mistakes_no_corrections")

    def test_build_script_condition_e_uses_tablet_for_explicit_memory_review(self):
        app = make_app()
        app.local_child_name = "Sam"
        app.local_condition = "E"
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["condition"] = "E"
        app.pull_um = lambda: dict(um)
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}, "mistakes": []}

        script = app.build_script()
        phase36 = next(turn for turn in script if turn.get("phase_id") == "3.6/7")
        review_segment = phase36["segments"][1]

        self.assertIn("Kijk maar op de tablet", app.turn_text(review_segment))
        self.assertTrue(review_segment["memory_review_from_access_scope"])
        self.assertFalse(review_segment["speak_memory_review_from_access_scope"])
        self.assertTrue(review_segment["activate_tablet_memory_access"])

    def test_part3_role_model_phase_uses_stored_role_model_and_scenario_fallback(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p3_rolemodel_recall": {
                    "default": "[STUB] Ik weet nog dat je moeder iemand is naar wie je opkijkt."
                },
                "p3_rolemodel_ack": {
                    "default": "[STUB] Dat klinkt als iemand die veel voor je betekent."
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase33 = script[15]
        segment = phase33["segments"][0]

        self.assertEqual(phase33["phase_id"], "3.3")
        self.assertEqual(phase33["script_phase"], "part3_rolemodel_rapport")
        self.assertEqual(len(phase33["segments"]), 1)
        self.assertIn(
            "Ik weet nog dat je moeder iemand is naar wie je opkijkt.",
            app.turn_text(segment),
        )
        self.assertIn("Wat maakt die persoon voor jou zo bijzonder?", app.turn_text(segment))
        self.assertEqual(segment["response_mode"], "acknowledge")
        self.assertTrue(segment["llm_turn"])
        self.assertEqual(segment["used_fields"], {"role_model": "haar moeder"})
        self.assertEqual(segment["l3"]["response_function"], "wrap_up")
        self.assertEqual(segment["l3"]["relevant_um_fields"], ["role_model"])
        self.assertEqual(segment["l3"]["fallback"], "Dat klinkt als iemand die veel voor je betekent.")

    def test_part3_role_model_phase_uses_no_role_model_branch_when_missing(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["role_model"] = CRI.UNKNOWN_VALUE
        app.pull_um = lambda: um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p3_norolemodel_ack": {
                    "default": "[STUB] Dat snap ik. Soms leer je van verschillende mensen iets."
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase33 = script[15]
        segments = phase33["segments"]

        self.assertEqual(phase33["phase_id"], "3.3")
        self.assertEqual(phase33["used_fields"], {})
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["response_mode"], "role_model_absence_check")
        self.assertEqual(segments[0]["memory_correction_field"], "role_model")
        self.assertIn("Als ik het goed heb", app.turn_text(segments[0]))
        self.assertIn("Klopt dat een beetje?", app.turn_text(segments[0]))
        self.assertEqual(segments[1]["response_mode"], "role_model_discovery")
        self.assertIn("Wie is dat voor jou?", app.turn_text(segments[1]))
        self.assertEqual(segments[1]["l3"]["script_phase"], "part3_rolemodel")
        self.assertEqual(segments[1]["l3"]["topic"], "rolemodel")
        self.assertEqual(segments[1]["l3"]["response_function"], "wrap_up")
        self.assertEqual(segments[1]["l3"]["relevant_um_fields"], [])
        self.assertEqual(
            segments[1]["l3"]["fallback"],
            "Dat snap ik. Soms leer je van verschillende mensen iets.",
        )

    def test_part3_role_model_absence_rejection_asks_who_child_looks_up_to(self):
        app = make_app()
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "role_model_absence_check",
            "memory_correction_available": True,
            "memory_correction_field": "role_model",
            "used_fields": {"role_model": "niemand"},
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="role_model", value=None, confidence=0.92),
            "Nee dat klopt niet",
            turn,
        )

        self.assertEqual(action["action"], "role_model_absence_ask_detail")
        self.assertEqual(action["leo_response"], "Oeps, wie is dan iemand naar wie je opkijkt?")
        self.assertTrue(action["follow_up_needed"])
        self.assertTrue(turn["memory_correction_requested"])

    def test_part3_role_model_absence_agreement_does_not_become_person(self):
        app = make_app()
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "role_model_absence_check",
            "memory_correction_available": True,
            "memory_correction_field": "role_model",
            "used_fields": {"role_model": "niemand"},
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja dat klopt",
            turn,
        )

        self.assertEqual(action["action"], "role_model_absence_continue")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, [])

    def test_part3_role_model_absence_inline_person_gets_confirmed_and_continues(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "role_model_absence_check",
            "memory_correction_available": True,
            "memory_correction_field": "role_model",
            "used_fields": {"role_model": "niemand"},
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="role_model", value="mijn vader", confidence=0.95),
            "Nee dat klopt niet, mijn vader",
            turn,
        )

        self.assertEqual(action["action"], "confirm_role_model_discovery")
        self.assertTrue(action["change_confirmed"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(action["change"]["new_value"], "mijn vader")
        self.assertIn("mijn vader iemand is naar wie je opkijkt", app.speech.spoken[0])

    def test_part3_role_model_discovery_confirms_clear_person_but_not_vague_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "role_model_discovery",
            "used_fields": {},
            "l3": {
                "script_phase": "part3_rolemodel",
                "topic": "rolemodel",
                "response_function": "wrap_up",
                "question_allowed": False,
                "relevant_um_fields": [],
                "fallback": "Dat snap ik wel. Soms kun je van allerlei mensen iets leren.",
            },
        }
        app.current_turn_context = turn

        clear_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="mijn trainer", confidence=0.92),
            "mijn trainer",
            turn,
        )

        self.assertEqual(clear_action["action"], "confirm_role_model_discovery")
        self.assertTrue(clear_action["change_confirmed"])
        self.assertTrue(clear_action["continue_phase_after_change"])
        self.assertEqual(clear_action["change"]["new_value"], "mijn trainer")

        vague_app = make_app()
        vague_turn = dict(turn)
        vague_turn["response_mode"] = "role_model_discovery"
        vague_action = vague_app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.92),
            "weet ik niet",
            vague_turn,
        )

        self.assertEqual(vague_action["action"], "role_model_discovery_no_person")
        self.assertFalse(vague_action.get("continue_phase_after_change", False))
        self.assertEqual(vague_app.speech.spoken[-1], "Dat snap ik wel. Soms kun je van allerlei mensen iets leren.")

    def test_part3_role_model_phase_uses_plural_grammar_for_multiple_role_models(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["role_model"] = "mijn vader  en moeder"
        app.pull_um = lambda: um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p3_rolemodel_recall": {
                    "default": "Ik weet nog dat mijn vader  en moeder voor jou iemand is naar wie je opkijkt."
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase33 = script[15]
        text = app.turn_text(phase33["segments"][0])

        self.assertIn("mijn vader en moeder voor jou mensen zijn", text)
        self.assertIn("Wat maakt hen voor jou zo bijzonder?", text)
        self.assertNotIn("vader  en", text)

    def test_part3_role_model_correction_reasks_question_with_new_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["role_model"] = "mijn moeder"
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "name": "role_model rapport",
            "segments": app.script.role_model_rapport_segments(app.last_um_preview),
            "used_fields": {"role_model": "mijn moeder"},
        }
        app.current_turn_context = app.segment_context(turn, turn["segments"][0], 1)

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "role_model",
                "old_value": "mijn moeder",
                "new_value": "Superman",
            },
        })

        self.assertEqual(turn["used_fields"], {"role_model": "Superman"})
        self.assertEqual(len(turn["segments"]), 2)
        self.assertIn("mijn moeder", app.turn_text(turn["segments"][0]))
        self.assertEqual(turn["segments"][1]["response_mode"], "acknowledge")
        self.assertEqual("Wat maakt die persoon voor jou zo bijzonder?", app.turn_text(turn["segments"][1]))

    def test_part3_mistake4_uses_scenario_wrong_aspiration(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M4",
                    "target_field": "aspiration",
                    "wrong_value": "juf",
                    "mistake_type": "completely-wrong",
                }
            ],
            "utterances": {
                "p3_m4_followup_wrong_aspiration": {
                    "default": "[STUB] En volgens mij wil jij later juf worden."
                },
            },
        }

        script = app.build_script()
        phase34 = script[16]
        segments = phase34["segments"]

        self.assertEqual(phase34["phase_id"], "3.4/5")
        self.assertEqual(phase34["phase_aliases"], ["3.4", "3.5"])
        self.assertEqual(phase34["script_phase"], "part3_mistake4_aspiration_reflection")
        self.assertEqual(phase34["mistake_id"], "M4")
        self.assertEqual(phase34["mistake_field"], "aspiration")
        self.assertEqual(phase34["mistake_type"], "completely-wrong")
        self.assertEqual(phase34["mistake_actual"], "dierenarts worden")
        self.assertEqual(phase34["mistake_wrong"], "juf worden")
        self.assertIn("juffen en meesters", app.turn_text(segments[0]))
        self.assertEqual(segments[0]["response_mode"], "listen_only")
        self.assertIn("Sommige mensen geven je echt ideeen", app.turn_text(segments[1]))
        self.assertFalse(segments[1]["expects_response"])
        self.assertEqual(app.turn_text(segments[2]), "En volgens mij wil jij later juf worden.")
        self.assertEqual(segments[2]["response_mode"], "mistake_interpretation")
        self.assertTrue(segments[2]["defer_corrected_response"])
        self.assertEqual(segments[2]["used_fields"], {"aspiration": "juf worden"})
        self.assertTrue(segments[3]["run_if_phase_confirmed_change"])
        self.assertTrue(segments[4]["skip_if_phase_confirmed_change"])
        self.assertEqual(app.mistake_correction_question(phase34), "Oeps, wat wil jij dan later worden?")

    def test_part3_mistake4_uses_fallback_wrong_aspiration_when_um_missing(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["aspiration"] = CRI.UNKNOWN_VALUE
        app.pull_um = lambda: um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M4",
                    "target_field": "aspiration",
                    "wrong_value": "architect worden",
                    "mistake_type": "completely-wrong",
                }
            ],
            "utterances": {},
        }

        script = app.build_script()
        phase34 = script[16]
        segments = phase34["segments"]

        self.assertEqual(phase34["phase_id"], "3.4/5")
        self.assertEqual(phase34["mistake_actual"], CRI.UNKNOWN_VALUE)
        self.assertEqual(phase34["mistake_wrong"], "architect worden")
        self.assertEqual(
            app.turn_text(segments[2]),
            "En volgens mij wil jij later architect worden.",
        )
        self.assertEqual(segments[2]["used_fields"], {"aspiration": "architect worden"})
        self.assertEqual(phase34["mistake_topic"]["memory_link"], "wat je later wilt worden")

    def test_part3_merged_phase_uses_profile_based_postcorrection_reflection(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p3_m4_postcorrection_reflection": {
                    "default": "[STUB] Stale scenario line that should not override the corrected aspiration."
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase34 = script[16]
        segments = phase34["segments"]

        self.assertEqual(phase34["phase_id"], "3.4/5")
        self.assertEqual(phase34["script_phase"], "part3_mistake4_aspiration_reflection")
        self.assertEqual(len(segments), 7)
        self.assertIn("Dat past ook wel mooi bij jou", app.turn_text(segments[3]))
        self.assertIn("dierenarts worden", app.turn_text(segments[3]))
        self.assertNotIn("Stale scenario line", app.turn_text(segments[3]))
        self.assertTrue(segments[3]["run_if_phase_confirmed_change"])
        self.assertEqual(segments[3]["condition_phase"], 17)
        self.assertEqual(segments[3]["used_fields"]["aspiration"], "dierenarts worden")
        self.assertIn("dieren", segments[3]["used_fields"]["interest"])
        self.assertTrue(segments[4]["skip_if_phase_confirmed_change"])
        self.assertEqual(segments[6]["response_mode"], "middle_school_feeling")
        self.assertTrue(segments[6]["llm_turn"])

    def test_part3_merged_phase_fallback_reflection_uses_profile_cues(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}, "mistakes": []}

        script = app.build_script()
        reflection_text = app.turn_text(script[16]["segments"][3])

        self.assertIn("Dat past ook wel mooi bij jou", reflection_text)
        self.assertIn("dierenarts worden", reflection_text)
        self.assertIn("Wat lijkt jou het mooiste aan", reflection_text)

    def test_part3_mistake4_inline_aspiration_update_is_confirmed_and_continues(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "juf worden",
            "used_fields": {"aspiration": "juf worden"},
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="tuinman", confidence=0.95),
            "Nee ik wil tuinman worden",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change_confirmed"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(action["change"]["new_value"], "tuinman")

    def test_part3_mistake4_multiple_aspirations_asks_for_one_profession(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["kok", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="kok", confidence=0.95),
            "Nee kok en voetballer",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change_confirmed"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(action["change"]["field"], "aspiration")
        self.assertEqual(action["change"]["new_value"], "kok")
        self.assertEqual(app.last_um_preview["aspiration"], "kok")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier een beroep onthouden. Wat wil jij later worden? Noem een ding.",
        )
        self.assertIn("verander naar kok", app.speech.spoken[1])

    def test_part3_mistake4_multiple_aspirations_without_clarification_does_not_update(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["kok of voetballer", ""]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="kok of voetballer", confidence=0.92),
            "Nee ik wil kok of voetballer worden",
            turn,
        )

        self.assertEqual(action["action"], "change_value_limit_unresolved")
        self.assertEqual(app.last_um_preview["aspiration"], "dierenarts worden")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier een beroep onthouden. Wat wil jij later worden? Noem een ding.",
        )
        self.assertEqual(
            app.speech.spoken[1],
            "Dat zijn er nog te veel. Ik kan hier een beroep onthouden. Wat wil jij later worden? Noem een ding.",
        )
        self.assertIn("Dan verander ik het nu nog niet", app.speech.spoken[2])

    def test_part3_mistake4_unknown_aspiration_answer_gets_confirmed_and_stored_unknown(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
            "memory_correction_requested": True,
            "memory_correction_field": "aspiration",
            "last_correction_question": "Oeps, wat wil jij dan later worden?",
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.95),
            "Weet ik niet",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change_confirmed"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(action["change"]["field"], "aspiration")
        self.assertEqual(action["change"]["new_value"], CRI.UNKNOWN_VALUE)
        self.assertTrue(action["change"]["sets_unknown_value"])
        self.assertEqual(app.last_um_preview["aspiration"], CRI.UNKNOWN_VALUE)
        self.assertEqual(
            app.speech.spoken[0],
            "Wil je dat ik onthoud dat je dat nog niet weet?",
        )

    def test_part3_mistake4_direct_unknown_aspiration_answer_is_not_stored_as_profession(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="niks", confidence=0.85),
            "niks",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change_confirmed"])
        self.assertEqual(action["change"]["new_value"], CRI.UNKNOWN_VALUE)
        self.assertNotEqual(action["change"]["new_value"], "niks")
        self.assertEqual(app.last_um_preview["aspiration"], CRI.UNKNOWN_VALUE)
        self.assertEqual(
            app.speech.spoken[0],
            "Wil je dat ik onthoud dat je dat nog niet weet?",
        )

    def test_part3_mistake4_unknown_aspiration_refreshes_neutral_route(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}, "mistakes": []}
        app.last_um_preview = sample_um()
        turn = app.build_script()[16]
        app.current_turn_context = app.segment_context(turn, turn["segments"][2], 3)

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "aspiration",
                "old_value": "kapper worden",
                "new_value": CRI.UNKNOWN_VALUE,
                "sets_unknown_value": True,
            },
        })

        self.assertEqual(turn["mistake_actual"], CRI.UNKNOWN_VALUE)
        self.assertEqual(turn["used_fields"]["aspiration"], CRI.UNKNOWN_VALUE)
        self.assertIn("dat je nog niet weet wat je later wilt worden", turn["mistake_topic"]["memory_link"])
        self.assertIn("Je hoeft dat nu nog niet te weten", app.turn_text(turn["segments"][3]))
        self.assertNotIn("bij jou past", app.turn_text(turn["segments"][3]))
        self.assertEqual(turn["segments"][-1]["response_mode"], "middle_school_feeling")

    def test_part3_mistake4_rejected_confirmation_clears_pending_correction_state(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["nee"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
            "memory_correction_requested": True,
            "memory_correction_field": "aspiration",
            "last_correction_question": "Oeps, wat wil jij dan later worden?",
        }
        app.current_turn_context = turn
        app.mistake_states = {"M4": {"id": "M4", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="kok", confidence=0.95),
            "Kok",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertFalse(action["change_confirmed"])
        self.assertTrue(action["repeat_current_segment_after_rejected_correction"])
        self.assertFalse(turn.get("memory_correction_requested", False))
        self.assertNotIn("memory_correction_field", turn)
        self.assertNotIn("last_correction_question", turn)
        self.assertFalse(app.mistake_states["M4"].get("wrong_value_rejected", False))
        self.assertEqual(app.last_um_preview["aspiration"], "dierenarts worden")

    def test_part3_mistake4_normal_answer_after_rejected_confirmation_is_not_update(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
        }
        app.current_turn_context = turn
        app.mistake_states = {"M4": {"id": "M4"}}

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
            "Dat ik lekker kan knippen",
            turn,
        )

        self.assertEqual(action["action"], "continue_wrong_value_followup")
        self.assertEqual(app.speech.spoken, [])
        self.assertIsNone(getattr(app, "pending_change", None))
        self.assertEqual(app.last_um_preview["aspiration"], "dierenarts worden")

    def test_part3_mistake4_rejected_correction_repeats_current_segment(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["nee"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "mistake_interpretation",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "used_fields": {"aspiration": "kapper worden"},
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="aspiration", value="niks", confidence=0.85),
            "niks",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertFalse(action["change_confirmed"])
        self.assertTrue(action["repeat_current_segment_after_rejected_correction"])
        self.assertFalse(action["continue_phase_after_change"])
        self.assertFalse(action["stop_phase_after_change"])

    def test_run_phase_retries_segment_after_rejected_mistake_correction(self):
        app = make_app()
        app.shutdown_event = SimpleNamespace(is_set=lambda: False)
        app.start_turn_log = lambda turn: None
        app.finish_turn_log = lambda: None
        app.record_mistake_outcome = lambda turn: None
        app.refresh_topic_after_change = lambda turn, action: None
        calls = []
        actions = iter([
            {"handled": True, "repeat_current_segment_after_rejected_correction": True},
            {"handled": True},
            {"handled": True},
        ])

        def fake_run_phase_segment(turn, segment, index=None):
            calls.append(index)
            return next(actions)

        app.run_phase_segment = fake_run_phase_segment
        phase = {
            "phase": 17,
            "phase_id": "3.4/5",
            "name": "Mistake 4 - aspiration + reflection",
            "layer": "L1 + L2-pregen WRONG + reflection",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "kapper worden",
            "segments": [
                {"content_plan": app.l1("Ik weet ook nog dat jij later kapper wilt worden.")},
                {"content_plan": app.l1("Ik weet ook nog dat jij later kapper wilt worden. Gaaf! Wat trekt je daarin aan?")},
            ],
        }

        app.run_phase(phase, phase_index=16, total_phases=20)

        self.assertEqual(calls, [1, 1, 2])

    def test_part3_mistake4_refreshes_reflection_after_corrected_aspiration(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}, "mistakes": []}
        app.last_um_preview = sample_um()
        turn = app.build_script()[16]
        app.current_turn_context = app.segment_context(turn, turn["segments"][2], 3)

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "aspiration",
                "old_value": "juf worden",
                "new_value": "tuinman",
            },
        })

        self.assertEqual(turn["mistake_actual"], "tuinman worden")
        self.assertEqual(len(turn["segments"]), 4)
        self.assertIn("tuinman worden", app.turn_text(turn["segments"][3]))
        self.assertIn("Wat lijkt jou het mooiste aan", app.turn_text(turn["segments"][3]))

    def test_part2_subject_phase_uses_plural_for_multiple_favorite_subjects(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["fav_subject"] = "gym en rekenen"
        app.pull_um = lambda: um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p2_fav_subject_comment_subject": {
                    "default": "[STUB] Gym en rekenen, dat snap ik wel."
                },
                "p2_subject_profile_link": {
                    "default": "[STUB] Dat past bij jou."
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase22 = script[10]
        phase23 = script[11]

        self.assertEqual(
            phase22["segments"][0]["l3"]["next_script_line"],
            "Ik weet ook nog dat gym en rekenen jouw lievelingsvakken zijn.",
        )
        self.assertIn(
            "Ik weet ook nog dat gym en rekenen jouw lievelingsvakken zijn.",
            app.turn_text(phase23["segments"][0]),
        )
        self.assertEqual(app.turn_text(phase23["segments"][1]), "Gym en rekenen, dat snap ik wel.")
        self.assertEqual(app.turn_text(phase23["segments"][2]), "Dat past bij jou.")

    def test_part2_subject_correction_rebuilds_remaining_lines_with_new_subject(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um = sample_um()
        um["fav_subject"] = "Rekenen en Gym"
        um["hobbies"] = "tekenen en lezen"
        um["interest"] = "dieren"
        app.pull_um = lambda: um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p2_fav_subject_comment_subject": {
                    "default": "Dat snap ik. Rekenen is best als een puzzel."
                },
                "p2_subject_profile_link": {
                    "default": "Grappig, rekenen en voetbal gaan eigenlijk best goed samen."
                },
            },
            "mistakes": [],
        }
        app.last_um_preview = dict(um)
        script = app.build_script()
        phase23 = script[11]

        app.refresh_topic_after_change(phase23, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "fav_subject",
                "old_value": "Rekenen en Gym",
                "new_value": "taal",
            },
        })

        self.assertTrue(phase23["force_topic_fallback"])
        comment_context = app.segment_context(phase23, phase23["segments"][1], 2)
        link_context = app.segment_context(phase23, phase23["segments"][2], 3)
        self.assertIn("Met taal kun je verhalen maken", app.turn_text(comment_context))
        self.assertIn("met taal kun je daar ook weer over vertellen", app.turn_text(link_context))
        self.assertNotIn("rekenen en voetbal", app.turn_text(link_context).lower())

    def test_part2_subject_bare_rejection_asks_for_favorite_subject(self):
        app = make_app()
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "topic_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_subject",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value=None, confidence=0.92),
            "Dat klopt niet",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertEqual(
            action["leo_response"],
            "Oeps, vertel mij maximaal twee lievelingsvakken die ik moet onthouden.",
        )

    def test_part2_subject_agreement_does_not_start_memory_change(self):
        for transcript in ("Ja dat klopt", "Oké", "Ja inderdaad goed onthouden"):
            with self.subTest(transcript=transcript):
                app = make_app()
                turn = {
                    "phase": 12,
                    "phase_id": "2.3",
                    "response_mode": "topic_interpretation",
                    "memory_correction_available": True,
                    "memory_correction_field": "fav_subject",
                    "used_fields": {"fav_subject": "Rekenen en Gym"},
                    "topic": {
                        "domain": "school_subject",
                        "label": "Rekenen en Gym",
                        "fields": ["fav_subject"],
                        "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                        "current_values": {"fav_subject": "Rekenen en Gym"},
                        "expected_value_count": {"fav_subject": 2},
                    },
                }

                action = app.action_handler(
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                    transcript,
                    turn,
                )

                self.assertEqual(action["action"], "no_memory_change")
                self.assertEqual(action["change"], {})
                self.assertEqual(app.speech.spoken, [])

    def test_part2_subject_one_subject_correction_asks_for_second_subject(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["gym", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "listen_only",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "memory_correction_requested": True,
            "memory_correction_field": "fav_subject",
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value="taal", confidence=0.94),
            "Taal",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "fav_subject")
        self.assertEqual(action["change"]["new_value"], "taal en gym")
        self.assertEqual(app.last_um_preview["fav_subject"], "taal en gym")
        self.assertEqual(app.speech.spoken[0], "Oké, taal. Vertel mij maximaal één ander lievelingsvak.")
        self.assertIn("je twee lievelingsvakken verander naar taal en gym", app.speech.spoken[1])

    def test_part2_subject_inline_single_subject_correction_asks_only_for_second_subject(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["gym", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "topic_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_subject",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="wrong_guess", confidence=0.92),
            "Nee dat is Taal",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "fav_subject")
        self.assertEqual(action["change"]["new_value"], "Taal en gym")
        self.assertEqual(app.speech.spoken[0], "Oké, Taal. Vertel mij maximaal één ander lievelingsvak.")
        self.assertNotIn("Oeps, vertel mij maximaal twee lievelingsvakken die ik moet onthouden.", app.speech.spoken)
        self.assertIn("je twee lievelingsvakken verander naar Taal en gym", app.speech.spoken[1])

    def test_part2_subject_inline_two_subject_correction_confirms_directly(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "topic_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_subject",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="wrong_guess", confidence=0.92),
            "Nee dat is Taal en Gym",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "fav_subject")
        self.assertEqual(action["change"]["new_value"], "Taal en Gym")
        self.assertFalse(any("Vertel mij maximaal één ander lievelingsvak" in text for text in app.speech.spoken))
        self.assertIn("je twee lievelingsvakken verander naar Taal en Gym", app.speech.spoken[0])

    def test_part2_subject_too_many_initial_subjects_asks_for_max_two(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["taal en gym", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "topic_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_subject",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value="Taal, gym en rekenen", confidence=0.94),
            "Taal, gym en rekenen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["new_value"], "taal en gym")
        self.assertEqual(action["change"]["field_label"], "je twee lievelingsvakken")
        self.assertEqual(app.last_um_preview["fav_subject"], "taal en gym")
        self.assertEqual(app.speech.heard, [])
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan er maximaal twee onthouden. Vertel mij maximaal twee lievelingsvakken die ik moet bewaren.",
        )
        self.assertIn("je twee lievelingsvakken verander naar taal en gym", app.speech.spoken[1])

    def test_part2_subject_completion_with_too_many_extra_subjects_asks_for_one_extra(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["gym en rekenen", "gym", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "listen_only",
            "used_fields": {"fav_subject": "Rekenen en Gym"},
            "memory_correction_requested": True,
            "memory_correction_field": "fav_subject",
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je twee lievelingsvakken"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
                "expected_value_count": {"fav_subject": 2},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value="taal", confidence=0.94),
            "Taal",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["new_value"], "taal en gym")
        self.assertEqual(action["change"]["field_label"], "je twee lievelingsvakken")
        self.assertEqual(app.last_um_preview["fav_subject"], "taal en gym")
        self.assertEqual(app.speech.spoken[0], "Oké, taal. Vertel mij maximaal één ander lievelingsvak.")
        self.assertEqual(app.speech.spoken[1], "Ik heb al taal. Vertel mij maximaal één ander lievelingsvak.")
        self.assertIn("je twee lievelingsvakken verander naar taal en gym", app.speech.spoken[2])

    def test_part2_subject_final_question_does_not_allow_memory_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "listen_only",
            "used_fields": {},
            "topic": {
                "domain": "school_subject",
                "label": "Rekenen en Gym",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je lievelingsvak"},
                "current_values": {"fav_subject": "Rekenen en Gym"},
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value="taal", confidence=0.94),
            "Nee, taal",
            turn,
        )

        self.assertEqual(action["action"], "listen_only")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, [])

    def test_normal_reactions_with_um_update_intent_do_not_start_memory_change(self):
        app = make_app()
        cases = [
            (
                {
                    "phase": 13,
                    "phase_id": "2.4",
                    "response_mode": "acknowledge",
                    "used_fields": {
                        "fav_subject": "Rekenen en Gym",
                        "school_difficulty": "begrijpend lezen",
                    },
                },
                IntentResult(intent="um_update", field="school_difficulty", value="moeilijk", confidence=0.92),
                "Ik vind het gewoon moeilijk",
            ),
            (
                {
                    "phase": 16,
                    "phase_id": "3.3",
                    "response_mode": "acknowledge",
                    "used_fields": {"role_model": "mijn moeder"},
                },
                IntentResult(intent="um_update", field="role_model", value="lief", confidence=0.92),
                "Omdat ze lief is",
            ),
            (
                {
                    "phase": 17,
                    "phase_id": "3.4/5",
                    "response_mode": "middle_school_feeling",
                    "used_fields": {"aspiration": "dierenarts worden"},
                },
                IntentResult(intent="um_update", field="aspiration", value="spannend", confidence=0.92),
                "Ik vind het spannend",
            ),
        ]

        for turn, result, transcript in cases:
            with self.subTest(phase_id=turn["phase_id"], transcript=transcript):
                self.assertEqual(app.actions.change_from_intent_result(result, turn, transcript), {})

    def test_explicit_memory_correction_cue_after_unflagged_um_mention_does_not_update(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: (_ for _ in ()).throw(AssertionError("unexpected write"))
        turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "acknowledge",
            "used_fields": {"role_model": "mijn moeder"},
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="role_model", value="mijn vader", confidence=0.94),
            "Nee dat klopt niet, mijn rolmodel is mijn vader",
            turn,
        )

        self.assertEqual(action["action"], "acknowledge")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.last_um_preview["role_model"], sample_um()["role_model"])

    def test_part2_mistake3_uses_default_scenario_lines_for_branch_paths(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M3",
                    "target_field": "school_strength",
                    "wrong_value": "rekenen",
                    "mistake_type": "completely-wrong",
                }
            ],
            "utterances": {
                "p2_m3_postcorrection_true_strength": {
                    "default": "[STUB] Oeps, goed dat je het zegt. Dan pas ik het aan. Taal! Wat vind jij het leukste aan taal?"
                },
                "p2_school_wrap_after_difficulty": {
                    "default": "[STUB] Dat snap ik wel. School kan soms plakken."
                },
            },
        }

        script = app.build_script()
        phase24 = script[12]
        segments = phase24["segments"]

        self.assertEqual(phase24["phase_id"], "2.4")
        self.assertEqual(phase24["mistake_field"], "school_strength")
        self.assertEqual(phase24["mistake_wrong"], "rekenen")
        self.assertEqual(phase24["mistake_actual"], "taal")
        self.assertEqual(
            app.turn_text(segments[0]),
            "En volgens mij ben jij vooral goed in rekenen.",
        )
        self.assertNotIn("Oeps, goed dat je het zegt.", app.turn_text(segments[1]))
        self.assertNotIn("Dan pas ik het aan.", app.turn_text(segments[1]))
        self.assertIn("Taal! Wat vind jij het leukste aan taal?", app.turn_text(segments[1]))
        self.assertIn("Gym vond ik altijd al moeilijk", app.turn_text(segments[2]))
        self.assertEqual(segments[2]["response_mode"], "acknowledge")
        self.assertEqual(segments[2]["l3"]["relevant_um_fields"], [])
        self.assertEqual(segments[2]["l3"]["fallback"], "Dat snap ik wel. School kan soms plakken.")
        self.assertIn("natuur jouw lievelingsvak is", app.turn_text(segments[3]))
        self.assertIn("rekenen voor jou soms wat lastiger voelt", app.turn_text(segments[3]))
        self.assertTrue(segments[3]["m3_school_difficulty_resolution"])
        self.assertEqual(segments[3]["l3"]["relevant_um_fields"], ["fav_subject", "school_difficulty"])
        self.assertEqual(phase24["mistake_topic"]["expected_value_count"], {"school_strength": 1})
        self.assertEqual(
            app.mistake_correction_question(phase24),
            "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
        )

    def test_part2_mistake3_no_correction_keeps_difficulty_when_it_does_not_conflict(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M3",
                    "target_field": "school_strength",
                    "wrong_value": "aardrijkskunde",
                    "mistake_type": "completely-wrong",
                }
            ],
            "utterances": {},
        }

        script = app.build_script()
        phase24 = script[12]
        no_correction_followup = app.turn_text(phase24["segments"][3])

        self.assertIn("rekenen voor jou soms wat lastiger voelt", no_correction_followup)
        self.assertEqual(phase24["segments"][3]["l3"]["relevant_um_fields"], ["fav_subject", "school_difficulty"])

    def test_mistake1_uses_cri_scenario_opener_and_branch_followups(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_um_preview = sample_um()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M1",
                    "target_field": "hobby_fav",
                    "wrong_value": "bakken",
                    "mistake_type": "related-but-wrong",
                }
            ],
            "utterances": {
                "p1_m1_wrong_hobby_opener": {
                    "default": "[STUB] Iets maken dat ook nog lekker ruikt, dat is best indrukwekkend."
                },
                "p1_m1_followup_wrong_hobby": {
                    "default": "[STUB] Wat vind jij het leukste om te bakken?"
                },
                "p1_followup_postcorrection_true_hobby": {
                    "default": "[STUB] Oeps, dan had ik dat verkeerd. Wat vind jij op dit moment het leukste aan tekenen?"
                },
            },
        }

        script = app.build_script()
        phase6 = script[5]
        segments = phase6["segments"]

        self.assertEqual(phase6["phase_id"], "1.6")
        self.assertEqual(phase6["mistake_field"], "hobby_fav")
        self.assertEqual(phase6["mistake_wrong"], "bakken")
        self.assertEqual(phase6["mistake_topic"]["expected_value_count"], {"hobby_fav": 1})
        self.assertEqual(len(segments), 4)
        self.assertEqual(
            app.turn_text(segments[0]),
            "En volgens mij is bakken jouw allerliefste hobby.",
        )
        self.assertIn("Dat snap ik trouwens wel.", app.turn_text(segments[1]))
        self.assertIn("Iets maken dat ook nog lekker ruikt", app.turn_text(segments[1]))
        self.assertNotIn("Oeps, dan had ik dat verkeerd.", app.turn_text(segments[2]))
        self.assertIn("Wat vind jij op dit moment het leukste aan tekenen?", app.turn_text(segments[2]))
        self.assertEqual(app.turn_text(segments[3]), "Wat vind jij het leukste om te bakken?")

    def test_mistake1_corrected_followup_uses_child_corrected_hobby_at_runtime(self):
        app = make_app()
        um = sample_um()
        um["hobby_fav"] = "dansen"
        um["hobbies"] = "dansen, schilderen, turnen en zingen"
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = lambda: dict(um)
        app.last_um_preview = dict(um)

        script = app.build_script()
        corrected_segment = script[5]["segments"][2]

        app.last_um_preview["hobby_fav"] = "tekenen"

        text = app.turn_text(corrected_segment)
        self.assertNotIn("Oeps, dan had ik dat verkeerd.", text)
        self.assertIn("Wat vind jij het leukste aan tekenen?", text)
        self.assertNotIn("dansen", text.lower())

    def test_mistake2_uses_default_scenario_lines_for_wrong_and_corrected_paths(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "mistakes": [
                {
                    "id": "M2",
                    "target_field": "fav_food",
                    "wrong_value": "pizza",
                    "mistake_type": "completely-wrong",
                }
            ],
            "utterances": {
                "p1_m2_followup_wrong_food": {
                    "default": "[STUB] Pizza is rond en warm. Wat vind jij daar lekker aan?"
                },
                "p1_m2_postcorrection_true_food": {
                    "default": "[STUB] Pannenkoeken klinken eerlijk gezegd ook meteen gezellig."
                },
            },
        }

        script = app.build_script()
        phase8 = script[7]
        segments = phase8["segments"]

        self.assertEqual(phase8["phase_id"], "1.8")
        self.assertEqual(phase8["mistake_field"], "fav_food")
        self.assertEqual(phase8["mistake_wrong"], "pizza")
        self.assertEqual(len(segments), 3)
        self.assertIn("Ik weet nog dat jouw lievelingseten pizza is.", app.turn_text(segments[0]))
        self.assertIn("Dat is op zich wel een lekkere keuze.", app.turn_text(segments[1]))
        self.assertIn("Pizza is rond en warm.", app.turn_text(segments[1]))
        self.assertIn("Pannenkoeken klinken eerlijk gezegd", app.turn_text(segments[2]))

        app.last_um_preview = sample_um()
        app.last_um_preview["fav_food"] = "boerenkool"
        corrected_text = app.turn_text(segments[2])
        self.assertIn("Dan houden we het bij boerenkool.", corrected_text)
        self.assertNotIn("Pannenkoeken", corrected_text)

        self.assertFalse(segments[2]["expects_response"])
        self.assertTrue(segments[2]["run_if_phase_confirmed_change"])

    def test_build_script_uses_condition_specific_tutorial_text(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        um_c = sample_um()
        um_c["condition"] = "C"
        app.pull_um = lambda: um_c

        script_c = app.build_script()
        tutorial_c = app.turn_text(script_c[1])

        self.assertEqual(script_c[1]["tutorial_condition"], "C")
        self.assertIn("Als mijn ogen wit zijn, luister ik niet.", tutorial_c)
        self.assertIn("Dan herhaal ik waar we het over gehad hebben.", tutorial_c)
        self.assertNotIn("geheugenboek op de tablet", tutorial_c)

        um_e = sample_um()
        um_e["condition"] = "E"
        app.pull_um = lambda: um_e

        script_e = app.build_script()
        tutorial_e = app.turn_text(script_e[1])

        self.assertEqual(script_e[1]["tutorial_condition"], "E")
        self.assertIn("Als mijn ogen wit zijn, luister ik niet.", tutorial_e)
        self.assertIn(
            "Dan kan je jouw geheugenboek op de tablet bekijken om te zien waar we het over gehad hebben.",
            tutorial_e,
        )
        self.assertNotIn("Dan herhaal ik waar we het over gehad hebben.", tutorial_e)

    def test_phase3_mini_story_uses_cri_scenario_with_response_gaps(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_leo_ministory_opening": {
                    "default": "[STUB] Opening uit GraphDB. Heb jij dat wel eens?"
                },
                "p1_leo_ministory_followup": {
                    "default": "[STUB] Follow-up uit GraphDB. Doe jij dat ook wel eens?"
                },
                "p1_leo_ministory_wrap": {
                    "default": "[STUB] Wrap uit GraphDB."
                },
            },
        }

        script = app.build_script()
        phase3 = script[2]
        segments = phase3["segments"]

        self.assertEqual(phase3["phase"], 3)
        self.assertEqual(phase3["name"], "Leo mini-story")
        self.assertEqual(len(segments), 3)
        self.assertTrue(segments[0]["expects_response"])
        self.assertTrue(segments[1]["expects_response"])
        self.assertFalse(segments[2]["expects_response"])
        self.assertEqual(app.turn_text(segments[0]), "Opening uit GraphDB. Heb jij dat wel eens?")
        self.assertEqual(app.turn_text(segments[1]), "Follow-up uit GraphDB. Doe jij dat ook wel eens?")
        self.assertEqual(app.turn_text(segments[2]), "Wrap uit GraphDB.")

    def test_topic_sport_segments_use_cri_scenario_utterances(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_sport_recall": {"default": "[STUB] Ik weet ook nog dat jij zelf hockey speelt."},
                "p1_sport_open": {"default": "[STUB] In welke positie speel jij?"},
                "p1_sport_followup": {"default": "[STUB] Ben jij dan meer van verdedigen of aanvallen?"},
            }
        }
        topic = app.topic_candidate(
            domain="sport",
            label="hockey",
            fields=["sports_fav_play"],
            field_labels={"sports_fav_play": "de sport die je graag doet"},
            current_values={"sports_fav_play": "hockey"},
            correct_values=["je iets met hockey hebt"],
            memory_link="hockey iets is waar jij iets mee hebt",
            options=["hockey", "sport"],
            reground="Ik houd goed vast dat hockey iets is waar jij iets mee hebt.",
        )

        segments = app.topic1_phase_segments(topic)

        self.assertEqual(app.turn_text(segments[0]), "Ik weet ook nog dat jij zelf hockey speelt.")
        self.assertEqual(app.turn_text(segments[1]), "In welke positie speel jij?")
        self.assertIn("sportrobot", app.turn_text(segments[2]))
        self.assertIn("Ben jij dan meer van verdedigen of aanvallen?", app.turn_text(segments[3]))
        self.assertEqual(app.turn_text(segments[4]), "Maar jij doet natuurlijk nog meer leuke dingen.")
        self.assertEqual(segments[0]["response_mode"], "topic_interpretation")
        self.assertEqual(segments[1]["used_fields"], {})
        self.assertEqual(segments[2]["used_fields"], {})
        self.assertEqual(segments[3]["used_fields"], {})

    def test_part1_topics_follow_cri_scenario_topic_sets(self):
        app = make_app()
        app.local_child_name = "Sam"
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_music_open": {"default": "[STUB] Ik weet ook nog dat jij piano speelt."},
                "p1_music_ack": {"default": "[STUB] Piano klinkt mooi."},
                "p1_music_followup": {"default": "[STUB] Speel jij rustige of vrolijke stukken?"},
                "p1_animals_open": {"default": "[STUB] Ik weet ook nog dat jij een kat hebt die Momo heet."},
                "p1_animals_followup": {"default": "[STUB] Wat voor kat is Momo eigenlijk?"},
            },
            "mistakes": [],
        }

        script = app.build_script()

        self.assertEqual(script[4]["topic"]["domain"], "muziek")
        self.assertEqual(script[6]["topic"]["domain"], "huisdier")
        self.assertEqual(
            app.turn_text(script[4]["segments"][0]),
            "Ik weet ook nog dat jij piano speelt.",
        )
        self.assertEqual(
            app.turn_text(script[6]["segments"][3]),
            "Ik weet ook nog dat jij een kat hebt die Momo heet.",
        )

    def test_part1_topics_can_use_generic_t1_t2_scenario_names(self):
        app = make_app()
        app.local_child_name = "Sam"
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "topic_1": {"default": "sport"},
                "topic_2": {"default": "huisdier"},
                "p1_t1_recall": {"default": "[STUB] T1 recall uit GraphDB."},
                "p1_t1_open": {"default": "[STUB] T1 open uit GraphDB?"},
                "p1_t1_question": {"default": "[STUB] T1 question uit GraphDB?"},
                "p1_t1_followup": {"default": "[STUB] T1 followup uit GraphDB?"},
                "p1_t2_recall": {"default": "[STUB] T2 recall uit GraphDB."},
                "p1_t2_open": {"default": "[STUB] T2 open uit GraphDB?"},
                "p1_t2_followup": {"default": "[STUB] T2 followup uit GraphDB?"},
                "p1_t2_close": {"default": "[STUB] T2 close uit GraphDB."},
            },
            "mistakes": [],
        }

        script = app.build_script()

        self.assertEqual(script[4]["topic"]["domain"], "sport")
        self.assertEqual(script[6]["topic"]["domain"], "huisdier")
        self.assertEqual(app.turn_text(script[4]["segments"][0]), "T1 recall uit GraphDB.")
        self.assertEqual(app.turn_text(script[4]["segments"][1]), "T1 open uit GraphDB?")
        self.assertEqual(app.turn_text(script[4]["segments"][2]), "T1 question uit GraphDB?")
        self.assertEqual(app.turn_text(script[4]["segments"][3]), "T1 followup uit GraphDB?")
        self.assertEqual(script[4]["segments"][0]["response_mode"], "topic_interpretation")
        self.assertEqual(script[4]["segments"][1]["used_fields"], {})
        self.assertEqual(script[4]["segments"][2]["used_fields"], {})
        self.assertEqual(script[4]["segments"][3]["used_fields"], {})
        topic2_texts = [app.turn_text(segment) for segment in script[6]["segments"]]
        self.assertIn("Mensen hebben vaak meer dan een ding", topic2_texts[0])
        self.assertIn("Bij boeken vind ik dat zo fijn", topic2_texts[1])
        self.assertEqual(topic2_texts[2], "En jij hebt volgens mij ook meer dingen die je leuk vindt.")
        self.assertNotIn("T2 recall uit GraphDB.", topic2_texts)
        self.assertIn("kat", topic2_texts[3].lower())
        self.assertNotEqual(topic2_texts[3], "T2 open uit GraphDB?")
        self.assertIn("kat", topic2_texts[4].lower())
        self.assertNotEqual(topic2_texts[4], "T2 followup uit GraphDB?")
        self.assertEqual(topic2_texts[5], "T2 close uit GraphDB.")

    def test_topic1_activity_sport_rejects_position_open_question(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["sports_fav_play"] = "vissen"
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_t1_recall": {"default": "[STUB] Ik weet ook nog dat jij zelf vissen doet."},
                "p1_t1_open": {"default": "[STUB] In welke positie speel jij?"},
                "p1_t1_question": {
                    "default": "[STUB] Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                    "maar ik val waarschijnlijk al om voor de warming-up. Wat vind jij zo leuk aan vissen?"
                },
                "p1_t1_followup": {
                    "default": "[STUB] Ben jij dan meer van snel rennen, goed overspelen, "
                    "of juist lekker fanatiek meedoen?"
                },
            },
            "mistakes": [],
        }
        topic = app.topic_candidate(
            domain="sport",
            label="vissen",
            fields=["sports_enjoys", "sports_fav_play"],
            field_labels={
                "sports_enjoys": "of je sport leuk vindt",
                "sports_fav_play": "de sport die je graag doet",
            },
            current_values={"sports_enjoys": "ja", "sports_fav_play": "vissen"},
            correct_values=["je iets met vissen hebt"],
            memory_link="vissen iets is waar jij iets mee hebt",
            options=["vissen", "sport"],
            reground="Ik houd goed vast dat vissen iets is waar jij iets mee hebt.",
        )

        segments = app.topic1_phase_segments(topic)

        open_text = app.turn_text(segments[1])
        followup_text = app.turn_text(segments[3])
        self.assertIn("vissen", open_text)
        self.assertNotIn("positie", open_text.lower())
        self.assertNotIn("overspelen", followup_text.lower())
        self.assertNotIn("snel rennen", followup_text.lower())

    def test_topic1_correction_to_activity_sport_rebuilds_with_activity_questions(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = app.topic_candidate(
            domain="sport",
            label="voetbal",
            fields=["sports_enjoys", "sports_fav_play"],
            field_labels={
                "sports_enjoys": "of je sport leuk vindt",
                "sports_fav_play": "de sport die je graag doet",
            },
            current_values={"sports_enjoys": "ja", "sports_fav_play": "voetbal"},
            correct_values=["je iets met voetbal hebt"],
            memory_link="voetbal iets is waar jij iets mee hebt",
            options=["voetbal", "sport"],
            reground="Ik houd goed vast dat voetbal iets is waar jij iets mee hebt.",
        )
        turn = {
            "phase": 5,
            "layer": "L2+L3",
            "name": "Topic 1",
            "topic": topic,
            "segments": app.topic1_phase_segments(topic),
        }

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "sports_fav_play",
                "old_value": "voetbal",
                "new_value": "vissen",
            },
        })

        open_text = app.turn_text(app.segment_context(turn, turn["segments"][1], 2))
        question_text = app.turn_text(app.segment_context(turn, turn["segments"][2], 3))
        followup_text = app.turn_text(app.segment_context(turn, turn["segments"][3], 4))
        self.assertIn("vissen", open_text)
        self.assertNotIn("positie", open_text.lower())
        self.assertIn("vissen", question_text)
        self.assertNotIn("voetbal", question_text.lower())
        self.assertNotIn("overspelen", followup_text.lower())

    def test_topic2_animals_uses_bridge_open_followup_then_topic_closing(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_animals_open": {
                    "default": "[STUB] Ik weet ook nog dat jij een kat hebt die Luna heet."
                },
                "p1_animals_followup": {
                    "default": "[STUB] Wat voor kat is Luna eigenlijk?"
                },
            }
        }
        topic = app.topic_candidate(
            domain="huisdier",
            label="Luna",
            fields=["pet_name", "pet_type"],
            field_labels={"pet_name": "de naam van je huisdier", "pet_type": "het soort huisdier"},
            current_values={"pet_name": "Luna", "pet_type": "kat"},
            correct_values=["Luna bij jou hoort"],
            memory_link="Luna belangrijk voor je is",
            options=["Luna", "dieren"],
            reground="Ik onthoud dat Luna belangrijk voor je is.",
        )

        segments = app.topic2_phase_segments(topic)

        self.assertEqual(len(segments), 6)
        self.assertFalse(segments[0]["expects_response"])
        self.assertFalse(segments[1]["expects_response"])
        self.assertFalse(segments[2]["expects_response"])
        self.assertFalse(segments[3]["expects_response"])
        self.assertTrue(segments[4]["expects_response"])
        self.assertFalse(segments[5]["expects_response"])
        self.assertIn("Mensen hebben vaak meer dan een ding", app.turn_text(segments[0]))
        self.assertIn("Bij boeken vind ik dat zo fijn", app.turn_text(segments[1]))
        self.assertEqual(app.turn_text(segments[2]), "En jij hebt volgens mij ook meer dingen die je leuk vindt.")
        self.assertEqual(app.turn_text(segments[3]), "Ik weet ook nog dat jij een kat hebt die Luna heet.")
        self.assertEqual(app.turn_text(segments[4]), "Wat voor kat is Luna eigenlijk?")
        self.assertIn("Ik vind dieren altijd fascinerend", app.turn_text(segments[5]))

    def test_topic2_pet_uses_generic_open_without_fallback_recall(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_t2_open": {
                    "default": "[STUB] En ik herinner me dat jij een vis als huisdier hebt."
                },
                "p1_t2_followup": {
                    "default": "[STUB] Hoe gaat het met jouw vis? Doet die nog leuke dingen?"
                },
                "p1_t2_close": {
                    "default": "[STUB] Bizar leuk. Ik vind dieren altijd fascinerend."
                },
            }
        }
        topic = app.topic_candidate(
            domain="huisdier",
            label="Vis",
            fields=["pet_name", "pet_type"],
            field_labels={"pet_name": "de naam van je huisdier", "pet_type": "het soort huisdier"},
            current_values={"pet_name": "Vis", "pet_type": "Vis"},
            correct_values=["Vis bij jou hoort"],
            memory_link="Vis belangrijk voor je is",
            options=["Vis", "dieren"],
            reground="Ik onthoud dat Vis belangrijk voor je is.",
        )

        segments = app.topic2_phase_segments(topic)

        self.assertEqual(len(segments), 6)
        self.assertFalse(segments[0]["expects_response"])
        self.assertFalse(segments[1]["expects_response"])
        self.assertFalse(segments[2]["expects_response"])
        self.assertFalse(segments[3]["expects_response"])
        self.assertTrue(segments[4]["expects_response"])
        self.assertFalse(segments[5]["expects_response"])
        self.assertIn("Mensen hebben vaak meer dan een ding", app.turn_text(segments[0]))
        self.assertEqual(app.turn_text(segments[3]), "En ik herinner me dat jij een vis als huisdier hebt.")
        self.assertEqual(app.turn_text(segments[4]), "Hoe gaat het met jouw vis? Doet die nog leuke dingen?")
        self.assertEqual(app.turn_text(segments[5]), "Bizar leuk. Ik vind dieren altijd fascinerend.")
        self.assertEqual(
            segments[4]["used_fields"],
            {"pet_name": "Vis", "pet_type": "Vis"},
        )

    def test_topic2_pet_stale_scenario_lines_are_rewritten_with_live_um(self):
        app = make_app(openai_payloads=[
            "En ik herinner me dat jij een kat hebt die Simba heet. Gaaf!",
            "Hoe is Simba als kat eigenlijk?",
        ])
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "kat", "pet_name": "Simba"})
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_t2_open": {
                    "default": "[STUB] En ik herinner me dat jij een vis als huisdier hebt. Gaaf!"
                },
                "p1_t2_followup": {
                    "default": "[STUB] Hoe gaat het met jouw vis? Doet die nog leuke dingen?"
                },
            }
        }
        topic = app.topic_candidate(
            domain="huisdier",
            label="Simba",
            fields=["pet_name", "pet_type"],
            field_labels={"pet_name": "de naam van je huisdier", "pet_type": "het soort huisdier"},
            current_values={"pet_name": "Simba", "pet_type": "kat"},
            correct_values=["Simba bij jou hoort"],
            memory_link="Simba belangrijk voor je is",
            options=["Simba", "dieren"],
            reground="Ik onthoud dat Simba belangrijk voor je is.",
        )

        segments = app.topic2_phase_segments(topic)

        open_text = app.turn_text(segments[3])
        followup_text = app.turn_text(segments[4])
        self.assertEqual(open_text, "En ik herinner me dat jij een kat hebt die Simba heet. Gaaf!")
        self.assertEqual(followup_text, "Hoe is Simba als kat eigenlijk?")
        self.assertNotIn("vis", open_text.lower())
        self.assertNotIn("vis", followup_text.lower())
        self.assertEqual(len(app.openai_client.requests), 2)

    def test_topic2_pet_fallback_uses_pet_type_and_name(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}}
        topic = app.topic_candidate(
            domain="huisdier",
            label="Luna",
            fields=["pet_name", "pet_type"],
            field_labels={"pet_name": "de naam van je huisdier", "pet_type": "het soort huisdier"},
            current_values={"pet_name": "Luna", "pet_type": "kat"},
            correct_values=["Luna bij jou hoort"],
            memory_link="Luna belangrijk voor je is",
            options=["Luna", "dieren"],
            reground="Ik onthoud dat Luna belangrijk voor je is.",
        )

        segments = app.topic2_phase_segments(topic)

        self.assertEqual(
            app.turn_text(segments[3]),
            (
                "Ik weet ook nog dat jij een kat hebt die Luna heet. "
                "Dat vind ik echt een mooie naam. Luna klinkt alsof die stiekem "
                "belangrijke plannen maakt als niemand kijkt."
            ),
        )
        self.assertEqual(app.turn_text(segments[4]), "Wat voor kat is Luna eigenlijk?")

    def test_topic2_corrected_pet_skips_old_graphdb_lines(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_t2_recall": {"default": "[STUB] Ik weet nog dat jouw huisdier een vis is."},
                "p1_t2_open": {"default": "[STUB] En ik herinner me dat jij een vis als huisdier hebt."},
                "p1_t2_followup": {"default": "[STUB] Hoe gaat het met jouw vis?"},
                "p1_t2_close": {"default": "[STUB] Bizar leuk over vissen."},
            }
        }
        topic = app.topic_candidate(
            domain="huisdier",
            label="Vis",
            fields=["pet_name", "pet_type"],
            field_labels={"pet_name": "de naam van je huisdier", "pet_type": "het soort huisdier"},
            current_values={"pet_name": "Vis", "pet_type": "Vis"},
            correct_values=["Vis bij jou hoort"],
            memory_link="Vis belangrijk voor je is",
            options=["Vis", "dieren"],
            reground="Ik onthoud dat Vis belangrijk voor je is.",
        )
        turn = {
            "phase": 7,
            "layer": "L2+L3",
            "name": "Topic 2",
            "topic": topic,
            "segments": app.topic2_phase_segments(topic),
        }

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "pet_name",
                "old_value": "Vis",
                "new_value": "Luna",
            },
        })

        self.assertFalse(turn["force_topic_fallback"])
        self.assertEqual(turn["topic"]["label"], "Luna")
        followup_segment = app.segment_context(turn, turn["segments"][4], 5)
        self.assertIn("Luna", app.turn_text(followup_segment))

    def test_topic2_sport_followup_carries_correction_context(self):
        app = make_app()
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_t2_open": {"default": "[STUB] Ik weet dat jij voetbal doet."},
                "p1_t2_followup": {"default": "[STUB] Hoe gaat het met voetbal?"},
                "p1_t2_close": {"default": "[STUB] Sport is leuk."},
            }
        }
        topic = app.topic_candidate(
            domain="sport",
            label="voetbal",
            fields=["sports_enjoys", "sports_fav_play"],
            field_labels={
                "sports_enjoys": "of je sport leuk vindt",
                "sports_fav_play": "de sport die je graag doet",
            },
            current_values={"sports_enjoys": "ja", "sports_fav_play": "voetbal"},
            correct_values=["voetbal"],
            memory_link="voetbal belangrijk voor je is",
            options=["voetbal"],
            reground="Ik onthoud dat voetbal belangrijk voor je is.",
        )

        segments = app.topic2_phase_segments(topic)

        self.assertEqual(segments[4]["response_mode"], "listen_only")
        self.assertEqual(
            segments[4]["used_fields"],
            {"sports_enjoys": "ja", "sports_fav_play": "voetbal"},
        )

    def test_topic2_closing_line_depends_on_domain(self):
        app = make_app()

        self.assertIn("Verhalen", app.segments.topic2_closing_line("boeken"))
        self.assertIn("Muziek", app.segments.topic2_closing_line("muziek"))
        self.assertIn("Sport", app.segments.topic2_closing_line("sport"))
        self.assertIn("dieren", app.segments.topic2_closing_line("huisdier"))

    def test_l3_runtime_prompt_uses_structured_context(self):
        app = make_app(openai_payloads=["Voorin, lekker! Daar moet je snel zijn."])
        app.last_um_preview = sample_um()
        app.last_um_preview["sports_fav_play"] = "hockey"
        app.last_leo_utterance = "In welke positie speel jij?"
        turn = {
            "phase": 5,
            "used_fields": {"sports_fav_play": "hockey"},
            "next_script_line": (
                "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                "maar ik val waarschijnlijk al om voor de warming-up."
            ),
            "l3": {
                "script_phase": "part1_topic1",
                "topic": "sport",
                "response_function": "acknowledge",
                "question_allowed": False,
                "relevant_um_fields": ["sports_fav_play"],
                "fallback": "Dat snap ik wel.",
            },
        }

        response = app.llm_response("Ik sta meestal voorin.", turn)

        self.assertEqual(response, "Voorin, lekker! Daar moet je snel zijn.")
        request = app.openai_client.requests[-1]
        self.assertIn("NEVER reveal or reference UM fields", request["messages"][0]["content"])
        self.assertIn("Use only plain letters and simple punctuation", request["messages"][0]["content"])
        self.assertIn("Script phase: part1_topic1", request["messages"][1]["content"])
        self.assertIn("Next scripted Leo line", request["messages"][1]["content"])
        self.assertIn('"sports_fav_play": "hockey"', request["messages"][1]["content"])

    def test_l3_runtime_sanitizes_symbols_accents_and_labels(self):
        app = make_app()

        clean = app.actions.l3.sanitize_output('Leo: "Oké — één café ✓"')

        self.assertEqual(clean, "Oke, een cafe")

    def test_l3_runtime_sanitizer_keeps_normal_short_output_readable(self):
        app = make_app()

        clean = app.actions.l3.sanitize_output("Dat snap ik wel!")

        self.assertEqual(clean, "Dat snap ik wel!")

    def test_l3_runtime_uses_fallback_when_output_breaks_safety_rules(self):
        app = make_app(openai_payloads=["Knap! Speel je vaak met Momo?"])
        app.last_um_preview = sample_um()
        app.last_um_preview["sports_fav_play"] = "hockey"
        app.conversation_log = {"events": []}
        turn = {
            "phase": 5,
            "used_fields": {"sports_fav_play": "hockey"},
            "next_script_line": "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn.",
            "l3": {
                "script_phase": "part1_topic1",
                "topic": "sport",
                "response_function": "acknowledge",
                "question_allowed": False,
                "relevant_um_fields": ["sports_fav_play"],
                "fallback": "Dat snap ik wel.",
            },
        }

        response = app.llm_response("Ik ben keeper.", turn)

        self.assertEqual(response, "Dat snap ik wel.")
        l3_events = [event for event in app.conversation_log["events"] if event["type"] == "l3_call"]
        self.assertEqual(len(l3_events), 1)
        self.assertTrue(l3_events[0]["used_fallback"])
        self.assertEqual(l3_events[0]["final_output"], "Dat snap ik wel.")

    def test_startup_memory_table_has_only_requested_columns_and_all_fields(self):
        app = make_app()
        app.simulated_persona_path = str(PACKAGE_DIR / "fake_personas" / "noor_1001.json")
        app.load_simulated_persona()
        script = app.build_script()
        rows = app.script_memory_table(script, app.last_um_preview)

        self.assertEqual(set(rows[0].keys()), {"field", "true_value", "script_value", "mistake"})
        self.assertNotIn("SPT", rows[0])
        self.assertNotIn("Used?", rows[0])
        fields = [row["field"] for row in rows]
        for field in app.SCRIPT_TABLE_FIELDS:
            self.assertIn(field, fields)
        self.assertIn("M1 related-but-wrong", [row["mistake"] for row in rows])
        self.assertIn("M2 completely-wrong", [row["mistake"] for row in rows])

    def test_prestart_db_retrieval_rows_show_um_and_scenario_missing_values(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["name"] = "Ali"
        app.last_um_preview["role_model"] = CRI.UNKNOWN_VALUE
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p1_m1_followup_wrong_hobby": {"default": "Wat vind jij leuk aan padel?"},
                "p1_m2_postcorrection_true_food": {"default": "Boerenkool klinkt ook gezellig."},
                "p1_t1_recall": {"default": "Ik weet nog dat jij voetbal doet."},
                "p3_future_theme_wrap": {"default": "Later voelt soms nog ver weg."},
                "unused_extra_step": {"default": "Deze regel hoort niet in het overzicht."},
            },
            "mistakes": [],
        }
        script = [
            {
                "content_plan": app.l2_pregen(
                    "m2_wrong_followup",
                    "fallback",
                    branch="not_corrected",
                ),
            },
            {
                "segments": [
                    {
                        "content_plan": app.l2_pregen(
                            "m2_corrected_followup",
                            "fallback",
                            branch="corrected",
                        ),
                    }
                ],
            },
        ]

        um_rows = {row["field"]: row["value"] for row in app.db_um_retrieval_rows(app.last_um_preview)}
        self.assertEqual(set(um_rows), set(app.STARTUP_OVERVIEW_UM_FIELDS))
        self.assertNotIn("condition", um_rows)
        self.assertNotIn("exposure", um_rows)
        self.assertEqual(um_rows["name"], "Ali")
        self.assertEqual(um_rows["role_model"], "<empty / unknown>")

        scenario_result = app.db_scenario_retrieval_rows(script)
        scenario_rows = {(row["step_id"], row["branch"]): row["value"] for row in scenario_result}
        self.assertEqual(
            list(scenario_rows),
            list(app.STARTUP_OVERVIEW_SCENARIO_FIELDS),
        )
        self.assertNotIn(("unused_extra_step", "default"), scenario_rows)
        self.assertEqual(
            scenario_rows[("p1_m1_followup_wrong_hobby", "default")],
            "Wat vind jij leuk aan padel?",
        )
        self.assertEqual(
            scenario_rows[("p1_m2_followup_wrong_food", "default")],
            "<missing>",
        )
        self.assertEqual(
            scenario_rows[("p1_m2_postcorrection_true_food", "default")],
            "Boerenkool klinkt ook gezellig.",
        )
        self.assertEqual(
            scenario_rows[("p1_t1_recall", "default")],
            "Ik weet nog dat jij voetbal doet.",
        )
        self.assertEqual(
            scenario_rows[("p3_future_theme_wrap", "default")],
            "Later voelt soms nog ver weg.",
        )

    def test_memory_access_returns_only_child_facing_mentioned_um_fields(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = {
            "name",
            "hobbies",
            "hobby_fav",
            "sports_enjoys",
            "sports_fav_play",
            "condition",
            "exposure",
        }
        turn = {
            "used_fields": {
                "sports_enjoys": "ja",
                "sports_fav_play": "zwemmen",
            }
        }
        result = IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95)

        response, scope, returned = app.memory_access_response(result, turn)

        self.assertNotIn("Ik heb vandaag al gebruikt", response)
        self.assertIn("je Noor heet", response)
        self.assertIn("je houdt van tekenen, tuinieren, lego bouwen", response)
        self.assertIn("tekenen jouw favoriete hobby is", response)
        self.assertIn("Over sport weet ik dat je zwemmen doet", response)
        self.assertNotIn("of je sport leuk vindt", response)
        self.assertNotIn("condition", scope)
        self.assertNotIn("exposure", scope)
        self.assertIn("sports_enjoys", scope)
        self.assertIn("freetime_fav", scope)
        self.assertIn("sports_enjoys", returned)

    def test_memory_access_scope_expands_one_field_to_whole_chapter(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = {"sports_fav_play"}

        scope = app.memory_access_scope({})

        self.assertIn("sports_enjoys", scope)
        self.assertIn("sports_fav_play", scope)
        self.assertNotIn("fav_food", scope)

    def test_memory_access_resume_replays_setup_bundle_before_current_question(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = {"interest"}
        app.speech = FakeSpeech(["kan ik je geheugen zien?", "ja hoor"])
        app.log_action_handler_result = lambda action: None

        class SequenceClassifier:
            def __init__(self):
                self.results = [
                    IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95),
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                ]

            def classify(self, text, turn_context=None):
                return self.results.pop(0)

            def classify_retry(self, text, turn_context=None):
                return self.classify(text, turn_context)

        app.clf = SequenceClassifier()
        phase = {
            "phase": 17,
            "phase_id": "3.4/5",
            "name": "Mistake 4 - aspiration + reflection",
            "layer": "test",
            "segments": [
                {
                    "content_plan": app.l1("Vorige vraag die niet opnieuw moet."),
                    "expects_response": True,
                    "response_mode": "listen_only",
                },
                {
                    "content_plan": app.l1("Later is trouwens niet alleen later-later."),
                    "expects_response": False,
                },
                {
                    "content_plan": app.l1("Voor kinderen in groep 7 en 8 komt de middelbare school dichtbij."),
                    "expects_response": False,
                },
                {
                    "content_plan": app.l1("Denk jij daar al een beetje over na?"),
                    "expects_response": True,
                    "response_mode": "listen_only",
                },
            ],
        }

        with patch("builtins.input", side_effect=[""]), patch("time.sleep", lambda *_args, **_kwargs: None):
            action = app.run_phase_segment(phase, phase["segments"][3], 4)

        self.assertEqual(action["action"], "listen_only")
        self.assertEqual(app.speech.spoken.count("Denk jij daar al een beetje over na?"), 2)
        self.assertIn("Later is trouwens niet alleen later-later.", app.speech.spoken)
        self.assertIn(
            "Voor kinderen in groep 7 en 8 komt de middelbare school dichtbij.",
            app.speech.spoken,
        )
        self.assertEqual(app.speech.spoken.count("Vorige vraag die niet opnieuw moet."), 0)
        replay_start = app.speech.spoken.index("Later is trouwens niet alleen later-later.")
        self.assertEqual(
            app.speech.spoken[replay_start:replay_start + 3],
            [
                "Later is trouwens niet alleen later-later.",
                "Voor kinderen in groep 7 en 8 komt de middelbare school dichtbij.",
                "Denk jij daar al een beetje over na?",
            ],
        )

    def test_memory_access_mentions_no_fixed_role_model_in_inspiration_chapter(self):
        app = make_app()
        um = sample_um()
        um["role_model"] = "niemand"
        um["interest"] = "Verbetering van technologie"
        um["aspiration"] = CRI.UNKNOWN_VALUE
        app.last_um_preview = um
        app.memory_fields_mentioned_so_far = {"aspiration"}
        result = IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95)

        response, scope, returned = app.memory_access_response(result, {})

        self.assertIn("role_model", scope)
        self.assertIn("role_model", returned)
        self.assertIn("je interesse hebt in Verbetering van technologie", response)
        self.assertIn("je niet echt een vaste persoon hebt naar wie je opkijkt", response)
        self.assertNotIn("niemand", response.lower())

    def test_explicit_memory_inspection_accepts_and_unblocks_review_segments(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = {
            "name",
            "age",
            "hobbies",
            "hobby_fav",
            "fav_food",
            "fav_subject",
            "school_strength",
            "school_difficulty",
            "role_model",
            "aspiration",
        }
        turn = {
            "phase": 19,
            "phase_id": "3.6",
            "response_mode": "explicit_memory_inspection_offer",
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja",
            turn,
        )

        self.assertEqual(action["action"], "explicit_memory_inspection_accepted")
        self.assertFalse(action["follow_up_needed"])
        self.assertEqual(action["tutorial_condition"], "C")
        self.assertTrue(app.memory_review_requested)
        self.assertEqual(
            set(action["memory_scope"]),
            {
                "hobby_fav",
                "freetime_fav",
                "role_model",
                "interest",
                "hobbies",
                "fav_subject",
                "fav_food",
                "school_strength",
                "name",
                "age",
                "aspiration",
                "school_difficulty",
            },
        )
        self.assertEqual(app.speech.spoken, [])

    def test_explicit_memory_inspection_decline_and_unclear_continue(self):
        app = make_app()
        app.last_um_preview = sample_um()

        decline = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Nee",
            {"phase": 19, "response_mode": "explicit_memory_inspection_offer"},
        )
        self.assertEqual(decline["action"], "explicit_memory_inspection_declined")
        self.assertEqual(decline["leo_response"], "Dat is ook goed. Dan gaan we gewoon nog even verder.")

        unclear = app.action_handler(
            IntentResult(intent="dialogue_none", field=None, value=None, confidence=0.9),
            "Weet ik niet",
            {"phase": 19, "response_mode": "explicit_memory_inspection_offer"},
        )
        self.assertEqual(unclear["action"], "explicit_memory_inspection_unclear")
        self.assertEqual(unclear["leo_response"], "Dat is goed. Je hoeft niet te kijken. Dan gaan we gewoon verder.")

    def test_explicit_memory_inspection_condition_e_accepts_tablet_review(self):
        app = make_app()
        um = sample_um()
        um["condition"] = "E"
        app.last_um_preview = um
        app.memory_fields_mentioned_so_far = {"name", "hobbies", "fav_food", "aspiration"}
        turn = {
            "phase": 19,
            "phase_id": "3.6",
            "response_mode": "explicit_memory_inspection_offer",
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja",
            turn,
        )

        self.assertEqual(action["action"], "explicit_memory_inspection_accepted")
        self.assertFalse(action["follow_up_needed"])
        self.assertEqual(action["tutorial_condition"], "E")
        self.assertTrue(app.memory_review_requested)
        self.assertEqual(
            set(action["memory_scope"]),
            {"name", "hobbies", "hobby_fav", "freetime_fav", "fav_food", "aspiration", "role_model", "interest"},
        )
        self.assertEqual(app.speech.spoken, [])

    def test_memory_review_phase_groups_fields_in_script_order(self):
        app = make_app()
        um = sample_um()
        app.last_um_preview = um

        segments, fields = app.script.memory_review_group_segments(um)

        self.assertEqual(
            [segment["memory_review_group"] for segment in segments],
            ["hobbies", "animals_food", "school", "future"],
        )
        self.assertEqual(
            set(fields),
            {
                "hobbies",
                "hobby_fav",
                "animal_fav",
                "pet_type",
                "pet_name",
                "fav_food",
                "fav_subject",
                "school_strength",
                "school_difficulty",
                "role_model",
                "aspiration",
            },
        )
        self.assertIn("tekenen jouw favoriete hobby is", app.turn_text(segments[0]))
        self.assertIn("je huisdier Momo een kat is", app.turn_text(segments[1]))
        self.assertIn("je lievelingseten pannenkoeken is", app.turn_text(segments[1]))
        self.assertIn("natuur jouw lievelingsvak is", app.turn_text(segments[2]))
        self.assertIn("rekenen soms wat lastiger voelt", app.turn_text(segments[2]))
        self.assertIn("haar moeder iemand is naar wie je opkijkt", app.turn_text(segments[3]))
        self.assertIn("je later dierenarts wilt worden", app.turn_text(segments[3]))
        for segment in segments:
            self.assertTrue(segment["memory_correction_available"])
            self.assertEqual(segment["memory_review_fields"], list(segment["used_fields"].keys()))

    def test_memory_review_uses_plural_for_multiple_favourite_subjects(self):
        app = make_app()
        um = sample_um()
        um["fav_subject"] = "gym en rekenen"
        app.last_um_preview = um

        lines = app.mem.memory_review_lines(["fav_subject"])

        self.assertIn("gym en rekenen jouw lievelingsvakken zijn", " ".join(lines))
        self.assertNotIn("gym en rekenen jouw lievelingsvak is", " ".join(lines))

    def test_memory_review_group_response_modes_confirm_repeat_and_ask_detail(self):
        app = make_app()
        turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "response_mode": "memory_review_group",
            "content_plan": app.l1("Over school weet ik iets. Klopt dat?"),
            "topic": {
                "fields": ["fav_subject", "school_strength"],
                "field_labels": {
                    "fav_subject": "je lievelingsvak",
                    "school_strength": "waar je goed in bent op school",
                },
                "current_values": {
                    "fav_subject": "natuur",
                    "school_strength": "taal",
                },
            },
        }

        confirmed = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja, dat klopt",
            turn,
        )
        self.assertEqual(confirmed["action"], "memory_review_group_confirmed")
        self.assertEqual(confirmed["leo_response"], "Fijn, dan laat ik dat zo staan.")

        app.speech.spoken.clear()
        repeat = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Kun je dat nog een keer zeggen?",
            turn,
        )
        self.assertEqual(repeat["action"], "memory_review_group_repeat")
        self.assertTrue(repeat["follow_up_needed"])
        self.assertEqual(app.speech.spoken, ["Over school weet ik iets. Klopt dat?"])

        detail = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Dat klopt niet",
            turn,
        )
        self.assertEqual(detail["action"], "memory_review_group_ask_correction_detail")
        self.assertTrue(detail["follow_up_needed"])

    def test_memory_review_final_can_decline_or_ask_for_extra_detail(self):
        app = make_app()
        turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "response_mode": "memory_review_add_final",
            "content_plan": app.l1("Is er nog iets dat ik moet onthouden?"),
            "topic": {"fields": list(CRI.UM_FIELDS), "field_labels": {}, "current_values": {}},
        }

        no_extra = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Nee",
            turn,
        )
        self.assertEqual(no_extra["action"], "memory_review_final_no_extra")

        asks_detail = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja",
            turn,
        )
        self.assertEqual(asks_detail["action"], "memory_review_final_ask_detail")
        self.assertTrue(asks_detail["follow_up_needed"])

        extra = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ik hou ook van dinosaurussen",
            turn,
        )
        self.assertEqual(extra["action"], "memory_review_final_extra_unmapped")

    def test_memory_review_segments_skip_unless_offer_was_accepted_and_can_activate_tablet(self):
        app = make_app()
        self.assertTrue(app.should_skip_phase({"condition": "run_if_memory_review_requested"}))
        app.memory_review_requested = True
        self.assertFalse(app.should_skip_phase({"condition": "run_if_memory_review_requested"}))

        captured = {}

        class FakeTabletState:
            def activate_memory_access(self, fields, phase=None):
                captured["fields"] = list(fields)
                captured["phase"] = phase

        app.tablet_state = FakeTabletState()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = {"fav_food"}
        phase = {"phase": 18, "phase_id": "3.6/7", "name": "Explicit memory inspection", "layer": "L1"}
        segment = {
            "content_plan": app.l1("Kijk maar op de tablet."),
            "expects_response": False,
            "condition": "run_if_memory_review_requested",
            "memory_review_from_access_scope": True,
            "speak_memory_review_from_access_scope": True,
            "activate_tablet_memory_access": True,
            "memory_review_fallback_fields": ["hobbies", "fav_food", "aspiration"],
        }

        app.memory_review_requested = False
        app.run_phase_segment(phase, segment)
        self.assertEqual(captured, {})
        self.assertEqual(app.speech.spoken, [])

        app.memory_review_requested = True
        app.run_phase_segment(phase, segment)

        self.assertEqual(captured["phase"], 18)
        self.assertEqual(captured["fields"], ["fav_food"])
        self.assertEqual(
            app.speech.spoken,
            ["Kijk maar op de tablet.", "Ik weet ook nog dat je lievelingseten pannenkoeken is."],
        )

    def test_explicit_memory_review_direct_start_does_not_fall_back_to_static_um_fields(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.memory_fields_mentioned_so_far = set()
        app.memory_review_requested = True
        captured = {}

        class FakeTabletState:
            def activate_memory_access(self, fields, phase=None):
                captured["fields"] = list(fields)
                captured["phase"] = phase

        app.tablet_state = FakeTabletState()
        phase = {"phase": 18, "phase_id": "3.6/7", "name": "Explicit memory inspection", "layer": "L1"}
        segment = {
            "content_plan": app.l1("Kijk maar op de tablet."),
            "expects_response": False,
            "condition": "run_if_memory_review_requested",
            "memory_review_from_access_scope": True,
            "speak_memory_review_from_access_scope": True,
            "activate_tablet_memory_access": True,
        }

        app.run_phase_segment(phase, segment)

        self.assertEqual(captured["phase"], 18)
        self.assertEqual(captured["fields"], [])
        self.assertEqual(
            app.speech.spoken,
            [
                "Kijk maar op de tablet.",
                "Ik heb vandaag nog niet zoveel uit mijn geheugen gebruikt.",
            ],
        )

    def test_spontaneous_memory_access_condition_e_uses_same_tablet_visible_fields(self):
        app = make_app()
        um = sample_um()
        um["condition"] = "E"
        app.last_um_preview = um
        app.memory_fields_mentioned_so_far = {"hobbies", "fav_food"}
        captured = {}

        class FakeTabletState:
            def activate_memory_access(self, fields, phase=None):
                captured["fields"] = list(fields)
                captured["phase"] = phase

        app.tablet_state = FakeTabletState()
        turn = {
            "phase": 12,
            "response_mode": "listen_only",
            "used_fields": {"fav_subject": "natuur"},
        }

        action = app.action_handler(
            IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95),
            "Wat weet je over mij?",
            turn,
        )

        self.assertEqual(action["action"], "memory_access_tablet")
        self.assertEqual(
            app.speech.spoken,
            [
                "Tik op mijn geheugenboek om te zien wat erin staat. "
                "Als er iets niet klopt, zeg het dan tegen mij."
            ],
        )
        self.assertEqual(
            set(action["visible_fields"]),
            {
                "hobbies",
                "hobby_fav",
                "freetime_fav",
                "fav_food",
                "fav_subject",
                "school_strength",
                "school_difficulty",
            },
        )
        self.assertEqual(set(captured["fields"]), set(action["visible_fields"]))
        self.assertEqual(captured["phase"], 12)

    def test_spontaneous_memory_access_condition_c_speaks_memory_with_tutorial(self):
        app = make_app()
        um = sample_um()
        um["condition"] = "C"
        app.last_um_preview = um
        app.memory_fields_mentioned_so_far = {"sports_fav_play"}

        action = app.action_handler(
            IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95),
            "Wat weet je over mij?",
            {"phase": 5, "response_mode": "topic_interpretation"},
        )

        self.assertEqual(action["action"], "memory_access")
        self.assertIn("Over sport weet ik dat je zwemmen doet", action["leo_response"])
        self.assertIn("Als iets niet klopt", action["leo_response"])
        self.assertIn("nog een keer wilt horen", action["leo_response"])
        self.assertIn("sports_enjoys", action["visible_fields"])
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_memory_access_change_context_allows_any_visible_field(self):
        app = make_app()
        app.last_um_preview = sample_um()
        action = {
            "visible_fields": ["sports_enjoys", "sports_fav_play"],
            "memory_scope": ["sports_enjoys", "sports_fav_play"],
        }

        context = app.memory_access_change_context(action, {"phase": 5, "phase_id": "1.5", "name": "Topic 1"})

        self.assertEqual(context["response_mode"], "memory_access_change")
        self.assertTrue(context["allow_memory_change"])
        self.assertTrue(context["memory_correction_requested"])
        self.assertEqual(context["memory_review_fields"], ["sports_enjoys", "sports_fav_play"])
        self.assertEqual(set(app.allowed_change_fields(context)), {"sports_enjoys", "sports_fav_play"})

    def test_memory_access_change_handles_ambiguous_yes_no_without_write(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.write_um_change = lambda *_args, **_kwargs: self.fail("Ambiguous answer must not write UM")
        turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "name": "Explicit memory inspection",
            "response_mode": "memory_access_change",
            "allow_memory_change": True,
            "memory_correction_requested": True,
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja, nee, of niet.",
            turn,
        )

        self.assertEqual(action["action"], "memory_access_change_clarify_yes_no")
        self.assertNotEqual(action["action"], "ask_correction_detail")
        self.assertTrue(action["follow_up_needed"])
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, ["Wil je iets veranderen? Zeg maar ja of nee."])

    def test_memory_access_change_no_yes_and_wrong_are_separate_routes(self):
        turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "name": "Explicit memory inspection",
            "response_mode": "memory_access_change",
            "allow_memory_change": True,
            "memory_correction_requested": True,
        }

        app = make_app()
        no_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "nee hoor",
            dict(turn),
        )
        self.assertEqual(no_action["action"], "memory_access_change_none")
        self.assertEqual(app.speech.spoken, ["Oké, dan laat ik alles zo."])

        app = make_app()
        with patch("time.sleep", lambda *_args, **_kwargs: None):
            yes_action = app.action_handler(
                IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                "ja",
                dict(turn),
            )
        self.assertEqual(yes_action["action"], "memory_access_change_no_value")
        self.assertEqual(app.speech.spoken[0], "Wat wil je veranderen?")

        app = make_app()
        wrong_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "dat klopt niet",
            dict(turn),
        )
        self.assertEqual(wrong_action["action"], "ask_correction_detail")
        self.assertEqual(wrong_action["leo_response"], "Oeps, wat klopt er dan niet?")

    def test_memory_access_change_to_already_stored_value_is_acknowledged(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["hobby_fav"] = "gamen"
        action = {
            "visible_fields": ["hobbies", "hobby_fav", "freetime_fav"],
            "memory_scope": ["hobbies", "hobby_fav", "freetime_fav"],
        }
        source_turn = {"phase": 6, "phase_id": "1.6", "name": "Mistake 1 - hobby_fav"}
        context = app.memory_access_change_context(action, source_turn)

        result = IntentResult(intent="um_update", field="hobby_fav", value="gamen", confidence=0.95)
        handled = app.action_handler(result, "Mijn lievelingshobby naar gamen", context)

        self.assertEqual(handled["action"], "memory_access_change_already_stored")
        self.assertTrue(handled["handled"])
        self.assertEqual(handled["change"]["field"], "hobby_fav")
        self.assertEqual(handled["change"]["new_value"], "gamen")
        self.assertEqual(
            app.speech.spoken[0],
            "Dat staat al zo in mijn geheugen. Ik heb al onthouden dat je favoriete hobby gamen is.",
        )

    def test_memory_access_change_from_visible_mistake_to_stored_value_is_correction(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["hobby_fav"] = "gamen"
        app.mistake_states = {
            "M1": {
                "id": "M1",
                "mentioned": True,
                "field": "hobby_fav",
                "actual": "gamen",
                "wrong": "padel",
                "corrected": False,
            }
        }
        action = {
            "visible_fields": ["hobbies", "hobby_fav", "freetime_fav"],
            "memory_scope": ["hobbies", "hobby_fav", "freetime_fav"],
        }
        source_turn = {"phase": 6, "phase_id": "1.6", "name": "Memory access"}
        context = app.memory_access_change_context(action, source_turn)

        self.assertEqual(context["topic"]["current_values"]["hobby_fav"], "padel")
        self.assertEqual(context["topic"]["stored_values"]["hobby_fav"], "gamen")

        result = IntentResult(intent="um_update", field="hobby_fav", value="gamen", confidence=0.95)
        change = app.actions.change_from_intent_result(result, context, "Mijn lievelingshobby naar gamen")

        self.assertEqual(change["action"], "update")
        self.assertEqual(change["old_value"], "padel")
        self.assertEqual(change["new_value"], "gamen")
        self.assertEqual(change["visible_mistake_id"], "M1")
        self.assertTrue(change["replace_field"])

        app.handle_confirmed_mistake_related_change(change, context)

        self.assertTrue(app.mistake_states["M1"]["corrected"])

    def test_memory_access_change_in_mistake_phase_reroutes_to_corrected_branch(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["condition"] = "E"
        app.last_um_preview["hobby_fav"] = "gamen"
        app.simulated_persona = dict(app.last_um_preview)
        app.speech = FakeSpeech([
            "Kan ik je geheugen zien?",
            "ja",
            "favoriete hobby naar spelletjes spelen",
            "ja",
            "nee",
        ])
        app.log_action_handler_result = lambda action: None

        class SequenceClassifier:
            def __init__(self):
                self.results = [
                    IntentResult(intent="um_inspect", field=None, value=None, confidence=0.95),
                    IntentResult(intent="um_update", field="hobby_fav", value="spelletjes spelen", confidence=0.95),
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.95),
                ]

            def classify(self, text, turn_context=None):
                return self.results.pop(0)

            def classify_retry(self, text, turn_context=None):
                return self.classify(text, turn_context)

        app.clf = SequenceClassifier()
        phase = {
            "phase": 6,
            "phase_id": "1.6",
            "name": "Mistake 1 - hobby_fav",
            "layer": "L2-slot WRONG + L2-pregen",
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "gamen",
            "mistake_wrong": "padel",
            "mistake_topic": {
                "fields": ["hobby_fav"],
                "field_labels": {"hobby_fav": "je favoriete hobby"},
                "current_values": {"hobby_fav": "gamen"},
                "domain": "hobby",
            },
            "used_fields": {"hobby_fav": "padel"},
            "segments": [
                {
                    "content_plan": app.l2_slot(
                        "En volgens mij is {wrong_hobby} jouw allerliefste hobby.",
                        {"wrong_hobby": "padel"},
                        wrong=True,
                    ),
                    "expects_response": True,
                    "response_mode": "mistake_interpretation",
                },
                {
                    "content_plan": app.sequence(
                        app.l1("Dat snap ik trouwens wel."),
                        app.l2_pregen(
                            "m1_wrong_opener",
                            "Leuk, padel! Wat vind je er het allerleukst aan?",
                            ["hobby_fav"],
                        ),
                    ),
                    "expects_response": True,
                    "response_mode": "mistake_interpretation",
                    "skip_if_phase_confirmed_change": True,
                },
                {
                    "content_plan": app.l2_pregen(
                        "m1_corrected_followup",
                        "Wat vind jij het leukste aan {hobby_fav}?",
                        ["hobby_fav"],
                        require_input_values=True,
                        branch="corrected",
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "run_if_phase_confirmed_change": True,
                    "used_fields": {"hobby_fav": "gamen"},
                },
            ],
        }
        app.register_mistake_phase(phase)

        with patch("builtins.input", side_effect=["c", ""]), patch("time.sleep", lambda *_args, **_kwargs: None):
            action = app.run_phase_segment(phase, phase["segments"][1], 2)

        wrong_followup = "Dat snap ik trouwens wel. Leuk, padel! Wat vind je er het allerleukst aan?"
        self.assertEqual(app.speech.spoken.count(wrong_followup), 1)
        self.assertEqual(action["action"], "memory_access_change_confirmed")
        self.assertTrue(action["change_confirmed"])
        self.assertFalse(action["stop_phase_after_change"])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"]["corrected"])
        self.assertEqual(app.mistake_states["M1"]["actual"], "spelletjes spelen")
        self.assertEqual(app.last_um_preview["hobby_fav"], "spelletjes spelen")
        self.assertEqual(phase["mistake_actual"], "spelletjes spelen")
        self.assertEqual(phase["segments"][2]["used_fields"]["hobby_fav"], "spelletjes spelen")

        corrected_context = app.segment_context(phase, phase["segments"][2], 3)
        self.assertEqual(
            app.turn_text(corrected_context),
            "Wat vind jij het leukste aan spelletjes spelen?",
        )

    def test_tablet_state_writes_visible_fields_for_memory_access(self):
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_tablet_state_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        state_path = Path(temp_dir) / "session_state.json"
        writer = cri_module.TabletStateWriter(
            state_path=str(state_path),
            get_child_id_fn=lambda: "test_boy_1",
            get_child_name_fn=lambda: "Sam",
            get_mistake_states_fn=lambda: {
                "M4": {"field": "aspiration", "wrong": "juf worden", "corrected": True}
            },
        )

        writer.update({"phase": 18, "used_fields": {"fav_food": "pannenkoeken"}})
        writer.activate_memory_access(["name", "fav_food", "aspiration"], phase=19)

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["child_id"], "test_boy_1")
        self.assertEqual(state["phase"], 19)
        self.assertTrue(state["memory_access_active"])
        self.assertEqual(state["memory_access_prompt_id"], 1)
        self.assertEqual(state["tablet_command"]["type"], "memory_access_home")
        self.assertEqual(state["visible_fields"], ["name", "fav_food", "aspiration"])
        self.assertIn("eten", state["unlocked_categories"])
        self.assertIn("aspiratie", state["unlocked_categories"])
        self.assertTrue(state["mistakes"]["M4"]["corrected"])
        self.assertEqual(state["mistakes"]["M4"]["wrong"], "juf worden")
        first_command_id = state["tablet_command"]["id"]

        writer.activate_memory_access(["fav_food"], phase=20)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["tablet_command"]["type"], "memory_access_home")
        self.assertNotEqual(state["tablet_command"]["id"], first_command_id)

    def test_tablet_state_writes_reveal_command_for_corrected_value(self):
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_tablet_reveal_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        state_path = Path(temp_dir) / "session_state.json"
        writer = cri_module.TabletStateWriter(
            state_path=str(state_path),
            get_child_id_fn=lambda: "701",
            get_child_name_fn=lambda: "Ali",
            get_mistake_states_fn=lambda: {
                "M1": {"field": "hobby_fav", "actual": "voetbal", "wrong": "padel", "corrected": True}
            },
        )

        writer.reveal_change(field="hobby_fav", old_value="padel", new_value="gamen", phase=6)

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], 6)
        self.assertIn("hobby", state["unlocked_categories"])
        self.assertEqual(state["tablet_reveal"]["id"], 1)
        self.assertEqual(state["tablet_reveal"]["field"], "hobby_fav")
        self.assertEqual(state["tablet_reveal"]["category"], "hobby")
        self.assertEqual(state["tablet_reveal"]["old_value"], "padel")
        self.assertEqual(state["tablet_reveal"]["new_value"], "gamen")
        self.assertIsInstance(state["tablet_reveal"]["created_at"], float)

    def test_tablet_state_writes_pending_reveal_and_clears_on_reveal(self):
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_tablet_pending_reveal_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        state_path = Path(temp_dir) / "session_state.json"
        writer = cri_module.TabletStateWriter(
            state_path=str(state_path),
            get_child_id_fn=lambda: "701",
            get_child_name_fn=lambda: "Ali",
        )

        writer.prepare_reveal_change(
            field="hobbies",
            old_value="voetbal, schaatsen",
            new_value="fietsen, zwemmen",
            phase=6,
        )

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIn("hobby", state["unlocked_categories"])
        self.assertIsNone(state["tablet_reveal"])
        self.assertEqual(state["tablet_reveal_pending"]["field"], "hobbies")
        self.assertEqual(state["tablet_reveal_pending"]["category"], "hobby")
        self.assertEqual(state["tablet_reveal_pending"]["old_value"], "voetbal, schaatsen")
        self.assertEqual(state["tablet_reveal_pending"]["new_value"], "fietsen, zwemmen")

        writer.reveal_change(
            field="hobbies",
            old_value="voetbal, schaatsen",
            new_value="fietsen, zwemmen",
            phase=6,
        )

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIsNone(state["tablet_reveal_pending"])
        self.assertEqual(state["tablet_reveal"]["field"], "hobbies")
        self.assertEqual(state["tablet_reveal"]["old_value"], "voetbal, schaatsen")
        self.assertEqual(state["tablet_reveal"]["new_value"], "fietsen, zwemmen")

    def test_tablet_state_reset_clears_stale_reveal_file(self):
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_tablet_reset_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        state_path = Path(temp_dir) / "session_state.json"
        writer = cri_module.TabletStateWriter(
            state_path=str(state_path),
            get_child_id_fn=lambda: "701",
            get_child_name_fn=lambda: "Ali",
        )

        writer.reveal_change(field="hobby_fav", old_value="padel", new_value="gamen", phase=6)
        writer.reset()

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["unlocked_categories"], [])
        self.assertFalse(state["memory_access_active"])
        self.assertEqual(state["memory_access_prompt_id"], 0)
        self.assertEqual(state["visible_fields"], [])
        self.assertIsNone(state["tablet_reveal"])
        self.assertIsNone(state["tablet_reveal_pending"])
        self.assertIsNone(state["tablet_command"])

    def test_tablet_state_carries_all_unresolved_mistake_overlays(self):
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_tablet_state_mistakes_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        state_path = Path(temp_dir) / "session_state.json"
        writer = cri_module.TabletStateWriter(
            state_path=str(state_path),
            get_child_id_fn=lambda: "701",
            get_child_name_fn=lambda: "Ali",
            get_mistake_states_fn=lambda: {
                "M1": {"field": "hobby_fav", "actual": "voetbal", "wrong": "padel", "corrected": False},
                "M2": {"field": "fav_food", "actual": "cag kebabi", "wrong": "frietjes", "corrected": False},
                "M3": {"field": "school_strength", "actual": "Rekenen", "wrong": "begrijpend lezen", "corrected": False},
                "M4": {"field": "aspiration", "actual": "dokter", "wrong": "kok", "corrected": False},
            },
        )

        writer.activate_memory_access(
            ["hobby_fav", "fav_food", "school_strength", "aspiration"],
            phase=12,
        )

        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertTrue(state["memory_access_active"])
        self.assertEqual(
            state["visible_fields"],
            ["hobby_fav", "fav_food", "school_strength", "aspiration"],
        )
        self.assertEqual(state["mistakes"]["M1"]["wrong"], "padel")
        self.assertEqual(state["mistakes"]["M2"]["wrong"], "frietjes")
        self.assertEqual(state["mistakes"]["M3"]["wrong"], "begrijpend lezen")
        self.assertEqual(state["mistakes"]["M4"]["wrong"], "kok")
        self.assertEqual(state["mistakes"]["M1"]["actual"], "voetbal")
        self.assertEqual(state["mistakes"]["M4"]["actual"], "dokter")

    def test_mistake_is_registered_before_tablet_state_update(self):
        app = make_app()
        state_seen_by_tablet_update = {}

        def fake_start_turn_log(turn):
            state_seen_by_tablet_update.update(app.mistake_states.get("M1", {}))

        app.start_turn_log = fake_start_turn_log
        app.finish_turn_log = lambda: None
        app.run_phase_segment = lambda turn, segment: None

        app.run_phase(
            {
                "phase": 6,
                "layer": "L2-slot WRONG",
                "name": "Mistake 1 - hobby_fav",
                "text": "En volgens mij is padel jouw allerliefste hobby.",
                "mistake_id": "M1",
                "mistake_field": "hobby_fav",
                "mistake_actual": "voetbal",
                "mistake_wrong": "padel",
                "mistake_type": "related-but-wrong",
            },
            phase_index=5,
            total_phases=21,
        )

        self.assertEqual(state_seen_by_tablet_update["field"], "hobby_fav")
        self.assertEqual(state_seen_by_tablet_update["wrong"], "padel")
        self.assertFalse(state_seen_by_tablet_update["corrected"])

    def test_mistake_latency_timer_starts_on_wrong_value_segment_only(self):
        app = make_app()
        app.log_timestamp = lambda: 42.5

        setup_context = {
            "mistake_id": "M4",
            "content_plan": app.l1("Eerst praat Leo over later nadenken."),
        }
        app.register_mistake_utterance_start(setup_context)

        self.assertNotIn("M4", app.mistake_states)

        wrong_context = {
            "mistake_id": "M4",
            "content_plan": app.l2_pregen(
                "p3_m4_followup_wrong_aspiration",
                "En volgens mij wil jij later kok worden.",
            ),
            "starts_mistake_timer": True,
        }
        app.register_mistake_utterance_start(wrong_context)

        self.assertEqual(app.mistake_states["M4"]["mistake_utterance_at"], 42.5)

    def test_tablet_js_keeps_mistake_visible_without_auto_animation(self):
        script = (OUTER_DIR / "UM-TABLET" / "webfiles" / "script.js").read_text(encoding="utf-8")

        self.assertNotIn("if (!memoryAccessActive) return null;", script)
        self.assertIn("const instantUpdatesByCategory", script)
        self.assertIn("if (prevValue !== newValue && unresolvedMistake)", script)
        self.assertIn("renderPills(currentCategory, true);", script)

    def test_action_handler_confirms_already_correct_visible_mistake_without_um_write(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        app.mistake_states = {"M1": {"id": "M1", "wrong_value_rejected": True}}
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
        }
        app.current_turn_context = turn
        result = IntentResult(intent="um_add", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Mijn favoriete hobby is tekenen", turn)

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change"]["skip_um_write"])
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_listen_only_ignores_hallucinated_um_add_from_topic_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 5,
            "response_mode": "listen_only",
            "topic": {
                "fields": ["hobbies"],
                "field_labels": {"hobbies": "je hobby's"},
                "current_values": {"hobbies": "turnen"},
            },
        }
        result = IntentResult(intent="um_add", field="hobbies", value="voetbal", confidence=0.9)

        action = app.action_handler(
            result,
            "Het is gezond en ik doe het graag met mijn vrienden",
            turn,
        )

        self.assertEqual(action["action"], "listen_only")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, [])

    def test_topic_value_statement_rejection_asks_for_correction_detail(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 5,
            "response_mode": "topic_interpretation",
            "used_fields": {"sports_fav_play": "voetbal"},
            "topic": {
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="sports_enjoys", value=None, confidence=0.93),
            "Nee dat is niet waar",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertTrue(action["follow_up_needed"])
        self.assertTrue(turn["memory_correction_requested"])
        self.assertEqual(turn["memory_correction_field"], "sports_fav_play")
        self.assertEqual(app.speech.spoken, ["Oeps, welke sport moet ik dan onthouden?"])

    def test_topic2_listen_only_pet_rejection_asks_broad_pet_correction_question(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="has_pet", value=None, confidence=0.92),
            "Nee dat klopt niet",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertTrue(action["follow_up_needed"])
        self.assertTrue(turn["memory_correction_requested"])
        self.assertNotIn("memory_correction_field", turn)
        self.assertEqual(
            action["leo_response"],
            "Oeps, wat moet ik dan onthouden over je huisdier?",
        )

    def test_topic2_listen_only_pet_plain_no_stays_normal_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.92),
            "Nee",
            turn,
        )

        self.assertEqual(action["action"], "listen_only")
        self.assertEqual(action["change"], {})
        self.assertNotIn("memory_correction_requested", turn)
        self.assertEqual(app.speech.spoken, [])

    def test_topic2_listen_only_pet_explicit_name_update_confirms_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["een kat", "ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="pet_name", value="Luna", confidence=0.95),
            "Mijn huisdier heet Luna",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "kat")
        self.assertEqual(writes[1]["new_value"], "Luna")
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(app.last_um_preview["pet_type"], "kat")
        self.assertEqual(app.last_um_preview["pet_name"], "Luna")

    def test_topic2_pet_type_classifier_result_keeps_pet_type_field(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "vis", "pet_name": "Vis"})
        app.speech.heard = ["Simba", "ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "memory_correction_requested": True,
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_add", field="pet_type", value="kat", confidence=0.95),
            "Het is een kat",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "kat")
        self.assertEqual(writes[1]["new_value"], "Simba")
        self.assertEqual(app.last_um_preview["pet_type"], "kat")
        self.assertEqual(app.last_um_preview["pet_name"], "Simba")
        self.assertEqual(app.speech.spoken[0], "Oké, een kat. Hoe heet je huisdier?")
        self.assertIn("een kat is en Simba heet", app.speech.spoken[1])

    def test_topic2_pet_name_only_asks_for_pet_type_before_confirming(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "vis", "pet_name": "Vis"})
        app.speech.heard = ["een kat", "ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "memory_correction_requested": True,
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="pet_name", value="Simba", confidence=0.95),
            "Hij heet Simba",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "kat")
        self.assertEqual(writes[1]["new_value"], "Simba")
        self.assertEqual(app.speech.spoken[0], "Oké, Simba. Wat voor huisdier is Simba?")

    def test_topic2_pet_type_and_name_can_be_corrected_in_one_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "vis", "pet_name": "Vis"})
        app.speech.heard = ["ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "memory_correction_requested": True,
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_add", field="pet_type", value="kat", confidence=0.95),
            "Het is een kat en hij heet Simba",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertTrue(action["change_confirmed"])
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "kat")
        self.assertEqual(writes[1]["new_value"], "Simba")
        self.assertEqual(app.last_um_preview["pet_type"], "kat")
        self.assertEqual(app.last_um_preview["pet_name"], "Simba")
        self.assertIn("een kat is en Simba heet", app.speech.spoken[0])
        self.assertTrue(action["continue_phase_after_change"])

    def test_topic2_pet_type_and_name_catches_reversed_dutch_name_order(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "vis", "pet_name": "Vis"})
        app.speech.heard = ["ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "memory_correction_requested": True,
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_add", field="pet_type", value="poes", confidence=0.95),
            "Dat het een poes is die lulu heet",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertTrue(action["change_confirmed"])
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "poes")
        self.assertEqual(writes[1]["new_value"], "Lulu")
        self.assertEqual(app.last_um_preview["pet_type"], "poes")
        self.assertEqual(app.last_um_preview["pet_name"], "Lulu")
        self.assertIn("een poes is en Lulu heet", app.speech.spoken[0])
        self.assertNotIn("Hoe heet je huisdier?", app.speech.spoken[0])
        self.assertTrue(action["continue_phase_after_change"])

    def test_topic2_pet_name_then_type_in_one_answer_does_not_use_name_as_type(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview.update({"pet_type": "vis", "pet_name": "Vis"})
        app.speech.heard = ["ja"]
        writes = []
        self.capture_pet_pair_writes(app, writes)
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "memory_correction_requested": True,
            "used_fields": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            "topic": {
                "domain": "huisdier",
                "label": "Vis",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_add", field="pet_name", value="lulu", confidence=0.95),
            "dat hij lulu heet en het is een poes",
            turn,
        )

        self.assertEqual(action["action"], "confirm_multi_update")
        self.assertTrue(action["change_confirmed"])
        self.assertEqual([write["field"] for write in writes], ["pet_type", "pet_name"])
        self.assertEqual(writes[0]["new_value"], "poes")
        self.assertEqual(writes[1]["new_value"], "Lulu")
        self.assertEqual(app.last_um_preview["pet_type"], "poes")
        self.assertEqual(app.last_um_preview["pet_name"], "Lulu")
        self.assertIn("een poes is en Lulu heet", app.speech.spoken[0])
        self.assertNotIn("een lulu is", app.speech.spoken[0])
        self.assertTrue(action["continue_phase_after_change"])

    def test_topic2_pet_correction_resumes_with_corrected_followup_not_open(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = {
            "domain": "huisdier",
            "label": "Vis",
            "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
            "field_labels": {
                "pet_name": "de naam van je huisdier",
                "pet_type": "het soort huisdier",
                "animal_fav": "je lievelingsdier",
                "has_pet": "of je een huisdier hebt",
            },
            "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
        }
        phase = {
            "phase": 7,
            "name": "Topic 2",
            "topic": topic,
            "segments": app.topic2_phase_segments(topic),
        }
        app.current_turn_context = {**phase, **phase["segments"][4], "segment": 5}
        action = {
            "continue_phase_after_change": True,
            "change": {
                "action": "multi_update",
                "topic_correction": True,
                "changes": [
                    {"field": "pet_type", "new_value": "kat"},
                    {"field": "pet_name", "new_value": "Simba"},
                ],
            },
        }

        app.refresh_topic_after_change(phase, action)

        followup_context = {**phase, **phase["segments"][5]}
        close_context = {**phase, **phase["segments"][6]}
        self.assertEqual(app.turn_text(followup_context), "Wat voor kat is Simba eigenlijk?")
        self.assertNotIn("Ik weet ook nog dat jij een kat hebt die Simba heet", app.turn_text(followup_context))
        self.assertIn("Bizar leuk", app.turn_text(close_context))

    def test_repeating_topic2_after_pet_correction_uses_corrected_pet_values(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = {
            "domain": "huisdier",
            "label": "Vis",
            "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
            "field_labels": {
                "pet_name": "de naam van je huisdier",
                "pet_type": "het soort huisdier",
                "animal_fav": "je lievelingsdier",
                "has_pet": "of je een huisdier hebt",
            },
            "current_values": {"pet_name": "Vis", "pet_type": "vis", "has_pet": "ja"},
        }
        phase = {
            "phase": 7,
            "name": "Topic 2",
            "topic": topic,
            "segments": app.topic2_phase_segments(topic),
        }
        app.current_turn_context = {**phase, **phase["segments"][4], "segment": 5}
        action = {
            "continue_phase_after_change": True,
            "change": {
                "action": "multi_update",
                "topic_correction": True,
                "changes": [
                    {"field": "pet_type", "new_value": "poes"},
                    {"field": "pet_name", "new_value": "Lulu"},
                ],
            },
        }

        app.refresh_topic_after_change(phase, action)
        app.reset_phase_attempt_state(phase)

        open_context = {**phase, **phase["segments"][3]}
        followup_context = {**phase, **phase["segments"][4]}
        self.assertIn("poes", app.turn_text(open_context))
        self.assertIn("Lulu", app.turn_text(open_context))
        self.assertEqual(app.turn_text(followup_context), "Wat voor poes is Lulu eigenlijk?")
        self.assertNotIn("vis", app.turn_text(open_context).lower())
        self.assertNotIn("vis", app.turn_text(followup_context).lower())

    def test_topic2_listen_only_sport_rejection_asks_sport_question(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 7,
            "response_mode": "listen_only",
            "used_fields": {"sports_enjoys": "ja", "sports_fav_play": "voetbal"},
            "topic": {
                "domain": "sport",
                "label": "voetbal",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="sports_enjoys", value=None, confidence=0.92),
            "Nee dat klopt niet",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertEqual(action["leo_response"], "Oeps, welke sport moet ik dan onthouden?")

    def test_topic_value_statement_agreement_continues_without_memory_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 5,
            "response_mode": "topic_interpretation",
            "used_fields": {"sports_fav_play": "voetbal"},
            "topic": {
                "domain": "sport",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.93),
            "Ja dat klopt",
            turn,
        )

        self.assertEqual(action["action"], "no_memory_change")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, [])
        self.assertEqual(app.last_um_preview["sports_fav_play"], "zwemmen")

    def test_topic_value_statement_inline_update_confirms_and_continues_topic_phase(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 5,
            "response_mode": "topic_interpretation",
            "used_fields": {"sports_fav_play": "voetbal"},
            "topic": {
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="sports_fav_play", value="hockey", confidence=0.96),
            "Nee, ik doe hockey",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change_confirmed"])
        self.assertFalse(action["stop_phase_after_change"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertTrue(action["change"]["topic_correction"])
        self.assertTrue(action["change"]["replace_field"])
        self.assertEqual(app.last_um_preview["sports_fav_play"], "hockey")
        self.assertIn(5, app.phases_with_confirmed_change)
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik de sport die je graag doet verander naar hockey?")

    def test_listen_only_um_mention_followup_bare_answer_updates_memory(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 5,
            "response_mode": "listen_only",
            "used_fields": {"sports_fav_play": "voetbal"},
            "memory_correction_requested": True,
            "memory_correction_field": "sports_fav_play",
            "topic": {
                "domain": "sport",
                "label": "voetbal",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="hockey", confidence=0.9),
            "hockey",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "sports_fav_play")
        self.assertEqual(action["change"]["new_value"], "hockey")
        self.assertFalse(action["stop_phase_after_change"])
        self.assertTrue(action["continue_phase_after_change"])
        self.assertEqual(app.last_um_preview["sports_fav_play"], "hockey")

    def test_topic_correction_prefers_spoken_sport_field_over_classifier_boolean(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 5,
            "response_mode": "listen_only",
            "used_fields": {"sports_fav_play": "voetbal"},
            "memory_correction_requested": True,
            "memory_correction_field": "sports_fav_play",
            "topic": {
                "domain": "sport",
                "label": "voetbal",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_add", field="sports_enjoys", value="hockey", confidence=0.92),
            "Ik doe hockey",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "sports_fav_play")
        self.assertEqual(action["change"]["old_value"], "voetbal")
        self.assertEqual(action["change"]["new_value"], "hockey")
        self.assertNotEqual(action["change"]["field"], "sports_enjoys")

    def test_confirmed_topic_correction_rebuilds_remaining_segments_with_new_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = app.topic_candidate(
            domain="sport",
            label="voetbal",
            fields=["sports_enjoys", "sports_fav_play"],
            field_labels={
                "sports_enjoys": "of je sport leuk vindt",
                "sports_fav_play": "de sport die je graag doet",
            },
            current_values={"sports_enjoys": "ja", "sports_fav_play": "voetbal"},
            correct_values=["je iets met voetbal hebt"],
            memory_link="voetbal iets is waar jij iets mee hebt",
            options=["voetbal", "sport"],
            reground="Ik houd goed vast dat voetbal iets is waar jij iets mee hebt.",
        )
        turn = {
            "phase": 5,
            "layer": "L2+L3",
            "name": "Topic 1",
            "topic": topic,
            "segments": app.topic1_phase_segments(topic),
        }

        app.refresh_topic_after_change(turn, {
            "continue_phase_after_change": True,
            "change_confirmed": True,
            "change": {
                "field": "sports_fav_play",
                "old_value": "voetbal",
                "new_value": "hockey",
            },
        })

        self.assertEqual(turn["topic"]["label"], "hockey")
        self.assertTrue(turn["force_topic_fallback"])
        second_segment = app.segment_context(turn, turn["segments"][1], 2)
        third_segment = app.segment_context(turn, turn["segments"][2], 3)
        self.assertIn("hockey", app.turn_text(second_segment))
        self.assertIn("hockey", app.turn_text(third_segment))

    def test_topic_followup_answer_does_not_update_corrected_sport(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = app.topic_candidate(
            domain="sport",
            label="hockey",
            fields=["sports_enjoys", "sports_fav_play"],
            field_labels={
                "sports_enjoys": "of je sport leuk vindt",
                "sports_fav_play": "de sport die je graag doet",
            },
            current_values={"sports_enjoys": "ja", "sports_fav_play": "hockey"},
            correct_values=["je iets met hockey hebt"],
            memory_link="hockey iets is waar jij iets mee hebt",
            options=["hockey", "sport"],
            reground="Ik houd goed vast dat hockey iets is waar jij iets mee hebt.",
        )
        turn = {
            "phase": 5,
            "layer": "L2+L3",
            "name": "Topic 1",
            "topic": topic,
            "segments": app.topic1_phase_segments(topic),
            "force_topic_fallback": True,
        }
        open_segment = app.segment_context(turn, turn["segments"][1], 2)

        action = app.action_handler(
            IntentResult(intent="um_add", field="sports_fav_play", value="achterin", confidence=0.92),
            "achterin",
            open_segment,
        )

        self.assertEqual(open_segment["used_fields"], {})
        self.assertEqual(action["action"], "acknowledge")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.last_um_preview["sports_fav_play"], "zwemmen")

    def test_topic_correction_without_value_continues_with_neutral_topic(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 5,
            "response_mode": "listen_only",
            "used_fields": {"sports_fav_play": "voetbal"},
            "memory_correction_requested": True,
            "memory_correction_field": "sports_fav_play",
            "topic": {
                "domain": "sport",
                "label": "voetbal",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "geen idee",
            turn,
        )

        self.assertEqual(action["action"], "topic_correction_no_value")
        self.assertTrue(action["continue_phase_after_change"])
        self.assertTrue(action["topic_neutral_after_correction"])
        self.assertEqual(action["change"], {})
        self.assertIn("algemener over sport", app.speech.spoken[-1])

    def test_mistake_correction_rejects_classifier_value_not_in_transcript(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
        }
        result = IntentResult(intent="um_update", field="hobby_fav", value="voetbal", confidence=0.9)

        action = app.action_handler(
            result,
            "Het is gezond en ik doe het graag met mijn vrienden",
            turn,
        )

        self.assertEqual(action["action"], "continue_wrong_value_followup")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.corrections_seen, 0)

    def test_deferred_mistake_correction_asks_confirmation_for_stored_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        result = IntentResult(intent="um_update", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Nee, tekenen", turn)

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change"]["skip_um_write"])
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake_correction_uses_scripted_actual_when_um_snapshot_is_incomplete(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["hobby_fav"] = CRI.UNKNOWN_VALUE
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "mistake_topic": app.topic_candidate(
                domain="hobby",
                label="tekenen",
                fields=["hobbies"],
                field_labels={"hobby_fav": "je favoriete hobby", "hobbies": "je hobby's"},
                current_values={"hobbies": "tekenen, tuinieren, lego bouwen"},
                correct_values=["je favoriete hobby tekenen is"],
                memory_link="tekenen bij jouw interesses hoort",
                options=["tekenen"],
                reground="Ik weet zeker dat tekenen bij jouw interesses hoort.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {"M1": {"id": "M1", "wrong_value_rejected": True}}
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        result = IntentResult(intent="um_add", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Mijn favoriete hobby is tekenen", turn)

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change"]["skip_um_write"])
        self.assertEqual(action["change"]["old_value"], "tekenen")
        self.assertEqual(action["change"]["new_value"], "tekenen")
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake_one_rejection_then_bare_value_asks_confirmation(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "zingen",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn

        first_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Nee dat klopt niet",
            turn,
        )
        self.assertEqual(first_action["action"], "ask_correction_detail")
        self.assertEqual(first_action["leo_response"], "Oeps, wat is dan je favoriete hobby?")
        self.assertTrue(app.mistake_states["M1"].get("wrong_value_rejected"))

        app.speech.spoken.clear()
        second_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="tekenen", confidence=0.9),
            "tekenen",
            turn,
        )

        self.assertEqual(second_action["action"], "confirm_update")
        self.assertTrue(second_action["change"]["skip_um_write"])
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake_rejection_sets_correction_context_for_classifier(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 13,
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "Rekenen en Gym",
            "mistake_wrong": "begrijpend lezen",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "school_strength",
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="school_strength", value=None, confidence=0.92),
            "dat is niet zo",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertTrue(turn["memory_correction_requested"])
        self.assertEqual(turn["memory_correction_field"], "school_strength")
        self.assertEqual(
            turn["last_correction_question"],
            "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
        )
        context = app.actions.classifier_context(turn)
        self.assertEqual(context["expected_correction_field"], "school_strength")
        self.assertEqual(
            context["correction_question"],
            "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
        )

    def test_fav_subject_soft_no_answers_offer_correction_detail(self):
        soft_answers = ("Beetje", "niet echt", "volgens mij niet")

        for answer in soft_answers:
            with self.subTest(answer=answer):
                app = make_app()
                app.last_um_preview = sample_um()
                app.write_um_change = lambda change: (_ for _ in ()).throw(AssertionError("unexpected write"))
                turn = {
                    "phase": 12,
                    "phase_id": "2.3",
                    "response_mode": "topic_interpretation",
                    "memory_correction_available": True,
                    "memory_correction_field": "fav_subject",
                    "used_fields": {"fav_subject": "natuur"},
                    "topic": {
                        "domain": "school_subject",
                        "label": "natuur",
                        "fields": ["fav_subject"],
                        "field_labels": {"fav_subject": "je lievelingsvak"},
                        "current_values": {"fav_subject": "natuur"},
                        "expected_value_count": {"fav_subject": 1},
                    },
                }

                action = app.action_handler(
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                    answer,
                    turn,
                )

                self.assertEqual(action["action"], "ask_correction_detail")
                self.assertTrue(action["follow_up_needed"])
                self.assertTrue(turn["memory_correction_requested"])
                self.assertEqual(turn["memory_correction_field"], "fav_subject")
                self.assertEqual(action["leo_response"], "Oeps, wat is dan je lievelingsvak?")

    def test_school_strength_soft_no_answers_offer_correction_detail(self):
        soft_answers = ("mwah", "niet helemaal", "ik weet niet of dat klopt")

        for answer in soft_answers:
            with self.subTest(answer=answer):
                app = make_app()
                app.last_um_preview = sample_um()
                app.write_um_change = lambda change: (_ for _ in ()).throw(AssertionError("unexpected write"))
                turn = {
                    "phase": 13,
                    "phase_id": "2.4",
                    "mistake_id": "M3",
                    "mistake_field": "school_strength",
                    "mistake_actual": "taal",
                    "mistake_wrong": "rekenen",
                    "response_mode": "mistake_interpretation",
                    "memory_correction_available": True,
                    "memory_correction_field": "school_strength",
                }

                action = app.action_handler(
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                    answer,
                    turn,
                )

                self.assertEqual(action["action"], "ask_correction_detail")
                self.assertTrue(action["follow_up_needed"])
                self.assertTrue(turn["memory_correction_requested"])
                self.assertEqual(turn["memory_correction_field"], "school_strength")
                self.assertEqual(
                    action["leo_response"],
                    "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
                )
                self.assertTrue(app.mistake_states["M3"]["wrong_value_rejected"])

    def test_soft_no_school_correction_does_not_store_beetje_as_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.write_um_change = lambda change: (_ for _ in ()).throw(AssertionError("unexpected write"))
        turn = {
            "phase": 12,
            "phase_id": "2.3",
            "response_mode": "topic_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_subject",
            "used_fields": {"fav_subject": "natuur"},
            "topic": {
                "domain": "school_subject",
                "label": "natuur",
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je lievelingsvak"},
                "current_values": {"fav_subject": "natuur"},
                "expected_value_count": {"fav_subject": 1},
            },
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_subject", value="beetje", confidence=0.94),
            "beetje",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.last_um_preview["fav_subject"], sample_um()["fav_subject"])

    def test_unflagged_mistake_followup_soft_no_does_not_trigger_memory_correction(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "phase_id": "1.6",
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "tuinieren",
            "response_mode": "mistake_interpretation",
            "used_fields": {"hobby_fav": "tuinieren"},
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "beetje",
            turn,
        )

        self.assertEqual(action["action"], "continue_wrong_value_followup")
        self.assertEqual(action["change"], {})
        self.assertNotIn("memory_correction_requested", turn)

    def test_unflagged_mistake_followup_um_value_does_not_trigger_memory_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "phase_id": "1.6",
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "tuinieren",
            "response_mode": "mistake_interpretation",
            "used_fields": {"hobby_fav": "tuinieren"},
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="hobby_fav", value="tekenen", confidence=0.95),
            "tekenen",
            turn,
        )

        self.assertEqual(action["action"], "continue_wrong_value_followup")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.last_um_preview["hobby_fav"], sample_um()["hobby_fav"])

    def test_negative_wrong_value_responses_ask_correction_detail_for_mistakes(self):
        cases = (
            (
                "M1",
                "hobby_fav",
                "tekenen",
                "surfen",
                "Ik wil niet surfen",
                IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
                "Oeps, wat is dan je favoriete hobby?",
            ),
            (
                "M2",
                "fav_food",
                "pannenkoeken",
                "pizza",
                "Ik wil geen pizza",
                IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
                "Oeps, wat is dan je lievelingseten?",
            ),
            (
                "M3",
                "school_strength",
                "taal",
                "begrijpend lezen",
                "Ik ben niet goed in begrijpend lezen",
                IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
                "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
            ),
            (
                "M4",
                "aspiration",
                "dierenarts worden",
                "architect worden",
                "Ik hoef geen architect te worden",
                IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
                "Oeps, wat wil jij dan later worden?",
            ),
            (
                "M4",
                "aspiration",
                "dierenarts worden",
                "kok worden",
                "Ik wil geen kok worden",
                IntentResult(intent="um_update", field="aspiration", value="kok", confidence=0.92),
                "Oeps, wat wil jij dan later worden?",
            ),
            (
                "M4",
                "aspiration",
                "dierenarts worden",
                "architect worden",
                "Ik koffer een architect te worden",
                IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
                "Oeps, wat wil jij dan later worden?",
            ),
        )

        for mistake_id, field, actual, wrong, transcript, result, expected_response in cases:
            with self.subTest(transcript=transcript):
                app = make_app()
                app.last_um_preview = sample_um()
                app.write_um_change = lambda change: (_ for _ in ()).throw(AssertionError("unexpected write"))
                turn = {
                    "phase": 17 if mistake_id == "M4" else 6,
                    "mistake_id": mistake_id,
                    "mistake_field": field,
                    "mistake_actual": actual,
                    "mistake_wrong": wrong,
                    "response_mode": "mistake_interpretation",
                    "memory_correction_available": True,
                    "memory_correction_field": field,
                    "used_fields": {field: wrong},
                }

                action = app.action_handler(result, transcript, turn)

                self.assertEqual(action["action"], "ask_correction_detail")
                self.assertEqual(action["leo_response"], expected_response)
                self.assertEqual(action["change"], {})
                self.assertTrue(turn["memory_correction_requested"])
                self.assertEqual(turn["memory_correction_field"], field)

    def test_negative_wrong_value_guard_keeps_positive_unflagged_and_concrete_replacement_paths(self):
        app = make_app()
        app.last_um_preview = sample_um()
        flagged_turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "mistake_actual": "dierenarts worden",
            "mistake_wrong": "architect worden",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
            "used_fields": {"aspiration": "architect worden"},
        }
        positive = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
            "Ja, architect lijkt me leuk",
            dict(flagged_turn),
        )
        self.assertEqual(positive["action"], "continue_wrong_value_followup")

        unflagged_turn = dict(flagged_turn)
        unflagged_turn.pop("memory_correction_available")
        unflagged_turn.pop("memory_correction_field")
        unflagged = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
            "Ik hoef geen architect te worden",
            unflagged_turn,
        )
        self.assertEqual(unflagged["action"], "continue_wrong_value_followup")
        self.assertNotIn("memory_correction_requested", unflagged_turn)

        replacement_turn = dict(flagged_turn)
        replacement_turn["mistake_wrong"] = "kok worden"
        replacement_turn["used_fields"] = {"aspiration": "kok worden"}
        replacement_result = IntentResult(intent="um_update", field="aspiration", value="bakker", confidence=0.94)
        self.assertFalse(
            app.actions.is_negative_wrong_value_response(
                replacement_result,
                "Ik wil geen kok worden, ik wil bakker worden",
                replacement_turn,
            )
        )
        replacement_change = app.actions.change_from_intent_result(
            replacement_result,
            replacement_turn,
            "Ik wil geen kok worden, ik wil bakker worden",
        )
        self.assertEqual(replacement_change["action"], "update")
        self.assertEqual(replacement_change["new_value"], "bakker")

    def test_soft_no_answers_work_for_other_memory_correction_opportunities(self):
        app = make_app()
        app.last_um_preview = sample_um()
        food_turn = {
            "phase": 8,
            "phase_id": "1.8",
            "mistake_id": "M2",
            "mistake_field": "fav_food",
            "mistake_actual": "broccoli",
            "mistake_wrong": "pizza",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_food",
        }

        food_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "niet helemaal",
            food_turn,
        )

        self.assertEqual(food_action["action"], "ask_correction_detail")
        self.assertEqual(food_action["leo_response"], "Oeps, wat is dan je lievelingseten?")
        self.assertTrue(food_turn["memory_correction_requested"])
        self.assertEqual(food_turn["memory_correction_field"], "fav_food")

        app = make_app()
        app.last_um_preview = sample_um()
        sport_turn = {
            "phase": 5,
            "response_mode": "listen_only",
            "used_fields": {"sports_fav_play": "voetbal"},
            "topic": {
                "domain": "sport",
                "label": "voetbal",
                "fields": ["sports_enjoys", "sports_fav_play"],
                "field_labels": {
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav_play": "de sport die je graag doet",
                },
                "current_values": {
                    "sports_enjoys": "ja",
                    "sports_fav_play": "voetbal",
                },
            },
        }

        sport_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "mwah",
            sport_turn,
        )

        self.assertEqual(sport_action["action"], "listen_only")
        self.assertEqual(sport_action["change"], {})
        self.assertNotIn("memory_correction_requested", sport_turn)

        app = make_app()
        available_turn = {
            "phase": 99,
            "response_mode": "acknowledge",
            "memory_correction_available": True,
            "memory_correction_field": "fav_food",
            "used_fields": {"fav_food": "broccoli"},
        }

        available_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "dat weet ik niet zeker",
            available_turn,
        )

        self.assertEqual(available_action["action"], "ask_correction_detail")
        self.assertEqual(available_action["leo_response"], "Oeps, wat is dan je lievelingseten?")
        self.assertTrue(available_turn["memory_correction_requested"])

    def test_soft_no_answers_work_for_review_rolemodel_inspection_and_nudge(self):
        app = make_app()
        review_turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "response_mode": "memory_review_group",
            "content_plan": app.l1("Over school weet ik iets. Klopt dat?"),
            "topic": {
                "fields": ["fav_subject"],
                "field_labels": {"fav_subject": "je lievelingsvak"},
                "current_values": {"fav_subject": "natuur"},
            },
        }
        review_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "een beetje",
            review_turn,
        )
        self.assertEqual(review_action["action"], "memory_review_group_ask_correction_detail")

        app = make_app()
        role_turn = {
            "phase": 16,
            "phase_id": "3.3",
            "response_mode": "role_model_absence_check",
            "memory_correction_available": True,
            "memory_correction_field": "role_model",
            "used_fields": {"role_model": "niemand"},
        }
        role_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "volgens mij niet",
            role_turn,
        )
        self.assertEqual(role_action["action"], "role_model_absence_ask_detail")
        self.assertTrue(role_turn["memory_correction_requested"])

        app = make_app()
        inspection_turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "response_mode": "explicit_memory_inspection_offer",
            "explicit_memory_inspection_active": True,
            "memory_review_lines": ["Ik weet nog dat natuur je lievelingsvak is."],
        }
        inspection_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "ik weet niet of dat klopt",
            inspection_turn,
        )
        self.assertEqual(inspection_action["action"], "explicit_memory_inspection_ask_correction_detail")

        app = make_app()
        nudge_turn = {
            "phase": 9,
            "response_mode": "nudge_interpretation",
        }
        nudge_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "niet echt",
            nudge_turn,
        )
        self.assertEqual(nudge_action["action"], "nudge_ask_correction_detail")
        self.assertTrue(nudge_turn["nudge_correction_requested"])

        app = make_app()
        nudge_detail_turn = {
            "phase": 9,
            "response_mode": "nudge_interpretation",
            "nudge_correction_requested": True,
        }
        nudge_detail = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "beetje",
            nudge_detail_turn,
        )
        self.assertEqual(nudge_detail["action"], "nudge_correction_detail_missing")
        self.assertNotIn("corrected_value", nudge_detail)

    def test_uncertain_or_unrelated_beetje_does_not_trigger_memory_correction(self):
        for answer in ("weet ik niet", "geen idee", "misschien"):
            with self.subTest(answer=answer):
                app = make_app()
                app.last_um_preview = sample_um()
                turn = {
                    "phase": 12,
                    "phase_id": "2.3",
                    "response_mode": "topic_interpretation",
                    "memory_correction_available": True,
                    "memory_correction_field": "fav_subject",
                    "used_fields": {"fav_subject": "natuur"},
                    "topic": {
                        "domain": "school_subject",
                        "label": "natuur",
                        "fields": ["fav_subject"],
                        "field_labels": {"fav_subject": "je lievelingsvak"},
                        "current_values": {"fav_subject": "natuur"},
                        "expected_value_count": {"fav_subject": 1},
                    },
                }

                action = app.action_handler(
                    IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
                    answer,
                    turn,
                )

                self.assertNotEqual(action["action"], "ask_correction_detail")
                self.assertNotIn("memory_correction_requested", turn)

        app = make_app()
        unrelated_turn = {
            "phase": 1,
            "response_mode": "acknowledge",
            "used_fields": {"name": "Sam"},
        }
        unrelated = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "een beetje",
            unrelated_turn,
        )

        self.assertNotEqual(unrelated["action"], "ask_correction_detail")
        self.assertNotIn("memory_correction_requested", unrelated_turn)

        for turn in (
            {"phase": 10, "response_mode": "school_joke_transition"},
            {"phase": 11, "response_mode": "robot_school_guess"},
            {"phase": 17, "response_mode": "middle_school_feeling"},
        ):
            with self.subTest(mode=turn["response_mode"]):
                app = make_app()
                action = app.action_handler(
                    IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.9),
                    "een beetje",
                    turn,
                )
                self.assertNotEqual(action["action"], "ask_correction_detail")
                self.assertNotIn("memory_correction_requested", turn)

        app = make_app()
        memory_change_turn = {
            "phase": 18,
            "phase_id": "3.6/7",
            "response_mode": "memory_access_change",
            "allow_memory_change": True,
            "memory_correction_requested": True,
        }
        memory_change = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "beetje",
            memory_change_turn,
        )
        self.assertNotEqual(memory_change["action"], "ask_correction_detail")

    def test_mistake_correction_followup_uses_raw_answer_when_classifier_says_unclear(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 13,
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "Rekenen en Gym",
            "mistake_wrong": "begrijpend lezen",
            "response_mode": "mistake_interpretation",
            "memory_correction_requested": True,
            "memory_correction_field": "school_strength",
            "last_correction_question": "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
        }
        app.current_turn_context = turn
        app.mistake_states = {"M3": {"id": "M3", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85),
            "Gym",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "school_strength")
        self.assertEqual(action["change"]["new_value"], "Gym")
        self.assertEqual(app.last_um_preview["school_strength"], "Gym")
        self.assertIn(13, app.phases_with_confirmed_change)
        self.assertIn("verander naar Gym", app.speech.spoken[0])

    def test_part2_mistake3_multiple_strengths_asks_for_one_strength(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["Gym", "ja"]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "Taal",
            "mistake_wrong": "begrijpend lezen",
            "response_mode": "mistake_interpretation",
            "memory_correction_requested": True,
            "memory_correction_field": "school_strength",
            "last_correction_question": "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
            "mistake_topic": app.topic_candidate(
                domain="school",
                label="waar je goed in bent op school",
                fields=["school_strength"],
                field_labels={"school_strength": "waar je goed in bent op school"},
                current_values={"school_strength": "Taal"},
                correct_values=["je vooral goed bent in Taal"],
                memory_link="je vooral goed bent in Taal",
                options=["Taal"],
                reground="Ik weet zeker dat Taal iets is waar je goed in bent op school.",
            ),
        }
        turn["mistake_topic"]["expected_value_count"] = {"school_strength": 1}
        app.current_turn_context = turn
        app.mistake_states = {"M3": {"id": "M3", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="um_update", field="school_strength", value="Rekenen en Gym", confidence=0.94),
            "Nee dat is Rekenen en Gym",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "school_strength")
        self.assertEqual(action["change"]["new_value"], "Gym")
        self.assertEqual(app.last_um_preview["school_strength"], "Gym")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier een ding onthouden. Waar ben jij vooral goed in op school? Noem een ding.",
        )
        self.assertIn("waar je goed in bent op school verander naar Gym", app.speech.spoken[1])

    def test_part2_mistake3_multiple_strengths_without_clarification_does_not_update(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["Rekenen en Gym", ""]
        app.write_um_change = lambda change: True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "Taal",
            "mistake_wrong": "begrijpend lezen",
            "response_mode": "mistake_interpretation",
            "memory_correction_requested": True,
            "memory_correction_field": "school_strength",
            "last_correction_question": "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
            "mistake_topic": app.topic_candidate(
                domain="school",
                label="waar je goed in bent op school",
                fields=["school_strength"],
                field_labels={"school_strength": "waar je goed in bent op school"},
                current_values={"school_strength": "Taal"},
                correct_values=["je vooral goed bent in Taal"],
                memory_link="je vooral goed bent in Taal",
                options=["Taal"],
                reground="Ik weet zeker dat Taal iets is waar je goed in bent op school.",
            ),
        }
        turn["mistake_topic"]["expected_value_count"] = {"school_strength": 1}
        app.current_turn_context = turn
        app.mistake_states = {"M3": {"id": "M3", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="um_update", field="school_strength", value="Rekenen en Gym", confidence=0.94),
            "Nee dat is Rekenen en Gym",
            turn,
        )

        self.assertEqual(action["action"], "change_value_limit_unresolved")
        self.assertNotEqual(app.last_um_preview["school_strength"], "Rekenen en Gym")
        self.assertNotEqual(app.last_um_preview["school_strength"], "Gym")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier een ding onthouden. Waar ben jij vooral goed in op school? Noem een ding.",
        )
        self.assertEqual(
            app.speech.spoken[1],
            "Dat zijn er nog te veel. Ik kan hier een ding onthouden. Waar ben jij vooral goed in op school? Noem een ding.",
        )
        self.assertIn("Dan verander ik het nu nog niet", app.speech.spoken[2])

    def test_mistake_phase_logs_not_corrected_outcome_with_leo_memory_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.conversation_log = {"session_id": "test-session", "events": []}
        app.current_turn_log = {"events": [], "utterances": []}
        app.conv_log.write_conversation_logs = lambda: None
        payloads = []
        app.um.log_cri_interaction_event = lambda payload: payloads.append(dict(payload)) or True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "layer": "L1 + L2-slot WRONG",
            "name": "Mistake 3 - school_strength",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "taal",
            "mistake_wrong": "begrijpend lezen",
            "mistake_type": "completely-wrong",
            "spt_layer": "exploratory affective",
        }

        app.register_mistake_phase(turn)
        app.mistake_states["M3"]["mistake_utterance_at"] = 100.0
        app.log_timestamp = lambda: 131.25
        app.record_mistake_outcome(turn)

        self.assertEqual(payloads[0]["event_type"], "mistake_not_corrected")
        self.assertEqual(payloads[0]["outcome"], "not_corrected")
        self.assertFalse(payloads[0]["corrected"])
        self.assertEqual(payloads[0]["real_value"], "taal")
        self.assertEqual(payloads[0]["mistake_value"], "begrijpend lezen")
        self.assertEqual(payloads[0]["spt_layer"], "exploratory affective")
        self.assertEqual(payloads[0]["mistake_type"], "completely-wrong")
        self.assertEqual(payloads[0]["latency_seconds"], 31.25)
        self.assertEqual(
            payloads[0]["leo_memory_key"],
            "M3_mistake_not_corrected_school_strength",
        )
        self.assertEqual(payloads[0]["leo_memory_value"], "begrijpend lezen")
        self.assertEqual(app.mistake_states["M3"]["outcome"], "not_corrected")
        self.assertEqual(
            app.mistake_states["M3"]["leo_memory_key"],
            "M3_mistake_not_corrected_school_strength",
        )
        self.assertEqual(app.mistake_states["M3"]["leo_memory_value"], "begrijpend lezen")
        self.assertEqual(app.mistake_states["M3"]["spt_layer"], "exploratory affective")
        self.assertEqual(app.mistake_states["M3"]["latency_seconds"], 31.25)
        self.assertTrue(app.mistake_states["M3"]["outcome_logged"])
        self.assertEqual(app.conversation_log["events"][0]["type"], "mistake_outcome")
        self.assertEqual(app.conversation_log["events"][0]["spt_layer"], "exploratory affective")
        self.assertEqual(app.conversation_log["events"][0]["latency_seconds"], 31.25)

    def test_mistake_phase_logs_corrected_outcome_with_confirmed_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["school_strength"] = "Gym"
        app.conversation_log = {"session_id": "test-session", "events": []}
        app.current_turn_log = {"events": [], "utterances": []}
        app.conv_log.write_conversation_logs = lambda: None
        payloads = []
        app.um.log_cri_interaction_event = lambda payload: payloads.append(dict(payload)) or True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "layer": "L1 + L2-slot WRONG",
            "name": "Mistake 3 - school_strength",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "taal",
            "mistake_wrong": "begrijpend lezen",
            "mistake_type": "completely-wrong",
        }

        app.register_mistake_phase(turn)
        app.mistake_states["M3"]["mistake_utterance_at"] = 200.0
        app.log_timestamp = lambda: 209.75
        app.current_turn_context = turn
        app.mark_current_mistake_corrected()
        app.log_timestamp = lambda: 240.0
        app.record_mistake_outcome(turn)

        self.assertEqual(payloads[0]["event_type"], "mistake_corrected")
        self.assertEqual(payloads[0]["outcome"], "corrected")
        self.assertTrue(payloads[0]["corrected"])
        self.assertEqual(payloads[0]["real_value"], "taal")
        self.assertEqual(payloads[0]["mistake_value"], "begrijpend lezen")
        self.assertEqual(payloads[0]["leo_memory_key"], "")
        self.assertEqual(payloads[0]["leo_memory_value"], "Gym")
        self.assertEqual(payloads[0]["latency_seconds"], 9.75)
        self.assertEqual(app.mistake_states["M3"]["outcome"], "corrected")
        self.assertEqual(app.mistake_states["M3"]["leo_memory_value"], "Gym")
        self.assertEqual(app.mistake_states["M3"]["latency_seconds"], 9.75)

    def test_mistake_phase_logs_rejected_unresolved_without_leo_memory_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.conversation_log = {"session_id": "test-session", "events": []}
        app.current_turn_log = {"events": [], "utterances": []}
        app.conv_log.write_conversation_logs = lambda: None
        payloads = []
        app.um.log_cri_interaction_event = lambda payload: payloads.append(dict(payload)) or True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "layer": "L1 + L2-slot WRONG",
            "name": "Mistake 3 - school_strength",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "taal",
            "mistake_wrong": "begrijpend lezen",
            "mistake_type": "completely-wrong",
        }

        app.register_mistake_phase(turn)
        app.mistake_states["M3"]["wrong_value_rejected"] = True
        app.record_mistake_outcome(turn)

        self.assertEqual(payloads[0]["event_type"], "mistake_rejected_unresolved")
        self.assertEqual(payloads[0]["outcome"], "rejected_unresolved")
        self.assertEqual(payloads[0]["leo_memory_key"], "")
        self.assertIsNone(payloads[0]["leo_memory_value"])
        self.assertEqual(payloads[0]["real_value"], "taal")
        self.assertEqual(payloads[0]["mistake_value"], "begrijpend lezen")

    def test_mistake3_school_difficulty_ack_resolves_mistake_without_memory_key(self):
        app = make_app(["Dat snap ik."])
        app.last_um_preview = sample_um()
        app.conversation_log = {"session_id": "test-session", "events": []}
        app.current_turn_log = {"events": [], "utterances": []}
        app.conv_log.write_conversation_logs = lambda: None
        payloads = []
        app.um.log_cri_interaction_event = lambda payload: payloads.append(dict(payload)) or True
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "name": "Mistake 3 - school_strength",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "taal",
            "mistake_wrong": "rekenen",
            "mistake_type": "completely-wrong",
            "m3_requires_school_difficulty_resolution": True,
            "m3_school_difficulty_resolution": True,
            "response_mode": "acknowledge",
            "llm_turn": True,
            "used_fields": {
                "fav_subject": "taal",
                "school_difficulty": "rekenen",
            },
        }

        app.register_mistake_phase(turn)
        app.current_turn_context = turn
        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja, dat klopt",
            turn,
        )
        app.record_mistake_outcome(turn)

        self.assertEqual(action["action"], "acknowledge")
        self.assertTrue(app.mistake_states["M3"]["corrected"])
        self.assertTrue(app.mistake_states["M3"]["corrected_by_school_difficulty_ack"])
        self.assertEqual(payloads[0]["event_type"], "mistake_corrected")
        self.assertEqual(payloads[0]["outcome"], "corrected")
        self.assertEqual(payloads[0]["leo_memory_key"], "")
        self.assertEqual(payloads[0]["leo_memory_value"], "taal")

    def test_mistake3_school_difficulty_change_marks_original_mistake_not_corrected(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.conversation_log = {"session_id": "test-session", "events": []}
        app.current_turn_log = {"events": [], "utterances": []}
        app.conv_log.write_conversation_logs = lambda: None
        payloads = []
        app.um.log_cri_interaction_event = lambda payload: payloads.append(dict(payload)) or True

        def write_um_change(change):
            app.last_um_preview[change["field"]] = change["new_value"]
            return True

        app.write_um_change = write_um_change
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "name": "Mistake 3 - school_strength",
            "mistake_id": "M3",
            "mistake_field": "school_strength",
            "mistake_actual": "taal",
            "mistake_wrong": "rekenen",
            "mistake_type": "completely-wrong",
            "m3_requires_school_difficulty_resolution": True,
            "m3_school_difficulty_resolution": True,
            "memory_correction_available": True,
            "response_mode": "acknowledge",
            "used_fields": {
                "fav_subject": "taal",
                "school_difficulty": "rekenen",
            },
        }

        app.register_mistake_phase(turn)
        app.current_turn_context = turn
        action = app.action_handler(
            IntentResult(intent="um_update", field="school_difficulty", value="spelling", confidence=0.94),
            "Nee, spelling is lastig",
            turn,
        )
        app.record_mistake_outcome(turn)

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(app.last_um_preview["school_difficulty"], "spelling")
        self.assertFalse(app.mistake_states["M3"].get("corrected", False))
        self.assertTrue(app.mistake_states["M3"]["m3_not_corrected_by_difficulty_change"])
        self.assertEqual(payloads[0]["event_type"], "mistake_not_corrected")
        self.assertEqual(payloads[0]["outcome"], "not_corrected")
        self.assertEqual(
            payloads[0]["leo_memory_key"],
            "M3_mistake_not_corrected_school_strength",
        )
        self.assertEqual(payloads[0]["leo_memory_value"], "rekenen")

    def test_listen_only_segment_with_mentioned_um_field_does_not_trigger_memory_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "response_mode": "listen_only",
            "used_fields": {"school_strength": "rekenen"},
        }

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.92),
            "Nee niet echt",
            turn,
        )

        self.assertEqual(action["action"], "listen_only")
        self.assertEqual(action["change"], {})
        self.assertEqual(app.speech.spoken, [])
        self.assertEqual(app.last_um_preview["school_strength"], "taal")

    def test_mistake_correction_change_replaces_field_values(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "zingen",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
        }

        change = app.change_from_intent_result(
            IntentResult(intent="um_update", field="hobby_fav", value="tennis", confidence=0.96),
            turn,
            "Nee, mijn favoriete hobby is tennis",
        )

        self.assertEqual(change["action"], "update")
        self.assertEqual(change["new_value"], "tennis")
        self.assertTrue(change["replace_field"])

    def test_inline_mistake_correction_asks_confirmation_before_continuing(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "padel",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="um_update", field="hobby_fav", value="gamen", confidence=0.94),
            "Nee dat is gamen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar gamen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertEqual(app.last_um_preview["hobby_fav"], "gamen")
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake1_inline_multiple_favorite_hobbies_asks_for_one_hobby(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["gamen", "ja"]
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "padel",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "defer_corrected_response": True,
        }
        turn["mistake_topic"]["expected_value_count"] = {"hobby_fav": 1}
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="rejects_joke_no", confidence=0.92),
            "nee dat is gamen en tekenen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "hobby_fav")
        self.assertEqual(action["change"]["new_value"], "gamen")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier één favoriete hobby onthouden. Wat is jouw allerliefste hobby? Noem één ding.",
        )
        self.assertEqual(app.speech.spoken[1], "Wil je dat ik je favoriete hobby verander naar gamen?")
        self.assertEqual(app.last_um_preview["hobby_fav"], "gamen")

    def test_mistake1_correction_detail_multiple_favorite_hobbies_asks_for_one_hobby(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["tekenen", "ja"]
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "padel",
            "response_mode": "mistake_interpretation",
            "memory_correction_requested": True,
            "memory_correction_field": "hobby_fav",
            "last_correction_question": "Oeps, wat is dan je favoriete hobby?",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "defer_corrected_response": True,
        }
        turn["mistake_topic"]["expected_value_count"] = {"hobby_fav": 1}
        app.current_turn_context = turn
        app.mistake_states = {"M1": {"id": "M1", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="um_update", field="hobby_fav", value="gamen en tekenen", confidence=0.94),
            "gamen en tekenen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(action["change"]["field"], "hobby_fav")
        self.assertEqual(action["change"]["new_value"], "tekenen")
        self.assertEqual(
            app.speech.spoken[0],
            "Ik kan hier één favoriete hobby onthouden. Wat is jouw allerliefste hobby? Noem één ding.",
        )
        self.assertEqual(app.speech.spoken[1], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertEqual(app.last_um_preview["hobby_fav"], "tekenen")

    def test_mistake_rejection_phrase_is_not_repaired_into_fake_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "zingen",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.93),
            "Nee dat is niet zo",
            turn,
        )

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertEqual(action["leo_response"], "Oeps, wat is dan je favoriete hobby?")
        self.assertEqual(action["change"], {})
        self.assertTrue(app.mistake_states["M1"].get("wrong_value_rejected"))

    def test_explicit_mistake_correction_answer_asks_confirmation_when_actual_is_missing(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["hobby_fav"] = CRI.UNKNOWN_VALUE
        app.speech.heard = ["ja"]
        app.simulated_persona = {}
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_wrong": "zingen",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "hobby_fav",
            "mistake_topic": app.topic_candidate(
                domain="hobby",
                label="je hobby's",
                fields=["hobbies"],
                field_labels={"hobby_fav": "je favoriete hobby", "hobbies": "je hobby's"},
                current_values={"hobbies": "tekenen, tuinieren, lego bouwen"},
                correct_values=["je favoriete hobby tekenen is"],
                memory_link="tekenen bij jouw interesses hoort",
                options=["tekenen"],
                reground="Ik wil goed onthouden wat jouw favoriete hobby is.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {"M1": {"id": "M1", "wrong_value_rejected": True}}

        action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value="tekenen", confidence=0.9),
            "tekenen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(app.simulated_persona["hobby_fav"], "tekenen")
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je favoriete hobby verander naar tekenen?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))
        self.assertTrue(action["change"]["replace_field"])

    def test_replace_field_write_validates_deletes_then_posts_update(self):
        class FakeResponse:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self._payload = payload or {}

            def json(self):
                return self._payload

        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.UM_API_BASE = "http://um.local"
        app.CHILD_ID = "701"
        app.last_um_preview = {"hobby_fav": "voetbal"}
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(("post", url, json))
            return FakeResponse(200, {"data": {"skipped": []}})

        def fake_delete(url, timeout=None):
            calls.append(("delete", url, None))
            return FakeResponse(200)

        change = {
            "action": "update",
            "field": "hobby_fav",
            "old_value": "voetbal",
            "new_value": "tennis, Playmobiel bouwen",
            "replace_field": True,
        }

        with patch("cri_um.client.requests.post", side_effect=fake_post), patch(
            "cri_um.client.requests.delete",
            side_effect=fake_delete,
        ):
            ok = app.write_um_change(change)

        self.assertTrue(ok)
        self.assertEqual(calls[0][0], "post")
        self.assertTrue(calls[0][2]["dry_run"])
        self.assertEqual(calls[1], ("delete", "http://um.local/api/um/701/field/hobby_fav", None))
        self.assertEqual(calls[2][0], "post")
        self.assertNotIn("dry_run", calls[2][2])
        self.assertEqual(calls[2][2]["fields"], {"hobby_fav": "tennis, Playmobiel bouwen"})
        self.assertEqual(app.last_um_preview["hobby_fav"], "tennis, Playmobiel bouwen")

    def test_pet_pair_write_uses_consolidated_pets_field(self):
        class FakeResponse:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self._payload = payload or {}

            def json(self):
                return self._payload

        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.UM_API_BASE = "http://um.local"
        app.CHILD_ID = "701"
        app.last_um_preview = {"pet_type": "Vis", "pet_name": "Vis", "pets": "Vis (Vis)"}
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(("post", url, json))
            return FakeResponse(200, {"data": {"skipped": []}})

        def fake_delete(url, timeout=None):
            calls.append(("delete", url, None))
            return FakeResponse(200)

        change = {
            "action": "multi_update",
            "changes": [
                {
                    "action": "update",
                    "field": "pet_type",
                    "old_value": "Vis",
                    "new_value": "poes",
                },
                {
                    "action": "update",
                    "field": "pet_name",
                    "old_value": "Vis",
                    "new_value": "Lulu",
                },
            ],
            "replace_field": True,
        }

        with patch("cri_um.client.requests.post", side_effect=fake_post), patch(
            "cri_um.client.requests.delete",
            side_effect=fake_delete,
        ):
            ok = app.write_pet_pair_change(change)

        self.assertTrue(ok)
        self.assertEqual(calls[0][0], "post")
        self.assertTrue(calls[0][2]["dry_run"])
        self.assertEqual(calls[0][2]["fields"], {"pets": "poes"})
        self.assertEqual(calls[0][2]["extra_props"], {"pets_petName": "Lulu"})
        self.assertEqual(calls[1], ("delete", "http://um.local/api/um/701/field/pets", None))
        self.assertEqual(calls[2][0], "post")
        self.assertEqual(calls[2][2]["fields"], {"pets": "poes"})
        self.assertEqual(calls[2][2]["extra_props"], {"pets_petName": "Lulu"})
        self.assertNotIn("dry_run", calls[2][2])
        self.assertEqual(app.last_um_preview["pet_type"], "poes")
        self.assertEqual(app.last_um_preview["pet_name"], "Lulu")
        self.assertEqual(app.last_um_preview["pets"], "Lulu (poes)")

    def test_cri_interaction_event_local_log_avoids_event_type_collision(self):
        class FakeResponse:
            status_code = 200

        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.UM_API_BASE = "http://um.local"
        app.CHILD_ID = "701"
        events = []
        app.log_conversation_event = lambda event_type, **data: events.append((event_type, data))
        payload = {
            "event_type": "mistake_corrected",
            "mistake_id": "M3",
            "field": "school_strength",
        }

        with patch("cri_um.client.requests.post", return_value=FakeResponse()):
            ok = app.um.log_cri_interaction_event(payload)

        self.assertTrue(ok)
        self.assertEqual(events[0][0], "cri_interaction_event_write")
        self.assertEqual(events[0][1]["logged_event_type"], "mistake_corrected")
        self.assertNotIn("event_type", events[0][1])

    def test_cri_interaction_event_failure_log_avoids_event_type_collision(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.UM_API_BASE = "http://um.local"
        app.CHILD_ID = "701"
        events = []
        app.log_conversation_event = lambda event_type, **data: events.append((event_type, data))
        payload = {
            "event_type": "mistake_corrected",
            "mistake_id": "M3",
            "field": "school_strength",
        }

        with patch("cri_um.client.requests.post", side_effect=RuntimeError("boom")):
            ok = app.um.log_cri_interaction_event(payload)

        self.assertFalse(ok)
        self.assertEqual(events[0][0], "cri_interaction_event_write")
        self.assertEqual(events[0][1]["logged_event_type"], "mistake_corrected")
        self.assertNotIn("event_type", events[0][1])
        self.assertIn("boom", events[0][1]["error"])

    def test_confirmed_change_updates_local_preview_even_when_save_fails(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: False
        app.current_turn_context = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_wrong": "padel",
        }
        change = {
            "action": "update",
            "field": "hobby_fav",
            "field_label": "je favoriete hobby",
            "old_value": "voetbal",
            "new_value": "playmobiel bouwen",
            "replace_field": True,
        }

        accepted = app.confirm_topic_change(change)

        self.assertTrue(accepted)
        self.assertEqual(app.last_um_preview["hobby_fav"], "playmobiel bouwen")
        self.assertIn("opslaan lukte nu niet", app.speech.spoken[-1])

    def test_tablet_condition_reveals_confirmed_change_after_acknowledgement(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["condition"] = "C2"
        app.local_condition = "C2"
        app.current_turn_context = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_wrong": "padel",
        }
        app.write_um_change = lambda change: True
        events = []

        class EventSpeech(FakeSpeech):
            def say(self, text):
                super().say(text)
                events.append(("say", text))

        app.speech = EventSpeech(["ja"])
        app.tablet_state = SimpleNamespace(
            refresh=lambda phase=None: events.append(("refresh", phase)),
            prepare_reveal_change=lambda **kwargs: events.append(("prepare", kwargs)),
            reveal_change=lambda **kwargs: events.append(("reveal", kwargs)),
        )
        change = {
            "action": "update",
            "field": "hobby_fav",
            "field_label": "je favoriete hobby",
            "old_value": "voetbal",
            "new_value": "playmobiel spelen",
            "replace_field": True,
        }

        accepted = app.confirm_topic_change(change)

        self.assertTrue(accepted)
        self.assertEqual(events[0], ("say", "Wil je dat ik je favoriete hobby verander naar playmobiel spelen?"))
        self.assertEqual(
            events[1],
            (
                "prepare",
                {
                    "field": "hobby_fav",
                    "old_value": "padel",
                    "new_value": "playmobiel spelen",
                    "phase": 6,
                },
            ),
        )
        self.assertEqual(
            events[2],
            ("say", "Dankjewel, ik heb dat aangepast. Kijk maar op de tablet, daar zie je het veranderen."),
        )
        self.assertEqual(
            events[3],
            (
                "reveal",
                {
                    "field": "hobby_fav",
                    "old_value": "padel",
                    "new_value": "playmobiel spelen",
                    "phase": 6,
                },
            ),
        )

    def test_tablet_condition_confirms_even_when_correction_already_in_um(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["condition"] = "C2"
        app.last_um_preview["hobby_fav"] = "gamen"
        app.local_condition = "C2"
        app.current_turn_context = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "gamen",
            "mistake_wrong": "padel",
        }
        app.write_um_change = lambda change: True
        events = []

        class EventSpeech(FakeSpeech):
            def say(self, text):
                super().say(text)
                events.append(("say", text))

        app.speech = EventSpeech(["ja"])
        app.tablet_state = SimpleNamespace(
            refresh=lambda phase=None: events.append(("refresh", phase)),
            reveal_change=lambda **kwargs: events.append(("reveal", kwargs)),
        )
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_actual": "gamen",
            "mistake_wrong": "padel",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "defer_corrected_response": True,
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="hobby_fav", value="gamen", confidence=0.94),
            "Nee dat is gamen",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertEqual(events[0], ("say", "Wil je dat ik je favoriete hobby verander naar gamen?"))
        self.assertEqual(
            events[2],
            (
                "reveal",
                {
                    "field": "hobby_fav",
                    "old_value": "padel",
                    "new_value": "gamen",
                    "phase": 6,
                },
            ),
        )

    def test_mistake_rejection_without_value_asks_for_detail_then_accepts_short_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        turn = {
            "phase": 8,
            "mistake_id": "M2",
            "mistake_field": "fav_food",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_food",
            "mistake_topic": app.topic_candidate(
                domain="eten",
                label="je lievelingseten",
                fields=["fav_food"],
                field_labels={"fav_food": "je lievelingseten"},
                current_values={"fav_food": "pannenkoeken"},
                correct_values=["je lievelingseten pannenkoeken is"],
                memory_link="je lievelingseten pannenkoeken is",
                options=["pannenkoeken"],
                reground="Ik weet zeker dat pannenkoeken met jouw lievelingseten te maken heeft.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {"M2": {"id": "M2"}}

        rejection = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)
        first_action = app.action_handler(rejection, "Nee", turn)

        self.assertEqual(first_action["action"], "ask_correction_detail")
        self.assertTrue(first_action["follow_up_needed"])
        self.assertEqual(first_action["leo_response"], "Oeps, wat is dan je lievelingseten?")
        self.assertTrue(app.mistake_states["M2"].get("wrong_value_rejected"))

        app.speech.spoken.clear()
        short_answer = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)
        second_action = app.action_handler(short_answer, "pannenkoeken", turn)

        self.assertEqual(second_action["action"], "confirm_update")
        self.assertTrue(second_action["change"]["skip_um_write"])
        self.assertEqual(second_action["change"]["new_value"], "pannenkoeken")
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M2"].get("corrected"))
        self.assertIn(8, app.phases_with_confirmed_change)
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je lievelingseten verander naar pannenkoeken?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])

    def test_food_mistake_correction_to_stored_value_confirms_and_reveals_visible_change(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["condition"] = "E"
        app.last_um_preview["fav_food"] = "pizza"
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        prepared = []
        revealed = []
        app.tablet_state = SimpleNamespace(
            reset=lambda: None,
            update=lambda turn: None,
            refresh=lambda phase=None: None,
            prepare_reveal_change=lambda **kwargs: prepared.append(kwargs),
            reveal_change=lambda **kwargs: revealed.append(kwargs),
        )
        turn = {
            "phase": 8,
            "mistake_id": "M2",
            "mistake_field": "fav_food",
            "mistake_actual": "pizza",
            "mistake_wrong": "broccoli",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "fav_food",
            "mistake_topic": app.topic_candidate(
                domain="eten",
                label="je lievelingseten",
                fields=["fav_food"],
                field_labels={"fav_food": "je lievelingseten"},
                current_values={"fav_food": "pizza"},
                correct_values=["je lievelingseten pizza is"],
                memory_link="je lievelingseten pizza is",
                options=["pizza"],
                reground="Ik weet zeker dat pizza met jouw lievelingseten te maken heeft.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {
            "M2": {
                "id": "M2",
                "mentioned": True,
                "field": "fav_food",
                "actual": "pizza",
                "wrong": "broccoli",
                "corrected": False,
            }
        }

        action = app.action_handler(
            IntentResult(intent="um_update", field="fav_food", value="pizza", confidence=0.95),
            "Pizza",
            turn,
        )

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change"]["skip_um_write"])
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik je lievelingseten verander naar pizza?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertTrue(app.mistake_states["M2"].get("corrected"))
        self.assertEqual(prepared[0]["old_value"], "broccoli")
        self.assertEqual(prepared[0]["new_value"], "pizza")
        self.assertEqual(revealed[0]["old_value"], "broccoli")
        self.assertEqual(revealed[0]["new_value"], "pizza")

    def test_aspiration_inline_correction_matches_stored_worden_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.speech.heard = ["ja"]
        app.write_um_change = lambda change: self.fail("already-stored visible mistake should not rewrite UM")
        turn = {
            "phase": 17,
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
            "mistake_topic": app.topic_candidate(
                domain="droom",
                label="dierenarts worden",
                fields=["aspiration"],
                field_labels={"aspiration": "wat je later wilt worden"},
                current_values={"aspiration": "dierenarts worden"},
                correct_values=["je later dierenarts worden wilt"],
                memory_link="je later dierenarts worden wilt",
                options=["dierenarts worden"],
                reground="Ik onthoud dat dierenarts worden iets is waar je later iets mee wilt.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {"M4": {"id": "M4"}}
        result = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)

        action = app.action_handler(result, "Nee, dierenarts", turn)

        self.assertEqual(action["action"], "confirm_update")
        self.assertTrue(action["change"]["skip_um_write"])
        self.assertEqual(action["change"]["new_value"], "dierenarts")
        self.assertEqual(app.speech.spoken[0], "Wil je dat ik wat je later wilt worden verander naar dierenarts?")
        self.assertIn("Dankjewel, ik heb dat aangepast.", app.speech.spoken[1])
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M4"].get("corrected"))
        self.assertIn(17, app.phases_with_confirmed_change)

    def test_aspiration_bare_rejection_asks_for_detail(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 17,
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "response_mode": "mistake_interpretation",
            "memory_correction_available": True,
            "memory_correction_field": "aspiration",
            "mistake_topic": app.topic_candidate(
                domain="droom",
                label="dierenarts worden",
                fields=["aspiration"],
                field_labels={"aspiration": "wat je later wilt worden"},
                current_values={"aspiration": "dierenarts worden"},
                correct_values=["je later dierenarts worden wilt"],
                memory_link="je later dierenarts worden wilt",
                options=["dierenarts worden"],
                reground="Ik onthoud dat dierenarts worden iets is waar je later iets mee wilt.",
            ),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        app.mistake_states = {"M4": {"id": "M4"}}
        result = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)

        action = app.action_handler(result, "Nee", turn)

        self.assertEqual(action["action"], "ask_correction_detail")
        self.assertTrue(action["follow_up_needed"])
        self.assertEqual(action["leo_response"], "Oeps, wat wil jij dan later worden?")
        self.assertTrue(app.mistake_states["M4"].get("wrong_value_rejected"))

    def test_middle_school_feeling_speaks_flexible_wrap_without_um_change(self):
        app = make_app()
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "middle_school_feeling",
        }
        result = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)

        action = app.action_handler(result, "Ik heb er zin in, maar het is ook spannend.", turn)

        self.assertEqual(action["action"], "middle_school_feeling")
        self.assertEqual(action["middle_school_feeling"], "mixed")
        self.assertEqual(action["leo_response"], "Dat snap ik. Je kunt er zin in hebben en het tegelijk spannend vinden.")
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_middle_school_feeling_detects_not_spannend_as_calm(self):
        app = make_app()
        turn = {
            "phase": 17,
            "phase_id": "3.4/5",
            "response_mode": "middle_school_feeling",
        }
        result = IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.85)

        action = app.action_handler(result, "Nee hoor vind ik helemaal niet spannend", turn)

        self.assertEqual(action["action"], "middle_school_feeling")
        self.assertEqual(action["middle_school_feeling"], "calm")
        self.assertEqual(action["leo_response"], "Fijn, dan kijk je er best rustig naar.")
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_school_joke_transition_uses_classifier_category(self):
        app = make_app()
        turn = {
            "phase": 10,
            "phase_id": "2.1",
            "response_mode": "school_joke_transition",
        }
        result = IntentResult(
            intent="dialogue_answer",
            field=None,
            value="dislikes_school",
            confidence=0.92,
        )

        action = app.action_handler(result, "School is stom.", turn)

        self.assertEqual(action["action"], "school_joke_transition")
        self.assertEqual(action["school_joke_category"], "dislikes_school")
        self.assertEqual(
            action["leo_response"],
            "Dat snap ik ook. Ik maakte een grapje; school hoeft echt niet altijd leuk te zijn.",
        )
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_school_joke_transition_has_offline_fallback_category(self):
        app = make_app()
        turn = {
            "phase": 10,
            "phase_id": "2.1",
            "response_mode": "school_joke_transition",
        }
        result = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)

        action = app.action_handler(result, "Nee echt niet.", turn)

        self.assertEqual(action["action"], "school_joke_transition")
        self.assertEqual(action["school_joke_category"], "rejects_joke_no")
        self.assertIn("School is meestal niet iemands hobby", action["leo_response"])
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_robot_school_guess_uses_classifier_category(self):
        app = make_app()
        turn = {
            "phase": 11,
            "phase_id": "2.2",
            "response_mode": "robot_school_guess",
        }
        result = IntentResult(
            intent="dialogue_answer",
            field=None,
            value="wrong_guess",
            confidence=0.93,
        )

        action = app.action_handler(result, "Rekenen?", turn)

        self.assertEqual(action["action"], "robot_school_guess")
        self.assertEqual(action["robot_school_guess_category"], "wrong_guess")
        self.assertEqual(
            action["leo_response"],
            "Haha, bijna. Ik was eigenlijk best goed in luisteren. De rest was soms wat lastiger.",
        )
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_robot_school_guess_unknown_uses_reveal_line(self):
        app = make_app()
        turn = {
            "phase": 11,
            "phase_id": "2.2",
            "response_mode": "robot_school_guess",
        }
        result = IntentResult(intent="dialogue_answer", field=None, value="unclear", confidence=0.95)

        action = app.action_handler(result, "Weet ik niet", turn)

        self.assertEqual(action["action"], "robot_school_guess")
        self.assertEqual(action["robot_school_guess_category"], "unclear")
        self.assertEqual(
            action["leo_response"],
            "Ik verklap het: luisteren ging best oké. De rest was soms wat lastiger.",
        )
        self.assertEqual(app.speech.spoken, [action["leo_response"]])

    def test_phase_confirmed_change_segment_flags_can_target_previous_phase(self):
        app = make_app()
        phase = {"phase": 18, "layer": "test", "name": "test"}
        run_segment = {
            "content_plan": app.l1("corrected branch"),
            "expects_response": False,
            "run_if_phase_confirmed_change": True,
            "condition_phase": 17,
        }
        skip_segment = {
            "content_plan": app.l1("uncorrected branch"),
            "expects_response": False,
            "skip_if_phase_confirmed_change": True,
            "condition_phase": 17,
        }

        app.run_phase_segment(phase, run_segment)
        app.run_phase_segment(phase, skip_segment)
        self.assertEqual(app.speech.spoken, ["uncorrected branch"])

        app.speech.spoken.clear()
        app.phases_with_confirmed_change = {17}
        app.run_phase_segment(phase, run_segment)
        app.run_phase_segment(phase, skip_segment)
        self.assertEqual(app.speech.spoken, ["corrected branch"])

    def test_repeat_phase_clears_prior_confirmed_change_state_for_that_phase(self):
        app = make_app()
        turn = {
            "phase": 13,
            "phase_id": "2.4",
            "mistake_id": "M3",
        }
        app.phases_with_confirmed_change = {12, 13, 17}
        app.mistake_states = {
            "M3": {
                "id": "M3",
                "corrected": True,
                "outcome_logged": True,
            },
            "M2": {
                "id": "M2",
                "corrected": False,
            },
        }

        app.reset_phase_attempt_state(turn)

        self.assertEqual(app.phases_with_confirmed_change, {12, 17})
        self.assertNotIn("M3", app.mistake_states)
        self.assertIn("M2", app.mistake_states)

    def test_nudge_phase_asks_for_missing_correction_or_offers_memory_access(self):
        app = make_app()
        app.last_um_preview = sample_um()
        topic = app.general_memory_topic(app.last_um_preview)
        turn = {
            "phase": 9,
            "response_mode": "nudge_interpretation",
            "topic": topic,
        }
        app.mistake_states = {
            "M2": {
                "id": "M2",
                "mentioned": True,
                "field": "fav_food",
                "actual": "pannenkoeken",
                "wrong": "pizza",
                "corrected": False,
            }
        }

        wrong_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Nee",
            turn,
        )

        self.assertEqual(wrong_action["action"], "nudge_ask_correction_detail")
        self.assertTrue(wrong_action["follow_up_needed"])
        self.assertEqual(wrong_action["leo_response"], "Oeps. Wil je zeggen wat er niet klopte?")
        self.assertTrue(turn["nudge_correction_requested"])

        detail_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "pannenkoeken",
            turn,
        )

        self.assertEqual(detail_action["action"], "nudge_correction_detail")
        self.assertEqual(detail_action["corrected_value"], "pannenkoeken")
        self.assertEqual(detail_action["mistake_id"], "M2")
        self.assertTrue(app.mistake_states["M2"]["corrected"])
        self.assertEqual(app.corrections_seen, 1)

        offer_turn = {
            "phase": 9,
            "response_mode": "nudge_interpretation",
            "topic": topic,
        }
        fine_action = app.action_handler(
            IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9),
            "Ja",
            offer_turn,
        )

        self.assertEqual(fine_action["action"], "nudge_memory_offer")
        self.assertTrue(fine_action["follow_up_needed"])
        self.assertEqual(fine_action["leo_response"], "We kunnen ook samen kijken wat ik over jou onthoud, als je wilt.")
        self.assertTrue(offer_turn["nudge_memory_offer_made"])

    def test_stub_classifier_detects_memory_access_phrases(self):
        clf = cri_module.StubIntentClassifier(valid_fields=list(CRI.UM_FIELDS))

        for phrase in (
            "Wat heb je over mij onthouden?",
            "Kan ik je geheugen zien?",
            "Wat weet je nog van mij?",
            "Mag ik in je geheugen kijken?",
            "Wat staat er in je geheugenboek?",
            "Kun je zeggen wat je van mij weet?",
            "Wat weet jij nog allemaal over mij?",
            "Laat eens zien wat je onthoudt.",
        ):
            with self.subTest(phrase=phrase):
                result = clf.classify(phrase)
                self.assertEqual(result.intent, "um_inspect")

    def test_gpt_classifier_coerces_memory_book_questions_to_um_inspect(self):
        gpt = object.__new__(cri_module.GPTIntentClassifier)

        for phrase in (
            "Kan ik je geheugen zien?",
            "Mag ik je geheugenboek zien?",
            "Kan ik in je geheugen kijken?",
        ):
            with self.subTest(phrase=phrase):
                coerced = gpt._coerce_result(
                    "dialogue_question",
                    None,
                    None,
                    0.92,
                    phrase,
                )
                self.assertEqual(coerced[0], "um_inspect")
                self.assertIsNone(coerced[1])
                self.assertIsNone(coerced[2])
                self.assertGreaterEqual(coerced[3], 0.95)

    def test_public_gpt_classifier_uses_context_aware_newer_version(self):
        self.assertEqual(cri_module.GPTIntentClassifier.__module__, "cri_classifier.gpt_newer")

    def test_action_handler_passes_turn_context_to_context_aware_classifier(self):
        app = make_app()

        class ContextAwareClassifier:
            def __init__(self):
                self.calls = []

            def classify(self, text, turn_context=None):
                self.calls.append(("classify", text, turn_context))
                return cri_module.IntentResult(
                    intent="dialogue_answer",
                    field=None,
                    value=None,
                    confidence=0.95,
                )

            def classify_retry(self, text, turn_context=None):
                self.calls.append(("classify_retry", text, turn_context))
                return cri_module.IntentResult(
                    intent="dialogue_answer",
                    field=None,
                    value=None,
                    confidence=0.95,
                )

        app.clf = ContextAwareClassifier()
        app.last_leo_utterance = "En volgens mij is padel jouw allerliefste hobby."
        turn = {
            "phase": 6,
            "text": app.last_leo_utterance,
            "response_mode": "mistake_interpretation",
            "mistake_field": "hobby_fav",
            "used_fields": {"hobbies": "voetbal, schaatsen, judo en padel"},
            "topic": {"kind": "hobby", "current_values": {"hobby_fav": "schaatsen"}},
        }

        app.actions.classify_with_repeat("Nee, schaatsen", turn)

        _, transcript, context = app.clf.calls[0]
        self.assertEqual(transcript, "Nee, schaatsen")
        self.assertEqual(context["leo_previous"], app.last_leo_utterance)
        self.assertEqual(context["response_mode"], "mistake_interpretation")
        self.assertEqual(context["topic"], "hobby")
        self.assertIn("hobby_fav", context["relevant_fields"])
        self.assertIn("hobbies", context["relevant_fields"])

    def test_short_meaningful_replies_are_dialogue_answers_not_none(self):
        stub = cri_module.StubIntentClassifier(valid_fields=list(CRI.UM_FIELDS))
        self.assertEqual(stub.classify("Ja tuurlijk").intent, "dialogue_answer")
        self.assertEqual(stub.classify("Mijn favoriete hobby is tekenen").intent, "um_update")

        gpt = object.__new__(cri_module.GPTIntentClassifier)
        coerced = gpt._coerce_result("dialogue_none", None, None, 0.9, "Ja tuurlijk")
        self.assertEqual(coerced[0], "dialogue_answer")
        self.assertEqual(coerced[1], None)
        self.assertEqual(coerced[2], None)

        filler = gpt._coerce_result("dialogue_none", None, None, 0.9, "um eh ja nee")
        self.assertEqual(filler[0], "dialogue_none")

    def test_conversation_logs_write_to_local_folder_and_keep_previous_log(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.CHILD_ID = "1001"
        app.local_child_name = "Noor"
        app.researcher_name = "Sander"
        app.last_um_preview = sample_um()
        app.session_config = {
            "child_name": "Noor",
            "first_name_cri": "Noor",
            "first_name_tablet": "Noortje",
        }
        app.resume_from_log_path = r"C:\previous\Noor.json"
        app.resume_source_log = {
            "session_id": "old",
            "events": [{"type": "utterance", "speaker": "LEO", "text": "Hoi", "timestamp": 1.0}],
            "turns": [{"phase": 1, "started_at": 1.0, "ended_at": 2.0}],
            "last_completed_phase": 1,
        }

        turn = {
            "phase": 2,
            "part": 1,
            "phase_id": "1.2",
            "name": "Tutorial",
            "layer": "L1",
            "dialogue_case": "um_template",
            "content_plan": app.l1("Test"),
        }
        app.start_conversation_log([turn])
        app.start_turn_log(turn)
        app.log_conversation_event("utterance", speaker="LEO", text="Hoi Noor")
        app.log_conversation_event("utterance", speaker="CHILD", text="Ja")
        app.finish_turn_log()

        log = app.conversation_log
        self.assertTrue(Path(log["folder"]).is_relative_to(Path(temp_dir)))
        self.assertEqual(Path(log["json_path"]).name, "N_1001_debug.json")
        self.assertEqual(Path(log["txt_path"]).name, "N_1001_conversation_debug.txt")
        self.assertEqual(Path(log["omr_log_path"]).suffix, ".log")
        self.assertTrue(log["previous_log_included"])
        self.assertEqual(log["events"][0]["text"], "Hoi")
        self.assertEqual(log["timestamp_unit"], "seconds_from_interaction_start")

        json_payload = Path(log["json_path"]).read_text(encoding="utf-8")
        transcript_payload = Path(log["conversation_debug_path"]).read_text(encoding="utf-8")
        omr_payload = Path(log["omr_log_path"]).read_text(encoding="utf-8")
        for payload in (json_payload, transcript_payload, omr_payload):
            self.assertNotIn("Noor", payload)
            self.assertNotIn("Noortje", payload)
        self.assertIn("[Leo] Hoi N_1001", transcript_payload)
        self.assertIn("[N_1001] Ja", transcript_payload)
        self.assertIn("Part 1 phase 2: Tutorial", transcript_payload)
        self.assertNotIn("[um_template]", transcript_payload)

        omr_log = json.loads(omr_payload)
        self.assertEqual(omr_log["session_metadata"]["child_label"], "N_1001")
        self.assertIn("mistakes_and_corrections", omr_log)
        self.assertIn("memory_acts", omr_log)
        self.assertIn("um_updates", omr_log)
        self.assertIn("tablet_events", omr_log)

    def test_omr_log_formats_timestamps_and_dedupes_confirmed_memory_acts(self):
        app = make_app()
        log = {
            "session_id": "session-test",
            "child_id": "1001",
            "child_label": "N_1001",
            "tutorial_condition": "E",
            "started_at": 0.0,
            "started_wall_time": "1970-01-01T00:00:00+00:00",
            "ended_at": 240.0,
            "ended_wall_time": "1970-01-01T00:04:00+00:00",
            "events": [
                {
                    "type": "mistake_outcome",
                    "timestamp": 120.2,
                    "phase": 6,
                    "mistake_id": "M1",
                    "field": "hobby_fav",
                    "wrong_value": "padel",
                    "real_value": "voetbal",
                    "corrected": True,
                    "spt_layer": "orientation",
                    "layer": "L2-slot WRONG",
                    "latency_seconds": 8.42,
                },
                {
                    "type": "action_handler",
                    "timestamp": 184.391,
                    "phase": 6,
                    "action": "confirm_update",
                    "change_confirmed": True,
                    "change": {"action": "update", "field": "hobby_fav", "new_value": "gamen"},
                },
                {
                    "type": "action_handler",
                    "timestamp": 193.031,
                    "phase": 6,
                    "action": "confirm_update",
                    "change_confirmed": True,
                    "change": {"action": "update", "field": "hobby_fav", "new_value": "gamen"},
                },
            ],
        }

        omr_log = app.conv_log.build_omr_log(log)

        self.assertEqual(omr_log["mistakes_and_corrections"][0]["timestamp"], "02:00")
        self.assertEqual(omr_log["mistakes_and_corrections"][0]["child_initiated"], "yes")
        self.assertEqual(omr_log["mistakes_and_corrections"][0]["spt_layer"], "orientation")
        self.assertNotEqual(omr_log["mistakes_and_corrections"][0]["spt_layer"], "L2-slot WRONG")
        self.assertEqual(omr_log["mistakes_and_corrections"][0]["latency_seconds"], 8.42)
        self.assertEqual(len(omr_log["memory_acts"]), 1)
        self.assertEqual(omr_log["memory_acts"][0]["timestamp"], "03:04")
        self.assertEqual(omr_log["memory_acts"][0]["target_field"], "hobby_fav")
        self.assertIn(
            {
                "type": "session_start",
                "timestamp": "00:00",
            },
            omr_log["tablet_events"],
        )
        self.assertIn(
            {
                "type": "session_end",
                "timestamp": "04:00",
            },
            omr_log["tablet_events"],
        )

    def test_omr_log_imports_matching_tablet_json_events(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_tablet_events_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        event_path = Path(temp_dir) / "tablet_events.jsonl"
        app.TABLET_EVENTS_LOG_PATH = str(event_path)
        event_path.write_text(
            "\n".join([
                json.dumps({
                    "type": "shown",
                    "session_id": "session-test",
                    "child_id": "1001",
                    "server_wall_time": 62.4,
                    "category": "hobby",
                    "memory_item": "Hobby's",
                    "screen": "category",
                    "phase": 6,
                }),
                json.dumps({
                    "type": "shown",
                    "session_id": "other-session",
                    "child_id": "1001",
                    "server_wall_time": 90.0,
                    "category": "school",
                }),
                json.dumps({
                    "type": "tablet_display_changed",
                    "session_id": "session-test",
                    "child_id": "1001",
                    "server_wall_time": 125.0,
                    "field": "hobby_fav",
                    "old_value": "padel",
                    "new_value": "gamen",
                }),
            ]),
            encoding="utf-8",
        )
        log = {
            "session_id": "session-test",
            "child_id": "1001",
            "child_label": "N_1001",
            "tutorial_condition": "E",
            "started_at": 0.0,
            "started_wall_time": "1970-01-01T00:00:00+00:00",
            "events": [],
        }

        omr_log = app.conv_log.build_omr_log(log)
        tablet_events = omr_log["tablet_events"]

        self.assertIn({
            "type": "shown",
            "timestamp": "01:02",
            "memory_item": "Hobby's",
            "category": "hobby",
            "screen": "category",
            "phase": 6,
        }, tablet_events)
        self.assertIn({
            "type": "tablet_display_changed",
            "timestamp": "02:05",
            "field": "hobby_fav",
            "old_value": "padel",
            "new_value": "gamen",
        }, tablet_events)
        self.assertNotIn("school", json.dumps(tablet_events))

    def test_conversation_log_write_error_does_not_crash_dialogue(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_log_error_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.last_um_preview = sample_um()
        app.conversation_log = {
            "session_id": "test",
            "json_path": str(Path(temp_dir) / "Noor.json"),
            "text_path": str(Path(temp_dir) / "Noor.txt"),
            "txt_path": str(Path(temp_dir) / "Noor.txt"),
            "events": [],
            "turns": [],
            "script_plan": [],
        }

        with patch("os.replace", side_effect=OSError(22, "Invalid argument")):
            app.finish_conversation_log()

        self.assertEqual(app.conversation_log["session_status"], "completed")
        self.assertIn("Invalid argument", app.conversation_log["last_log_write_error"])

    def test_resume_helpers_clean_quoted_paths_and_replay_previous_transcript(self):
        app = make_app()
        clean = app.clean_pasted_path('"C:\\Users\\Sander\\log.json"')
        self.assertEqual(clean, r"C:\Users\Sander\log.json")

        log = {
            "events": [
                {"type": "phase_start", "phase": 1, "name": "Greeting", "timestamp": 1.0},
                {"type": "utterance", "speaker": "LEO", "text": "Hoi Noor", "timestamp": 2.0},
                {"type": "utterance", "speaker": "CHILD", "text": "ja", "timestamp": 3.0},
                {"type": "phase_end", "phase": 1, "timestamp": 4.0},
            ]
        }

        lines = app.resume_console_transcript_lines(log)

        self.assertIn("[00:01] Phase 1: Greeting", lines)
        self.assertIn("[00:02] LEO: Hoi Noor", lines)
        self.assertIn("[00:03] CHILD: ja", lines)
        self.assertIn("[00:04] Phase 1 finished", lines)

    def test_local_launcher_points_to_inner_shareable_package_and_local_env(self):
        spec = importlib.util.spec_from_file_location("run_cri_dialogue2_for_tests", LAUNCHER_PATH)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        self.assertEqual(launcher.DIALOGUE_DIR, PACKAGE_DIR)
        self.assertEqual(launcher.DIALOGUE_FILE, PACKAGE_DIR / "CRI-BRANCH-BASIC4_0.py")
        self.assertEqual(launcher.DEFAULT_ENV, LOCAL_DIR / ".env")

    def test_shared_launcher_points_to_inner_shareable_package_and_local_env(self):
        spec = importlib.util.spec_from_file_location("run_cri_dialogue2_shared_for_tests", SHARED_LAUNCHER_PATH)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        self.assertEqual(launcher.DIALOGUE_DIR, PACKAGE_DIR)
        self.assertEqual(launcher.DIALOGUE_FILE, PACKAGE_DIR / "CRI-BRANCH-BASIC4_0.py")
        self.assertEqual(launcher.DEFAULT_ENV, LOCAL_DIR / ".env")

    def test_realtimestt_backend_mac_auto_uses_cpu_int8_small_without_cuda_probe(self):
        torch = FakeTorchModule(mps_available=True, fail_cuda_check=True)
        ctranslate2 = FakeCTranslate2Module(fail_cuda_count=True)

        backend = cri_module.resolve_realtimestt_backend(
            requested_device="auto",
            requested_compute_type="auto",
            requested_model="auto",
            system_name="Darwin",
            torch_module=torch,
            ctranslate2_module=ctranslate2,
        )

        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.compute_type, "int8")
        self.assertEqual(backend.model, "small")
        self.assertTrue(backend.mac_gpu_available)
        self.assertIn("Apple GPU/MPS", backend.note)
        self.assertEqual(torch.cuda.available_checks, 0)
        self.assertEqual(ctranslate2.cuda_count_checks, 0)

    def test_realtimestt_backend_windows_auto_with_cuda_uses_cuda_float16_small(self):
        torch = FakeTorchModule(cuda_available=True, cuda_name="RTX 3050")
        ctranslate2 = FakeCTranslate2Module(cuda_count=1)

        backend = cri_module.resolve_realtimestt_backend(
            requested_device="auto",
            requested_compute_type="auto",
            requested_model="auto",
            system_name="Windows",
            torch_module=torch,
            ctranslate2_module=ctranslate2,
        )

        self.assertEqual(backend.device, "cuda")
        self.assertEqual(backend.compute_type, "float16")
        self.assertEqual(backend.model, "small")
        self.assertEqual(backend.cuda_device_name, "RTX 3050")

    def test_realtimestt_backend_auto_without_cuda_uses_cpu_int8_small(self):
        torch = FakeTorchModule(cuda_available=False)
        ctranslate2 = FakeCTranslate2Module(cuda_count=0)

        backend = cri_module.resolve_realtimestt_backend(
            requested_device="auto",
            requested_compute_type="auto",
            requested_model="auto",
            system_name="Windows",
            torch_module=torch,
            ctranslate2_module=ctranslate2,
        )

        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.compute_type, "int8")
        self.assertEqual(backend.model, "small")

    def test_realtimestt_backend_explicit_cuda_without_cuda_fails_clearly(self):
        torch = FakeTorchModule(cuda_available=False)
        ctranslate2 = FakeCTranslate2Module(cuda_count=0)

        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            cri_module.resolve_realtimestt_backend(
                requested_device="cuda",
                requested_compute_type="auto",
                requested_model="auto",
                system_name="Darwin",
                torch_module=torch,
                ctranslate2_module=ctranslate2,
            )

    def test_realtimestt_halo_spinner_is_disabled_in_recorder_kwargs(self):
        recorder_kwargs = {}

        class DummyRecorder(FakeSTTRecorder):
            def __init__(self, **kwargs):
                super().__init__("")
                recorder_kwargs.update(kwargs)

        with patch.dict(
            cri_module.SpeechIO.__init__.__globals__,
            {"AudioToTextRecorder": DummyRecorder},
        ):
            speech = cri_module.SpeechIO()

        self.assertIs(recorder_kwargs["spinner"], False)
        self.assertEqual(recorder_kwargs["model"], "small")
        self.assertEqual(recorder_kwargs["device"], "cpu")
        self.assertEqual(recorder_kwargs["compute_type"], "int8")
        self.assertEqual(recorder_kwargs["post_speech_silence_duration"], 1.6)
        self.assertEqual(speech._recorder.microphone_states, [False])
        self.assertIsNotNone(speech._stt_spinner_lock)

    def test_stt_normalize_transcript_removes_symbols_accents_and_non_latin_text(self):
        transcript = "Oké — één café ✓ Привет, dat klopt!"

        clean = cri_module.SpeechIO.normalize_transcript(transcript)

        self.assertEqual(clean, "Oke een cafe, dat klopt!")

    def test_clean_stt_transcript_normalizes_after_repair_and_keeps_normal_answers(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)

        self.assertEqual(speech._clean_stt_transcript("basta — oké"), "pasta oke")
        self.assertEqual(speech._clean_stt_transcript("gym en rekenen"), "gym en rekenen")

    def test_say_clears_stale_realtimestt_spinner_before_output(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        recorder = FakeSTTRecorder()
        halo = recorder.halo
        speech._recorder = recorder

        with patch("builtins.print"):
            speech.say("Hallo!")

        self.assertTrue(halo.stopped)
        self.assertIsNone(recorder.halo)
        self.assertIs(recorder.spinner, False)

    def test_nao_say_prints_leo_terminal_once_and_sends_tts(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )

        with patch("builtins.print") as print_mock, patch("time.sleep", lambda *_args, **_kwargs: None):
            speech.say("Hoi daar.")

        print_mock.assert_called_once_with("\n[LEO]: Hoi daar.\n")
        self.assertEqual(len(speech.nao.tts.requests), 1)

    def test_nao_system_prompt_prints_leo_terminal_once_and_sends_tts(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )

        with patch("builtins.print") as print_mock, patch("time.sleep", lambda *_args, **_kwargs: None):
            speech._say_system("Kun je het nog een keer zeggen?")

        print_mock.assert_called_once_with("\n[LEO]: Kun je het nog een keer zeggen?\n")
        self.assertEqual(len(speech.nao.tts.requests), 1)

    def test_non_nao_say_prints_leo_terminal_once(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)

        with patch("builtins.print") as print_mock:
            speech.say("Hoi zonder robot.")

        print_mock.assert_called_once_with("\n[LEO]: Hoi zonder robot.\n")

    def test_leo_terminal_print_clears_spinner_before_output_and_tts(self):
        events = []

        class EventTTS:
            def request(self, request):
                events.append("tts")

        speech = cri_module.SpeechIO(
            nao=SimpleNamespace(tts=EventTTS(), leds=FakeNaoChannel()),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        speech._stop_stt_spinner = lambda: events.append("spinner")

        with (
            patch("builtins.print", side_effect=lambda *_args, **_kwargs: events.append("print")),
            patch("time.sleep", lambda *_args, **_kwargs: None),
        ):
            speech.say("Eerst lezen, dan spreken.")

        self.assertEqual(events[:3], ["spinner", "print", "tts"])

    def test_tts_sanitizer_handles_phase12_sample_and_preserves_dutch_accents(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)

        cleaned = speech._sanitize_tts_text(
            "Dat snap ik. Rekenen is best als een puzzel \u2014 "
            "je zoekt steeds de oplossing. Ok\u00e9, \u00e9\u00e9n."
        )

        self.assertEqual(
            cleaned,
            "Dat snap ik. Rekenen is best als een puzzel, "
            "je zoekt steeds de oplossing. Ok\u00e9, \u00e9\u00e9n.",
        )
        self.assertNotIn("\u2014", cleaned)
        self.assertIn("Ok\u00e9", cleaned)
        self.assertIn("\u00e9\u00e9n", cleaned)

    def test_tts_sanitizer_removes_risky_symbols_and_repairs_mojibake(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        raw = (
            "Hij zei \u201cHoi\u201d\u2026 A\u00a0B \u2192 punt \u2022 "
            "\u2713 \u2717 \u26a0 \U0001f512 \u2500 "
            "moji \u00e2\u20ac\u201d dash \u00e2\u20ac\u201c en \u00c3\u00a9."
        )

        cleaned = speech._sanitize_tts_text(raw)

        for bad in (
            "\u201c", "\u201d", "\u2026", "\u2192", "\u2022",
            "\u2713", "\u2717", "\u26a0", "\U0001f512", "\u2500",
            "\u00a0", "\u00e2\u20ac\u201d", "\u00e2\u20ac\u201c",
        ):
            self.assertNotIn(bad, cleaned)
        self.assertIn('"Hoi"...', cleaned)
        self.assertIn("A B", cleaned)
        self.assertIn("moji, dash, en \u00e9.", cleaned)

    def test_nao_tts_receives_sanitized_text_and_terminal_matches(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        raw = (
            "Dat snap ik. Rekenen is best als een puzzel \u2014 "
            "je zoekt steeds de oplossing. \u201cHoi\u201d."
        )

        with patch("builtins.print") as print_mock, patch("time.sleep", lambda *_args, **_kwargs: None):
            speech.say(raw)

        terminal_line = print_mock.call_args.args[0]
        terminal_text = terminal_line.strip()[len("[LEO]: "):]
        request_text = speech.nao.tts.requests[0].text
        spoken_text = request_text.replace("\\rspd=92\\", "", 1)
        self.assertEqual(spoken_text, terminal_text)
        self.assertNotIn("\u2014", request_text)
        self.assertNotIn("\u201c", request_text)
        self.assertIn('"Hoi"', request_text)

    def test_nao_tts_retries_ascii_fallback_after_encoding_error(self):
        class FailingOnceTTS:
            def __init__(self):
                self.requests = []

            def request(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    raise RuntimeError("encoding problem")

        tts = FailingOnceTTS()
        speech = cri_module.SpeechIO(
            nao=SimpleNamespace(tts=tts, leds=FakeNaoChannel()),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )

        with patch("builtins.print"), patch("time.sleep", lambda *_args, **_kwargs: None):
            speech.say("Ok\u00e9 \u00e9\u00e9n \u2014 test.")

        self.assertEqual(len(tts.requests), 2)
        self.assertIn("Ok\u00e9 \u00e9\u00e9n, test.", tts.requests[0].text)
        self.assertIn("Oke een, test.", tts.requests[1].text)
        tts.requests[1].text.encode("ascii")

    def test_nao_tts_repeated_failure_does_not_raise_and_prints_error(self):
        class AlwaysFailTTS:
            def __init__(self):
                self.requests = []

            def request(self, request):
                self.requests.append(request)
                raise RuntimeError("still broken")

        tts = AlwaysFailTTS()
        speech = cri_module.SpeechIO(
            nao=SimpleNamespace(tts=tts, leds=FakeNaoChannel()),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )

        with patch("builtins.print") as print_mock:
            speech.say("Ok\u00e9 \u2014 test.")

        self.assertEqual(len(tts.requests), 2)
        self.assertTrue(
            any("[TTS ERROR]" in str(call.args[0]) for call in print_mock.call_args_list)
        )

    def test_nao_startup_keeps_solitary_life_and_disables_blinking(self):
        app = make_app()
        app.CONNECT_NAO = True
        app.simulation_mode = False
        app.nao = FakeNao()
        eye_colors = []
        app.speech = SimpleNamespace(_set_eyes=eye_colors.append)
        app.log_conversation_event = lambda *_args, **_kwargs: None

        app.prepare_nao_for_dialogue()

        autonomous_requests = app.nao.autonomous.requests
        self.assertIsInstance(autonomous_requests[0], cri_module.NaoWakeUpRequest)
        self.assertIsInstance(autonomous_requests[1], cri_module.NaoSetAutonomousLifeRequest)
        self.assertEqual(autonomous_requests[1].state, "solitary")
        blink_values = [
            request.value
            for request in autonomous_requests
            if isinstance(request, cri_module.NaoBlinkingRequest)
        ]
        self.assertEqual(blink_values, [False, False])
        self.assertNotIn(
            "disabled",
            [
                request.state
                for request in autonomous_requests
                if isinstance(request, cri_module.NaoSetAutonomousLifeRequest)
            ],
        )
        self.assertEqual(app.nao.motion.requests[0].animation_path, "animations/Stand/Gestures/Hey_1")
        self.assertEqual(eye_colors, ["white", "white"])

    def test_greeting_wave_disables_blinking_and_resets_white_after_animation(self):
        app = make_app()
        app.CONNECT_NAO = True
        app.simulation_mode = False
        app.nao = FakeNao()
        eye_colors = []
        app.speech = SimpleNamespace(_set_eyes=eye_colors.append)
        app.log_conversation_event = lambda *_args, **_kwargs: None

        app.perform_greeting_wave()

        self.assertEqual(app.nao.motion.requests[0].animation_path, "animations/Stand/Gestures/Hey_1")
        self.assertIsInstance(app.nao.autonomous.requests[-1], cri_module.NaoBlinkingRequest)
        self.assertIs(app.nao.autonomous.requests[-1].value, False)
        self.assertEqual(eye_colors, ["white"])

    def test_nao_cleanup_restores_blinking_before_disabling_life_and_resting(self):
        app = make_app()
        app.CONNECT_NAO = True
        app.simulation_mode = False
        app.nao = FakeNao()
        app.log_conversation_event = lambda *_args, **_kwargs: None

        app.cleanup_nao_after_dialogue()

        autonomous_requests = app.nao.autonomous.requests
        self.assertIsInstance(autonomous_requests[0], cri_module.NaoBlinkingRequest)
        self.assertIs(autonomous_requests[0].value, True)
        self.assertIsInstance(autonomous_requests[1], cri_module.NaoSetAutonomousLifeRequest)
        self.assertEqual(autonomous_requests[1].state, "disabled")
        self.assertIsInstance(autonomous_requests[2], cri_module.NaoRestRequest)

    def test_eye_say_sets_white_and_never_green_for_nao_speech(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        eye_colors = []
        speech._set_eyes = eye_colors.append

        with patch("time.sleep", lambda *_args, **_kwargs: None), patch("builtins.print"):
            speech.say("Hallo!")
            speech.say("Nog een zin.")

        self.assertEqual(eye_colors, ["white", "white"])
        self.assertNotIn("green", eye_colors)
        self.assertEqual(len(speech.nao.tts.requests), 2)

    def test_eye_system_retry_prompt_sets_white_and_never_green(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        eye_colors = []
        speech._set_eyes = eye_colors.append

        with patch("time.sleep", lambda *_args, **_kwargs: None), patch("builtins.print"):
            speech._say_system("Kun je het nog een keer zeggen?")

        self.assertEqual(eye_colors, ["white"])
        self.assertNotIn("green", eye_colors)
        self.assertEqual(len(speech.nao.tts.requests), 1)

    def test_eye_listen_turns_green_only_after_mic_opens_then_white_on_success(self):
        events = []

        class ObservedRecorder(FakeSTTRecorder):
            def set_microphone(self, enabled):
                super().set_microphone(enabled)
                events.append(("mic", enabled))

            def text(self):
                events.append(("text", self.text_value))
                return super().text()

        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        speech._use_keyboard_input = lambda: False
        speech._recorder = ObservedRecorder("ja")
        speech._set_eyes = lambda color: events.append(("eyes", color))

        with patch("builtins.print"):
            transcript = speech.listen()

        self.assertEqual(transcript, "ja")
        self.assertLess(events.index(("mic", True)), events.index(("eyes", "green")))
        self.assertLess(events.index(("eyes", "green")), events.index(("text", "ja")))
        self.assertEqual(events[-1], ("eyes", "white"))

    def test_cri_spinner_starts_after_mic_and_green_then_stops_before_transcript(self):
        events = []

        class ObservedRecorder(FakeSTTRecorder):
            def set_microphone(self, enabled):
                super().set_microphone(enabled)
                events.append(("mic", enabled))

            def text(self):
                events.append(("text", self.text_value))
                return super().text()

        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        speech._use_keyboard_input = lambda: False
        speech._recorder = ObservedRecorder("ja")
        speech._set_eyes = lambda color: events.append(("eyes", color))
        speech._start_stt_spinner = lambda text="recording": events.append(("spinner", "start", text))
        speech._stop_stt_spinner = lambda: events.append(("spinner", "stop"))

        with patch("builtins.print", side_effect=lambda *_args, **_kwargs: events.append(("print", None))):
            transcript = speech.listen()

        self.assertEqual(transcript, "ja")
        self.assertLess(events.index(("mic", True)), events.index(("eyes", "green")))
        self.assertLess(events.index(("eyes", "green")), events.index(("spinner", "start", "recording")))
        self.assertLess(events.index(("spinner", "start", "recording")), events.index(("text", "ja")))
        first_stop = events.index(("spinner", "stop"))
        first_print = next(i for i, event in enumerate(events) if event[0] == "print")
        self.assertLess(first_stop, first_print)

    def test_listen_timeout_and_error_paths_end_with_white_eyes(self):
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
            stt_timeout=0.01,
        )
        speech._use_keyboard_input = lambda: False
        speech._recorder = FakeSTTRecorder("late text", sleep_seconds=0.2)
        timeout_eye_colors = []
        speech._set_eyes = timeout_eye_colors.append

        with patch("builtins.print"):
            timeout_transcript = speech.listen()

        self.assertEqual(timeout_transcript, "")
        self.assertIn("green", timeout_eye_colors)
        self.assertEqual(timeout_eye_colors[-1], "white")

        class FailingRecorder(FakeSTTRecorder):
            def text(self):
                raise RuntimeError("boom")

        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        speech._use_keyboard_input = lambda: False
        speech._recorder = FailingRecorder()
        error_eye_colors = []
        speech._set_eyes = error_eye_colors.append

        with patch("builtins.print"):
            error_transcript = speech.listen()

        self.assertEqual(error_transcript, "")
        self.assertIn("green", error_eye_colors)
        self.assertEqual(error_eye_colors[-1], "white")

    def test_eye_transcript_review_retry_does_not_set_green_before_listen(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        eye_colors = []
        speech._set_eyes = eye_colors.append
        speech.listen_with_retry = lambda: "opnieuw gehoord"

        with patch("builtins.input", side_effect=["r", ""]), patch("builtins.print"):
            reviewed = speech.review_transcript("eerste poging")

        self.assertEqual(reviewed, "opnieuw gehoord")
        self.assertNotIn("green", eye_colors)

    def test_review_prompt_stops_spinner_before_researcher_input(self):
        events = []
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        speech._stop_stt_spinner = lambda: events.append("stop")

        def fake_input(_prompt):
            events.append("input")
            return ""

        with patch("builtins.input", side_effect=fake_input), patch(
            "builtins.print",
            side_effect=lambda *_args, **_kwargs: events.append("print"),
        ):
            reviewed = speech.review_transcript("dat klopt")

        self.assertEqual(reviewed, "dat klopt")
        input_index = events.index("input")
        self.assertGreater(input_index, 0)
        self.assertEqual(events[input_index - 1], "stop")

    def test_shutdown_stops_spinner_mutes_eyes_and_closes_recorder(self):
        events = []
        speech = cri_module.SpeechIO(
            nao=FakeNao(),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
        )
        recorder = FakeSTTRecorder("ja")
        speech._recorder = recorder
        speech._stop_stt_spinner = lambda: events.append("stop")
        speech._set_mic = lambda enabled: events.append(("mic", enabled))
        speech._set_eyes = lambda color: events.append(("eyes", color))

        speech.shutdown()

        self.assertEqual(events[:3], ["stop", ("mic", False), ("eyes", "white")])
        self.assertTrue(recorder.aborted)
        self.assertTrue(recorder.shutdown_called)
        self.assertIsNone(speech._recorder)

    def test_tts_handoff_waits_before_queue_clear_and_mic_open(self):
        events = []

        class EventTTS:
            def request(self, request):
                events.append(("tts", request.text))

        class EventRecorder(FakeSTTRecorder):
            def set_microphone(self, enabled):
                super().set_microphone(enabled)
                events.append(("mic", enabled))

            def clear_audio_queue(self):
                super().clear_audio_queue()
                events.append(("clear_queue", None))

            def text(self):
                events.append(("text", self.text_value))
                return super().text()

        speech = cri_module.SpeechIO(
            nao=SimpleNamespace(tts=EventTTS(), leds=FakeNaoChannel()),
            use_nao_output=True,
            use_keyboard_input_fn=lambda: True,
            tts_char_seconds=0.1,
            tts_tail_buffer_seconds=0.5,
        )
        speech._recorder = EventRecorder("ja")
        speech._use_keyboard_input = lambda: False
        speech._set_eyes = lambda color: events.append(("eyes", color))

        with (
            patch("time.monotonic", side_effect=[10.0, 10.0]),
            patch("time.sleep", lambda seconds: events.append(("sleep", round(seconds, 3)))),
            patch("builtins.print"),
        ):
            speech.say("abcd")
            transcript = speech.listen()

        self.assertEqual(transcript, "ja")
        self.assertIn(("sleep", 0.9), events)
        sleep_index = events.index(("sleep", 0.9))
        mic_open_index = events.index(("mic", True))
        clear_indices = [
            i for i, event in enumerate(events)
            if event == ("clear_queue", None)
        ]
        self.assertTrue(any(sleep_index < i < mic_open_index for i in clear_indices))
        self.assertLess(mic_open_index, events.index(("eyes", "green")))
        self.assertEqual(events[-1], ("eyes", "white"))

    def test_tts_handoff_knobs_can_come_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "CRI_TTS_CHAR_SECONDS": "0.07",
                "CRI_TTS_TAIL_BUFFER_SECONDS": "1.1",
                "CRI_LEO_ECHO_SIMILARITY": "0.9",
            },
        ):
            speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)

        self.assertEqual(speech._tts_char_seconds, 0.07)
        self.assertEqual(speech._tts_tail_buffer_seconds, 1.1)
        self.assertEqual(speech._leo_echo_similarity, 0.9)

    def test_stt_rejects_likely_leo_echo_transcript(self):
        events = []
        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            log_event_fn=lambda event_type, **data: events.append((event_type, data)),
        )
        speech._last_leo_text = "Wil je iets veranderen? Zeg maar ja of nee."

        cleaned = speech._clean_stt_transcript("Wil je iets veranderen zeg maar ja of nee")

        self.assertEqual(cleaned, "")
        self.assertEqual(events[-1][0], "stt_rejected")
        self.assertEqual(events[-1][1]["reason"], "leo_echo")

        repeated = speech._clean_stt_transcript(
            "Wil je iets veranderen zeg maar ja of nee, "
            "Wil je iets veranderen zeg maar ja of nee, "
            "Wil je iets veranderen zeg maar ja of nee"
        )
        self.assertEqual(repeated, "")
        self.assertEqual(events[-1][1]["reason"], "leo_echo")

    def test_stt_echo_filter_keeps_short_and_normal_child_answers(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        speech._last_leo_text = "Wil je iets veranderen? Zeg maar ja of nee."

        for transcript in ("ja", "nee", "pizza", "dat klopt niet", "gym en rekenen"):
            self.assertEqual(speech._clean_stt_transcript(transcript), transcript)

        speech._last_leo_text = "Wat is jouw lievelingseten?"
        self.assertEqual(
            speech._clean_stt_transcript("mijn lievelingseten is pizza"),
            "mijn lievelingseten is pizza",
        )

    def test_listen_clears_spinner_and_collapses_repeated_no_loop(self):
        events = []
        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            log_event_fn=lambda event_type, **data: events.append((event_type, data)),
        )
        speech._use_keyboard_input = lambda: False
        recorder = FakeSTTRecorder("Nee, nee, nee, nee, nee, nee, nee, nee, nee, nee.")
        halo = recorder.halo
        speech._recorder = recorder

        with patch("builtins.print"):
            transcript = speech.listen()

        self.assertEqual(transcript, "nee")
        self.assertTrue(recorder.queue_cleared)
        self.assertEqual(recorder.microphone_states[-1], False)
        self.assertTrue(halo.stopped)
        self.assertIsNone(recorder.halo)
        self.assertIs(recorder.spinner, False)
        self.assertIn("stt_repetition_collapsed", [event_type for event_type, _ in events])

    def test_reset_stt_audio_state_drains_realtimestt_carryover(self):
        events = []
        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            log_event_fn=lambda event_type, **data: events.append((event_type, data)),
        )
        recorder = FakeSTTRecorder()
        recorder.audio_queue.put(b"old raw")
        recorder.recorded_audio_queue.put({"frames": [b"old complete"]})
        recorder.frames = [b"frame"]
        recorder.last_frames = [b"last"]
        recorder.audio_buffer = [b"buffer"]
        recorder.audio_buffer_metadata = [{"old": True}]
        recorder.last_words_buffer = ["oude", "woorden"]
        recorder.text_storage = ["old text"]
        recorder.audio = b"cached"
        recorder.last_transcription_bytes = b"cached"
        recorder.last_transcription_bytes_b64 = "cached"
        recorder.last_transcription_metadata = {"old": True}
        recorder.last_preroll_selection = object()
        recorder._pending_preroll_selection = object()
        recorder.realtime_stabilized_text = "old"
        recorder.realtime_stabilized_safetext = "old"
        recorder.continuous_listening = True
        recorder.start_recording_on_voice_activity = True
        recorder.stop_recording_on_voice_deactivity = True
        recorder.is_recording = True
        recorder.is_webrtc_speech_active = True
        recorder.is_silero_speech_active = True
        recorder.wakeword_detected = True
        recorder.listen_start = 123
        recorder.recording_start_time = 123
        recorder.recording_start_monotonic = 123
        recorder.recording_stop_time = 124
        recorder.last_recording_start_time = 123
        recorder.last_recording_stop_time = 124
        recorder.backdate_stop_seconds = 1.0
        recorder.backdate_resume_seconds = 1.0
        recorder.speech_end_silence_start = 123
        recorder.speech_end_silence_candidate_start = 123
        recorder.wake_word_detect_time = 123
        recorder.silero_check_time = 123
        recorder.start_recording_event.set()
        recorder.stop_recording_event.set()
        speech._recorder = recorder

        speech._reset_stt_audio_state("unit_test")

        self.assertTrue(recorder.audio_queue.empty())
        self.assertTrue(recorder.recorded_audio_queue.empty())
        for attr in (
            "frames",
            "last_frames",
            "audio_buffer",
            "audio_buffer_metadata",
            "last_words_buffer",
            "text_storage",
        ):
            self.assertEqual(getattr(recorder, attr), [])
        for attr in (
            "audio",
            "last_transcription_bytes",
            "last_transcription_bytes_b64",
            "last_transcription_metadata",
            "last_preroll_selection",
            "_pending_preroll_selection",
        ):
            self.assertIsNone(getattr(recorder, attr))
        self.assertEqual(recorder.realtime_stabilized_text, "")
        self.assertEqual(recorder.realtime_stabilized_safetext, "")
        for attr in (
            "continuous_listening",
            "start_recording_on_voice_activity",
            "stop_recording_on_voice_deactivity",
            "is_recording",
            "is_webrtc_speech_active",
            "is_silero_speech_active",
            "wakeword_detected",
        ):
            self.assertIs(getattr(recorder, attr), False)
        for attr in (
            "listen_start",
            "recording_start_time",
            "recording_start_monotonic",
            "recording_stop_time",
            "last_recording_start_time",
            "last_recording_stop_time",
            "speech_end_silence_start",
            "speech_end_silence_candidate_start",
            "wake_word_detect_time",
            "silero_check_time",
        ):
            self.assertEqual(getattr(recorder, attr), 0)
        self.assertEqual(recorder.backdate_stop_seconds, 0.0)
        self.assertEqual(recorder.backdate_resume_seconds, 0.0)
        self.assertFalse(recorder.start_recording_event.is_set())
        self.assertFalse(recorder.stop_recording_event.is_set())
        self.assertEqual(recorder.microphone_states[-1], False)
        self.assertIn("stt_audio_reset", [event_type for event_type, _ in events])

    def test_listen_resets_before_mic_opens_and_mutes_before_printing_child(self):
        events = []

        class EventRecorder(FakeSTTRecorder):
            def set_microphone(self, enabled):
                super().set_microphone(enabled)
                events.append(("mic", enabled))

            def clear_audio_queue(self):
                super().clear_audio_queue()
                events.append(("clear_queue", None))

            def text(self):
                events.append(("text", self.text_value))
                return super().text()

        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        speech._use_keyboard_input = lambda: False
        speech._recorder = EventRecorder("ja")

        with patch("builtins.print", side_effect=lambda *_args, **_kwargs: events.append(("print", None))):
            transcript = speech.listen()

        self.assertEqual(transcript, "ja")
        self.assertLess(events.index(("clear_queue", None)), events.index(("mic", True)))
        self.assertLess(events.index(("mic", True)), events.index(("text", "ja")))
        text_index = events.index(("text", "ja"))
        first_print = next(i for i, event in enumerate(events) if event[0] == "print")
        self.assertTrue(
            any(event == ("mic", False) for event in events[text_index:first_print])
        )

    def test_listen_rejects_stale_audio_from_before_current_turn(self):
        events = []

        class StaleRecorder(FakeSTTRecorder):
            def text(self):
                self.last_recording_start_time = time.time() - 10
                self.recording_start_time = self.last_recording_start_time
                return super().text()

        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            log_event_fn=lambda event_type, **data: events.append((event_type, data)),
        )
        speech._use_keyboard_input = lambda: False
        speech._recorder = StaleRecorder("oude zin")

        with patch("builtins.print") as print_mock:
            transcript = speech.listen()

        self.assertEqual(transcript, "")
        self.assertIn("stt_rejected", [event_type for event_type, _ in events])
        self.assertEqual(events[-1][1]["reason"], "stale_audio")
        self.assertFalse(
            any("CHILD:" in str(call) for call in print_mock.call_args_list)
        )

    def test_listen_keeps_current_turn_audio(self):
        class CurrentRecorder(FakeSTTRecorder):
            def text(self):
                self.last_recording_start_time = time.time()
                self.recording_start_time = self.last_recording_start_time
                return super().text()

        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)
        speech._use_keyboard_input = lambda: False
        speech._recorder = CurrentRecorder("nieuw antwoord")

        with patch("builtins.print"):
            transcript = speech.listen()

        self.assertEqual(transcript, "nieuw antwoord")

    def test_listen_timeout_aborts_mutes_and_returns_empty(self):
        events = []
        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            stt_timeout=0.01,
            log_event_fn=lambda event_type, **data: events.append((event_type, data)),
        )
        speech._use_keyboard_input = lambda: False
        recorder = FakeSTTRecorder("late text", sleep_seconds=0.2)
        speech._recorder = recorder

        with patch("builtins.print"):
            transcript = speech.listen()

        self.assertEqual(transcript, "")
        self.assertTrue(recorder.aborted)
        self.assertEqual(recorder.microphone_states[-1], False)
        self.assertIn("stt_timeout", [event_type for event_type, _ in events])

    def test_stt_transcript_cleanup_keeps_normal_short_answers(self):
        speech = cri_module.SpeechIO(use_keyboard_input_fn=lambda: True)

        self.assertEqual(
            speech._clean_stt_transcript("Nee, nee, nee, nee, nee, nee, nee."),
            "nee",
        )
        self.assertEqual(
            speech._clean_stt_transcript("dat klopt niet, dat klopt niet, dat klopt niet"),
            "dat klopt niet",
        )
        self.assertEqual(speech._clean_stt_transcript("nee hoor"), "nee hoor")
        self.assertEqual(speech._clean_stt_transcript("gym en rekenen"), "gym en rekenen")
        self.assertEqual(speech._clean_stt_transcript("ik weet het niet"), "ik weet het niet")

    def test_pl_config_applies_stt_and_review_settings(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_pl_stt_config_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        config_path = Path(temp_dir) / "test_config.pl"
        config_path.write_text(
            "\n".join([
                "userId('710').",
                "localVariable(first_name_cri, \"Stef\").",
                "localVariable(first_name_tablet, \"Stef\").",
                "localVariable(operator_name, \"Julianna\").",
                "localVariable(nao_ip, \"169.254.248.247\").",
                "condition(experimental).",
                "continueSession(false).",
                "sttTimeout(20).",
                "sttPhraseLimit(18).",
                "reviewTranscripts(true).",
            ]),
            encoding="utf-8",
        )

        parsed = app.session_setup._parse_pl_config(str(config_path))
        app.apply_session_config({
            "child_id": parsed["child_id"],
            "condition": parsed["condition"],
            "stt_timeout": parsed["stt_timeout"],
            "stt_phrase_limit": parsed["stt_phrase_limit"],
            "review_transcripts": parsed["review_transcripts"],
            "start_phase_index": 0,
        })

        self.assertEqual(parsed["stt_timeout"], 20)
        self.assertEqual(parsed["stt_phrase_limit"], 18)
        self.assertIs(parsed["review_transcripts"], True)
        self.assertEqual(app.STT_TIMEOUT, 20)
        self.assertEqual(app.STT_PHRASE_LIMIT, 18)
        self.assertIs(app.REVIEW_TRANSCRIPTS, True)

    def test_speech_output_strips_emoji_before_nao_or_terminal(self):
        spoken = []
        speech = cri_module.SpeechIO(
            use_keyboard_input_fn=lambda: True,
            use_desktop_mic=True,
            log_event_fn=lambda event_type, **data: spoken.append(data.get("text", "")),
            set_last_utterance_fn=spoken.append,
        )

        with patch("builtins.print"):
            speech.say("Hallo \U0001F999!")

        self.assertEqual(spoken[0], "Hallo!")
        self.assertEqual(spoken[1], "Hallo!")


if __name__ == "__main__":
    unittest.main()
