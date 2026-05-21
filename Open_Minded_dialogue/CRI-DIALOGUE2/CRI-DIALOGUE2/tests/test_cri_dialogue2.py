import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUTER_DIR = PACKAGE_DIR.parent
LOCAL_DIR = OUTER_DIR / "_local"
SCRIPT_PATH = PACKAGE_DIR / "CRI-BRANCH-BASIC4_0.py"
LAUNCHER_PATH = LOCAL_DIR / "launchers" / "run_cri_dialogue2.py"


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


cri_module = load_cri_module()
CRI = cri_module.CRI_ScriptedDialogue
IntentResult = cri_module.IntentResult


class DummyLogger:
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
    app.tablet_state = SimpleNamespace(reset=lambda: None, update=lambda turn: None)

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

    app.SIMULATED_PERSONA_DIR = str(PACKAGE_DIR / "fake_personas")
    app.SIMULATION_WRITE_PERSONA_FILE = False
    app.USE_FAKE_PERSONA_UM = True
    app.CONVERSATION_LOG_ENABLED = True
    app.CONVERSATION_LOG_ROOT = str(LOCAL_DIR / "conversations")
    app.SESSION_CONFIG_PATH = str(LOCAL_DIR / "session_config.local.json")
    app.SESSION_STATE_PATH = str(LOCAL_DIR / "session_state.json")
    app.ROSTER_PATH = str(LOCAL_DIR / "test_config.txt")

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
        "condition": "C1",
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
        self.assertTrue((PACKAGE_DIR / ".example_env").exists())
        self.assertFalse((PACKAGE_DIR / ".env").exists())
        self.assertFalse((PACKAGE_DIR / "conversations").exists())

    def test_fake_personas_are_bundled_balanced_and_complete(self):
        app = make_app()
        personas = app.available_fake_personas()

        self.assertEqual([persona["child_id"] for persona in personas], ["1001", "1002", "1003", "1004"])
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "C1"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "C2"), 2)
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
        self.assertEqual(app.session_condition_from_um("1002"), "C2")

    def test_greeting_and_tutorial_route_by_exposure_and_condition(self):
        app = make_app()
        returning_c1 = {"name": "Noor", "exposure": "returning", "condition": "C1"}
        new_c2 = {"name": "Mila", "exposure": "new", "condition": "C2"}

        self.assertIn("weer te zien", app.greeting_text(returning_c1))
        self.assertIn("Volgens mijn geheugen heet jij Mila", app.greeting_text(new_c2))
        self.assertIn("vertel ik wat ik mij herinner", app.tutorial_text(returning_c1))
        self.assertIn("geheugenboek bekijken op de tablet", app.tutorial_text(new_c2))

    def test_build_script_uses_part1_phase_sequence_and_mistakes(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um

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
            ],
        )
        self.assertEqual([turn["phase"] for turn in script], list(range(1, 10)))
        self.assertEqual(script[5]["mistake_id"], "M1")
        self.assertEqual(script[5]["mistake_field"], "hobby_fav")
        self.assertEqual(script[7]["mistake_id"], "M2")
        self.assertEqual(script[7]["mistake_field"], "fav_food")
        self.assertEqual(script[8]["condition"], "run_if_two_mistakes_no_corrections")

    def test_topic_sport_segments_use_general_wording(self):
        app = make_app()
        topic = app.topic_candidate(
            domain="sport",
            label="zwemmen",
            fields=["sports_fav_play"],
            field_labels={"sports_fav_play": "de sport die je graag doet"},
            current_values={"sports_fav_play": "zwemmen"},
            correct_values=["je iets met zwemmen hebt"],
            memory_link="zwemmen iets is waar jij iets mee hebt",
            options=["zwemmen", "sport"],
            reground="Ik houd goed vast dat zwemmen iets is waar jij iets mee hebt.",
        )

        segments = app.topic1_phase_segments(topic)

        self.assertIn("jij aan zwemmen doet", app.turn_text(segments[0]))
        self.assertEqual(app.turn_text(segments[1]), "Waarom zit je op zwemmen?")
        self.assertIn("bezig zijn, beter worden", app.turn_text(segments[3]))
        self.assertNotIn("zwemmen speelt", app.turn_text(segments[0]))
        self.assertNotIn("goed overspelen", app.turn_text(segments[3]))

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

        self.assertIn("je naam: Noor", response)
        self.assertIn("je hobby's: tekenen, tuinieren, lego bouwen", response)
        self.assertIn("de sport die je graag doet: zwemmen", response)
        self.assertNotIn("of je sport leuk vindt", response)
        self.assertNotIn("condition", scope)
        self.assertNotIn("exposure", scope)
        self.assertNotIn("sports_enjoys", returned)

    def test_action_handler_logs_already_correct_mistake_without_um_write(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.mistake_states = {"M1": {"id": "M1", "wrong_value_rejected": True}}
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
        }
        app.current_turn_context = turn
        result = IntentResult(intent="um_add", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Mijn favoriete hobby is tekenen", turn)

        self.assertEqual(action["action"], "mistake_corrected_no_um_change")
        self.assertEqual(action["leo_response"], "O ja, dankjewel.")
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))
        self.assertIn("O ja, dankjewel.", app.speech.spoken)

    def test_stub_classifier_detects_memory_access_phrases(self):
        clf = cri_module.StubIntentClassifier(valid_fields=list(CRI.UM_FIELDS))

        for phrase in (
            "Wat heb je over mij onthouden?",
            "Kan ik je geheugen zien?",
            "Wat weet je nog van mij?",
        ):
            with self.subTest(phrase=phrase):
                result = clf.classify(phrase)
                self.assertEqual(result.intent, "um_inspect")

    def test_conversation_logs_write_to_local_folder_and_keep_previous_log(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_dialogue2_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.CHILD_ID = "1001"
        app.local_child_name = "Noor"
        app.researcher_name = "Sander"
        app.last_um_preview = sample_um()
        app.resume_from_log_path = r"C:\previous\Noor.json"
        app.resume_source_log = {
            "session_id": "old",
            "events": [{"type": "utterance", "speaker": "LEO", "text": "Hoi", "timestamp": 1.0}],
            "turns": [{"phase": 1, "started_at": 1.0, "ended_at": 2.0}],
            "last_completed_phase": 1,
        }

        app.start_conversation_log([{"phase": 2, "name": "Tutorial", "layer": "L1", "content_plan": app.l1("Test")}])

        log = app.conversation_log
        self.assertTrue(Path(log["folder"]).is_relative_to(Path(temp_dir)))
        self.assertEqual(Path(log["json_path"]).name, "Noor.json")
        self.assertEqual(Path(log["txt_path"]).name, "Noor.txt")
        self.assertTrue(log["previous_log_included"])
        self.assertEqual(log["events"][0]["text"], "Hoi")
        self.assertEqual(log["timestamp_unit"], "seconds_from_interaction_start")

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

    def test_speech_output_strips_emoji_before_nao_or_terminal(self):
        spoken = []
        speech = cri_module.SpeechIO(
            use_desktop_mic=True,
            log_event_fn=lambda event_type, **data: spoken.append(data.get("text", "")),
            set_last_utterance_fn=spoken.append,
        )

        with patch("builtins.print"):
            speech.say("Hallo \U0001F999!")

        self.assertEqual(spoken[0], "Hallo !")
        self.assertEqual(spoken[1], "Hallo !")


if __name__ == "__main__":
    unittest.main()
