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
    app.memory_review_requested = False
    app.last_cri_scenario = {}
    app.last_cri_scenario_loaded = False

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

    def test_tutorial_condition_can_fall_back_to_local_config(self):
        app = make_app()
        app.local_condition = "E"
        um = sample_um()
        um["condition"] = ""

        self.assertEqual(app.tutorial_condition(um), "E")
        self.assertIn("geheugenboek op de tablet bekijken", app.tutorial_text(um))

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
                    "not_corrected": "[STUB] Wat is jouw allerliefste game?"
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
            app.pregenerated_utterance("m1_wrong_followup", "fallback"),
            "Wat is jouw allerliefste game?",
        )

        mistake = app.script_plan_mistake("M1")
        self.assertEqual(mistake["field"], "hobby_fav")
        self.assertEqual(mistake["type"], "related-but-wrong")
        self.assertEqual(mistake["wrong_value"], "gamen")

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
                "Mistake 4 - aspiration",
                "Space for correction + personalized reflection",
                "Explicit memory inspection",
                "Memory review / co-construction",
                "Closing",
            ],
        )
        self.assertEqual([turn["phase"] for turn in script], list(range(1, 22)))
        self.assertEqual(
            [turn["phase_id"] for turn in script],
            [f"1.{phase}" for phase in range(1, 10)]
            + ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"],
        )
        self.assertEqual(script[5]["part"], 1)
        self.assertEqual(script[5]["phase_id"], "1.6")
        self.assertEqual(script[5]["script_phase"], "part1_mistake1")
        self.assertEqual(script[9]["part"], 2)
        self.assertEqual(script[9]["phase_id"], "2.1")
        self.assertEqual(script[9]["script_phase"], "part2_school_joke_transition")
        self.assertEqual(script[9]["segments"][0]["response_mode"], "listen_only")
        self.assertFalse(script[9]["segments"][1]["expects_response"])
        self.assertEqual(script[9]["layer"], "L1")
        self.assertIn(
            "Zeg... als we het toch over jouw hobby's en lievelingsdingen hebben",
            app.turn_text(script[9]["segments"][0]),
        )
        phase22 = script[10]
        self.assertEqual(phase22["part"], 2)
        self.assertEqual(phase22["phase_id"], "2.2")
        self.assertEqual(phase22["script_phase"], "part2_robot_school_self_disclosure")
        self.assertEqual(phase22["segments"][0]["response_mode"], "acknowledge")
        self.assertTrue(phase22["segments"][0]["llm_turn"])
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
        self.assertEqual(len(phase23["segments"]), 4)
        self.assertIn("Ik weet ook nog dat natuur jouw lievelingsvak is.", app.turn_text(phase23["segments"][0]))
        self.assertEqual(phase23["segments"][0]["response_mode"], "listen_only")
        self.assertEqual(phase23["segments"][1]["response_mode"], "listen_only")
        self.assertEqual(phase23["segments"][2]["response_mode"], "listen_only")
        self.assertFalse(phase23["segments"][3]["expects_response"])
        phase24 = script[12]
        self.assertEqual(phase24["part"], 2)
        self.assertEqual(phase24["phase_id"], "2.4")
        self.assertEqual(phase24["script_phase"], "part2_mistake3_school_strength")
        self.assertEqual(phase24["mistake_id"], "M3")
        self.assertEqual(phase24["mistake_field"], "school_strength")
        self.assertEqual(len(phase24["segments"]), 4)
        self.assertEqual(phase24["segments"][0]["response_mode"], "mistake_interpretation")
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
        self.assertEqual(phase34["phase_id"], "3.4")
        self.assertEqual(phase34["script_phase"], "part3_mistake4_aspiration")
        self.assertEqual(phase34["mistake_id"], "M4")
        self.assertEqual(phase34["mistake_field"], "aspiration")
        self.assertEqual(phase34["mistake_actual"], "dierenarts worden")
        self.assertEqual(len(phase34["segments"]), 4)
        self.assertEqual(phase34["segments"][0]["response_mode"], "listen_only")
        self.assertFalse(phase34["segments"][1]["expects_response"])
        self.assertEqual(phase34["segments"][2]["response_mode"], "mistake_interpretation")
        self.assertTrue(phase34["segments"][3]["run_if_phase_confirmed_change"])
        self.assertIn("Snap jij een beetje wat ik bedoel?", app.turn_text(phase34["segments"][0]))
        self.assertIn("En volgens mij wil jij later", app.turn_text(phase34["segments"][2]))
        phase35 = script[17]
        self.assertEqual(phase35["part"], 3)
        self.assertEqual(phase35["phase_id"], "3.5")
        self.assertEqual(phase35["script_phase"], "part3_aspiration_reflection")
        self.assertEqual(len(phase35["segments"]), 7)
        self.assertTrue(phase35["segments"][0]["run_if_phase_confirmed_change"])
        self.assertEqual(phase35["segments"][0]["condition_phase"], 17)
        self.assertTrue(phase35["segments"][1]["run_if_phase_confirmed_change"])
        self.assertEqual(phase35["segments"][1]["condition_phase"], 17)
        self.assertTrue(phase35["segments"][2]["skip_if_phase_confirmed_change"])
        self.assertEqual(phase35["segments"][2]["condition_phase"], 17)
        self.assertEqual(phase35["segments"][4]["response_mode"], "middle_school_feeling")
        self.assertFalse(phase35["segments"][6]["expects_response"])
        self.assertIn("Wat lijkt jou daar dan zo leuk aan?", app.turn_text(phase35["segments"][0]))
        self.assertIn("middelbare school", app.turn_text(phase35["segments"][3]))
        self.assertIn("goed kijken of ik nu alles goed over jou heb onthouden", app.turn_text(phase35["segments"][6]))
        phase36 = script[18]
        self.assertEqual(phase36["part"], 3)
        self.assertEqual(phase36["phase_id"], "3.6")
        self.assertEqual(phase36["script_phase"], "part3_explicit_memory_inspection")
        self.assertEqual(phase36["response_mode"], "explicit_memory_inspection_offer")
        self.assertEqual(phase36["used_fields"], {})
        self.assertEqual(
            app.turn_text(phase36),
            "Wil je misschien zien wat ik allemaal over jou onthoud?",
        )
        phase37 = script[19]
        self.assertEqual(phase37["part"], 3)
        self.assertEqual(phase37["phase_id"], "3.7")
        self.assertEqual(phase37["script_phase"], "part3_memory_review_co_construction")
        self.assertEqual(phase37["condition"], "run_if_memory_review_requested")
        self.assertEqual(phase37["segments"][0]["expects_response"], False)
        self.assertEqual(phase37["segments"][1]["response_mode"], "memory_review_group")
        self.assertEqual(phase37["segments"][-1]["response_mode"], "memory_review_add_final")
        self.assertIn("Klopt dat een beetje", app.turn_text(phase37["segments"][1]))
        self.assertIn("wat ik nog niet heb onthouden", app.turn_text(phase37["segments"][-1]))
        phase38 = script[20]
        self.assertEqual(phase38["part"], 3)
        self.assertEqual(phase38["phase_id"], "3.8")
        self.assertEqual(phase38["script_phase"], "part3_closing")
        self.assertEqual(phase38["layer"], "L1 + L2-slot: first_name")
        self.assertEqual(len(phase38["segments"]), 2)
        self.assertFalse(phase38["segments"][0]["expects_response"])
        self.assertFalse(phase38["segments"][1]["expects_response"])
        self.assertIn("Dank je wel dat je met mij hebt gepraat.", app.turn_text(phase38["segments"][0]))
        self.assertEqual(app.turn_text(phase38["segments"][1]), "Tot de volgende keer, Sam.")
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
        self.assertEqual(script[5]["mistake_id"], "M1")
        self.assertEqual(script[5]["mistake_field"], "hobby_fav")
        self.assertEqual(script[7]["mistake_id"], "M2")
        self.assertEqual(script[7]["mistake_field"], "fav_food")
        self.assertEqual(script[8]["condition"], "run_if_two_mistakes_no_corrections")

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
        self.assertEqual(segments[0]["response_mode"], "listen_only")
        self.assertIn("Als ik het goed heb", app.turn_text(segments[0]))
        self.assertIn("Klopt dat een beetje?", app.turn_text(segments[0]))
        self.assertEqual(segments[1]["response_mode"], "acknowledge")
        self.assertTrue(segments[1]["llm_turn"])
        self.assertIn("Wie is dat voor jou?", app.turn_text(segments[1]))
        self.assertEqual(segments[1]["l3"]["script_phase"], "part3_rolemodel")
        self.assertEqual(segments[1]["l3"]["topic"], "rolemodel")
        self.assertEqual(segments[1]["l3"]["response_function"], "wrap_up")
        self.assertEqual(segments[1]["l3"]["relevant_um_fields"], [])
        self.assertEqual(
            segments[1]["l3"]["fallback"],
            "Dat snap ik. Soms leer je van verschillende mensen iets.",
        )

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

        self.assertEqual(phase34["phase_id"], "3.4")
        self.assertEqual(phase34["script_phase"], "part3_mistake4_aspiration")
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

        self.assertEqual(phase34["phase_id"], "3.4")
        self.assertEqual(phase34["mistake_actual"], CRI.UNKNOWN_VALUE)
        self.assertEqual(phase34["mistake_wrong"], "architect worden")
        self.assertEqual(
            app.turn_text(segments[2]),
            "En volgens mij wil jij later architect worden.",
        )
        self.assertEqual(segments[2]["used_fields"], {"aspiration": "architect worden"})
        self.assertEqual(phase34["mistake_topic"]["memory_link"], "wat je later wilt worden")

    def test_part3_phase35_uses_postcorrection_reflection_scenario(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {
            "utterances": {
                "p3_m4_postcorrection_reflection": {
                    "default": "[STUB] Dat past echt bij jou. Wat lijkt jou daar het mooiste aan?"
                },
            },
            "mistakes": [],
        }

        script = app.build_script()
        phase35 = script[17]
        segments = phase35["segments"]

        self.assertEqual(phase35["phase_id"], "3.5")
        self.assertEqual(phase35["script_phase"], "part3_aspiration_reflection")
        self.assertEqual(len(segments), 7)
        self.assertTrue(segments[0]["run_if_phase_confirmed_change"])
        self.assertEqual(segments[0]["condition_phase"], 17)
        self.assertEqual(segments[0]["response_mode"], "listen_only")
        self.assertEqual(app.turn_text(segments[1]), "Dat past echt bij jou. Wat lijkt jou daar het mooiste aan?")
        self.assertTrue(segments[1]["run_if_phase_confirmed_change"])
        self.assertEqual(segments[1]["condition_phase"], 17)
        self.assertEqual(segments[1]["used_fields"]["aspiration"], "dierenarts worden")
        self.assertIn("dieren", segments[1]["used_fields"]["interest"])
        self.assertTrue(segments[2]["skip_if_phase_confirmed_change"])
        self.assertEqual(segments[4]["response_mode"], "middle_school_feeling")
        self.assertFalse(segments[6]["expects_response"])

    def test_part3_phase35_fallback_reflection_uses_profile_cues(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        app.last_cri_scenario_loaded = True
        app.last_cri_scenario = {"utterances": {}, "mistakes": []}

        script = app.build_script()
        reflection_text = app.turn_text(script[17]["segments"][1])

        self.assertIn("Dat past ook wel mooi bij jou", reflection_text)
        self.assertIn("dierenarts bij jou past", reflection_text)
        self.assertIn("Wat lijkt jou daar het mooiste aan?", reflection_text)

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
        self.assertIn("Gym en rekenen, dat snap ik wel.", app.turn_text(phase23["segments"][0]))
        self.assertEqual(app.turn_text(phase23["segments"][1]), "Dat past bij jou.")

    def test_part2_mistake3_uses_scenario_and_school_l3_branches(self):
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
                    "corrected": "[STUB] Taal! Wat vind jij het leukste aan taal?"
                },
                "p2_school_wrap_after_difficulty": {
                    "not_corrected": "[STUB] Dat snap ik wel. School kan soms plakken."
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
        self.assertIn("Oeps, goed dat je het zegt.", app.turn_text(segments[1]))
        self.assertIn("Taal! Wat vind jij het leukste aan taal?", app.turn_text(segments[1]))
        self.assertIn("Taal vond ik altijd al moeilijk", app.turn_text(segments[2]))
        self.assertEqual(segments[2]["response_mode"], "acknowledge")
        self.assertEqual(segments[2]["l3"]["relevant_um_fields"], [])
        self.assertEqual(segments[2]["l3"]["fallback"], "Dat snap ik wel. School kan soms plakken.")
        self.assertIn("natuur jouw lievelingsvak is", app.turn_text(segments[3]))
        self.assertIn("rekenen voor jou soms wat lastiger voelt", app.turn_text(segments[3]))
        self.assertEqual(segments[3]["l3"]["relevant_um_fields"], ["fav_subject", "school_difficulty"])
        self.assertEqual(app.mistake_correction_question(phase24), "Oeps, waar ben jij dan vooral goed in op school?")

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
                    "not_corrected": "[STUB] Wat vind jij het leukste om te bakken?"
                },
                "p1_followup_postcorrection_true_hobby": {
                    "corrected": "[STUB] Dansen! Wat voor dans doe je?"
                },
            },
        }

        script = app.build_script()
        phase6 = script[5]
        segments = phase6["segments"]

        self.assertEqual(phase6["phase_id"], "1.6")
        self.assertEqual(phase6["mistake_field"], "hobby_fav")
        self.assertEqual(phase6["mistake_wrong"], "bakken")
        self.assertEqual(len(segments), 4)
        self.assertEqual(
            app.turn_text(segments[0]),
            "En volgens mij is bakken jouw allerliefste hobby.",
        )
        self.assertIn("Dat snap ik trouwens wel.", app.turn_text(segments[1]))
        self.assertIn("Iets maken dat ook nog lekker ruikt", app.turn_text(segments[1]))
        self.assertIn("Oeps, dan had ik dat verkeerd.", app.turn_text(segments[2]))
        self.assertIn("Wat vind jij het leukste aan tekenen?", app.turn_text(segments[2]))
        self.assertNotIn("Dansen", app.turn_text(segments[2]))
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
        self.assertIn("Oeps, dan had ik dat verkeerd.", text)
        self.assertIn("Wat vind jij het leukste aan tekenen?", text)
        self.assertNotIn("dansen", text.lower())

    def test_mistake2_uses_cri_scenario_wrong_and_corrected_branches(self):
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
                    "not_corrected": "[STUB] Pizza is rond en warm. Wat vind jij daar lekker aan?"
                },
                "p1_m2_postcorrection_true_food": {
                    "corrected": "[STUB] Pannenkoeken klinken eerlijk gezegd ook meteen gezellig."
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
        self.assertIn("Dat is op zich wel een sterke keuze.", app.turn_text(segments[1]))
        self.assertIn("Pizza is rond en warm.", app.turn_text(segments[1]))
        self.assertIn("Oeps, goed dat je het zegt. Dan pas ik het aan.", app.turn_text(segments[2]))
        self.assertIn("Pannenkoeken klinken eerlijk gezegd", app.turn_text(segments[2]))
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
            app.turn_text(script[6]["segments"][0]),
            "Ik weet ook nog dat jij een kat hebt die Momo heet.",
        )

    def test_topic2_animals_uses_open_followup_then_topic_closing(self):
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

        self.assertEqual(len(segments), 3)
        self.assertFalse(segments[0]["expects_response"])
        self.assertTrue(segments[1]["expects_response"])
        self.assertFalse(segments[2]["expects_response"])
        self.assertEqual(app.turn_text(segments[0]), "Ik weet ook nog dat jij een kat hebt die Luna heet.")
        self.assertEqual(app.turn_text(segments[1]), "Wat voor kat is Luna eigenlijk?")
        self.assertIn("Ik vind dieren altijd fascinerend", app.turn_text(segments[2]))

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
        self.assertIn("Script phase: part1_topic1", request["messages"][1]["content"])
        self.assertIn("Next scripted Leo line", request["messages"][1]["content"])
        self.assertIn('"sports_fav_play": "hockey"', request["messages"][1]["content"])

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

        self.assertIn("Ik heb vandaag al gebruikt", response)
        self.assertIn("je Noor heet", response)
        self.assertIn("je houdt van tekenen, tuinieren, lego bouwen", response)
        self.assertIn("tekenen jouw favoriete hobby is", response)
        self.assertIn("Over sport weet ik dat je zwemmen doet", response)
        self.assertNotIn("of je sport leuk vindt", response)
        self.assertNotIn("condition", scope)
        self.assertNotIn("exposure", scope)
        self.assertNotIn("sports_enjoys", returned)

    def test_explicit_memory_inspection_accepts_and_defers_to_phase_37(self):
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
                "role_model",
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
        self.assertEqual(
            app.speech.spoken,
            ["Goed. Dan gaan we zo rustig samen door mijn geheugen heen."],
        )

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
        self.assertEqual(set(action["memory_scope"]), {"name", "hobbies", "fav_food", "aspiration"})
        self.assertEqual(
            app.speech.spoken,
            ["Goed. Dan kunnen we zo samen op de tablet naar mijn geheugenboek kijken."],
        )

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

    def test_memory_review_group_response_modes_confirm_repeat_and_ask_detail(self):
        app = make_app()
        turn = {
            "phase": 20,
            "phase_id": "3.7",
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
            "phase": 20,
            "phase_id": "3.7",
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

    def test_memory_review_phase_skips_unless_offer_was_accepted_and_can_activate_tablet(self):
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
        phase = {"phase": 20, "phase_id": "3.7", "name": "Memory review / co-construction", "layer": "L1"}
        segment = {
            "content_plan": app.l1("Kijk maar op de tablet."),
            "expects_response": False,
            "activate_tablet_memory_access": True,
            "memory_review_fields": ["hobbies", "fav_food", "aspiration"],
        }

        app.run_phase_segment(phase, segment)

        self.assertEqual(captured["phase"], 20)
        self.assertEqual(captured["fields"], ["hobbies", "fav_food", "aspiration"])
        self.assertEqual(app.speech.spoken, ["Kijk maar op de tablet."])

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
        self.assertEqual(set(action["visible_fields"]), {"hobbies", "fav_food", "fav_subject"})
        self.assertEqual(set(captured["fields"]), {"hobbies", "fav_food", "fav_subject"})
        self.assertEqual(captured["phase"], 12)

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
        self.assertEqual(state["visible_fields"], ["name", "fav_food", "aspiration"])
        self.assertIn("eten", state["unlocked_categories"])
        self.assertIn("aspiratie", state["unlocked_categories"])
        self.assertTrue(state["mistakes"]["M4"]["corrected"])
        self.assertEqual(state["mistakes"]["M4"]["wrong"], "juf worden")

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

    def test_deferred_mistake_correction_marks_phase_without_generic_reply(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "response_mode": "mistake_interpretation",
            "mistake_topic": app.hobby_mistake_topic(app.last_um_preview),
            "defer_corrected_response": True,
        }
        app.current_turn_context = turn
        result = IntentResult(intent="um_update", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Nee, tekenen", turn)

        self.assertEqual(action["action"], "mistake_corrected_no_um_change")
        self.assertIsNone(action["leo_response"])
        self.assertEqual(app.speech.spoken, [])
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
        result = IntentResult(intent="um_add", field="hobby_fav", value="tekenen", confidence=0.96)

        action = app.action_handler(result, "Mijn favoriete hobby is tekenen", turn)

        self.assertEqual(action["action"], "mistake_corrected_no_um_change")
        self.assertIsNone(action["leo_response"])
        self.assertEqual(action["change"]["old_value"], "tekenen")
        self.assertEqual(action["change"]["new_value"], "tekenen")
        self.assertEqual(app.speech.spoken, [])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake_one_rejection_then_bare_value_does_not_ask_confirmation(self):
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

        self.assertEqual(second_action["action"], "mistake_corrected_no_um_change")
        self.assertIsNone(second_action["leo_response"])
        self.assertEqual(app.speech.spoken, [])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

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

    def test_explicit_mistake_correction_answer_updates_directly_when_actual_is_missing(self):
        app = make_app()
        app.last_um_preview = sample_um()
        app.last_um_preview["hobby_fav"] = CRI.UNKNOWN_VALUE
        app.simulated_persona = {}
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "mistake_field": "hobby_fav",
            "mistake_wrong": "zingen",
            "response_mode": "mistake_interpretation",
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

        self.assertEqual(action["action"], "mistake_corrected_update")
        self.assertTrue(action["write_success"])
        self.assertIsNone(action["leo_response"])
        self.assertEqual(app.simulated_persona["hobby_fav"], "tekenen")
        self.assertEqual(app.speech.spoken, [])
        self.assertIn(6, app.phases_with_confirmed_change)
        self.assertTrue(app.mistake_states["M1"].get("corrected"))

    def test_mistake_rejection_without_value_asks_for_detail_then_accepts_short_answer(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 8,
            "mistake_id": "M2",
            "mistake_field": "fav_food",
            "response_mode": "mistake_interpretation",
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

        self.assertEqual(second_action["action"], "mistake_corrected_no_um_change")
        self.assertIsNone(second_action["leo_response"])
        self.assertEqual(second_action["change"]["new_value"], "pannenkoeken")
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M2"].get("corrected"))
        self.assertIn(8, app.phases_with_confirmed_change)
        self.assertEqual(app.speech.spoken, [])

    def test_aspiration_inline_correction_matches_stored_worden_value(self):
        app = make_app()
        app.last_um_preview = sample_um()
        turn = {
            "phase": 17,
            "mistake_id": "M4",
            "mistake_field": "aspiration",
            "response_mode": "mistake_interpretation",
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

        self.assertEqual(action["action"], "mistake_corrected_no_um_change")
        self.assertIsNone(action["leo_response"])
        self.assertEqual(action["change"]["new_value"], "dierenarts")
        self.assertEqual(app.speech.spoken, [])
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

    def test_middle_school_feeling_is_static_session_context_not_um_change(self):
        app = make_app()
        turn = {
            "phase": 18,
            "phase_id": "3.5",
            "response_mode": "middle_school_feeling",
        }
        result = IntentResult(intent="dialogue_answer", field=None, value=None, confidence=0.9)

        action = app.action_handler(result, "Ik heb er zin in, maar het is ook spannend.", turn)

        self.assertEqual(action["action"], "middle_school_feeling")
        self.assertEqual(action["middle_school_feeling"], "mixed")
        self.assertEqual(app.speech.spoken, [])

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
