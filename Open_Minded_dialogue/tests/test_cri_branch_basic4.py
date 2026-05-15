import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


DIALOGUE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = DIALOGUE_DIR / "CRI-BRANCH-BASIC4.0.py"


def load_cri_module():
    spec = importlib.util.spec_from_file_location("cri_branch_basic4", SCRIPT_PATH)
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
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            content = payload
        else:
            content = json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


def make_app(openai_payloads=None):
    # Bypass CRI_ScriptedDialogue.__init__ and SICApplication.__new__ so tests
    # never start SIC services and each test gets an isolated object.
    app = object.__new__(CRI)
    app.logger = DummyLogger()
    app.openai_client = FakeOpenAIClient(openai_payloads or [])
    app.conversation_log = None
    app.current_turn_log = None
    app.conversation_log_started_monotonic = None
    app.session_config = {}
    app.resume_from_log_path = None
    app.resume_source_log = {}
    app.local_child_name = ""
    app.researcher_name = ""
    app.local_condition = ""
    app.start_phase_index = 0
    app.last_um_preview = {}
    app.pending_change = None
    app.corrections_seen = 0
    app.mistakes_mentioned = 0
    app.mistake_states = {}
    app.simulation_mode = False
    app.child_input_mode = "microphone"
    app.simulated_persona = {}
    app.simulated_persona_path = str(DIALOGUE_DIR / "fake_personas" / "noor_1001.json")
    app.simulated_history = []
    app.last_leo_utterance = ""
    app.current_turn_context = None
    app.phases_with_confirmed_change = set()
    app.memory_fields_mentioned_so_far = set()
    app.clf = None
    return app


def sample_um():
    unknown = CRI.UNKNOWN_VALUE
    um = {field: unknown for field in CRI.UM_FIELDS}
    um.update({
        "child_name": unknown,
        "name": unknown,
        "exposure": "returning",
        "condition": "C1",
        "age": "10",
        "hobbies": "scuba diving, tekenen",
        "hobby_fav": "scuba diving",
        "sports_enjoys": unknown,
        "sports_fav_play": unknown,
        "books_enjoys": unknown,
        "books_fav_title": "Dog Man",
        "music_enjoys": unknown,
        "animals_enjoys": unknown,
        "animal_fav": "tropical fish",
        "has_pet": unknown,
        "pet_type": unknown,
        "pet_name": "Blubby en Bluey",
        "freetime_fav": unknown,
        "fav_food": "chocolate, pasta, pizza",
        "fav_subject": "taal",
        "school_strength": "taal",
        "school_difficulty": "rekenen",
        "aspiration": "find the secret of the ocean",
        "role_model": "haar moeder",
        "interest": "dieren en natuur",
        "has_best_friend": "ja",
    })
    return um


class CRIBranchBasic4Tests(unittest.TestCase):
    def test_build_script_uses_walkthrough_sequence_and_two_mistakes(self):
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
        self.assertEqual(script[5]["mistake_wrong"], "tekenen")
        self.assertEqual(script[7]["mistake_id"], "M2")
        self.assertEqual(script[7]["mistake_field"], "fav_food")
        self.assertEqual(script[7]["mistake_wrong"], "pizza")
        self.assertEqual(script[8]["condition"], "run_if_two_mistakes_no_corrections")
        self.assertTrue(all("phase" in turn for turn in script))

    def test_part1_phases_have_explicit_content_layers_and_runtime_branches(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um

        script = app.build_script()

        self.assertEqual(script[0]["dialogue_case"], CRI.CASE_UM_TEMPLATE)
        self.assertEqual(script[1]["dialogue_case"], CRI.CASE_FULLY_SCRIPTED)
        self.assertEqual(script[3]["dialogue_case"], CRI.CASE_MIXED_SEQUENCE)
        self.assertEqual(script[3]["content_plan"]["parts"][1]["type"], CRI.CASE_LLM_PREGENERATED)
        self.assertTrue(script[4]["segments"])
        self.assertTrue(app.requires_runtime_llm(script[4]))
        self.assertTrue(script[5]["segments"][1]["skip_if_phase_confirmed_change"])

    def test_sport_topic_uses_general_do_sport_phrase(self):
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

        text = app.turn_text(app.topic1_phase_segments(topic)[0])
        followup = app.turn_text(app.topic1_phase_segments(topic)[1])
        choice_followup = app.turn_text(app.topic1_phase_segments(topic)[3])

        self.assertIn("jij aan zwemmen doet", text)
        self.assertNotIn("zwemmen speelt", text)
        self.assertEqual(followup, "Waarom zit je op zwemmen?")
        self.assertNotIn("positie", followup.lower())
        self.assertIn("bezig zijn, beter worden", choice_followup)
        self.assertNotIn("goed overspelen", choice_followup)

    def test_content_plan_renders_slot_and_pregenerated_utterance(self):
        app = make_app()
        app.last_um_preview = {
            "child_name": "Mila",
            "pregen_hobbies_bridge": "Dit kwam uit GraphDB.",
        }
        plan = app.sequence(
            app.l2_slot("Hoi {child_name}."),
            app.l2_pregen("hobbies_bridge", "Fallback.", ["hobbies"]),
        )

        self.assertEqual(app.render_content_plan(plan), "Hoi Mila. Dit kwam uit GraphDB.")

    def test_simulated_persona_preserves_pregenerated_fields(self):
        app = make_app()
        app.simulated_persona = sample_um()
        app.simulated_persona["pregen_hobbies_bridge"] = "Voorgemaakte zin."

        profile = app.simulated_um_profile()

        self.assertEqual(profile["pregen_hobbies_bridge"], "Voorgemaakte zin.")

    def test_exposure_selects_different_greetings_with_same_tutorial(self):
        app = make_app()
        new_child = sample_um()
        returning_child = sample_um()
        unclear_child = sample_um()
        new_child["exposure"] = "new"
        returning_child["exposure"] = "returning"
        unclear_child["exposure"] = CRI.UNKNOWN_VALUE
        new_child["child_name"] = "Mila"
        returning_child["child_name"] = "Mila"
        unclear_child["child_name"] = "Mila"

        new_greeting = app.greeting_text(new_child)
        returning_greeting = app.greeting_text(returning_child)
        unclear_greeting = app.greeting_text(unclear_child)

        self.assertIn("Wat leuk dat je er bent", new_greeting)
        self.assertIn("weer te zien", returning_greeting)
        self.assertIn("Wat leuk dat je er bent", unclear_greeting)
        self.assertNotEqual(new_greeting, returning_greeting)

        new_script_app = make_app()
        returning_script_app = make_app()
        new_script_app.USE_FAKE_PERSONA_UM = False
        returning_script_app.USE_FAKE_PERSONA_UM = False
        new_script_app.pull_um = lambda: new_child
        returning_script_app.pull_um = lambda: returning_child
        self.assertEqual(
            new_script_app.turn_text(new_script_app.build_script()[1]),
            returning_script_app.turn_text(returning_script_app.build_script()[1]),
        )

    def test_tutorial_condition_routes_c1_and_c2_text_from_um(self):
        app = make_app()
        c1 = sample_um()
        c2 = sample_um()
        c2["condition"] = "C2"

        c1_text = app.tutorial_text(c1)
        c2_text = app.tutorial_text(c2)

        self.assertEqual(app.tutorial_condition(c1), "C1")
        self.assertEqual(app.tutorial_condition(c2), "C2")
        self.assertIn("vertel ik wat ik mij herinner", c1_text)
        self.assertIn("geheugenboek bekijken op de tablet", c2_text)
        self.assertNotIn("vertel ik wat ik mij herinner", c2_text)

    def test_tutorial_condition_only_reads_condition_field(self):
        app = make_app()
        um = sample_um()

        um["condition"] = "C2"
        self.assertEqual(app.tutorial_condition(um), "C2")
        um["condition"] = "C1"
        self.assertEqual(app.tutorial_condition(um), "C1")

    def test_local_session_config_sets_child_name_and_condition_fallback(self):
        app = make_app()

        app.apply_session_config({
            "child_id": "1002",
            "child_name": "Mila",
            "researcher_name": "Sander",
            "condition": "2",
            "start_phase_index": 4,
        })

        self.assertEqual(app.CHILD_ID, "1002")
        self.assertEqual(app.child_display_name(sample_um()), "Mila")
        self.assertEqual(app.researcher_name, "Sander")
        self.assertEqual(app.start_phase_index, 4)
        self.assertEqual(app.local_condition, "C2")

        um_condition_wins = sample_um()
        um_condition_wins["condition"] = "C1"
        local_fallback = sample_um()
        local_fallback["condition"] = CRI.UNKNOWN_VALUE
        self.assertEqual(app.tutorial_condition(um_condition_wins), "C1")
        self.assertEqual(app.tutorial_condition(local_fallback), "C2")

    def test_new_session_interface_saves_local_config_without_teacher_name(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_session_config_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        config_path = Path(temp_dir) / "session_config.local.json"
        app.SESSION_CONFIG_PATH = str(config_path)
        app.SIMULATED_PERSONA_DIR = temp_dir
        app.check_child_in_um_api = lambda child_id: None
        Path(temp_dir, "mila_1005.json").write_text(
            json.dumps({
                "child_id": 1005,
                "name": "Mila",
                "exposure": "new",
                "condition": "C2",
            }),
            encoding="utf-8",
        )

        answers = iter(["1005", "Mila", "Sander", ""])
        with patch("builtins.input", lambda prompt="": next(answers)), patch("builtins.print"):
            app.run_new_session_interface()

        saved = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["child_id"], "1005")
        self.assertEqual(saved["child_name"], "Mila")
        self.assertEqual(saved["researcher_name"], "Sander")
        self.assertEqual(saved["start_phase_index"], 0)
        self.assertTrue(saved["fake_persona_path"].endswith("mila_1005.json"))
        self.assertNotIn("teacher_name", saved)
        self.assertNotIn("session_number", saved)
        self.assertNotIn("condition", saved)
        self.assertEqual(app.CHILD_ID, "1005")
        self.assertEqual(app.start_phase_index, 0)

    def test_condition_mismatch_alerts_but_um_condition_wins(self):
        app = make_app()
        app.apply_session_config({"condition": "C2"})
        um = sample_um()
        um["condition"] = "C1"

        with patch("builtins.input", return_value=""), patch("builtins.print") as printed:
            app.alert_condition_mismatch(um)

        printed_text = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("CONDITION MISMATCH", printed_text)
        self.assertEqual(app.tutorial_condition(um), "C1")

    def test_resume_log_restarts_active_phase_and_restores_memory_access_state(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_resume_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        log_path = Path(temp_dir) / "Noor.json"
        log_path.write_text(
            json.dumps({
                "child_id": "1001",
                "child_name": "Noor",
                "child_input_mode": "keyboard",
                "tutorial_condition": "C2",
                "session_config": {
                    "child_id": "1001",
                    "child_name": "Noor",
                    "researcher_name": "Sander",
                    "condition": "C2",
                },
                "mistakes_mentioned": 1,
                "corrections_seen": 0,
                "mistake_states": {"M1": {"wrong_value_rejected": True}},
                "phases_with_confirmed_change": [3],
                "memory_fields_mentioned_so_far": ["hobby_fav", "pet_name", "sports_fav_play"],
                "events": [
                    {"timestamp": 0.0, "type": "phase_start", "phase": 1, "name": "Greeting"},
                    {"timestamp": 0.5, "type": "utterance", "speaker": "LEO", "text": "Hoi Noor."},
                    {"timestamp": 1.2, "type": "utterance", "speaker": "CHILD", "text": "Ja."},
                    {"timestamp": 2.0, "type": "phase_end", "phase": 1},
                    {"timestamp": 3.0, "type": "phase_start", "phase": 6, "name": "Mistake 1"},
                ],
            }),
            encoding="utf-8",
        )

        with patch("builtins.input", return_value=""), patch("builtins.print") as printed:
            app.run_resume_session_interface(f'"{log_path}"')

        self.assertEqual(app.CHILD_ID, "1001")
        self.assertEqual(app.resume_from_log_path, str(log_path))
        self.assertEqual(app.local_child_name, "Noor")
        self.assertEqual(app.researcher_name, "Sander")
        self.assertEqual(app.local_condition, "C2")
        self.assertEqual(app.start_phase_index, 5)
        self.assertEqual(app.mistakes_mentioned, 1)
        self.assertEqual(app.mistake_states["M1"]["wrong_value_rejected"], True)
        self.assertEqual(app.phases_with_confirmed_change, {3})
        self.assertEqual(app.memory_fields_mentioned_so_far, {"hobby_fav", "pet_name", "sports_fav_play"})
        printed_text = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertNotIn("PREVIOUS CONVERSATION FROM LOG", printed_text)

        with patch("builtins.print") as replay_printed:
            app.print_resume_context_before_interaction()

        replay_text = "\n".join(str(call.args[0]) for call in replay_printed.call_args_list if call.args)
        self.assertIn("PREVIOUS CONVERSATION FROM LOG", replay_text)
        self.assertIn("LEO: Hoi Noor.", replay_text)
        self.assertIn("CHILD: Ja.", replay_text)

    def test_clean_pasted_path_accepts_common_quoted_windows_paths(self):
        app = make_app()
        path = r"C:\Users\Sander\dev\open_minded\Open_Minded_dialogue\conversations\Noor.json"

        self.assertEqual(app.clean_pasted_path(f'"{path}"'), path)
        self.assertEqual(app.clean_pasted_path(f"'{path}'"), path)
        self.assertEqual(app.clean_pasted_path(f"“{path}”"), path)
        self.assertEqual(app.clean_pasted_path(f"<{path}>"), path)

    def test_session_mode_accepts_pasted_resume_json_path_directly(self):
        app = make_app()
        app.ASK_SESSION_INTERFACE_AT_START = True
        app.run_new_session_interface = lambda: self.fail("Should resume instead of starting a new session")
        called = []
        resume_path = r"C:\Users\Sander\dev\open_minded\Open_Minded_dialogue\conversations\Noor.json"
        app.run_resume_session_interface = lambda path="": called.append(path)

        with patch.dict("os.environ", {}, clear=True), patch("builtins.input", return_value=f'"{resume_path}"'), patch("builtins.print"):
            app.configure_session_interface()

        self.assertEqual(called, [f'"{resume_path}"'])

    def test_resume_phase_moves_to_next_after_completed_phase(self):
        app = make_app()
        log = {
            "events": [
                {"type": "phase_start", "phase": 4},
                {"type": "phase_end", "phase": 4},
            ]
        }

        self.assertEqual(app.compute_resume_phase_from_log(log), 5)

    def test_script_memory_table_marks_used_fields_and_mistakes(self):
        app = make_app()
        app.USE_FAKE_PERSONA_UM = False
        app.pull_um = sample_um
        script = app.build_script()

        rows = {
            row["field"]: row
            for row in app.script_memory_table(script, sample_um())
        }

        self.assertEqual(rows["hobby_fav"]["true_value"], "scuba diving")
        self.assertIn("tekenen", rows["hobby_fav"]["script_value"])
        self.assertEqual(rows["hobby_fav"]["mistake"], "M1 related-but-wrong")
        self.assertNotIn("spt", rows["hobby_fav"])
        self.assertNotIn("used", rows["hobby_fav"])
        self.assertEqual(rows["fav_food"]["true_value"], "chocolate, pasta, pizza")
        self.assertIn("pizza", rows["fav_food"]["script_value"])
        self.assertEqual(rows["fav_food"]["mistake"], "M2 completely-wrong")

    def test_fake_persona_script_plan_drives_mistakes_and_start_table(self):
        app = make_app()

        script = app.build_script()
        rows = {
            row["field"]: row
            for row in app.script_memory_table(script, app.last_um_preview)
        }

        self.assertEqual(script[5]["mistake_wrong"], "tuinieren")
        self.assertEqual(script[7]["mistake_wrong"], "pizza")
        self.assertEqual(rows["hobby_fav"]["true_value"], "tekenen")
        self.assertEqual(rows["hobby_fav"]["script_value"], "tuinieren")
        self.assertEqual(rows["hobby_fav"]["mistake"], "M1 related-but-wrong")
        self.assertEqual(rows["fav_food"]["script_value"], "pizza")
        self.assertEqual(rows["fav_food"]["mistake"], "M2 completely-wrong")
        self.assertIn("books_fav_title", rows)
        self.assertEqual(rows["books_fav_title"]["true_value"], "De Gorgels")
        self.assertEqual(rows["books_fav_title"]["script_value"], "-")
        self.assertEqual(rows["books_fav_title"]["mistake"], "-")

    def test_related_wrong_hobby_value_never_reuses_true_favorite(self):
        app = make_app()
        um = sample_um()
        um["hobby_fav"] = "tekenen"
        um["hobbies"] = "tekenen, tuinieren, lego bouwen"

        wrong = app.related_wrong_hobby_value(um)

        self.assertNotEqual(wrong.lower(), "tekenen")
        self.assertNotIn("tekenen", wrong.lower())

    def test_should_skip_conditional_correction_space_after_confirmed_change(self):
        app = make_app()
        turn = {
            "condition": "skip_if_change_after_phase",
            "condition_phase": 6,
        }

        self.assertFalse(app.should_skip_phase(turn))
        app.phases_with_confirmed_change.add(6)
        self.assertTrue(app.should_skip_phase(turn))

    def test_nudge_only_runs_after_two_mistakes_and_no_corrections(self):
        app = make_app()
        turn = {"condition": "run_if_two_mistakes_no_corrections"}

        self.assertTrue(app.should_skip_phase(turn))
        app.mistakes_mentioned = 2
        self.assertFalse(app.should_skip_phase(turn))
        app.corrections_seen = 1
        self.assertTrue(app.should_skip_phase(turn))

    def test_mistake_acceptance_continues_to_wrong_value_followup_without_speaking(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "response_mode": "mistake_interpretation",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "bakken",
            "mistake_type": "related-but-wrong",
            "mistake_topic": {
                "fields": ["hobby_fav"],
                "field_labels": {"hobby_fav": "je favoriete hobby"},
                "current_values": {"hobby_fav": "tekenen"},
            },
        }

        result = app.action_handler(
            IntentResult("dialogue_answer", None, "oké", 0.8),
            "Oké.",
            turn,
        )

        self.assertEqual(result["action"], "continue_wrong_value_followup")
        self.assertFalse(result["follow_up_needed"])
        self.assertEqual(spoken, [])

    def test_final_nudge_phase_does_not_create_second_explicit_nudge(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.mistake_states = {
            "M1": {
                "id": "M1",
                "mentioned": True,
                "corrected": False,
                "nudge_count": 0,
                "field": "hobby_fav",
                "field_label": "je favoriete hobby",
                "wrong": "bakken",
            }
        }
        turn = {
            "phase": 9,
            "name": "Nudge",
            "condition": "run_if_two_mistakes_no_corrections",
            "response_mode": "topic_interpretation",
            "topic": {"label": "mijn geheugen over jou"},
        }

        result = app.action_handler(
            IntentResult("dialogue_answer", None, "ik weet het niet", 0.8),
            "Ik weet het niet.",
            turn,
        )

        self.assertNotIn(result["action"], ("gentle_mistake_nudge", "explicit_mistake_nudge"))
        self.assertEqual(result["action"], "no_memory_change")

    def test_topic_candidate_and_deliberate_mistake_topic_are_different(self):
        app = make_app()
        topics = app.topic_candidates(sample_um())
        discussed = next(topic for topic in topics if topic["domain"] == "huisdier")

        mistake_topic = app.select_deliberate_mistake_topic(sample_um(), discussed)

        self.assertNotEqual(app.topic_key(discussed), app.topic_key(mistake_topic))
        field, actual = app.preferred_memory_item(mistake_topic)
        self.assertTrue(field)
        self.assertTrue(app.is_known(actual))

    def test_preferred_memory_item_skips_booleanish_fields(self):
        app = make_app()
        topic = app.topic_candidate(
            domain="huisdier",
            label="Bluey",
            fields=["has_pet", "pet_name"],
            field_labels={"has_pet": "of je een huisdier hebt", "pet_name": "de naam van je huisdier"},
            current_values={"has_pet": "ja", "pet_name": "Bluey"},
            correct_values=[],
            memory_link="",
            options=[],
            reground="",
        )

        self.assertEqual(app.preferred_memory_item(topic), ("pet_name", "Bluey"))

    def test_mistake_correction_question_is_field_specific(self):
        app = make_app()
        turn = {
            "mistake_field": "pet_name",
            "mistake_topic": {
                "field_labels": {"pet_name": "de naam van je huisdier"}
            },
        }

        self.assertEqual(
            app.mistake_correction_question(turn),
            "Oeps, dan had ik dat verkeerd. Hoe heet je huisdier?",
        )

    def test_mistake_rejection_uses_field_specific_followup(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        turn = {
            "response_mode": "mistake_interpretation",
            "mistake_field": "pet_name",
            "mistake_actual": "Blubby en Bluey",
            "mistake_wrong": "Fluffy en Spot",
            "mistake_topic": {
                "domain": "huisdier",
                "label": "Blubby en Bluey",
                "fields": ["pet_name", "pet_type", "animal_fav", "has_pet"],
                "field_labels": {"pet_name": "de naam van je huisdier"},
                "current_values": {"pet_name": "Blubby en Bluey"},
            },
        }

        result = app.action_handler(
            IntentResult("um_update", None, None, 0.8),
            "Dat zijn mijn huisdieren helemaal niet.",
            turn,
        )

        self.assertEqual(result["action"], "ask_correction_detail")
        self.assertTrue(result["follow_up_needed"])
        self.assertEqual(spoken[-1], "Oeps, dan had ik dat verkeerd. Hoe heet je huisdier?")

    def test_mistake_correction_to_existing_um_value_counts_as_corrected(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.last_um_preview = {"hobby_fav": "tekenen"}
        app.current_turn_context = {
            "phase": 6,
            "mistake_id": "M1",
        }
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "response_mode": "mistake_interpretation",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "tuinieren",
            "mistake_topic": {
                "domain": "hobby",
                "label": "tekenen",
                "fields": ["hobby_fav"],
                "field_labels": {"hobby_fav": "je favoriete hobby"},
                "current_values": {"hobby_fav": "tekenen"},
            },
        }
        app.register_mistake_phase(turn)

        result = app.action_handler(
            IntentResult("um_add", "hobby_fav", "tekenen", 0.96),
            "Mijn favoriete hobby is tekenen.",
            turn,
        )

        self.assertEqual(result["action"], "mistake_corrected_no_um_change")
        self.assertEqual(spoken[-1], "O ja, dankjewel.")
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M1"]["corrected"])
        self.assertIn(6, app.phases_with_confirmed_change)

    def test_short_answer_after_mistake_rejection_uses_mistake_field(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.last_um_preview = {"hobby_fav": "tekenen"}
        app.current_turn_context = {
            "phase": 6,
            "mistake_id": "M1",
        }
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "response_mode": "mistake_interpretation",
            "mistake_field": "hobby_fav",
            "mistake_actual": "tekenen",
            "mistake_wrong": "tuinieren",
            "mistake_topic": {
                "domain": "hobby",
                "label": "tekenen",
                "fields": ["hobby_fav"],
                "field_labels": {"hobby_fav": "je favoriete hobby"},
                "current_values": {"hobby_fav": "tekenen"},
            },
        }
        app.register_mistake_phase(turn)

        rejection = app.action_handler(
            IntentResult("dialogue_none", None, None, 0.9),
            "Nee dat klopt niet.",
            turn,
        )
        correction = app.action_handler(
            IntentResult("dialogue_answer", None, "tekenen", 0.87),
            "Tekenen.",
            turn,
        )

        self.assertEqual(rejection["action"], "ask_correction_detail")
        self.assertEqual(correction["action"], "mistake_corrected_no_um_change")
        self.assertEqual(spoken[-1], "O ja, dankjewel.")
        self.assertEqual(app.corrections_seen, 1)
        self.assertTrue(app.mistake_states["M1"]["corrected"])

    def test_mistake_acceptance_continues_to_wrong_value_followup_instead_of_nudge(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        turn = {
            "phase": 6,
            "mistake_id": "M1",
            "response_mode": "mistake_interpretation",
            "mistake_field": "hobby_fav",
            "mistake_actual": "scuba diving",
            "mistake_wrong": "sneeuwpop maken",
            "mistake_topic": {
                "domain": "hobby",
                "label": "scuba diving",
                "fields": ["hobby_fav"],
                "field_labels": {"hobby_fav": "je favoriete hobby"},
                "current_values": {"hobby_fav": "scuba diving"},
            },
        }

        result = app.action_handler(
            IntentResult("dialogue_answer", None, "ja dat klopt", 1.0),
            "Ja dat klopt.",
            turn,
        )

        self.assertEqual(result["action"], "continue_wrong_value_followup")
        self.assertFalse(result["follow_up_needed"])
        self.assertEqual(spoken, [])

    def test_confirmation_text_removes_old_value_and_yes_no_instruction(self):
        app = make_app()
        change = {
            "action": "update",
            "field_label": "je lievelingseten",
            "old_value": "chocolate, pasta, pizza",
            "new_value": "pizza, pannenkoeken",
            "confirmation_question": (
                "Wil je dat ik je lievelingseten verander van chocolate, pasta, pizza "
                "naar pizza, pannenkoeken? Zeg ja of nee."
            ),
        }

        question = app.confirmation_text(change)

        self.assertEqual(
            question,
            "Wil je dat ik je lievelingseten verander naar pizza, pannenkoeken?",
        )
        self.assertNotIn("chocolate, pasta, pizza", question)
        self.assertNotIn("Zeg ja of nee", question)

    def test_confirmation_decision_uses_intent_classifier_result(self):
        app = make_app()
        change = {
            "action": "update",
            "field": "fav_food",
            "field_label": "je lievelingseten",
            "old_value": "pizza",
            "new_value": "pannenkoeken",
            "confirmation_question": "Wil je dat ik pannenkoeken onthoud als je lievelingseten?",
        }

        result = app.confirmation_decision_from_intent(
            IntentResult("dialogue_answer", None, "doe dat maar", 0.9),
            "Doe dat maar.",
            change,
        )

        self.assertEqual(result["action"], "confirm_change")
        self.assertEqual(result["change"], change)

    def test_confirm_topic_change_repeats_until_intent_decision_is_clear(self):
        app = make_app()
        spoken = []
        heard = iter(["misschien", "doe dat maar"])
        classifier_results = iter([
            IntentResult("dialogue_none", None, None, 1.0),
            IntentResult("dialogue_answer", None, "doe dat maar", 0.9),
        ])
        writes = []
        change = {
            "action": "update",
            "field": "fav_food",
            "field_label": "je lievelingseten",
            "old_value": "chocolate, pasta, pizza",
            "new_value": "pizza, pannenkoeken",
            "confirmation_question": "Wil je dat ik pizza en pannenkoeken onthoud als je lievelingseten?",
        }

        app.say = spoken.append
        app.listen_with_review = lambda: next(heard)
        app.classify_with_repeat = lambda transcript: next(classifier_results)
        app.write_um_change = lambda pending: writes.append(pending) or True

        self.assertTrue(app.confirm_topic_change(change))
        self.assertEqual(len(writes), 1)
        self.assertGreaterEqual(
            spoken.count("Wil je dat ik pizza en pannenkoeken onthoud als je lievelingseten?"),
            2,
        )
        for line in spoken:
            self.assertNotIn("chocolate, pasta, pizza", line)
            self.assertNotIn("Zeg ja of nee", line)

    def test_change_from_intent_result_validates_allowed_field_and_real_change(self):
        app = make_app()
        turn = {
            "topic": {
                "fields": ["fav_food"],
                "field_labels": {"fav_food": "je lievelingseten"},
                "current_values": {"fav_food": "pizza"},
            }
        }

        accepted = app.change_from_intent_result(
            IntentResult("um_update", "fav_food", "pannenkoeken", 0.1),
            turn,
            "Ik vind pannenkoeken lekker.",
        )
        same_value = app.change_from_intent_result(
            IntentResult("um_update", "fav_food", "pizza", 1.0),
            turn,
            "Pizza.",
        )
        wrong_field = app.change_from_intent_result(
            IntentResult("um_update", "pet_name", "Snuffie", 1.0),
            turn,
            "Mijn huisdier heet Snuffie.",
        )

        self.assertEqual(accepted["field"], "fav_food")
        self.assertEqual(accepted["new_value"], "pannenkoeken")
        self.assertEqual(same_value, {})
        self.assertEqual(wrong_field, {})

    def test_conversation_log_uses_timestamp_folder_and_child_named_files(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.CHILD_ID = "1001"
        app.last_um_preview = {"child_name": "Julianna", "age": "10"}
        script = [
            {
                "phase": 1,
                "name": "Greeting",
                "layer": "L1",
                "leo_text": "Hoi Julianna.",
                "expects_response": False,
            }
        ]

        app.start_conversation_log(script)
        app.start_turn_log(script[0])
        app.log_conversation_event("utterance", speaker="LEO", text="Hoi Julianna.")
        app.finish_turn_log()
        app.finish_conversation_log()

        folder = Path(app.conversation_log["folder"])
        self.assertRegex(
            folder.name,
            r"^Julianna_1001_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",
        )
        self.assertEqual(app.conversation_log["session_id"], folder.name)
        self.assertEqual(app.conversation_log["timestamp_unit"], "seconds_from_interaction_start")
        self.assertIsInstance(app.conversation_log["started_at"], float)
        self.assertIsInstance(app.conversation_log["ended_at"], float)
        self.assertEqual(Path(app.conversation_log["txt_path"]).name, "Julianna.txt")
        self.assertEqual(Path(app.conversation_log["json_path"]).name, "Julianna.json")
        self.assertTrue(Path(app.conversation_log["txt_path"]).exists())
        self.assertTrue(Path(app.conversation_log["json_path"]).exists())

        timestamps = [event["timestamp"] for event in app.conversation_log["events"]]
        self.assertTrue(all(isinstance(timestamp, float) for timestamp in timestamps))
        self.assertEqual(timestamps, sorted(timestamps))

    def test_resumed_conversation_log_includes_previous_conversation(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.CHILD_ID = "1001"
        app.researcher_name = "Sander"
        app.resume_from_log_path = r"C:\old\Noor.json"
        app.start_phase_index = 4
        app.last_um_preview = {"child_name": "Noor"}
        app.resume_source_log = {
            "session_id": "Noor_old",
            "turns": [{"phase": 1, "name": "Greeting", "events": []}],
            "events": [
                {"timestamp": 0.0, "type": "phase_start", "phase": 1, "name": "Greeting", "layer": "L1"},
                {"timestamp": 1.2, "type": "utterance", "speaker": "LEO", "text": "Hoi Noor."},
                {"timestamp": 3.4, "type": "phase_end", "phase": 1},
            ],
        }
        script = [
            {
                "phase": 5,
                "name": "Topic 1",
                "layer": "L2+L3",
                "leo_text": "Ik weet ook nog dat jij aan zwemmen doet.",
                "expects_response": True,
            }
        ]

        app.start_conversation_log(script)
        app.start_turn_log(script[0])

        events = app.conversation_log["events"]
        self.assertTrue(app.conversation_log["previous_log_included"])
        self.assertEqual(app.conversation_log["previous_session_id"], "Noor_old")
        self.assertEqual(events[1]["text"], "Hoi Noor.")
        boundary = next(event for event in events if event["type"] == "resume_boundary")
        self.assertGreater(boundary["timestamp"], 3.4)
        self.assertIn("Hoi Noor.", Path(app.conversation_log["txt_path"]).read_text(encoding="utf-8"))
        self.assertIn("RESUME: continuing from previous log", Path(app.conversation_log["txt_path"]).read_text(encoding="utf-8"))

    def test_intent_classifier_and_action_handler_are_logged_with_change_choice(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.last_um_preview = {"child_name": "Julianna"}
        script = [
            {
                "phase": 3,
                "name": "First richer topic domain",
                "layer": "L2+L3",
                "leo_text": "Ik denk dat Blubby een goed onderwerp is.",
                "expects_response": True,
            }
        ]
        topic = {
            "domain": "huisdier",
            "label": "Blubby",
            "fields": ["pet_name", "pet_type", "animal_fav"],
            "field_labels": {"pet_name": "de naam van je huisdier"},
            "current_values": {"pet_name": "Blubby"},
            "correct_values": ["Blubby hoort bij jou"],
            "memory_link": "Blubby belangrijk voor je is",
        }
        turn = dict(script[0], topic=topic, response_mode="topic_interpretation")
        app.say = lambda text: None
        app.listen_with_review = lambda: "ja"
        app.classify_with_repeat = lambda transcript: IntentResult("dialogue_answer", None, "ja", 0.9)
        app.write_um_change = lambda pending: True

        app.start_conversation_log(script)
        app.current_turn_context = turn
        intent_result = IntentResult("um_update", "pet_name", "Bluey", 0.88)
        app.log_intent_classifier_result("Nee, mijn vis heet Bluey.", intent_result)
        action = app.action_handler(intent_result, "Nee, mijn vis heet Bluey.", turn)
        app.log_action_handler_result(action)

        intent_events = [
            event for event in app.conversation_log["events"]
            if event["type"] == "intent_classifier"
        ]
        action_events = [
            event for event in app.conversation_log["events"]
            if event["type"] == "action_handler"
        ]
        self.assertEqual(len(intent_events), 1)
        self.assertEqual(intent_events[0]["result"]["intent"], "um_update")
        self.assertEqual(intent_events[0]["result"]["field"], "pet_name")
        self.assertEqual(intent_events[0]["result"]["value"], "Bluey")
        self.assertGreaterEqual(len(action_events), 1)
        self.assertEqual(action_events[-1]["action"], "confirm_update")
        self.assertEqual(action_events[-1]["change"]["field"], "pet_name")
        self.assertEqual(action_events[-1]["change"]["new_value"], "Bluey")

    def test_memory_access_uses_current_phase_and_previously_mentioned_fields(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.last_um_preview = {
            "hobby_fav": "scuba diving",
            "pet_name": "Blubby",
            "fav_food": "pizza",
            "animal_fav": "tropical fish",
            "age": "10",
            "hobbies": "scuba diving, tekenen",
            "sports_enjoys": "ja",
            "sports_fav_play": "zwemmen",
        }
        app.memory_fields_mentioned_so_far = {
            "hobby_fav", "animal_fav", "age", "hobbies",
            "sports_enjoys", "sports_fav_play",
        }
        turn = {
            "topic": {
                "fields": ["pet_name", "fav_food"],
                "field_labels": {"pet_name": "de naam van je huisdier"},
                "current_values": {"pet_name": "Blubby", "fav_food": "pizza"},
            },
            "used_fields": {"pet_name": "Blubby"},
        }

        action = app.action_handler(
            IntentResult("um_inspect", None, None, 0.95),
            "Wat weet je nog over mij?",
            turn,
        )

        self.assertEqual(action["action"], "memory_access")
        self.assertIn("hobby_fav", action["memory_scope"])
        self.assertIn("pet_name", action["memory_scope"])
        self.assertIn("sports_fav_play", action["memory_scope"])
        self.assertNotIn("fav_food", action["memory_scope"])
        self.assertNotIn("sports_enjoys", action["memory_scope"])
        self.assertIn("scuba diving", spoken[-1])
        self.assertIn("Blubby", spoken[-1])
        self.assertIn("tropical fish", spoken[-1])
        self.assertIn("10", spoken[-1])
        self.assertIn("zwemmen", spoken[-1])
        self.assertNotIn("pizza", spoken[-1])
        self.assertNotIn("of je over sport", spoken[-1])
        self.assertNotIn("of je sport leuk", spoken[-1])
        self.assertEqual(len(action["returned_fields"]), 6)

    def test_memory_access_scope_uses_spoken_fields_not_whole_topic(self):
        app = make_app()
        turn = {
            "topic": {
                "fields": ["sports_fav_play", "sports_enjoys"],
                "current_values": {
                    "sports_fav_play": "zwemmen",
                    "sports_enjoys": "ja",
                },
            },
            "used_fields": {"sports_fav_play": "zwemmen"},
        }

        scope = app.memory_access_scope(turn)

        self.assertIn("sports_fav_play", scope)
        self.assertNotIn("sports_enjoys", scope)

    def test_memory_access_blocks_requested_field_outside_current_scope(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.last_um_preview = {
            "hobby_fav": "scuba diving",
            "fav_food": "pizza",
        }
        app.memory_fields_mentioned_so_far = {"hobby_fav"}
        turn = {"used_fields": {"hobby_fav": "scuba diving"}}

        action = app.action_handler(
            IntentResult("um_inspect", "fav_food", None, 0.95),
            "Wat is mijn lievelingseten?",
            turn,
        )

        self.assertEqual(action["action"], "memory_access")
        self.assertEqual(action["requested_field"], "fav_food")
        self.assertNotIn("fav_food", action["memory_scope"])
        self.assertEqual(action["returned_fields"], [])
        self.assertIn("nog niet over gehad", spoken[-1])
        self.assertIn("scuba diving", spoken[-1])
        self.assertNotIn("pizza", spoken[-1])

    def test_c2_memory_access_routes_to_tablet_instead_of_verbal_memory(self):
        app = make_app()
        spoken = []
        app.say = spoken.append
        app.last_um_preview = {
            "condition": "C2",
            "hobby_fav": "scuba diving",
            "pet_name": "Blubby",
        }
        app.memory_fields_mentioned_so_far = {"hobby_fav"}
        turn = {
            "topic": {
                "fields": ["pet_name"],
                "field_labels": {"pet_name": "de naam van je huisdier"},
                "current_values": {"pet_name": "Blubby"},
            }
        }

        action = app.action_handler(
            IntentResult("um_inspect", None, None, 0.95),
            "Wat weet je nog over mij?",
            turn,
        )

        self.assertEqual(action["action"], "memory_access_tablet")
        self.assertEqual(action["tutorial_condition"], "C2")
        self.assertEqual(spoken[-1], "Je kunt mijn geheugenboek op de tablet bekijken.")
        self.assertNotIn("scuba diving", spoken[-1])
        self.assertNotIn("Blubby", spoken[-1])

    def test_memory_access_interrupt_repeats_then_resumes_same_phase(self):
        app = make_app()
        spoken = []
        heard = iter(["Kan ik je geheugen zien?", "Ja, dat klopt."])
        classified = iter([
            IntentResult("um_inspect", None, None, 0.95),
            IntentResult("dialogue_answer", None, "ja, dat klopt", 0.9),
        ])
        app.say = spoken.append
        app.listen_with_review = lambda: next(heard)
        app.classify_with_repeat = lambda transcript: next(classified)
        app.last_um_preview = {
            "condition": "C1",
            "hobby_fav": "tekenen",
            "hobbies": "tekenen, tuinieren, lego bouwen",
        }
        app.memory_fields_mentioned_so_far = {"hobby_fav"}
        turn = {
            "phase": 4,
            "name": "Correct hobby bridge",
            "layer": "L1 + L2-slot + L2-pregen",
            "content_plan": app.l1("Ik weet al dat jij van tekenen houdt."),
            "expects_response": True,
            "response_mode": "listen_only",
            "used_fields": {"hobby_fav": "tekenen"},
        }

        with patch("builtins.input", side_effect=["m", ""]):
            app.run_phase_segment(turn, turn)

        memory_replies = [reply for reply in spoken if "Ik heb vandaag al genoemd" in reply]
        self.assertEqual(len(memory_replies), 2)
        self.assertEqual(memory_replies[0], memory_replies[1])
        self.assertGreaterEqual(spoken.count("Ik weet al dat jij van tekenen houdt."), 2)

    def test_memory_access_mentioned_fields_exclude_internal_exposure(self):
        app = make_app()
        turn = {
            "used_fields": {
                "name": "Noor",
                "exposure": "returning",
                "condition": "C2",
                "age": "10",
            }
        }

        app.register_mentioned_memory_fields(turn)

        self.assertIn("name", app.memory_fields_mentioned_so_far)
        self.assertIn("age", app.memory_fields_mentioned_so_far)
        self.assertNotIn("exposure", app.memory_fields_mentioned_so_far)
        self.assertNotIn("condition", app.memory_fields_mentioned_so_far)

    def test_stub_intent_classifier_detects_general_memory_access_phrases(self):
        clf = cri_module.StubIntentClassifier(valid_fields=list(CRI.UM_FIELDS))

        for phrase in (
            "Wat heb je over mij onthouden?",
            "Wat weet je nog van mij?",
            "Wat zei je net over mij?",
        ):
            with self.subTest(phrase=phrase):
                result = clf.classify(phrase)
                self.assertEqual(result.intent, "um_inspect")
                self.assertIsNone(result.field)

    def test_simulated_persona_loads_numeric_child_id_and_all_um_fields(self):
        app = make_app()
        app.simulation_mode = True
        app.simulated_persona_path = str(DIALOGUE_DIR / "fake_personas" / "noor_1001.json")

        app.load_simulated_persona()
        profile = app.simulated_um_profile()

        self.assertEqual(app.simulated_persona["child_id"], 1001)
        self.assertEqual(profile["name"], "Noor")
        self.assertEqual(profile["exposure"], "returning")
        self.assertEqual(set(app.UM_FIELDS), set(profile.keys()))
        self.assertTrue(all(profile[field] for field in app.UM_FIELDS))

    def test_available_fake_personas_lists_child_ids_and_exposure(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_persona_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.SIMULATED_PERSONA_DIR = temp_dir
        Path(temp_dir, "noor_1001.json").write_text(
            json.dumps({"child_id": 1001, "name": "Noor", "exposure": "returning", "condition": "C1"}),
            encoding="utf-8",
        )
        Path(temp_dir, "mila_1002.json").write_text(
            json.dumps({"child_id": 1002, "name": "Mila", "exposure": "new", "condition": "C2"}),
            encoding="utf-8",
        )

        personas = app.available_fake_personas()

        self.assertEqual([persona["child_id"] for persona in personas], ["1001", "1002"])
        self.assertEqual(personas[0]["name"], "Noor")
        self.assertEqual(personas[0]["exposure"], "returning")
        self.assertEqual(personas[0]["condition"], "C1")
        self.assertEqual(personas[1]["name"], "Mila")
        self.assertEqual(personas[1]["exposure"], "new")
        self.assertEqual(personas[1]["condition"], "C2")

    def test_bundled_fake_personas_are_complete_and_balanced(self):
        app = make_app()
        personas = app.available_fake_personas()

        self.assertEqual([persona["child_id"] for persona in personas], ["1001", "1002", "1003", "1004"])
        self.assertEqual(
            {persona["child_id"]: (persona["exposure"], persona["condition"]) for persona in personas},
            {
                "1001": ("returning", "C1"),
                "1002": ("new", "C2"),
                "1003": ("new", "C1"),
                "1004": ("returning", "C2"),
            },
        )
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "C1"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["condition"] == "C2"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["exposure"] == "new"), 2)
        self.assertEqual(sum(1 for persona in personas if persona["exposure"] == "returning"), 2)

        for persona in personas:
            app.simulated_persona_path = persona["path"]
            app.simulated_persona = {}
            app.load_simulated_persona()
            for field in app.UM_FIELDS:
                self.assertIn(field, app.simulated_persona)
                self.assertTrue(app.simulated_persona[field])
            self.assertIn("script_plan", app.simulated_persona)
            self.assertIn("used_fields", app.simulated_persona["script_plan"])
            self.assertIn("mistakes", app.simulated_persona["script_plan"])
            self.assertEqual(
                {mistake["id"] for mistake in app.simulated_persona["script_plan"]["mistakes"]},
                {"M1", "M2"},
            )

    def test_configure_simulated_persona_selects_by_typed_child_id(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_persona_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.SIMULATED_PERSONA_DIR = temp_dir
        noor_path = Path(temp_dir, "noor_1001.json")
        mila_path = Path(temp_dir, "mila_1002.json")
        noor_path.write_text(
            json.dumps({"child_id": 1001, "child_name": "Noor", "exposure": "returning"}),
            encoding="utf-8",
        )
        mila_path.write_text(
            json.dumps({"child_id": 1002, "child_name": "Mila", "exposure": "new"}),
            encoding="utf-8",
        )

        with patch("builtins.input", return_value="1002"), patch("builtins.print"):
            app.configure_simulated_persona()

        self.assertEqual(Path(app.simulated_persona_path), mila_path)

    def test_configure_run_mode_offers_keyboard_without_simulation_prompt(self):
        app = make_app()
        app.ASK_RUN_MODE_AT_START = True
        app.simulation_mode = False

        with patch.dict("os.environ", {}, clear=True), patch("builtins.input", return_value="k"), patch("builtins.print") as printed:
            app.configure_run_mode()

        printed_text = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertEqual(app.child_input_mode, "keyboard")
        self.assertFalse(app.simulation_mode)
        self.assertIn("keyboard", printed_text.lower())
        self.assertNotIn("fake-child simulation", printed_text.lower())

    def test_keyboard_input_mode_returns_typed_child_response_without_review(self):
        app = make_app()
        events = []
        app.child_input_mode = "keyboard"
        app.log_conversation_event = lambda event_type, **data: events.append((event_type, data))
        app.review_transcript = lambda transcript: self.fail("Keyboard input should not open transcript review.")

        with patch("builtins.input", return_value="Nee, dat klopt niet."):
            transcript = app.listen_with_review()

        self.assertEqual(transcript, "Nee, dat klopt niet.")
        self.assertEqual(events[-1][0], "utterance")
        self.assertEqual(events[-1][1]["speaker"], "CHILD")
        self.assertEqual(events[-1][1]["input_mode"], "keyboard")

    def test_simulation_um_write_updates_fake_persona_without_graphdb(self):
        app = make_app()
        temp_dir = tempfile.mkdtemp(prefix="cri_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.simulation_mode = True
        app.simulated_persona = {"child_id": 1001, "fav_food": "pannenkoeken"}
        app.last_um_preview = {"child_name": "Noor", "fav_food": "pannenkoeken"}
        app.start_conversation_log([])

        ok = app.write_um_change(
            {
                "action": "update",
                "field": "fav_food",
                "old_value": "pannenkoeken",
                "new_value": "pizza",
            }
        )

        self.assertTrue(ok)
        self.assertEqual(app.simulated_persona["fav_food"], "pizza")
        self.assertEqual(app.last_um_preview["fav_food"], "pizza")

        write_events = [
            event for event in app.conversation_log["events"]
            if event["type"] == "um_write"
        ]
        self.assertEqual(len(write_events), 1)
        self.assertEqual(write_events[0]["status_code"], "fake_persona")

    def test_simulation_listen_uses_llm_fake_child_response(self):
        app = make_app(["Nee, mijn kat heet Momo."])
        temp_dir = tempfile.mkdtemp(prefix="cri_log_test_")
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        app.CONVERSATION_LOG_ROOT = temp_dir
        app.simulation_mode = True
        app.simulated_persona = sample_um()
        app.simulated_persona["child_id"] = 1001
        app.last_um_preview = {"child_name": "Noor"}
        app.current_turn_context = {"phase": 4, "name": "Deliberate memory mistake"}
        app.last_leo_utterance = "Ik dacht dat je huisdier Fluffy heet."
        app.start_conversation_log([])

        transcript = app.listen()

        self.assertEqual(transcript, "Nee, mijn kat heet Momo.")
        child_events = [
            event for event in app.conversation_log["events"]
            if event["type"] == "utterance" and event.get("speaker") == "CHILD"
        ]
        self.assertEqual(len(child_events), 1)
        self.assertTrue(child_events[0]["simulated"])

    def test_llm_response_uses_openai_client_when_sic_gpt_is_missing(self):
        app = make_app(["Wat grappig, lama's zijn soms heel nieuwsgierig. \U0001F999\u2728"])
        app.gpt = None

        response = app.llm_response("Wat voor gekke dingen doen ze?")

        self.assertEqual(response, "Wat grappig, lama's zijn soms heel nieuwsgierig.")
        self.assertEqual(len(app.openai_client.requests), 1)

    def test_say_strips_emoji_for_nao_tts_safety(self):
        app = make_app()
        spoken = []
        app.simulation_mode = True
        app.USE_DESKTOP_MIC = True
        app.log_conversation_event = lambda *args, **kwargs: spoken.append(kwargs.get("text", ""))

        with patch("builtins.print"):
            app.say("Hallo \U0001F999\u2728")

        self.assertEqual(app.last_leo_utterance, "Hallo ")
        self.assertEqual(spoken[-1], "Hallo ")


if __name__ == "__main__":
    unittest.main()

