import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


DIALOGUE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = DIALOGUE_DIR / "CRI-BRANCH-BASIC4.0.py"


def load_cri_module():
    spec = importlib.util.spec_from_file_location("cri_branch_basic4", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cri_module = load_cri_module()
CRI = cri_module.CRI_ScriptedDialogue


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
    app.last_um_preview = {}
    app.pending_change = None
    app.corrections_seen = 0
    app.mistakes_mentioned = 0
    return app


def sample_um():
    unknown = CRI.UNKNOWN_VALUE
    return {
        "child_name": unknown,
        "name": unknown,
        "age": "10",
        "hobbies": unknown,
        "hobby_fav": "scuba diving",
        "sports_enjoys": unknown,
        "sports_fav": unknown,
        "sports_plays": unknown,
        "sports_fav_play": unknown,
        "books_enjoys": unknown,
        "books_fav_genre": unknown,
        "books_fav_title": unknown,
        "music_enjoys": unknown,
        "music_talk": unknown,
        "music_plays_instrument": unknown,
        "music_instrument": unknown,
        "animals_enjoys": unknown,
        "animal_talk": unknown,
        "animal_fav": "tropical fish",
        "has_pet": unknown,
        "pet_type": unknown,
        "pet_name": "Blubby en Bluey",
        "freetime_fav": unknown,
        "fav_food": "chocolate, pasta, pizza",
        "aspiration": "find the secret of the ocean",
    }


class CRIBranchBasic4Tests(unittest.TestCase):
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
        app = make_app([
            {
                "response_type": "correction_unclear",
                "wrong_value_rejected": True,
                "wrong_value_accepted": False,
                "leo_response": "Wat moet ik onthouden over je huisdieren?",
                "confidence": 0.8,
                "change": {"action": "none"},
            }
        ])
        turn = {
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

        result = app.interpret_mistake_response(
            "Dat zijn mijn huisdieren helemaal niet.",
            turn,
        )

        self.assertEqual(result["response_type"], "correction_unclear")
        self.assertEqual(
            result["leo_response"],
            "Oeps, dan had ik dat verkeerd. Hoe heet je huisdier?",
        )

    def test_mistake_acceptance_uses_llm_confirmation_and_repeats_value(self):
        app = make_app([
            {
                "response_type": "no_change",
                "wrong_value_rejected": False,
                "wrong_value_accepted": True,
                "accepted_wrong_value_confirmation": "Oh leuk, zal ik dat zo onthouden?",
                "leo_response": None,
                "confidence": 1.0,
                "change": {"action": "none"},
            }
        ])
        turn = {
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

        result = app.interpret_mistake_response("Ja dat klopt.", turn)

        self.assertEqual(result["response_type"], "possible_update")
        self.assertTrue(result["wrong_value_accepted"])
        self.assertEqual(result["change"]["new_value"], "sneeuwpop maken")
        self.assertIn("sneeuwpop maken", result["change"]["confirmation_question"])
        self.assertNotIn("scuba diving", result["change"]["confirmation_question"])

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

    def test_confirmation_classifier_is_llm_driven(self):
        app = make_app([
            {
                "decision": "confirm_yes",
                "confidence": 0.2,
                "leo_response": None,
                "reason": "The LLM decides the child agreed.",
            }
        ])
        change = {
            "action": "update",
            "field": "fav_food",
            "field_label": "je lievelingseten",
            "old_value": "pizza",
            "new_value": "pannenkoeken",
            "confirmation_question": "Wil je dat ik pannenkoeken onthoud als je lievelingseten?",
        }

        result = app.interpret_confirmation_response("Doe dat maar.", change)

        self.assertEqual(result["decision"], "confirm_yes")
        self.assertEqual(result["confidence"], 0.2)

    def test_confirm_topic_change_repeats_until_llm_decision_is_clear(self):
        app = make_app()
        spoken = []
        heard = iter(["misschien", "doe dat maar"])
        decisions = iter([
            {
                "decision": "unclear",
                "leo_response": "Wil je dat ik pizza en pannenkoeken onthoud?",
                "confidence": 0.4,
            },
            {
                "decision": "confirm_yes",
                "leo_response": None,
                "confidence": 0.9,
            },
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
        app.interpret_confirmation_response = lambda transcript, pending: next(decisions)
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

    def test_normalize_topic_change_validates_allowed_field_and_real_change(self):
        app = make_app()
        topic = {
            "fields": ["fav_food"],
            "field_labels": {"fav_food": "je lievelingseten"},
            "current_values": {"fav_food": "pizza"},
        }

        accepted = app.normalize_topic_change(
            {
                "response_type": "possible_update",
                "change": {
                    "action": "update",
                    "field": "fav_food",
                    "old_value": "pizza",
                    "new_value": "pannenkoeken",
                    "confidence": 0.1,
                },
            },
            "Ik vind pannenkoeken lekker.",
            topic,
        )
        same_value = app.normalize_topic_change(
            {
                "change": {
                    "action": "update",
                    "field": "fav_food",
                    "old_value": "pizza",
                    "new_value": "pizza",
                    "confidence": 1.0,
                },
            },
            "Pizza.",
            topic,
        )
        wrong_field = app.normalize_topic_change(
            {
                "change": {
                    "action": "update",
                    "field": "pet_name",
                    "old_value": "Bluey",
                    "new_value": "Snuffie",
                    "confidence": 1.0,
                },
            },
            "Mijn huisdier heet Snuffie.",
            topic,
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
        app.last_um_preview = {"child_name": "Julianna", "age": "10"}
        script = [
            {
                "step": 1,
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
        self.assertTrue(folder.name.startswith("20"))
        self.assertEqual(Path(app.conversation_log["txt_path"]).name, "Julianna.txt")
        self.assertEqual(Path(app.conversation_log["json_path"]).name, "Julianna.json")
        self.assertTrue(Path(app.conversation_log["txt_path"]).exists())
        self.assertTrue(Path(app.conversation_log["json_path"]).exists())


if __name__ == "__main__":
    unittest.main()
