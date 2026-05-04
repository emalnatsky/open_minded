"""
Test suite for the Dutch stub intent classifier.

Run with:
    pytest test_stub_intent_classifier.py -v

All test utterances are in Dutch to match real child speech.
Intent naming convention (v3.0.0):
    um_*        — touches GraphDB (read or write)
    dialogue_*  — conversation only, no database operation
"""

import pytest
from intent_classifier import (
    StubIntentClassifier,
    IntentResult,
    DialogueManager,
)


@pytest.fixture
def clf():
    return StubIntentClassifier()

@pytest.fixture
def manager():
    return DialogueManager()


# ===========================================================================
# 1. INTENT detection — Dutch utterances
# ===========================================================================

class TestIntentDetection:

    # ── um_add ──────────────────────────────────────────────────────────────
    def test_add_lievelingseten(self, clf):
        assert clf.classify("Mijn lievelingseten is pizza").intent == "um_add"

    def test_add_ik_heb(self, clf):
        assert clf.classify("Ik heb een kat").intent == "um_add"

    def test_add_ik_speel(self, clf):
        assert clf.classify("Ik speel gitaar").intent == "um_add"

    def test_add_ik_wil_later(self, clf):
        assert clf.classify("Ik wil later dokter worden").intent == "um_add"

    def test_add_ik_hou_van(self, clf):
        assert clf.classify("Ik hou van voetbal").intent == "um_add"

    def test_add_mijn_lievelings(self, clf):
        assert clf.classify("Mijn lievelingssport is zwemmen").intent == "um_add"

    # ── um_update ────────────────────────────────────────────────────────────
    def test_update_eigenlijk(self, clf):
        assert clf.classify("Eigenlijk is mijn lievelingseten sushi niet pizza").intent == "um_update"

    def test_update_klopt_niet(self, clf):
        assert clf.classify("Dat klopt niet, ik vind tekenen leuker").intent == "um_update"

    def test_update_niet_meer(self, clf):
        assert clf.classify("Niet meer pizza, nu is het sushi").intent == "um_update"

    # ── um_delete ────────────────────────────────────────────────────────────
    def test_delete_vergeet(self, clf):
        assert clf.classify("Vergeet wat ik zei over mijn huisdier").intent == "um_delete"

    def test_delete_verwijder(self, clf):
        assert clf.classify("Verwijder mijn lievelingseten").intent == "um_delete"

    def test_delete_wis(self, clf):
        assert clf.classify("Wis mijn favoriete sport").intent == "um_delete"

    # ── um_inspect ───────────────────────────────────────────────────────────
    def test_inspect_wat_weet_je(self, clf):
        assert clf.classify("Wat weet je over mijn hobby?").intent == "um_inspect"

    def test_inspect_weet_je_nog(self, clf):
        assert clf.classify("Weet je nog wat mijn lievelingseten is?").intent == "um_inspect"

    def test_inspect_vertel_me(self, clf):
        assert clf.classify("Vertel me wat je weet over mijn sport").intent == "um_inspect"

    # ── dialogue_update ──────────────────────────────────────────────────────
    def test_dialogue_update_nee_wacht(self, clf):
        assert clf.classify("Pizza... nee wacht, ik bedoel sushi").intent == "dialogue_update"

    def test_dialogue_update_laat_maar(self, clf):
        assert clf.classify("Nee toch, laat maar").intent == "dialogue_update"

    def test_dialogue_update_ik_bedoel(self, clf):
        assert clf.classify("Ik bedoel eigenlijk tekenen").intent == "dialogue_update"

    # ── dialogue_question ────────────────────────────────────────────────────
    def test_question_waarom(self, clf):
        assert clf.classify("Waarom wil je dat weten?").intent == "dialogue_question"

    def test_question_ben_jij(self, clf):
        assert clf.classify("Ben jij echt een robot?").intent == "dialogue_question"

    # ── dialogue_social ──────────────────────────────────────────────────────
    def test_social_haha(self, clf):
        assert clf.classify("Haha dat is grappig").intent == "dialogue_social"

    def test_social_oke(self, clf):
        assert clf.classify("Oké!").intent == "dialogue_social"

    # ── dialogue_none ────────────────────────────────────────────────────────
    def test_none_empty(self, clf):
        assert clf.classify("").intent == "dialogue_none"

    def test_none_whitespace(self, clf):
        assert clf.classify("   ").intent == "dialogue_none"

    def test_none_gibberish(self, clf):
        assert clf.classify("bla bla bla").intent == "dialogue_none"


# ===========================================================================
# 2. FIELD detection — Dutch utterances
# ===========================================================================

class TestFieldDetection:

    def test_field_fav_food(self, clf):
        assert clf.classify("Mijn lievelingseten is pizza").field == "fav_food"

    def test_field_aspiration(self, clf):
        assert clf.classify("Ik wil later dokter worden").field == "aspiration"

    def test_field_hobby_fav(self, clf):
        assert clf.classify("Mijn lievelingshobby is tekenen").field == "hobby_fav"

    def test_field_music_instrument(self, clf):
        assert clf.classify("Ik speel gitaar").field == "music_instrument"

    def test_field_pet_type_kat(self, clf):
        assert clf.classify("Ik heb een kat").field == "pet_type"

    def test_field_pet_type_hond(self, clf):
        assert clf.classify("Mijn huisdier is een hond").field == "pet_type"

    def test_field_sports_fav(self, clf):
        assert clf.classify("Mijn lievelingssport is voetbal").field == "sports_fav"

    def test_field_fav_subject(self, clf):
        assert clf.classify("Mijn lievelingsvak is gym").field == "fav_subject"

    def test_field_school_strength(self, clf):
        assert clf.classify("Ik ben goed in tekenen").field == "school_strength"

    def test_field_school_difficulty(self, clf):
        assert clf.classify("Ik vind rekenen moeilijk").field == "school_difficulty"

    def test_field_animal_fav(self, clf):
        assert clf.classify("Mijn lievelingsdier is een panda").field == "animal_fav"

    def test_field_books_fav_title(self, clf):
        assert clf.classify("Mijn lievelingsboek is Overleven").field == "books_fav_title"

    def test_field_none_when_unknown(self, clf):
        assert clf.classify("bla bla bla").field is None

    def test_field_age(self, clf):
        r = clf.classify("Ik ben 9 jaar oud")
        assert r.field == "age"


# ===========================================================================
# 3. VALUE extraction
# ===========================================================================

class TestValueExtraction:

    def test_value_fav_food(self, clf):
        r = clf.classify("Mijn lievelingseten is pizza")
        assert r.value is not None
        assert "pizza" in r.value.lower()

    def test_value_aspiration(self, clf):
        r = clf.classify("Mijn droom is dokter worden")
        assert r.value is not None
        assert "dokter" in r.value.lower()

    def test_value_instrument(self, clf):
        r = clf.classify("Mijn instrument is gitaar")
        assert r.value is not None
        assert "gitaar" in r.value.lower()

    def test_no_value_for_inspect(self, clf):
        r = clf.classify("Wat weet je over mijn lievelingseten?")
        assert r.value is None

    def test_no_value_for_delete(self, clf):
        r = clf.classify("Vergeet mijn huisdier")
        assert r.value is None

    def test_no_value_for_none(self, clf):
        r = clf.classify("um ja")
        assert r.value is None


# ===========================================================================
# 4. JSON envelope — matches the contract
# ===========================================================================

class TestJsonEnvelope:

    REQUIRED_KEYS = {"intent", "field", "value", "confidence"}
    VALID_INTENTS = {
        "um_add", "um_update", "um_delete", "um_inspect",
        "dialogue_update", "dialogue_answer", "dialogue_question",
        "dialogue_social", "dialogue_none",
    }

    @pytest.mark.parametrize("text", [
        "Mijn lievelingseten is pizza",
        "Wat weet je over mijn hobby?",
        "Vergeet mijn huisdier",
        "Eigenlijk vind ik sushi leuker",
        "",
        "bla bla",
    ])
    def test_dict_has_required_keys(self, clf, text):
        result = clf.classify(text).to_dict()
        assert self.REQUIRED_KEYS == set(result.keys())

    @pytest.mark.parametrize("text", [
        "Mijn lievelingseten is pizza",
        "Vergeet mijn huisdier",
        "",
    ])
    def test_intent_is_valid(self, clf, text):
        assert clf.classify(text).intent in self.VALID_INTENTS

    def test_confidence_is_1_for_stub(self, clf):
        assert clf.classify("Mijn lievelingseten is pizza").confidence == 1.0

    def test_confidence_is_1_for_none(self, clf):
        assert clf.classify("").confidence == 1.0

    def test_to_dict_value_types(self, clf):
        result = clf.classify("Mijn lievelingseten is pizza").to_dict()
        assert isinstance(result["intent"], str)
        assert isinstance(result["confidence"], float)
        assert result["field"] is None or isinstance(result["field"], str)
        assert result["value"] is None or isinstance(result["value"], str)


# ===========================================================================
# 5. Schema loading
# ===========================================================================

class TestSchemaLoading:

    def test_loads_without_schema_file(self):
        """Classifier works even if schema file is missing."""
        clf = StubIntentClassifier(schema_path="nonexistent_file.json")
        result = clf.classify("Mijn lievelingseten is pizza")
        assert result.intent == "um_add"
        assert result.field  == "fav_food"

    def test_valid_fields_loaded(self, clf):
        assert "fav_food"    in clf.valid_fields
        assert "aspiration"  in clf.valid_fields
        assert "hobby_fav"   in clf.valid_fields
        assert "pet_type"    in clf.valid_fields

    def test_field_not_returned_if_not_in_schema(self):
        """If valid_fields is overridden to empty, field detection returns None."""
        clf = StubIntentClassifier()
        clf.valid_fields = []
        result = clf.classify("Mijn lievelingseten is pizza")
        assert result.field is None


# ===========================================================================
# 6. DialogueManager
# ===========================================================================

class TestDialogueManager:

    def test_add_generates_dutch_response(self, manager):
        resp = manager.handle("Mijn lievelingseten is pizza")
        assert "onthoud" in resp.lower()

    def test_update_generates_dutch_response(self, manager):
        resp = manager.handle("Eigenlijk is mijn lievelingseten sushi")
        assert "pas" in resp.lower() or "aanpas" in resp.lower()

    def test_delete_generates_dutch_response(self, manager):
        resp = manager.handle("Vergeet mijn huisdier")
        assert "vergeet" in resp.lower()

    def test_inspect_generates_dutch_response(self, manager):
        resp = manager.handle("Wat weet je over mijn lievelingseten?")
        assert isinstance(resp, str) and len(resp) > 0

    def test_none_generates_fallback(self, manager):
        resp = manager.handle("bla bla bla")
        assert "nog een keer" in resp.lower() or "begreep" in resp.lower()

    def test_custom_classifier_injection(self):
        class AlwaysAdd:
            def classify(self, text):
                return IntentResult(intent="um_add", field="fav_food", value="pizza")

        mgr  = DialogueManager(classifier=AlwaysAdd())
        resp = mgr.handle("anything")
        assert "fav_food" in resp or "onthoud" in resp


# ===========================================================================
# 7. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_mixed_case(self, clf):
        assert clf.classify("MIJN LIEVELINGSETEN IS PIZZA").intent == "um_add"

    def test_punctuation(self, clf):
        r = clf.classify("Mijn lievelingseten is pizza!")
        assert r.intent == "um_add"

    def test_delete_wins_over_add(self, clf):
        r = clf.classify("Vergeet dat ik mijn lievelingseten pizza heb gezegd")
        assert r.intent == "um_delete"

    def test_classify_never_raises(self, clf):
        for text in ["   \n\t  ", "a" * 500, "null", "None"]:
            try:
                result = clf.classify(text)
                assert isinstance(result, IntentResult)
            except Exception as e:
                pytest.fail(f"classify({text!r}) raised {e}")
