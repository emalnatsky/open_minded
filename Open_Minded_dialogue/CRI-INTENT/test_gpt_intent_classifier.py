import pytest
import json
from unittest.mock import MagicMock, patch
from gpt_intent_classifier import GPTIntentClassifier, REPEAT_SENTINEL
from stub_intent_classifier import IntentResult

# --- Test Data ---

NINE_INTENTS = [
    ("Mijn lievelingseten is pizza", "um_add", "fav_food", "pizza"),
    ("Nee, liever sushi", "um_update", "fav_food", "sushi"),
    ("Vergeet mijn hobby", "um_delete", "hobby_fav", None),
    ("Wat weet je van mijn school?", "um_inspect", "fav_subject", None),
    ("Dat is niet waar", "dialogue_update", None, None),
    ("Ja hoor", "dialogue_answer", None, None),
    ("Hoe oud ben jij?", "dialogue_question", None, None),
    ("Haha wat grappig", "dialogue_social", None, None),
    ("Ik weet het niet", "dialogue_none", None, None),
]

# --- Fixtures ---

@pytest.fixture
def mock_openai():
    with patch("gpt_intent_classifier.OpenAI") as mock:
        yield mock

@pytest.fixture
def classifier(mock_openai, tmp_path):
    # Create temporary schema and contract files for initialization
    schema_file = tmp_path / "um_field_schema.json"
    
    schema_file.write_text(json.dumps({"valid_fields": ["fav_food", "hobby_fav", "aspiration", "fav_subject"]}))
    
    contract_file = tmp_path / "intent_classification_contract.json"
    contract_file.write_text(json.dumps({
        "gpt_classifier": {
            "system_prompt": "test",
            "output_format": "test",
            "few_shot_examples": []
        }
    }))
    
    # We also need to mock the external call to _load_valid_fields to ensure consistency
    with patch("gpt_intent_classifier._load_valid_fields", return_value=["fav_food", "hobby_fav", "aspiration", "fav_subject"]):
        clf = GPTIntentClassifier(
            openai_key="test-key",
            schema_path=str(schema_file),
            contract_path=str(contract_file)
        )
        yield clf

# --- Tests ---

@pytest.mark.parametrize("text, expected_intent, expected_field, expected_value", NINE_INTENTS)
def test_all_intents_success(classifier, text, expected_intent, expected_field, expected_value):
    """Test that all 9 intents are correctly parsed when confidence is high."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({
        "intent": expected_intent,
        "field": expected_field,
        "value": expected_value,
        "confidence": 0.95
    })
    classifier.client.chat.completions.create.return_value = mock_resp
    
    result = classifier.classify(text)
    
    assert result.intent == expected_intent
    assert result.field == expected_field
    assert result.value == expected_value
    assert result.confidence == 0.95

def test_low_confidence_first_attempt(classifier):
    """Test that confidence < 0.7 triggers REPEAT_SENTINEL on classify()."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({
        "intent": "um_add",
        "field": "fav_food",
        "value": "kaas",
        "confidence": 0.5
    })
    classifier.client.chat.completions.create.return_value = mock_resp
    
    result = classifier.classify("Ik hou van kaas")
    
    assert result.intent == REPEAT_SENTINEL
    assert result.confidence == 0.5

def test_low_confidence_retry_fallback(classifier):
    """Test that classify_retry() falls back to stub on second low confidence."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({
        "intent": "um_add",
        "field": "fav_food",
        "value": "kaas",
        "confidence": 0.4
    })
    classifier.client.chat.completions.create.return_value = mock_resp
    
    # ADDED 'None, None' here to satisfy the required arguments
    classifier._stub.classify = MagicMock(return_value=IntentResult("dialogue_none", None, None, 0.4))
    
    result = classifier.classify_retry("Ik zei kaas!")
    
    assert result.intent == "dialogue_none"
    classifier._stub.classify.assert_called_once_with("Ik zei kaas!")

def test_hard_failure_immediate_fallback(classifier):
    """Test that an API Exception triggers an immediate fallback to the Stub."""
    classifier.client.chat.completions.create.side_effect = Exception("API Error")
    
    # Mock the stub return
    classifier._stub.classify = MagicMock(return_value=IntentResult("um_add", "fav_food", "pizza"))
    
    result = classifier.classify("Ik wil pizza")
    
    assert result.intent == "um_add"
    assert result.field == "fav_food"
    classifier._stub.classify.assert_called_once_with("Ik wil pizza")

def test_invalid_json_fallback(classifier):
    """Test that malformed GPT output triggers the Stub fallback."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Not a JSON string"
    classifier.client.chat.completions.create.return_value = mock_resp
    
    # ADDED 'None, None' here
    classifier._stub.classify = MagicMock(return_value=IntentResult("dialogue_none", None, None))
    
    result = classifier.classify("... uhm ...")
    
    assert result.intent == "dialogue_none"
    classifier._stub.classify.assert_called_once()
