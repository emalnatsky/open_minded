import os
import time
import json
import random
import requests
import unicodedata
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoWakeUpRequest,
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)
from sic_framework.services.openai_whisper_stt.whisper_stt import (
    GetTranscript,
    SICWhisper,
    WhisperConf,
)
from sic_framework.services.llm import GPT, GPTConf, GPTRequest
from openai import OpenAI

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))


# Embedded CRI intent classifier.
#
# These classes used to live in Open_Minded_dialogue/CRI-INTENT. They are kept
# here so CRI-BRANCH-BASIC4.0.py is self-contained at runtime: the classifier
# reports the child's intent, and the ActionHandler below decides what to do.

CLASSIFIER_LOGGER = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.7
GPT_INTENT_MODEL = "gpt-4o-mini"
GPT_INTENT_MAX_TOKENS = 120
GPT_INTENT_TEMPERATURE = 0.0
REPEAT_SENTINEL = "dialogue_repeat"

VALID_INTENTS = [
    "um_add", "um_update", "um_delete", "um_inspect",
    "dialogue_update", "dialogue_answer", "dialogue_question",
    "dialogue_social", "dialogue_none",
]

EMBEDDED_VALID_FIELDS = [
    "child_name", "name", "exposure", "hobbies",
    "condition",
    "hobby_1", "hobby_2", "hobby_3", "hobby_4", "hobby_fav",
    "sports_enjoys", "sports_talk", "sports_fav", "sports_plays", "sports_fav_play",
    "books_enjoys", "books_talk", "books_fav_genre", "books_fav_title",
    "music_enjoys", "music_talk", "music_plays_instrument", "music_instrument",
    "animals_enjoys", "animal_talk", "animal_fav",
    "has_pet", "pet_talk", "pet_type", "pet_name",
    "freetime_fav", "fav_food", "fav_subject",
    "school_strength", "school_difficulty",
    "interest", "aspiration", "role_model", "has_best_friend",
    "age",
]

FIELD_ALIASES = {
    "hobby_fav": ["lievelingshobby", "favoriete hobby", "liefste hobby", "leukste hobby", "mijn favoriete hobby"],
    "hobby_1": ["hobby 1", "eerste hobby"],
    "hobby_2": ["hobby 2", "tweede hobby"],
    "hobby_3": ["hobby 3", "derde hobby"],
    "hobby_4": ["hobby 4", "vierde hobby"],
    "hobbies": ["hobby's", "hobbies", "wat vind je leuk"],
    "sports_enjoys": ["sport leuk", "van sport", "sportief", "hou je van sport", "vind je sport"],
    "sports_talk": ["over sport praten", "sport praten"],
    "sports_fav": ["lievelingssport", "favoriete sport", "leukste sport", "mijn lievelingssport"],
    "sports_plays": ["zelf sport", "doe je een sport", "speel je sport"],
    "sports_fav_play": ["sport doe je zelf", "liefste sport om te doen", "sport speel je zelf"],
    "music_enjoys": ["muziek leuk", "van muziek", "muziek mooi", "houd je van muziek", "ik hou van muziek"],
    "music_talk": ["over muziek praten", "muziek praten"],
    "music_plays_instrument": ["instrument speel", "speel je een instrument", "ik speel een instrument"],
    "music_instrument": ["welk instrument", "instrument", "gitaar", "piano", "viool", "drums", "fluit", "cello", "trompet", "saxofoon", "ukelele", "keyboard"],
    "books_enjoys": ["lezen leuk", "graag lees", "lees je graag", "boeken leuk", "ik lees graag"],
    "books_talk": ["over boeken praten", "boeken praten"],
    "books_fav_genre": ["soort boeken", "genre", "boeken genre", "soort boek", "avontuur boek", "fantasy boek"],
    "books_fav_title": ["lievelingsboek", "favoriete boek", "boek heet", "boek is", "welk boek", "mijn lievelingsboek"],
    "freetime_fav": ["vrije tijd", "als ik vrij ben", "liefst doen", "doe je liefst", "gamen", "buiten spelen", "knutselen", "dansen", "puzzelen", "filmpjes kijken"],
    "has_best_friend": ["beste vriend", "bff", "beste vriendin", "vrienden"],
    "animals_enjoys": ["dieren leuk", "van dieren", "houd je van dieren", "ik hou van dieren"],
    "animal_talk": ["over dieren praten", "dieren praten"],
    "animal_fav": ["lievelingsdier", "favoriete dier", "leukste dier", "mijn lievelingsdier"],
    "pet_type": ["soort huisdier", "wat voor huisdier", "mijn huisdier is een", "mijn huisdier is", "hond", "kat", "konijn", "hamster", "vogel", "vis", "reptiel"],
    "pet_name": ["naam huisdier", "hoe heet je huisdier", "huisdier heet", "naam van je huisdier", "mijn huisdier heet"],
    "has_pet": ["heb je een huisdier", "eigen dier", "ik heb een huisdier", "huisdier"],
    "pet_talk": ["over je huisdier praten", "huisdier praten"],
    "fav_food": ["lievelingseten", "lievelings eten", "favoriete eten", "lekkerste eten", "het liefst eet", "mijn lievelingseten"],
    "fav_subject": ["lievelingsvak", "favoriete vak", "leukste vak", "mijn lievelingsvak"],
    "school_strength": ["goed in", "vakken ben je goed", "sterk in", "makkelijk vak", "waar je goed in bent", "ik ben goed in"],
    "school_difficulty": ["moeilijk vak", "lastig vak", "vak dat je lastig vindt", "school moeilijk", "vind ik lastig", "vind ik moeilijk"],
    "interest": ["interesseert", "interessant", "meer over weten", "nieuwsgierig naar", "vind je interessant"],
    "aspiration": ["later worden", "beroep", "droom", "wil je worden", "wat wil je worden", "ik wil later", "later wil ik"],
    "role_model": ["voorbeeld", "kijk je op", "bewonder", "held", "opkijkt naar", "ik kijk op"],
    "age": ["leeftijd", "jaar oud", "hoe oud", "ik ben", "ik word"],
    "child_name": ["naam", "hoe heet ik", "mijn naam"],
    "name": ["naam", "hoe heet ik", "mijn naam"],
    "condition": ["condition", "conditie"],
}

INTENT_PATTERNS = [
    (re.compile(r"\b(vergeet|verwijder|wis|schrap|gooi weg|dat klopt niet meer|niet meer waar)\b", re.IGNORECASE), "um_delete"),
    (re.compile(r"\b(eigenlijk niet meer|dat was fout|dat klopt niet|corrigeer|dat is veranderd|niet meer|ik had het fout|was vroeger|nu is het anders)\b", re.IGNORECASE), "um_update"),
    (re.compile(r"\b(nee wacht|ik bedoel|laat maar|nee toch|ik bedoelde|wacht nee|eigenlijk bedoel ik|nee ik bedoel)\b", re.IGNORECASE), "dialogue_update"),
    (re.compile(
        r"\b(wat weet je|weet je nog|herinner je|wat heb je|vertel me|laat zien|"
        r"wat staat er|klopt het dat|weet je wat mijn|wat is mijn|"
        r"wat onthoud je|wat heb je onthouden|heb je iets onthouden|"
        r"wat zei je net over mij|wat weet je nog van mij|wat weet je over mij|"
        r"wat weet je allemaal over mij)\b",
        re.IGNORECASE,
    ), "um_inspect"),
    (re.compile(r"\b(waarom|hoe heet jij|wat kan jij|ben jij|hoe werkt|wat doe jij|waarom wil je|wie ben jij|wat ben jij|kan jij)\b", re.IGNORECASE), "dialogue_question"),
    (re.compile(r"\b(haha|hehe|grappig|leuk|cool|wow|super|geweldig|okÃ©|oke|jaja|nee nee|echt waar|wauw|tof|nice)\b", re.IGNORECASE), "dialogue_social"),
    (re.compile(r"\b(ik vind|mijn .{1,30} is|ik heet|ik ben|ik heb|ik doe|ik speel|ik lees|ik wil|ik hou van|mijn lievelings|ik word later|later wil ik)\b", re.IGNORECASE), "um_add"),
    (re.compile(r"^\s*(um|uh|eh|hmm?|ahh?|uhh?)(\s+(um|uh|eh|hmm?|ahh?|uhh?|ja|nee))*\s*$", re.IGNORECASE), "dialogue_none"),
    (re.compile(r"^[a-zÃ Ã¡Ã¢Ã¤Ã¨Ã©ÃªÃ«Ã¬Ã­Ã®Ã¯Ã²Ã³Ã´Ã¶Ã¹ÃºÃ»Ã¼A-Z0-9 '\-]{1,40}$", re.IGNORECASE), "dialogue_answer"),
]

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a child-robot interaction system. "
    "A NAO robot called Leo is talking to a Dutch child aged 8-11. "
    "The child speaks Dutch. Classify what the child said into exactly one intent, "
    "identify the UM field if relevant, extract the value if relevant, and return "
    "a confidence score 0.0-1.0. Naming: um_* intents touch the database, "
    "dialogue_* intents are conversation only. Classify memory-access phrases such "
    "as 'wat weet je nog over mij', 'wat heb je onthouden', and "
    "'wat zei je net over mij' as um_inspect. Return ONLY valid JSON."
)

INTENT_FEW_SHOT_EXAMPLES = [
    ("Mijn lievelingseten is pizza", {"intent": "um_add", "field": "fav_food", "value": "pizza", "confidence": 0.97}),
    ("Ik wil later dokter worden", {"intent": "um_add", "field": "aspiration", "value": "dokter", "confidence": 0.95}),
    ("Eigenlijk niet meer pizza, nu is het sushi", {"intent": "um_update", "field": "fav_food", "value": "sushi", "confidence": 0.95}),
    ("Vergeet wat ik zei over mijn huisdier", {"intent": "um_delete", "field": "has_pet", "value": None, "confidence": 0.98}),
    ("Wat weet je over mijn lievelingshobby?", {"intent": "um_inspect", "field": "hobby_fav", "value": None, "confidence": 0.96}),
    ("Wat weet je nog over mij?", {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.95}),
    ("Wat heb je over mij onthouden?", {"intent": "um_inspect", "field": None, "value": None, "confidence": 0.94}),
    ("Pizza... nee wacht, ik bedoel sushi", {"intent": "dialogue_update", "field": "fav_food", "value": "sushi", "confidence": 0.92}),
    ("Voetbal", {"intent": "dialogue_answer", "field": None, "value": "voetbal", "confidence": 0.88}),
    ("Waarom wil je dat weten?", {"intent": "dialogue_question", "field": None, "value": None, "confidence": 0.95}),
    ("Haha dat is grappig", {"intent": "dialogue_social", "field": None, "value": None, "confidence": 0.99}),
    ("um eh ja nee", {"intent": "dialogue_none", "field": None, "value": None, "confidence": 0.91}),
]


@dataclass
class IntentResult:
    intent: str
    field: Optional[str]
    value: Optional[str]
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
        }


class StubIntentClassifier:
    """Embedded rule-based classifier with the same contract as the GPT classifier."""

    VALUE_RE = re.compile(
        r"(?:is|ben|heet|zijn|wordt|vind ik|doe ik|speel ik|lees ik|hou ik van|heb ik)"
        r"\s+(?:een\s+)?([a-zÃ Ã¡Ã¢Ã¤Ã¨Ã©ÃªÃ«Ã¬Ã­Ã®Ã¯Ã²Ã³Ã´Ã¶Ã¹ÃºÃ»Ã¼A-Z0-9 '\-]{1,80}?)(?:[.,!?]|$)",
        re.IGNORECASE,
    )
    WORDEN_RE = re.compile(
        r"(?:wil\s+(?:later\s+)?|word\s+later\s+)([a-zÃ Ã¡Ã¢Ã¤Ã¨Ã©ÃªÃ«Ã¬Ã­Ã®Ã¯Ã²Ã³Ã´Ã¶Ã¹ÃºÃ»Ã¼A-Z0-9'\-][a-zÃ Ã¡Ã¢Ã¤Ã¨Ã©ÃªÃ«Ã¬Ã­Ã®Ã¯Ã²Ã³Ã´Ã¶Ã¹ÃºÃ»Ã¼A-Z0-9 '\-]{0,40}?)\s+worden",
        re.IGNORECASE,
    )

    def __init__(self, schema_path: str = None, valid_fields: list = None):
        self.valid_fields = set(valid_fields or EMBEDDED_VALID_FIELDS)

    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None)

        text_clean = text.strip()
        intent = self.detect_intent(text_clean)
        field = self.detect_field(text_clean)
        value = (
            self.extract_value(text_clean)
            if intent in ("um_add", "um_update", "dialogue_update", "dialogue_answer")
            else None
        )
        return IntentResult(intent=intent, field=field, value=value, confidence=1.0)

    def detect_intent(self, text: str) -> str:
        for pattern, intent in INTENT_PATTERNS:
            if pattern.search(text):
                return intent
        return "dialogue_none"

    def detect_field(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for field, aliases in FIELD_ALIASES.items():
            if field not in self.valid_fields:
                continue
            for alias in aliases:
                if alias.lower() in text_lower:
                    return field
        return None

    def extract_value(self, text: str) -> Optional[str]:
        match = self.VALUE_RE.search(text)
        if match:
            value = match.group(1).strip()
            for prefix in ("het ", "de ", "een "):
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
            return value if value else None

        match = self.WORDEN_RE.search(text)
        if match:
            return match.group(1).strip()
        return None


class GPTIntentClassifier:
    """Embedded GPT classifier that falls back to the embedded stub classifier."""

    def __init__(
        self,
        openai_key: Optional[str] = None,
        valid_fields: list = None,
        schema_path: str = None,
        contract_path: str = None,
    ):
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required.")
        self.client = OpenAI(api_key=key)
        self.valid_fields = set(valid_fields or EMBEDDED_VALID_FIELDS)
        self.stub = StubIntentClassifier(valid_fields=list(self.valid_fields))

        self.system_prompt = (
            INTENT_SYSTEM_PROMPT
            + f"\n\nValid intents: {VALID_INTENTS}"
            + f"\nValid fields: {sorted(self.valid_fields)}"
            + "\nReturn ONLY a JSON object with keys: intent, field, value, confidence."
        )
        self.few_shot_messages = []
        for utterance, result in INTENT_FEW_SHOT_EXAMPLES:
            self.few_shot_messages.append({"role": "user", "content": utterance})
            self.few_shot_messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})

    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self.call_gpt(text.strip())
        if result is None:
            CLASSIFIER_LOGGER.warning("GPT hard failure on first attempt; falling back to embedded stub.")
            return self.stub.classify(text)

        intent, field, value, confidence = result
        if confidence >= CONFIDENCE_THRESHOLD:
            return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

        return IntentResult(intent=REPEAT_SENTINEL, field=None, value=None, confidence=confidence)

    def classify_retry(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None, confidence=1.0)

        result = self.call_gpt(text.strip())
        if result is None:
            return self.stub.classify(text)

        intent, field, value, confidence = result
        if confidence < CONFIDENCE_THRESHOLD:
            return self.stub.classify(text)

        return IntentResult(intent=intent, field=field, value=value, confidence=confidence)

    def call_gpt(self, text: str) -> Optional[tuple]:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.few_shot_messages,
                {"role": "user", "content": text},
            ]
            response = self.client.chat.completions.create(
                model=GPT_INTENT_MODEL,
                messages=messages,
                max_tokens=GPT_INTENT_MAX_TOKENS,
                temperature=GPT_INTENT_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            return self.parse_response(raw, text)
        except Exception as e:
            CLASSIFIER_LOGGER.error("GPT intent classifier error: %s", e)
            return None

    def parse_response(self, raw: str, original_text: str) -> Optional[tuple]:
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            CLASSIFIER_LOGGER.warning("Invalid intent JSON for '%s': %s", original_text[:50], e)
            return None

        for key in ("intent", "field", "value", "confidence"):
            if key not in parsed:
                return None

        intent = str(parsed["intent"]).strip()
        field = parsed["field"]
        value = parsed["value"]
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if intent not in VALID_INTENTS:
            return None
        if field is not None and field not in self.valid_fields:
            field = None

        confidence = max(0.0, min(1.0, confidence))
        return intent, field, value, confidence


class CRI_ScriptedDialogue(SICApplication):
    """
    CRI 4.0 walkthrough interaction flow.

    The script pulls known child-memory fields from the UM API before starting,
    prints a preview, then runs the explicit Mila-style dialogue structure:
    scripted turns, UM-template turns, pre-generated LLM utterances, and
    runtime LLM follow-up branches for unpredictable child responses.

    UM connection:
        GET http://localhost:8000/api/um/{child_id}/field/{field_name}
        No API key needed for reads.
        If a field is not set, robot says "dat weet ik nog niet".
    """

    # UM connection
    UM_API_BASE = "http://localhost:8000"
    CHILD_ID = "Julianna"
    UNKNOWN_VALUE = "dat weet ik nog niet"
    UM_FIELDS = (
        "child_name", "name", "exposure", "condition",
        "age", "hobbies", "hobby_fav",
        "sports_enjoys", "sports_talk", "sports_fav", "sports_plays", "sports_fav_play",
        "books_enjoys", "books_talk", "books_fav_genre", "books_fav_title",
        "music_enjoys", "music_talk", "music_plays_instrument", "music_instrument",
        "animals_enjoys", "animal_talk", "animal_fav",
        "has_pet", "pet_talk", "pet_type", "pet_name",
        "freetime_fav", "fav_food", "fav_subject",
        "school_strength", "school_difficulty",
        "aspiration", "role_model", "interest", "has_best_friend",
    )
    SCRIPT_TABLE_FIELDS = (
        ("name", "Orientation"),
        ("age", "Orientation"),
        ("hobbies", "Orientation"),
        ("hobby_fav", "Exploratory/Affective"),
        ("sports_enjoys", "Orientation"),
        ("sports_fav_play", "Exploratory/Affective"),
        ("music_enjoys", "Orientation"),
        ("books_enjoys", "Orientation"),
        ("books_fav_title", "Exploratory/Affective"),
        ("freetime_fav", "Orientation"),
        ("animals_enjoys", "Orientation"),
        ("animal_fav", "Exploratory/Affective"),
        ("has_pet", "Orientation"),
        ("pet_type", "Orientation"),
        ("pet_name", "Exploratory/Affective"),
        ("fav_food", "Orientation"),
        ("fav_subject", "Exploratory/Affective"),
        ("school_strength", "Exploratory/Affective"),
        ("school_difficulty", "Affective/Exchange"),
        ("aspiration", "Affective/Exchange"),
        ("role_model", "Affective/Exchange"),
        ("interest", "Affective/Exchange"),
        ("has_best_friend", "Affective/Exchange"),
    )
    FIELD_LABELS = {
        "child_name": "je naam",
        "name": "je naam",
        "exposure": "of we elkaar al eerder hebben gezien",
        "condition": "de conditie",
        "age": "je leeftijd",
        "hobbies": "je hobby's",
        "hobby_fav": "je favoriete hobby",
        "sports_enjoys": "of je sport leuk vindt",
        "sports_talk": "of je over sport wilt praten",
        "sports_fav": "je lievelingssport",
        "sports_plays": "of je een sport doet",
        "sports_fav_play": "de sport die je graag doet",
        "books_enjoys": "of je boeken leuk vindt",
        "books_talk": "of je over boeken wilt praten",
        "books_fav_genre": "je favoriete soort boeken",
        "books_fav_title": "je favoriete boek",
        "music_enjoys": "of je muziek leuk vindt",
        "music_talk": "of je over muziek wilt praten",
        "music_plays_instrument": "of je een instrument speelt",
        "music_instrument": "welk instrument je speelt",
        "animals_enjoys": "of je dieren leuk vindt",
        "animal_talk": "of je over dieren wilt praten",
        "animal_fav": "je lievelingsdier",
        "has_pet": "of je een huisdier hebt",
        "pet_talk": "of je over je huisdier wilt praten",
        "pet_type": "het soort huisdier",
        "pet_name": "de naam van je huisdier",
        "freetime_fav": "wat je graag in je vrije tijd doet",
        "fav_food": "je lievelingseten",
        "fav_subject": "je lievelingsvak",
        "school_strength": "waar je goed in bent op school",
        "school_difficulty": "wat je moeilijk vindt op school",
        "aspiration": "wat je later wilt doen of worden",
        "role_model": "naar wie je opkijkt",
        "interest": "je interesses",
        "has_best_friend": "of je een beste vriend hebt",
    }

    # Whisper
    STT_TIMEOUT = 20
    STT_PHRASE_LIMIT = 18

    # LLM
    LLM_FALLBACK = "Wauw, dat klinkt heel leuk!"
    LLM_SYSTEM = (
        "Jij bent een vriendelijke robot genaamd Leo en je praat tegen een Nederlands kind van 8 tot 11 jaar oud. "
        "Geef antwoord in een korte zin (maximaal 25 woorden). "
        "Wees warm, enthousiast en geschikt voor de leeftijden tussen 8 en 11. "
        "Vraag geen vragen. Praat in het Nederlands. Gebruik geen emoji's."
    )

    TOPIC_DOMAIN_ORDER = ("pet", "sports", "books", "music", "animals", "hobby", "freetime")
    BOOLEANISH_FIELDS = (
        "has_pet", "sports_enjoys", "sports_plays", "books_enjoys",
        "music_enjoys", "music_talk", "music_plays_instrument", "animals_enjoys",
    )
    TUTORIAL_CONDITION_FIELD = "condition"
    MEMORY_ACCESS_EXCLUDED_FIELDS = (
        "exposure",
        "condition",
    )
    OPPOSITE_VALUE_FALLBACKS = {
        "huisdier": ("een robotdinosaurus", "een steen", "Rover"),
        "sport": ("schaken op de bank", "stilzitten", "ballet"),
        "boeken": ("Harry Potter", "een kookboek", "een boek over vrachtwagens"),
        "muziek": ("helemaal geen muziek", "drummen", "opera zingen"),
        "hobby": ("stilzitten", "postzegels sorteren", "breien"),
        "eten": ("pizza", "spruitjes", "broccoli"),
        "school": ("rekenen", "aardrijkskunde", "gym"),
        "droom": ("juf", "bankdirecteur worden", "robots repareren"),
    }
    TOPIC_CHANGE_MODEL = "gpt-4o-mini"
    TOPIC_RESPONSE_TYPES = (
        "no_change", "story", "possible_update",
        "possible_delete", "correction_unclear",
        "wants_other_topic", "question", "unclear",
    )
    CONTENT_PLAN_SEQUENCE = "sequence"
    CASE_FULLY_SCRIPTED = "fully_scripted"
    CASE_UM_TEMPLATE = "um_template"
    CASE_PREAUTHORED_POOL = "preauthored_pool"
    CASE_LLM_PREGENERATED = "llm_pregenerated"
    CASE_RUNTIME_LLM_BRANCH = "runtime_llm_branch"
    CASE_MIXED_SEQUENCE = "mixed_sequence"
    PREGENERATED_UTTERANCE_PREFIXES = ("pregen_", "script_", "llm_pregen_")

    # Desktop mic flag
    USE_DESKTOP_MIC = False
    ASK_RUN_MODE_AT_START = True
    SIMULATION_MODE = False
    SIMULATED_PERSONA_DIR = os.path.abspath(os.path.join(_HERE, "fake_personas"))
    SIMULATED_PERSONA_PATH = os.path.join(SIMULATED_PERSONA_DIR, "noor_1001.json")
    SIMULATION_WRITE_PERSONA_FILE = False
    WAIT_FOR_PREVIEW_CONFIRMATION = True
    REVIEW_TRANSCRIPTS = True
    POST_PHASE_TEST_CONTROLS = True
    CHILD_INPUT_MODE = "microphone"
    SCRIPT_VERSION = "CRI-BRANCH-BASIC4.0"
    ASK_SESSION_INTERFACE_AT_START = True
    SESSION_CONFIG_PATH = os.path.abspath(os.path.join(_HERE, "session_config.local.json"))
    TOTAL_SCRIPT_PHASES = 9

    # Conversation logging
    CONVERSATION_LOG_ENABLED = True
    CONVERSATION_LOG_ROOT = os.path.abspath(os.path.join(_HERE, "conversations"))

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
        self.session_config = {}
        self.resume_from_log_path = None
        self.resume_source_log = {}
        self.local_child_name = ""
        self.researcher_name = ""
        self.session_number = 1
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
        self.set_log_level(sic_logging.INFO)
        self.configure_session_interface()
        self.configure_run_mode()
        self.setup()

    # Setup

    def load_local_session_config(self) -> dict:
        try:
            with open(self.SESSION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except Exception:
            return {}

    def save_local_session_config(self, config: dict):
        try:
            with open(self.SESSION_CONFIG_PATH, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("Could not save local session config: %s", e)

    def ask_session_value(self, label: str, default: str = "") -> str:
        default_hint = f" [{default}]" if default else ""
        try:
            value = input(f"  {label}{default_hint}: ").strip()
            return value if value else default
        except (EOFError, KeyboardInterrupt):
            return default

    def normalize_condition_value(self, value: str, default: str = "C1") -> str:
        clean = str(value or "").strip().lower()
        if clean in ("2", "c2", "condition 2", "with tablet", "tablet"):
            return "C2"
        if clean in ("1", "c1", "condition 1", "no tablet", "without tablet", "geen tablet"):
            return "C1"
        return default

    def condition_display(self, condition: str) -> str:
        normalized = self.normalize_condition_value(condition)
        return f"{normalized} ({'with tablet' if normalized == 'C2' else 'no tablet'})"

    def parse_session_number(self, value: str, default: int = 1) -> int:
        try:
            number = int(str(value).strip())
            return number if number > 0 else default
        except (TypeError, ValueError):
            return default

    def parse_phase_index(self, value: str, default_index: int = 0) -> int:
        try:
            phase = int(str(value).strip())
            if 1 <= phase <= self.TOTAL_SCRIPT_PHASES:
                return phase - 1
        except (TypeError, ValueError):
            pass
        return default_index

    def apply_session_config(self, config: dict):
        self.session_config = dict(config or {})
        child_id = str(self.session_config.get("child_id") or "").strip()
        if child_id:
            self.CHILD_ID = child_id
        self.local_child_name = str(self.session_config.get("child_name") or "").strip()
        self.researcher_name = str(self.session_config.get("researcher_name") or "").strip()
        self.session_number = self.parse_session_number(self.session_config.get("session_number"), 1)
        self.local_condition = self.normalize_condition_value(
            self.session_config.get("condition"),
            default="",
        )
        self.start_phase_index = int(self.session_config.get("start_phase_index", 0) or 0)
        self.start_phase_index = max(0, min(self.start_phase_index, self.TOTAL_SCRIPT_PHASES - 1))

    def check_child_in_um_api(self, child_id: str):
        print(f"\nChecking child '{child_id}' in UM API ({self.UM_API_BASE})...")
        try:
            response = requests.get(f"{self.UM_API_BASE}/api/um/{child_id}", timeout=3)
            if response.status_code == 200:
                print("  Child found.")
            elif response.status_code == 404:
                print("  Child not found.")
            else:
                print(f"  UM API returned {response.status_code}.")
        except Exception:
            print("  UM API is not reachable right now.")

    def run_new_session_interface(self):
        previous = self.load_local_session_config()
        print("\n" + "=" * 72)
        print("CRI SESSION SETUP")
        print("This local setup is for child ID, local first name, researcher, condition, and resume phase.")
        child_id = self.ask_session_value("Child ID", previous.get("child_id", self.CHILD_ID))
        child_name = self.ask_session_value("Child first name (local only)", previous.get("child_name", ""))
        researcher = self.ask_session_value("Researcher name", previous.get("researcher_name", ""))
        session_number = self.parse_session_number(
            self.ask_session_value("Session number", str(previous.get("session_number", 1))),
            1,
        )
        condition = self.normalize_condition_value(
            self.ask_session_value("Condition [1/C1 = no tablet, 2/C2 = with tablet]", previous.get("condition", "C1"))
        )
        start_default = int(previous.get("start_phase_index", 0) or 0) + 1
        start_phase_index = self.parse_phase_index(
            self.ask_session_value(f"Start from phase [1-{self.TOTAL_SCRIPT_PHASES}]", str(start_default)),
            default_index=0,
        )

        self.check_child_in_um_api(child_id)
        print("\n" + "-" * 56)
        print(f"  Child ID:    {child_id}")
        print(f"  Child name:  {child_name or '(not set)'}")
        print(f"  Researcher:  {researcher or '(not set)'}")
        print(f"  Session:     #{session_number}")
        print(f"  Condition:   {self.condition_display(condition)}")
        print(f"  Start phase: {start_phase_index + 1}")
        print("-" * 56)
        input("\nPress Enter to continue...")

        config = {
            "mode": "new",
            "child_id": child_id,
            "child_name": child_name,
            "researcher_name": researcher,
            "session_number": session_number,
            "condition": condition,
            "start_phase_index": start_phase_index,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.save_local_session_config(config)
        self.apply_session_config(config)

    def clean_pasted_path(self, path: str) -> str:
        return str(path or "").strip().strip('"').strip("'")

    def load_conversation_log_file(self, path: str) -> dict:
        clean_path = self.clean_pasted_path(path)
        with open(clean_path, "r", encoding="utf-8") as log_file:
            return json.load(log_file)

    def compute_resume_phase_from_log(self, log: dict) -> int:
        explicit = log.get("resume_phase")
        if explicit:
            return max(1, min(int(explicit), self.TOTAL_SCRIPT_PHASES))

        events = log.get("events") or []
        last_start_index = None
        last_start_phase = None
        for index, event in enumerate(events):
            if event.get("type") == "phase_start" and event.get("phase"):
                last_start_index = index
                last_start_phase = int(event["phase"])

        if last_start_phase is None:
            return max(1, min(int(log.get("last_completed_phase", 0) or 0) + 1, self.TOTAL_SCRIPT_PHASES))

        ended_after_start = any(
            event.get("type") == "phase_end" and int(event.get("phase") or 0) == last_start_phase
            for event in events[last_start_index + 1:]
        )
        if ended_after_start:
            return min(last_start_phase + 1, self.TOTAL_SCRIPT_PHASES)
        return last_start_phase

    def session_config_from_resume_log(self, log: dict, resume_path: str) -> dict:
        config = dict(log.get("session_config") or {})
        config.setdefault("child_id", log.get("child_id", self.CHILD_ID))
        config.setdefault("child_name", log.get("child_name", ""))
        config.setdefault("researcher_name", log.get("researcher_name", ""))
        config.setdefault("session_number", log.get("session_number", 1))
        config.setdefault("condition", log.get("tutorial_condition", "C1"))
        resume_phase = self.compute_resume_phase_from_log(log)
        config["mode"] = "resume"
        config["resume_from_log"] = self.clean_pasted_path(resume_path)
        config["resume_phase"] = resume_phase
        config["start_phase_index"] = resume_phase - 1
        config["resumed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return config

    def restore_runtime_state_from_log(self, log: dict):
        self.mistakes_mentioned = int(log.get("mistakes_mentioned", 0) or 0)
        self.corrections_seen = int(log.get("corrections_seen", 0) or 0)
        self.mistake_states = dict(log.get("mistake_states") or {})
        self.phases_with_confirmed_change = set(log.get("phases_with_confirmed_change") or [])
        self.memory_fields_mentioned_so_far = set(log.get("memory_fields_mentioned_so_far") or [])

    def run_resume_session_interface(self, resume_path: str = ""):
        print("\n" + "=" * 72)
        print("CRI SESSION RESUME")
        path = resume_path or self.ask_session_value("Paste previous conversation JSON path", "")
        log = self.load_conversation_log_file(path)
        config = self.session_config_from_resume_log(log, path)
        self.resume_from_log_path = self.clean_pasted_path(path)
        self.resume_source_log = log
        self.apply_session_config(config)
        self.restore_runtime_state_from_log(log)

        print("\nLoaded previous conversation log.")
        print(f"  Child ID:       {self.CHILD_ID}")
        print(f"  Child name:     {self.local_child_name or '(not set)'}")
        print(f"  Researcher:     {self.researcher_name or '(not set)'}")
        print(f"  Condition:      {self.condition_display(self.local_condition)}")
        print(f"  Resume phase:   {self.start_phase_index + 1}")
        print(f"  Mentioned UM fields restored: {len(self.memory_fields_mentioned_so_far)}")
        input("\nPress Enter to continue from this phase...")

    def configure_session_interface(self):
        if not self.ASK_SESSION_INTERFACE_AT_START:
            return

        env_resume = self.clean_pasted_path(os.environ.get("CRI_RESUME_LOG_PATH", ""))
        if env_resume:
            self.run_resume_session_interface(env_resume)
            return

        env_mode = os.environ.get("CRI_SESSION_MODE", "").strip().lower()
        if env_mode in ("skip", "none", "off", "0"):
            return
        if env_mode in ("resume", "r"):
            self.run_resume_session_interface()
            return
        if env_mode in ("new", "n"):
            self.run_new_session_interface()
            return

        print("\n" + "=" * 72)
        print("CRI SESSION")
        print("Press Enter for a new session.")
        print("Type R + Enter to resume from a previous conversation JSON log.")
        choice = input("Session mode: ").strip().lower()
        print("=" * 72)
        if choice in ("r", "resume"):
            self.run_resume_session_interface()
        else:
            self.run_new_session_interface()

    def configure_run_mode(self):
        """Ask at startup whether child responses should come from microphone or keyboard."""
        if not self.ASK_RUN_MODE_AT_START:
            return

        input_mode = os.environ.get("CRI_CHILD_INPUT_MODE", "").strip().lower()
        if input_mode in ("keyboard", "key", "k", "typed", "type"):
            self.child_input_mode = "keyboard"
            return
        if input_mode in ("microphone", "mic", "m", "whisper", "speech"):
            self.child_input_mode = "microphone"
            return

        env_choice = os.environ.get("CRI_SIMULATION_MODE", "").strip().lower()
        if env_choice in ("1", "true", "yes", "y", "sim", "simulation"):
            self.simulation_mode = True
            self.child_input_mode = "simulation"
            self.configure_simulated_persona()
            return
        if env_choice in ("0", "false", "no", "n", "real", "normal"):
            self.simulation_mode = False

        print("\n" + "=" * 72)
        print("CRI 4.0 CHILD INPUT MODE")
        print("Press Enter for microphone/Whisper input.")
        print("Type K + Enter for keyboard input.")
        choice = input("Child input mode: ").strip().lower()
        print("=" * 72)
        if choice in ("k", "key", "keyboard", "typed", "type"):
            self.child_input_mode = "keyboard"
        else:
            self.child_input_mode = "microphone"

    def use_keyboard_input(self) -> bool:
        return self.child_input_mode == "keyboard"

    def use_microphone_input(self) -> bool:
        return self.child_input_mode == "microphone" and not self.simulation_mode

    def persona_summary_from_file(self, path: str) -> dict:
        """Read minimal display metadata for a fake persona JSON file."""
        with open(path, "r", encoding="utf-8") as persona_file:
            persona = json.load(persona_file)
        child_id = persona.get("child_id")
        return {
            "child_id": str(child_id) if child_id is not None else "",
            "name": persona.get("child_name") or persona.get("name") or "unknown",
            "exposure": persona.get("exposure") or self.UNKNOWN_VALUE,
            "path": path,
        }

    def available_fake_personas(self) -> list:
        """Return all fake persona files with child ID, name, exposure, and path."""
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
            print(f"Using fake persona {selected['child_id']} - {selected['name']} ({selected['exposure']})")
            return

        personas = self.available_fake_personas()
        if not personas:
            print("No fake personas found; using default fake persona path.")
            return

        print("\nAVAILABLE FAKE PERSONAS")
        for persona in personas:
            print(f"  {persona['child_id']}: {persona['name']} ({persona['exposure']})")
        default_id = personas[0]["child_id"]
        choice = input(f"Type child ID for simulation, or press Enter for {default_id}: ").strip()
        selected_id = choice or default_id
        selected = self.select_simulated_persona_by_child_id(selected_id)
        print(f"Selected fake persona: {selected['child_id']} - {selected['name']} ({selected['exposure']})")

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
            self.logger.info("GPTIntentClassifier ready.")
        except Exception as e:
            self.logger.warning("GPTIntentClassifier failed (%s) - using stub.", e)
            self.clf = StubIntentClassifier(valid_fields=list(self.UM_FIELDS))

        self.logger.info("UM: LIVE - %s, child=%s", self.UM_API_BASE, self.CHILD_ID)
        self.logger.info("Child input mode: %s", self.child_input_mode)

        # NAO
        if not self.USE_DESKTOP_MIC:
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
        self.logger.info("Setup complete.")

    # UM pulling

    def load_simulated_persona(self):
        """Load the fake child profile used by LLM simulation mode."""
        path = self.simulated_persona_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Simulated persona file not found: {path}")
        with open(path, "r", encoding="utf-8") as persona_file:
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

    def get_field(self, field: str) -> str:
        """
        Pull a single UM field from Eunike's API.
        GET /api/um/{child_id}/field/{field_name}; no API key needed.

        Returns the value as a Dutch string.
        Returns 'dat weet ik nog niet' if field not set or API unreachable.
        """
        if not field:
            return self.UNKNOWN_VALUE
        if self.simulation_mode:
            return self.simulated_um_profile().get(field, self.UNKNOWN_VALUE)

        url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/field/{field}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                value = resp.json().get("data", {}).get("value")
                if value:
                    self.logger.info("UM[%s] = %s", field, value)
                    return str(value)
                return self.UNKNOWN_VALUE
            elif resp.status_code == 404:
                self.logger.info("UM field '%s' not set for child '%s'.", field, self.CHILD_ID)
                return self.UNKNOWN_VALUE
            else:
                self.logger.warning("UM API returned %d for field '%s'.", resp.status_code, field)
                return self.UNKNOWN_VALUE
        except requests.exceptions.ConnectionError:
            self.logger.error("UM API not reachable at %s - is Eunike's main.py running?", self.UM_API_BASE)
            return self.UNKNOWN_VALUE
        except Exception as e:
            self.logger.error("UM error for field '%s': %s", field, e)
            return self.UNKNOWN_VALUE

    def field_value_from_profile(self, profile: dict, field: str) -> str:
        """Extract one field from a bulk UM profile response."""
        scalar_entry = profile.get("scalars", {}).get(field)
        if isinstance(scalar_entry, dict):
            value = scalar_entry.get("value")
            return str(value) if value else self.UNKNOWN_VALUE

        node_entries = profile.get("nodes", {}).get(field)
        if isinstance(node_entries, list) and node_entries:
            values = [
                str(entry.get("value"))
                for entry in node_entries
                if isinstance(entry, dict) and entry.get("value")
            ]
            if values:
                return self.format_dutch_list(values)

        return self.UNKNOWN_VALUE

    def is_pregenerated_utterance_field(self, field: str) -> bool:
        """Fields with these prefixes are L2-pregen utterances stored in UM/GraphDB."""
        return any(str(field).startswith(prefix) for prefix in self.PREGENERATED_UTTERANCE_PREFIXES)

    def pregenerated_fields_from_profile(self, profile: dict) -> dict:
        """Pull L2-pregen utterance fields even when they are not core UM fields."""
        extra = {}
        for container_name in ("scalars", "nodes"):
            container = profile.get(container_name, {})
            if not isinstance(container, dict):
                continue
            for field in container:
                if self.is_pregenerated_utterance_field(field):
                    extra[field] = self.field_value_from_profile(profile, field)
        return extra

    def pull_um_bulk(self) -> dict:
        """Fetch all UM fields in one request using GET /api/um/{child_id}."""
        url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(f"UM profile returned {resp.status_code}")

        profile = resp.json().get("data", {}).get("profile", {})
        if not profile:
            raise RuntimeError("UM profile response did not contain data.profile")

        um = {
            field: self.field_value_from_profile(profile, field)
            for field in self.UM_FIELDS
        }
        um.update(self.pregenerated_fields_from_profile(profile))

        known_count = sum(1 for value in um.values() if self.is_known(value))
        self.logger.info("UM bulk profile pulled: %d/%d fields set.", known_count, len(self.UM_FIELDS))
        for field, value in um.items():
            if self.is_known(value):
                self.logger.info("UM[%s] = %s", field, value)
        return um

    def is_known(self, value: str) -> bool:
        """Return True when a UM value is present enough to safely mention."""
        if value is None:
            return False
        clean = str(value).strip()
        return bool(clean) and clean.lower() != self.UNKNOWN_VALUE

    def pull_um(self) -> dict:
        """Fetch all UM fields used by the 4.0 early interaction flow."""
        if self.simulation_mode:
            um = self.simulated_um_profile()
            known_count = sum(1 for value in um.values() if self.is_known(value))
            self.logger.info("Simulated UM profile loaded: %d/%d fields set.", known_count, len(self.UM_FIELDS))
            for field, value in um.items():
                if self.is_known(value):
                    self.logger.info("SIM UM[%s] = %s", field, value)
            self.last_um_preview = um
            return um

        try:
            um = self.pull_um_bulk()
        except Exception as e:
            self.logger.warning("UM bulk profile pull failed (%s); falling back to per-field reads.", e)
            um = {field: self.get_field(field) for field in self.UM_FIELDS}

        self.last_um_preview = um
        return um

    def known(self, um: dict, field: str, fallback: str = "") -> str:
        value = um.get(field, self.UNKNOWN_VALUE)
        return value if self.is_known(value) else fallback

    def first_known(self, um: dict, fields: list, fallback: str = "") -> tuple:
        for field in fields:
            value = self.known(um, field)
            if value:
                return field, value
        return "", fallback

    def yesish(self, value: str) -> bool:
        return self.is_known(value) and str(value).strip().lower() in ("ja", "yes", "true")

    def pick_wrong_value(self, actual: str, candidates: list) -> str:
        actual_clean = str(actual or "").strip().lower()
        for candidate in candidates:
            if candidate.lower() != actual_clean:
                return candidate
        return candidates[0]

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
        if self.yesish(um.get("sports_enjoys")) or self.known(um, "sports_fav") or self.known(um, "sports_fav_play"):
            clusters.append("sport")
        if self.yesish(um.get("books_enjoys")) or self.known(um, "books_fav_title") or self.known(um, "books_fav_genre"):
            clusters.append("boeken")
        if self.yesish(um.get("music_enjoys")) or self.known(um, "music_instrument"):
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
        for hobby in self.all_hobbies(um):
            if (
                hobby.strip().lower() in ("bakken", "koken", "taarten bakken")
                and (not actual or hobby.strip().lower() != actual.strip().lower())
            ):
                return hobby
        for hobby in self.all_hobbies(um):
            if actual and hobby.strip().lower() != actual.strip().lower():
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
        return self.opposite_memory_value(topic, "hobby_fav", actual or "tekenen")

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
        return self.known(um, "child_name") or self.known(um, "name") or self.CHILD_ID

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
        """Read C1/C2 from local session config first, then UM/GraphDB."""
        local_condition = self.normalize_condition_value(getattr(self, "local_condition", ""), default="")
        if local_condition in ("C1", "C2"):
            return local_condition
        profile = um or self.last_um_preview or {}
        value = self.known(profile, self.TUTORIAL_CONDITION_FIELD)
        clean = str(value or "").strip().lower()
        if "c2" in clean:
            return "C2"
        if "c1" in clean:
            return "C1"
        return "C1"

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
        base = (
            "Ik zal eerst uitleggen hoe je met mij kunt praten. "
            "Ik kan je alleen verstaan nadat ik een vraag heb gesteld. "
            "Als je antwoord geeft, doe dat dan luid en duidelijk. "
            "En wees niet te snel, anders mis ik het misschien. "
            "Mijn ogen worden groen als ik luister. "
            "Soms heb ik moeite om mensen te verstaan, want ik ben dat nog aan het leren. "
            "Vandaag ga ik mijn geheugen best veel gebruiken. "
            "Je mag altijd vragen wat ik over jou onthoud. "
        )
        if condition == "C2":
            return (
                base
                + "Als je dat vraagt, kun je mijn geheugenboek bekijken op de tablet. "
                "Daar kun je zien wat ik over jou heb onthouden. "
                "Als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen. "
                "Goed, dan gaan we beginnen."
            )

        return (
            base
            + "Als je dat vraagt, vertel ik wat ik mij herinner. "
            "Als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen. "
            "Goed, dan gaan we beginnen."
        )

    def alert_condition_mismatch(self, um: dict):
        local_condition = self.normalize_condition_value(getattr(self, "local_condition", ""), default="")
        if local_condition not in ("C1", "C2"):
            return

        profile = um or {}
        value = self.known(profile, self.TUTORIAL_CONDITION_FIELD)
        clean = str(value or "").strip().lower()
        if "c2" in clean:
            um_condition = "C2"
        elif "c1" in clean:
            um_condition = "C1"
        else:
            return

        if um_condition != local_condition:
            print("\n" + "!" * 72)
            print("CONDITION MISMATCH")
            print(f"Local session config says: {local_condition}")
            print(f"UM/GraphDB profile says:  {um_condition}")
            print("Leo will use the local session config, but please check this before continuing.")
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
                for field in ("pet_name", "pet_type", "animal_fav", "has_pet", "pet_talk", "animal_talk")
                if self.known(um, field)
            }
            correct_values = [f"{subject} bij jou hoort"]
            if animal:
                correct_values.append(f"je {animal} leuk vindt")
            candidates.append(self.topic_candidate(
                domain="huisdier",
                label=subject,
                fields=["pet_name", "pet_type", "animal_fav", "has_pet", "pet_talk", "animal_talk"],
                field_labels={
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                    "pet_talk": "of je over je huisdier wilt praten",
                    "animal_talk": "of je over dieren wilt praten",
                },
                current_values=current,
                correct_values=correct_values,
                memory_link=f"{subject} belangrijk voor je is",
                options=[subject, animal or pet_type or "dieren"],
                reground=f"Wat ik zeker wil onthouden, is dat {subject} belangrijk voor je is.",
            ))

        sport = self.known(um, "sports_fav_play") or self.known(um, "sports_fav")
        if sport or self.yesish(um.get("sports_enjoys")):
            label = sport or "sport"
            current = {
                field: self.known(um, field)
                for field in ("sports_enjoys", "sports_talk", "sports_fav", "sports_plays", "sports_fav_play")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="sport",
                label=label,
                fields=["sports_enjoys", "sports_talk", "sports_fav", "sports_plays", "sports_fav_play"],
                field_labels={
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_talk": "of je over sport wilt praten",
                    "sports_fav": "je lievelingssport",
                    "sports_plays": "of je een sport doet",
                    "sports_fav_play": "de sport die je graag doet",
                },
                current_values=current,
                correct_values=[f"je iets met {label} hebt", "sport eerder in jouw geheugen stond"],
                memory_link=f"{label} iets is waar jij iets mee hebt",
                options=[label, "sport"],
                reground=f"Ik houd goed vast dat {label} iets is waar jij iets mee hebt.",
            ))

        book = self.known(um, "books_fav_title") or self.known(um, "books_fav_genre")
        if book or self.yesish(um.get("books_enjoys")):
            label = book or "boeken"
            current = {
                field: self.known(um, field)
                for field in ("books_enjoys", "books_talk", "books_fav_genre", "books_fav_title")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="boeken",
                label=label,
                fields=["books_enjoys", "books_talk", "books_fav_genre", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_talk": "of je over boeken wilt praten",
                    "books_fav_genre": "je favoriete soort boeken",
                    "books_fav_title": "je favoriete boek",
                },
                current_values=current,
                correct_values=[f"{label} bij jouw boekenwereld hoort", "je eerder iets over boeken vertelde"],
                memory_link=f"{label} bij jouw boekenwereld hoort",
                options=[label, "boeken"],
                reground=f"Ik weet in elk geval dat {label} bij jouw boekenwereld hoort.",
            ))

        music = self.known(um, "music_instrument")
        if music or self.yesish(um.get("music_enjoys")):
            label = music or "muziek"
            current = {
                field: self.known(um, field)
                for field in ("music_enjoys", "music_talk", "music_plays_instrument", "music_instrument")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="muziek",
                label=label,
                fields=["music_enjoys", "music_talk", "music_plays_instrument", "music_instrument"],
                field_labels={
                    "music_enjoys": "of je muziek leuk vindt",
                    "music_talk": "of je over muziek wilt praten",
                    "music_plays_instrument": "of je een instrument speelt",
                    "music_instrument": "welk instrument je speelt",
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
        talk_fields = {
            "huisdier": ("pet_talk", "animal_talk"),
            "sport": ("sports_talk",),
            "boeken": ("books_talk",),
            "muziek": ("music_talk",),
        }.get(domain, ())
        wants_talk = any(self.yesish(um.get(field)) for field in talk_fields)
        base_order = {
            "huisdier": 0,
            "sport": 1,
            "boeken": 2,
            "muziek": 3,
            "hobby": 4,
            "eten": 5,
            "droom": 6,
        }.get(domain, 20)
        return (0 if wants_talk else 1, base_order)

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
                fields=["books_enjoys", "books_talk", "books_fav_genre", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_talk": "of je over boeken wilt praten",
                    "books_fav_genre": "je favoriete soort boeken",
                    "books_fav_title": "je favoriete boek",
                },
                current_values={
                    field: self.known(um, field)
                    for field in ("books_enjoys", "books_talk", "books_fav_genre", "books_fav_title")
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
        """Mila Part 1 priority: talk preference + enough detail, with sport first when it is a hobby."""
        domain = topic.get("domain")
        hobbies = " ".join(self.all_hobbies(um)).lower()
        sport_value = self.known(um, "sports_fav_play") or self.known(um, "sports_fav")
        sport_is_hobby = sport_value and sport_value.lower() in hobbies
        if domain == "sport" and self.yesish(um.get("sports_talk")) and sport_is_hobby:
            return (0, 0)

        talk_fields = {
            "sport": ("sports_talk",),
            "muziek": ("music_talk",),
            "huisdier": ("pet_talk", "animal_talk"),
            "boeken": ("books_talk",),
        }.get(domain, ())
        wants_talk = any(self.yesish(um.get(field)) for field in talk_fields)
        order = {"sport": 1, "muziek": 2, "huisdier": 3, "boeken": 4}.get(domain, 20)
        return (0 if wants_talk else 1, order)

    def select_part1_topic1(self, um: dict) -> dict:
        """Select Topic 1 from sport/music/animals/books, falling back to a hobby."""
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
        """Select a second correct topic, preferring animals/pet for the comfort re-ground."""
        first_key = self.topic_key(first_topic)
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

    def topic1_phase_segments(self, topic: dict) -> list:
        """Build the multi-turn Topic 1 phase from the Part 1 script."""
        domain = topic.get("domain")
        label = topic.get("label")
        if domain == "sport":
            sport = topic.get("current_values", {}).get("sports_fav_play") or label
            return [
                {
                    "content_plan": self.l2_slot(
                        "Ik weet ook nog dat jij zelf {sport} speelt.",
                        {"sport": sport},
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                },
                {
                    "content_plan": self.l2_pregen(
                        "sport_position_question",
                        "In welke positie speel jij?",
                        ["sports_fav_play"],
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                },
                {
                    "content_plan": self.l2_slot(
                        "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                        "maar ik val waarschijnlijk al om voor de warming-up. Wat vind jij zo leuk aan {sport}?",
                        {"sport": sport},
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                },
                {
                    "content_plan": self.sequence(
                        self.l1("Dat snap ik helemaal. Dat klinkt ook echt leuk."),
                        self.l2_pregen(
                            "sport_followup_choice",
                            "Ben jij dan meer van snel rennen, goed overspelen, of juist lekker fanatiek meedoen?",
                            ["sports_fav_play"],
                        ),
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                },
                {
                    "content_plan": self.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                    "expects_response": False,
                },
            ]

        return [
            {
                "content_plan": self.l2_slot(
                    "Ik weet ook nog dat {topic} iets is waar jij eerder over vertelde.",
                    {"topic": label},
                ),
                "expects_response": True,
                "response_mode": "listen_only",
            },
            {
                "content_plan": self.l2_pregen(
                    "topic1_followup",
                    f"Wat vind jij zo leuk aan {label}?",
                    list((topic.get("current_values") or {}).keys()),
                ),
                "expects_response": True,
                "response_mode": "acknowledge",
                "llm_turn": True,
            },
            {
                "content_plan": self.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                "expects_response": False,
            },
        ]

    def topic2_phase_segments(self, topic: dict) -> list:
        """Build the correct re-ground topic before M2."""
        label = topic.get("label")
        if topic.get("domain") == "huisdier":
            pet_name = topic.get("current_values", {}).get("pet_name") or label
            return [
                {
                    "content_plan": self.sequence(
                        self.l2_slot(
                            "Ik weet ook nog dat jij een huisdier hebt die {pet_name} heet.",
                            {"pet_name": pet_name},
                        ),
                        self.l2_pregen(
                            "pet_name_reaction",
                            f"Dat vind ik echt een mooie naam. {pet_name} klinkt alsof er stiekem belangrijke plannen worden gemaakt als niemand kijkt.",
                            ["pet_name"],
                        ),
                    ),
                    "expects_response": False,
                },
                {
                    "content_plan": self.l2_pregen(
                        "pet_kind_question",
                        f"Wat voor huisdier is {pet_name} eigenlijk?",
                        ["pet_name", "pet_type"],
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                },
                {
                    "content_plan": self.l1(
                        "Bizar leuk. Ik vind dieren altijd fascinerend. "
                        "Ze doen vaak alsof zij precies weten wat er aan de hand is, en ik net niet."
                    ),
                    "expects_response": False,
                },
            ]

        return [
            {
                "content_plan": self.l2_slot(
                    "Ik weet ook nog dat {topic} bij jou hoort.",
                    {"topic": label},
                ),
                "expects_response": True,
                "response_mode": "acknowledge",
                "llm_turn": True,
            }
        ]

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
            if self.is_known(wrong_value) and wrong_value.lower() != str(actual).strip().lower():
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

    def deliberate_mistake_text(self, topic: dict, field: str, wrong_value: str) -> str:
        field_label = topic.get("field_labels", {}).get(field, "iets over jou")
        return (
            f"Ik dacht dat ik ook had onthouden dat {wrong_value} iets te maken heeft met {field_label}. "
            "Kan je mij hier wat meer over vertellen?"
        )

    def topic_followup_question(self, topic: dict) -> str:
        domain = topic.get("domain")
        label = topic.get("label", "dit onderwerp")
        current = topic.get("current_values", {})
        if domain == "sport":
            sport = current.get("sports_fav_play") or current.get("sports_fav") or label
            return f"Wat vind jij zo leuk aan {sport}?"
        if domain == "huisdier":
            pet = current.get("pet_name") or label
            if current.get("pet_type"):
                return f"Wat voor {current.get('pet_type')} is {pet} eigenlijk?"
            return f"Wat maakt {pet} leuk?"
        if domain == "boeken":
            book = current.get("books_fav_title") or label
            return f"Wat vind jij leuk aan {book}?"
        if domain == "muziek":
            return f"Wat vind jij zo leuk aan {label}?"
        if domain == "hobby":
            return f"Wat vind jij het leukste aan {label}?"
        return f"Wat vind jij leuk aan {label}?"

    def topic_self_disclosure(self, topic: dict) -> str:
        domain = topic.get("domain")
        if domain == "sport":
            return (
                "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                "maar ik val waarschijnlijk al om voor de warming-up."
            )
        if domain == "huisdier":
            return (
                "Ik vind dieren altijd fascinerend. Ze doen vaak alsof zij precies weten "
                "wat er aan de hand is, en ik net niet."
            )
        if domain == "boeken":
            return (
                "Als Bookworm word ik natuurlijk meteen nieuwsgierig van boeken. "
                "Mijn boekenkast is ooit zelfs een beetje overstroomd."
            )
        if domain == "muziek":
            return (
                "Muziek vind ik bijzonder, ook al dansen mijn robotbenen soms sneller dan mijn hoofd."
            )
        if domain == "hobby":
            return (
                "Ik vind het leuk als iemand echt ergens in op kan gaan. Dat probeer ik zelf ook te leren."
            )
        return "Ik vind het leuk om te ontdekken wat kinderen belangrijk vinden."

    def topic_memory_fragment(self, topic: dict) -> str:
        values = self.unique_values(topic.get("correct_values", []), limit=2)
        if len(values) >= 2:
            return f"je zei dat {values[0]} en dat {values[1]}"
        if len(values) == 1:
            return f"je zei dat {values[0]}"
        return "er al iets over dat onderwerp in mijn geheugen staat"

    def preferred_topic_text(self, topic: dict) -> str:
        return (
            f"Ik weet ook nog dat {self.topic_memory_fragment(topic)}. "
            f"{self.topic_self_disclosure(topic)} "
            f"{self.topic_followup_question(topic)}"
        )

    def second_topic_text(self, topic: dict) -> str:
        return (
            f"Maar jij hebt natuurlijk nog meer dingen verteld. "
            f"{self.topic_examples_sentence(topic)} "
            f"{self.topic_self_disclosure(topic)} "
            f"{self.topic_followup_question(topic)}"
        )

    def mistake_followup_text(self, wrong_value: str) -> str:
        return (
            f"Ik ben benieuwd naar {wrong_value}. "
            "Wat vind jij daar zo leuk aan?"
        )

    def hobby_mistake_text(self, wrong_hobby: str) -> str:
        return (
            f"En volgens mij is {wrong_hobby} jouw allerliefste hobby. "
            "Kun je mij daar wat meer over vertellen?"
        )

    def books_mistake_text(self, wrong_book: str) -> str:
        return (
            f"Ik weet nog dat jouw favoriete boek {wrong_book} is. "
            "Ik vind boeken met een duidelijke naam altijd makkelijk te onthouden. "
            "Waar gaat dat boek ook alweer over?"
        )

    def food_mistake_text(self, wrong_food: str) -> str:
        return (
            f"Van al dat praten krijg ik trouwens trek. "
            f"Ik weet nog dat jouw lievelingseten {wrong_food} is. "
            "Wanneer eet je dat het liefst?"
        )

    def topic_examples_sentence(self, topic: dict) -> str:
        values = self.unique_values(topic.get("correct_values", []), limit=2)
        if len(values) >= 2:
            return f"Bijvoorbeeld: je zei dat {values[0]} en dat {values[1]}."
        if len(values) == 1:
            return f"Bijvoorbeeld: je zei dat {values[0]}."
        return "Bijvoorbeeld: er staat al iets over dat onderwerp in mijn geheugen."

    def first_topic_question(self, topic: dict) -> str:
        return (
            f"Ik denk dat {topic['label']} een goed onderwerp is om mee te beginnen, "
            "want daar heb je eerder best veel over verteld. "
            f"{self.topic_examples_sentence(topic)} "
            "Dat vond ik leuk om te onthouden, omdat het iets zegt over wat jij belangrijk of grappig vindt. "
            "Ik ben benieuwd: is daar sinds de vorige keer nog iets nieuws over gebeurd?"
        )

    def easier_topic_question(self, topic: dict) -> str:
        options = self.unique_values(topic.get("options", []), limit=2)
        while len(options) < 2:
            options.append("iets anders")
        return (
            "Dat is ook goed. Dan maak ik het makkelijker: "
            f"wil je liever vertellen over {options[0]}, {options[1]} of iets heel anders?"
        )

    def is_low_content_response(self, transcript: str) -> bool:
        clean = str(transcript or "").strip().lower()
        if not clean:
            return True
        low_content = {
            "ja", "nee", "ok", "oke", "okÃ©", "inderdaad",
            "geen idee", "weet ik niet", "dat weet ik niet",
        }
        return clean in low_content or len(clean.split()) <= 2

    def topic_correction_question(self, topic: dict) -> str:
        current_values = topic.get("current_values", {}) or {}
        value_parts = []
        for field in topic.get("fields", []):
            value = current_values.get(field)
            if self.is_known(value) and field not in self.BOOLEANISH_FIELDS:
                label = topic.get("field_labels", {}).get(field, field)
                value_parts.append(f"{label} is {value}")

        if value_parts:
            remembered = self.format_dutch_list(value_parts[:2])
            return (
                f"OkÃ©, dan wil ik het goed maken. Wat klopt er niet precies: {remembered}? "
                "Moet ik iets veranderen of vergeten?"
            )

        return (
            "OkÃ©, dan wil ik het goed maken. Wat klopt er niet precies? "
            "Moet ik iets veranderen of vergeten?"
        )

    def mistake_correction_question(self, turn: dict) -> str:
        topic = turn.get("mistake_topic", {})
        field = turn.get("mistake_field")
        field_label = topic.get("field_labels", {}).get(field, "dit onderwerp")
        field_questions = {
            "pet_name": "Hoe heet je huisdier?",
            "pet_type": "Wat voor huisdier heb je?",
            "animal_fav": "Wat is je lievelingsdier?",
            "fav_food": "Wat is je lievelingseten?",
            "hobby_fav": "Wat is je favoriete hobby?",
            "hobbies": "Welke hobby's wil je dat ik onthoud?",
            "freetime_fav": "Wat doe je graag in je vrije tijd?",
            "sports_fav": "Wat is je lievelingssport?",
            "sports_fav_play": "Welke sport doe je graag?",
            "books_fav_title": "Wat is je favoriete boek?",
            "books_fav_genre": "Welke soort boeken vind je leuk?",
            "music_instrument": "Welk instrument speel je?",
            "aspiration": "Wat wil je later doen of worden?",
        }
        question = field_questions.get(
            field,
            f"Wat moet ik over {field_label} onthouden?",
        )
        return (
            "Oeps, dan had ik dat verkeerd. "
            f"{question}"
        )

    def mistake_acceptance_change(self, turn: dict, confirmation_question: str = "") -> dict:
        """Build a focused confirmation change when the child accepts Leo's deliberate mistake."""
        topic = turn.get("mistake_topic", {})
        field = turn.get("mistake_field")
        field_label = topic.get("field_labels", {}).get(field, field or "dit onderwerp")
        old_value = turn.get("mistake_actual") or self.UNKNOWN_VALUE
        new_value = turn.get("mistake_wrong")
        fallback_question = f"Zal ik onthouden dat {new_value} bij jou past?"
        if self.is_known(confirmation_question) and str(new_value).lower() not in str(confirmation_question).lower():
            confirmation_question = f"{confirmation_question.rstrip(' ?')}: {new_value}?"

        return {
            "action": "update",
            "field": field,
            "field_label": field_label,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "confidence": 1.0,
            "reason": "Child appeared to accept Leo's deliberate wrong value; asking focused confirmation.",
            "source_text": "",
            "confirmation_question": confirmation_question or fallback_question,
        }

    def topic_no_update_response(self, topic: dict) -> str:
        return (
            f"OkÃ©, dan is er niets nieuws over {topic['label']}. "
            "Dan onthoud ik wat ik al wist."
        )

    def fallback_topic_response(self, response_type: str, topic: dict, transcript: str = "") -> str:
        if response_type == "no_change":
            return self.topic_no_update_response(topic)
        if response_type == "wants_other_topic":
            return "Dat is goed. Dan hoeven we hier niet over te praten."
        if response_type == "question":
            return "Dat is een goede vraag. Ik vertel kort wat ik weet, en dan gaan we verder."
        if response_type == "correction_unclear":
            return self.topic_correction_question(topic)
        if response_type == "unclear":
            return "Ik weet niet helemaal zeker wat je bedoelt, dus ik verander nu niets."
        if transcript:
            return (
                f"Ah, dus {transcript}. "
                f"Dat past wel bij wat ik al van je weet, want {topic.get('memory_link', 'dit belangrijk voor je is')}."
            )
        return self.LLM_FALLBACK

    def turn_text(self, turn: dict) -> str:
        """Single source of truth for the scripted text Leo says for a phase/segment."""
        if turn.get("content_plan"):
            return self.render_content_plan(turn["content_plan"], turn)
        if turn.get("segments"):
            return " ".join(
                self.turn_text({**turn, **segment, "segments": []}).strip()
                for segment in turn["segments"]
                if self.turn_text({**turn, **segment, "segments": []}).strip()
            )
        return turn.get("leo_text", "")

    def turn_phase(self, turn: dict):
        """Return the dialogue phase number."""
        return turn.get("phase")

    def dialogue_case(self, turn: dict) -> str:
        """Explicit dialogue structure case from the Mila content-layer reference."""
        if turn.get("dialogue_case"):
            return turn["dialogue_case"]
        if turn.get("segments"):
            return self.CASE_MIXED_SEQUENCE
        plan = turn.get("content_plan") or {}
        if isinstance(plan, list):
            return self.CASE_MIXED_SEQUENCE
        if isinstance(plan, dict) and plan.get("type"):
            plan_type = plan["type"]
            return self.CASE_MIXED_SEQUENCE if plan_type == self.CONTENT_PLAN_SEQUENCE else plan_type
        if turn.get("llm_turn"):
            return self.CASE_RUNTIME_LLM_BRANCH
        return self.CASE_FULLY_SCRIPTED

    def requires_runtime_llm(self, turn: dict) -> bool:
        """True when this phase has an L3 runtime branch after child input."""
        if turn.get("runtime_llm"):
            return True
        if turn.get("llm_turn"):
            return True
        if self.dialogue_case(turn) == self.CASE_RUNTIME_LLM_BRANCH:
            return True
        return any(segment.get("runtime_llm") or segment.get("llm_turn") for segment in turn.get("segments", []))

    def content_plan_log(self, plan):
        """Compact JSON-safe description of L1/L2-slot/L2-pregen/L3 structure."""
        if not plan:
            return None
        if isinstance(plan, list):
            return [self.content_plan_log(part) for part in plan]
        if not isinstance(plan, dict):
            return plan
        logged = {
            key: value
            for key, value in plan.items()
            if key not in ("text", "template", "fallback")
        }
        for text_key in ("text", "template", "fallback"):
            if text_key in plan:
                logged[text_key] = plan[text_key]
        return logged

    def render_template_text(self, template: str, values: dict = None) -> str:
        """Render an L2-slot template with UM values."""
        values = values or {}
        merged = dict(self.last_um_preview or {})
        merged.update(values)
        safe_values = {
            field: (value if self.is_known(value) else self.UNKNOWN_VALUE)
            for field, value in merged.items()
        }
        try:
            return template.format(**safe_values)
        except KeyError as e:
            missing = str(e).strip("'")
            safe_values[missing] = self.UNKNOWN_VALUE
            return template.format(**safe_values)

    def pregenerated_utterance(self, key: str, fallback: str = "") -> str:
        """Read an L2-pregen utterance from UM/GraphDB, with local fallback."""
        for prefix in self.PREGENERATED_UTTERANCE_PREFIXES:
            field = f"{prefix}{key}"
            value = self.last_um_preview.get(field)
            if self.is_known(value):
                return str(value)
        return fallback

    def render_content_plan(self, plan, turn: dict = None) -> str:
        """Render explicit L1/L2/L2-pregen content plans into one Leo utterance."""
        turn = turn or {}
        if isinstance(plan, list):
            return " ".join(
                rendered
                for rendered in (self.render_content_plan(part, turn).strip() for part in plan)
                if rendered
            )

        plan_type = plan.get("type")

        if plan_type == self.CONTENT_PLAN_SEQUENCE:
            return " ".join(
                rendered
                for rendered in (self.render_content_plan(part, turn).strip() for part in plan.get("parts", []))
                if rendered
            )

        if plan_type == self.CASE_FULLY_SCRIPTED:
            return plan.get("text", "")

        if plan_type == self.CASE_UM_TEMPLATE:
            return self.render_template_text(plan.get("template", ""), plan.get("values", {}))

        if plan_type == self.CASE_LLM_PREGENERATED:
            return self.pregenerated_utterance(plan.get("key", ""), plan.get("fallback", ""))

        if plan_type == self.CASE_PREAUTHORED_POOL:
            return plan.get("text", turn.get("leo_text", ""))

        return turn.get("leo_text", "")

    def l1(self, text: str) -> dict:
        return {"type": self.CASE_FULLY_SCRIPTED, "layer": "L1", "text": text}

    def l2_slot(self, template: str, values: dict = None, wrong: bool = False) -> dict:
        return {
            "type": self.CASE_UM_TEMPLATE,
            "layer": "L2-slot WRONG" if wrong else "L2-slot",
            "template": template,
            "values": values or {},
            "wrong_slot": bool(wrong),
        }

    def l2_pregen(self, key: str, fallback: str, input_fields: list = None) -> dict:
        return {
            "type": self.CASE_LLM_PREGENERATED,
            "layer": "L2-pregen",
            "key": key,
            "input_fields": input_fields or [],
            "fallback": fallback,
        }

    def sequence(self, *parts) -> dict:
        return {"type": self.CONTENT_PLAN_SEQUENCE, "parts": [part for part in parts if part]}

    def public_memory_fields(self, fields) -> list:
        """Keep memory access away from internal control fields such as exposure."""
        excluded = set(self.MEMORY_ACCESS_EXCLUDED_FIELDS)
        return [
            field for field in fields
            if field and field in self.UM_FIELDS and field not in excluded
        ]

    def current_phase_memory_fields(self, turn: dict) -> list:
        """Fields tied to the currently active script phase/category."""
        fields = []
        fields.extend((turn.get("used_fields") or {}).keys())

        topic = turn.get("topic") or turn.get("mistake_topic") or {}
        fields.extend(topic.get("fields") or [])
        fields.extend((topic.get("current_values") or {}).keys())

        if turn.get("mistake_field"):
            fields.append(turn["mistake_field"])

        return self.public_memory_fields(self.unique_values(fields))

    def register_mentioned_memory_fields(self, turn: dict):
        """Remember which UM fields Leo has already brought into this conversation."""
        mentioned = getattr(self, "memory_fields_mentioned_so_far", None)
        if mentioned is None:
            self.memory_fields_mentioned_so_far = set()
            mentioned = self.memory_fields_mentioned_so_far

        for field in self.current_phase_memory_fields(turn):
            mentioned.add(field)

    def memory_access_scope(self, turn: dict) -> list:
        """Memory Leo may reveal now: previous mentions plus current phase fields."""
        mentioned = getattr(self, "memory_fields_mentioned_so_far", set())
        fields = list(mentioned) + self.current_phase_memory_fields(turn)
        return self.public_memory_fields(self.unique_values(fields))

    def memory_value(self, field: str) -> str:
        """Read from the already-pulled UM snapshot first, then fall back to the API."""
        value = self.last_um_preview.get(field)
        if self.is_known(value):
            return str(value)
        return self.get_field(field)

    def memory_access_summary(self, fields: list, limit: int = 4) -> str:
        parts = []
        for field in fields:
            value = self.memory_value(field)
            if self.is_known(value):
                parts.append(f"{self.field_label(field)}: {value}")
            if len(parts) >= limit:
                break
        return "; ".join(parts)

    def memory_access_response(self, result, turn: dict) -> tuple[str, list, list]:
        """Return a phase-aware memory answer for a um_inspect intent."""
        scope = self.memory_access_scope(turn)
        requested_field = getattr(result, "field", None)

        if requested_field and requested_field in scope:
            value = self.memory_value(requested_field)
            returned = [requested_field] if self.is_known(value) else []
            if returned:
                return f"Ik weet dat {self.field_label(requested_field)} {value} is.", scope, returned
            return f"Over {self.field_label(requested_field)} weet ik nu nog niets zeker.", scope, returned

        summary = self.memory_access_summary(scope)
        if requested_field and requested_field not in scope:
            if summary:
                return (
                    "Daar hebben we het vandaag nog niet over gehad. "
                    f"Ik kan wel vertellen wat ik al genoemd heb: {summary}."
                ), scope, []
            return (
                "Daar hebben we het vandaag nog niet over gehad, "
                "en ik heb vandaag nog niet zoveel uit mijn geheugen genoemd."
            ), scope, []

        if summary:
            returned = [
                field for field in scope
                if self.is_known(self.memory_value(field))
            ][:4]
            return f"Ik heb vandaag al genoemd: {summary}.", scope, returned

        return "Ik heb vandaag nog niet zoveel uit mijn geheugen genoemd.", scope, []

    def table_true_value(self, um: dict, field: str) -> str:
        if field == "name":
            return self.known(um, "child_name") or self.known(um, "name") or self.UNKNOWN_VALUE
        return self.known(um, field) or self.UNKNOWN_VALUE

    def script_memory_table(self, script: list, um: dict) -> list:
        script_values = {}
        used_fields = set()
        mistakes = {}

        for turn in script:
            for field, value in (turn.get("used_fields") or {}).items():
                if not self.is_known(value):
                    continue
                script_values.setdefault(field, []).append(str(value))
                used_fields.add(field)

            mistake_field = turn.get("mistake_field")
            if mistake_field:
                wrong = turn.get("mistake_wrong")
                if self.is_known(wrong):
                    script_values.setdefault(mistake_field, []).append(str(wrong))
                used_fields.add(mistake_field)
                if turn.get("mistake_id"):
                    mistakes[mistake_field] = (
                        f"{turn.get('mistake_id')} {turn.get('mistake_type', 'wrong')}"
                    )

        rows = []
        for field, spt in self.SCRIPT_TABLE_FIELDS:
            values = self.unique_values(script_values.get(field, []))
            rows.append({
                "field": field,
                "spt": spt,
                "true_value": self.table_true_value(um, field),
                "script_value": self.format_dutch_list(values, "-") if values else "-",
                "mistake": mistakes.get(field, "-"),
                "used": "yes" if field in used_fields else "-",
            })
        return rows

    def print_script_memory_table(self, rows: list):
        headers = ("Field", "SPT", "True Value", "Script Value", "Mistake?", "Used?")
        widths = [18, 22, 24, 28, 22, 6]
        print("\nWalkthrough memory table:")
        print("  " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
        print("  " + "-+-".join("-" * width for width in widths))
        for row in rows:
            values = (
                row["field"],
                row["spt"],
                row["true_value"],
                row["script_value"],
                row["mistake"],
                row["used"],
            )
            clipped = [
                (str(value)[:width - 3] + "...") if len(str(value)) > width else str(value)
                for value, width in zip(values, widths)
            ]
            print("  " + " | ".join(clipped[i].ljust(widths[i]) for i in range(len(widths))))

    def print_prestart_preview(self, script: list):
        """Print the walkthrough memory table before interaction starts."""
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
        print(f"Child id: {self.CHILD_ID}")
        print(f"Child:    {self.child_display_name(self.last_um_preview)}")
        print(f"Researcher: {self.researcher_name or '(not set)'}")
        print(f"Session:  #{self.session_number}")
        print(f"Start:    phase {self.start_phase_index + 1}")
        if self.resume_from_log_path:
            print(f"Resume:   {self.resume_from_log_path}")
            print(f"Restored mentioned memory fields: {len(self.memory_fields_mentioned_so_far)}")
        print(f"UM API:   {self.UM_API_BASE}")

        self.print_script_memory_table(self.script_memory_table(script, self.last_um_preview))

        print("=" * 72)
        input("Press Enter to start the interaction...")
        print()

    def build_script(self) -> list:
        """
        Mila Part 1 walkthrough flow.

        This follows the content-layer reference directly:
        L1 scripted text, L2-slot UM templates, L2-pregen validated stored
        utterances, and L3 runtime branches after unpredictable child input.
        """
        um = self.pull_um()
        self.alert_condition_mismatch(um)
        name = self.child_display_name(um)
        first_topic = self.select_part1_topic1(um)
        second_topic = self.select_part1_topic2(um, first_topic)

        m1_topic = self.hobby_mistake_topic(um)
        m1_field = "hobby_fav"
        m1_actual = m1_topic.get("current_values", {}).get(m1_field) or m1_topic.get("label")
        m1_wrong = self.related_wrong_hobby_value(um)

        m2_field = "fav_food"
        m2_actual = self.known(um, m2_field) or "pannenkoeken"
        m2_wrong = self.pick_wrong_value(m2_actual, ["pizza", "pasta", "spruitjes"])
        m2_topic = self.topic_candidate(
            domain="eten",
            label="je lievelingseten",
            fields=["fav_food"],
            field_labels={"fav_food": "je lievelingseten"},
            current_values={"fav_food": m2_actual},
            correct_values=[f"je lievelingseten {m2_actual} is"],
            memory_link=f"je lievelingseten {m2_actual} is",
            options=[m2_actual, "iets anders dat je lekker vindt"],
            reground=f"Ik weet zeker dat {m2_actual} met jouw lievelingseten te maken heeft.",
        )
        general_topic = self.general_memory_topic(um)
        story_activity = self.preferred_story_activity(um)
        story_activity_spoken = story_activity[:1].upper() + story_activity[1:] if story_activity else "Iets nieuws proberen"
        story_is_baking = story_activity.strip().lower() in ("bakken", "koken", "taarten bakken")
        story_problem = "een klein deegdrama" if story_is_baking else "een klein robotdrama"
        story_question = "Heb jij eigenlijk ooit een lama zien eten?" if story_is_baking else "Heb jij eigenlijk ooit een lama iets geks zien doen?"
        hobbies = self.format_dutch_list(self.all_hobbies(um), "leuke dingen")

        self.logger.info(
            "UM pulled for Part 1 phase flow - child:%s topic1:%s topic2:%s m1:%s m2:%s",
            name,
            first_topic["domain"],
            second_topic["domain"],
            m1_field,
            m2_field,
        )

        return [
            {
                "phase": 1,
                "name": "Greeting",
                "layer": "L1 + L2-slot: first_name",
                "dialogue_case": self.CASE_UM_TEMPLATE,
                "content_plan": self.l2_slot(
                    "Hoi {first_name}! Wat fijn om je weer te zien. Heb je een beetje zin om met mij te kletsen?",
                    {"first_name": name},
                ),
                "follow_up": "Ik heb er zelf ook veel zin in. Kom, dan gaan we lekker beginnen.",
                "response_mode": "acknowledge",
                "llm_turn": False,
                "used_fields": {"name": name},
                "example_child": "Ja.",
                "example_leo_after": "Ik heb er zelf ook veel zin in. Kom, dan gaan we lekker beginnen.",
            },
            {
                "phase": 2,
                "name": "Tutorial",
                "layer": "L1",
                "dialogue_case": self.CASE_FULLY_SCRIPTED,
                "content_plan": self.sequence(
                    self.l1(
                        "Ik zal eerst uitleggen hoe je met mij kunt praten. "
                        "Ik kan je alleen verstaan nadat ik een vraag heb gesteld. "
                        "Mijn ogen worden groen als ik luister. "
                        "Als je antwoord geeft, doe dat dan luid en duidelijk."
                    ),
                    self.l1("Vandaag ga ik mijn geheugen best veel gebruiken. Je mag altijd vragen wat ik over jou onthoud."),
                    self.l1(
                        "Ik probeer alles netjes op de goede plek te bewaren, maar soms gaat dat nog een beetje robotachtig mis. "
                        "Dus als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen."
                    ),
                    self.l1("Goed, dan gaan we beginnen."),
                ),
                "tutorial_condition": self.tutorial_condition(um),
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
            {
                "phase": 3,
                "name": "Leo mini-story",
                "layer": "L1+L3",
                "dialogue_case": self.CASE_PREAUTHORED_POOL,
                "segments": [
                    {
                        "content_plan": self.l1(
                            f"Weet je wat ik laatst weer probeerde? {story_activity_spoken}. "
                            f"Dat klinkt heel indrukwekkend, maar eerlijk gezegd was het meer {story_problem}. "
                            f"Mijn lama-vrienden vonden het wel een succes, want die eten bijna alles op. {story_question}"
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.l1(
                            "Ik vind het gewoon leuk om nieuwe dingen uit te proberen, ook als het een beetje mislukt. "
                            "Doe jij dat ook wel eens?"
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.l1(
                            "Dat snap ik wel. Nieuwe dingen proberen kan leuk zijn, maar soms ook een beetje spannend."
                        ),
                        "expects_response": False,
                    },
                ],
                "example_child": "Nee, nog nooit.",
                "used_fields": {"hobbies": self.known(um, "hobbies")},
            },
            {
                "phase": 4,
                "name": "Correct hobby bridge",
                "layer": "L1 + L2-slot + L2-pregen",
                "dialogue_case": self.CASE_MIXED_SEQUENCE,
                "content_plan": self.sequence(
                    self.l2_slot(
                        "Ik weet al dat jij ook van leuke dingen houdt. Jij houdt van {hobbies}.",
                        {"hobbies": hobbies},
                    ),
                    self.l2_pregen(
                        "hobbies_bridge",
                        "Dat vind ik echt een gezellige combinatie. Daar zit van alles in: bewegen, bedenken en iets maken.",
                        ["hobbies"],
                    ),
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": {
                    "hobbies": self.known(um, "hobbies"),
                    "hobby_fav": self.known(um, "hobby_fav"),
                    "freetime_fav": self.known(um, "freetime_fav"),
                },
            },
            {
                "phase": 5,
                "name": "Topic 1",
                "layer": "L2+L3",
                "dialogue_case": self.CASE_MIXED_SEQUENCE,
                "segments": self.topic1_phase_segments(first_topic),
                "topic": first_topic,
                "memory_link": first_topic["memory_link"],
                "llm_turn": True,
                "used_fields": first_topic.get("current_values", {}),
                "example_child": "Daar is niets nieuws mee gebeurd.",
            },
            {
                "phase": 6,
                "name": "Mistake 1 - hobby_fav",
                "layer": "L2-slot WRONG + L2-pregen",
                "dialogue_case": self.CASE_MIXED_SEQUENCE,
                "mistake_id": "M1",
                "mistake_type": "related-but-wrong",
                "mistake_field": m1_field,
                "mistake_actual": m1_actual,
                "mistake_wrong": m1_wrong,
                "mistake_topic": m1_topic,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.l2_slot(
                            "En volgens mij is {wrong_hobby} jouw allerliefste hobby.",
                            {"wrong_hobby": m1_wrong},
                            wrong=True,
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                    },
                    {
                        "content_plan": self.sequence(
                            self.l1("Dat snap ik trouwens wel."),
                            self.l2_pregen(
                                "m1_wrong_followup",
                                f"Iets maken met {m1_wrong} klinkt best indrukwekkend. Wat vind jij daar zo leuk aan?",
                                [m1_field],
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                    },
                ],
                "used_fields": {m1_field: m1_wrong},
                "example_child": f"Nee, {m1_wrong} klopt niet.",
                "example_leo_after": "Oeps, dan had ik dat verkeerd. Wat is je favoriete hobby?",
            },
            {
                "phase": 7,
                "name": "Topic 2",
                "layer": "L2+L3",
                "dialogue_case": self.CASE_MIXED_SEQUENCE,
                "segments": self.topic2_phase_segments(second_topic),
                "topic": second_topic,
                "memory_link": second_topic["memory_link"],
                "llm_turn": True,
                "used_fields": second_topic.get("current_values", {}),
                "example_child": "Ja, dat klopt.",
            },
            {
                "phase": 8,
                "name": "Mistake 2 - fav_food",
                "layer": "L1 + L2-slot WRONG + L2-pregen",
                "dialogue_case": self.CASE_MIXED_SEQUENCE,
                "mistake_id": "M2",
                "mistake_type": "completely-wrong",
                "mistake_field": m2_field,
                "mistake_actual": m2_actual,
                "mistake_wrong": m2_wrong,
                "mistake_topic": m2_topic,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.sequence(
                            self.l1("Van al dat praten krijg ik trouwens trek."),
                            self.l2_slot(
                                "Ik weet nog dat jouw lievelingseten {wrong_food} is.",
                                {"wrong_food": m2_wrong},
                                wrong=True,
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                    },
                    {
                        "content_plan": self.sequence(
                            self.l1("Dat is op zich wel een sterke keuze."),
                            self.l2_pregen(
                                "m2_wrong_followup",
                                f"Rond, warm, handig. Wat vind jij daar eigenlijk zo lekker aan?",
                                [m2_field],
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                    },
                ],
                "used_fields": {m2_field: m2_wrong},
                "example_child": f"Nee, {m2_wrong} klopt niet.",
                "example_leo_after": "Oeps, dan had ik dat verkeerd. Wat moet ik daarover onthouden?",
            },
            {
                "phase": 9,
                "name": "Nudge",
                "layer": "L1",
                "condition": "run_if_two_mistakes_no_corrections",
                "condition_label": "only if both mistakes passed without correction",
                "dialogue_case": self.CASE_FULLY_SCRIPTED,
                "content_plan": self.l1("Zeg, ik heb al een paar dingen over jou gezegd vandaag. Klopte eigenlijk alles wat ik zei?"),
                "follow_up": "We kunnen ook samen kijken wat ik over jou onthoud, als je wilt.",
                "response_mode": "topic_interpretation",
                "topic": general_topic,
                "memory_link": general_topic["memory_link"],
                "llm_turn": True,
                "example_child": "Nee, er klopte iets niet.",
                "example_leo_after": "Oeps. Wil je zeggen wat er niet klopte?",
            },
        ]

    # Conversation logging

    def log_timestamp(self) -> float:
        start = getattr(self, "conversation_log_started_monotonic", None)
        if start is None:
            return 0.0
        return round(max(0.0, time.monotonic() - start), 3)

    def format_log_timestamp(self, timestamp) -> str:
        try:
            total_seconds = max(0, int(float(timestamp)))
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes:02d}:{seconds:02d}"
        except (TypeError, ValueError):
            return "00:00"

    def safe_filename_part(self, value: str) -> str:
        clean = str(value or "").strip()
        safe = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in clean
        )
        return safe.strip("_") or "child"

    def conversation_child_name(self) -> str:
        local_name = str(getattr(self, "local_child_name", "") or "").strip()
        if local_name:
            return local_name
        return (
            self.known(self.last_um_preview, "child_name")
            or self.known(self.last_um_preview, "name")
            or self.CHILD_ID
        )

    def conversation_session_id(self, child_name: str, started: datetime) -> str:
        child_part = self.safe_filename_part(child_name)
        child_id_part = self.safe_filename_part(self.CHILD_ID)
        readable_time = started.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{child_part}_{child_id_part}_{readable_time}"

    def planned_turn_log(self, turn: dict) -> dict:
        entry = {
            "phase": self.turn_phase(turn),
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "dialogue_case": self.dialogue_case(turn),
            "runtime_llm": self.requires_runtime_llm(turn),
            "content_plan": self.content_plan_log(turn.get("content_plan")),
            "leo_text": self.turn_text(turn),
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
                    "dialogue_case": self.dialogue_case({**turn, **segment}),
                    "runtime_llm": self.requires_runtime_llm({**turn, **segment}),
                    "content_plan": self.content_plan_log(segment.get("content_plan")),
                    "leo_text": self.turn_text({**turn, **segment}),
                    "expects_response": segment.get("expects_response", True),
                    "response_mode": segment.get("response_mode", turn.get("response_mode")),
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
            "mistakes_mentioned": self.mistakes_mentioned,
            "corrections_seen": self.corrections_seen,
            "mistake_states": dict(self.mistake_states),
            "phases_with_confirmed_change": sorted(self.phases_with_confirmed_change),
            "memory_fields_mentioned_so_far": sorted(self.memory_fields_mentioned_so_far),
        }

    def sync_runtime_state_to_log(self):
        if not getattr(self, "conversation_log", None):
            return
        self.conversation_log.update(self.runtime_state_snapshot())

    def start_conversation_log(self, script: list):
        if not self.CONVERSATION_LOG_ENABLED:
            return

        started = datetime.now().astimezone()
        self.conversation_log_started_monotonic = time.monotonic()
        child_name = self.conversation_child_name()
        session_id = self.conversation_session_id(child_name, started)
        session_dir = os.path.join(self.CONVERSATION_LOG_ROOT, session_id)
        counter = 2
        while os.path.exists(session_dir):
            session_dir = os.path.join(self.CONVERSATION_LOG_ROOT, f"{session_id}_{counter}")
            counter += 1

        file_base = self.safe_filename_part(child_name)
        os.makedirs(session_dir, exist_ok=True)

        self.conversation_log = {
            "session_id": os.path.basename(session_dir),
            "script_version": self.SCRIPT_VERSION,
            "child_id": self.CHILD_ID,
            "child_name": child_name,
            "child_input_mode": self.child_input_mode,
            "researcher_name": self.researcher_name,
            "session_number": self.session_number,
            "session_config": dict(self.session_config),
            "resume_from_log": self.resume_from_log_path,
            "start_phase": self.start_phase_index + 1,
            "current_phase": None,
            "last_completed_phase": None,
            "resume_phase": self.start_phase_index + 1,
            "tutorial_condition": self.tutorial_condition(self.last_um_preview),
            "started_at": self.log_timestamp(),
            "started_wall_time": started.isoformat(timespec="seconds"),
            "ended_at": None,
            "timestamp_unit": "seconds_from_interaction_start",
            "folder": session_dir,
            "txt_path": os.path.join(session_dir, f"{file_base}.txt"),
            "json_path": os.path.join(session_dir, f"{file_base}.json"),
            "um_snapshot_start": dict(self.last_um_preview),
            "planned_phases": [self.planned_turn_log(turn) for turn in script],
            "turns": [],
            "events": [],
        }
        self.conversation_log.update(self.runtime_state_snapshot())
        self.current_turn_log = None
        self.write_conversation_logs()
        self.logger.info("Conversation log folder: %s", session_dir)

    def finish_conversation_log(self):
        if not getattr(self, "conversation_log", None):
            return
        if self.current_turn_log:
            self.finish_turn_log()
        self.conversation_log["ended_at"] = self.log_timestamp()
        self.write_conversation_logs()

    def start_turn_log(self, turn: dict):
        if not getattr(self, "conversation_log", None):
            return
        phase = self.turn_phase(turn)
        self.conversation_log["current_phase"] = phase
        self.conversation_log["resume_phase"] = phase
        self.sync_runtime_state_to_log()
        self.current_turn_log = {
            "phase": phase,
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "dialogue_case": self.dialogue_case(turn),
            "started_at": self.log_timestamp(),
            "ended_at": None,
            "leo_text": self.turn_text(turn),
            "events": [],
        }
        self.conversation_log["turns"].append(self.current_turn_log)
        self.log_conversation_event(
            "phase_start",
            phase=phase,
            name=turn.get("name"),
            layer=turn.get("layer"),
            dialogue_case=self.dialogue_case(turn),
        )

    def finish_turn_log(self):
        if not getattr(self, "conversation_log", None) or not self.current_turn_log:
            return
        phase = self.current_turn_log.get("phase")
        name = self.current_turn_log.get("name")
        self.current_turn_log["ended_at"] = self.log_timestamp()
        self.conversation_log["last_completed_phase"] = phase
        self.conversation_log["current_phase"] = None
        self.conversation_log["resume_phase"] = min(int(phase or 0) + 1, self.TOTAL_SCRIPT_PHASES)
        self.sync_runtime_state_to_log()
        self.log_conversation_event("phase_end", phase=phase, name=name)
        self.current_turn_log = None

    def log_conversation_event(self, event_type: str, **data):
        if not getattr(self, "conversation_log", None):
            return

        event = {
            "timestamp": self.log_timestamp(),
            "type": event_type,
        }
        event.update(data)
        self.conversation_log["events"].append(event)
        if self.current_turn_log is not None:
            self.current_turn_log.setdefault("events", []).append(event)
        self.sync_runtime_state_to_log()
        self.write_conversation_logs()

    def render_conversation_text(self) -> str:
        log = self.conversation_log or {}
        lines = [
            f"Session: {log.get('session_id', '')}",
            f"Script: {log.get('script_version', '')}",
            f"Child id: {log.get('child_id', '')}",
            f"Child name: {log.get('child_name', '')}",
            f"Researcher: {log.get('researcher_name', '')}",
            f"Session number: {log.get('session_number', '')}",
            f"Child input mode: {log.get('child_input_mode', '')}",
            f"Tutorial condition: {log.get('tutorial_condition', '')}",
            f"Resume from log: {log.get('resume_from_log') or ''}",
            f"Resume phase: {log.get('resume_phase', '')}",
            f"Started: {self.format_log_timestamp(log.get('started_at', 0.0))}",
            f"Ended: {self.format_log_timestamp(log.get('ended_at')) if log.get('ended_at') is not None else ''}",
            f"Wall start: {log.get('started_wall_time', '')}",
            "",
            "UM snapshot at start:",
        ]

        for field, value in sorted((log.get("um_snapshot_start") or {}).items()):
            lines.append(f"  {field}: {value}")

        lines.extend(["", "Planned phases:"])
        for turn in log.get("planned_phases", []):
            lines.append(
                f"  {turn.get('phase')}. {turn.get('name')} "
                f"[{turn.get('layer')} | {turn.get('dialogue_case')}]"
            )
            lines.append(f"     {turn.get('leo_text')}")
            for segment in turn.get("segments", []):
                lines.append(
                    f"       {turn.get('phase')}.{segment.get('index')} "
                    f"[{segment.get('dialogue_case')}] {segment.get('leo_text')}"
                )

        lines.extend(["", "Conversation:"])
        for event in log.get("events", []):
            timestamp = self.format_log_timestamp(event.get("timestamp", 0.0))
            event_type = event.get("type")
            if event_type == "phase_start":
                lines.append("")
                lines.append(
                    f"[{timestamp}] Phase {event.get('phase')}: {event.get('name')} "
                    f"[{event.get('layer')} | {event.get('dialogue_case')}]"
                )
            elif event_type == "utterance":
                lines.append(f"[{timestamp}] {event.get('speaker')}: {event.get('text')}")
            elif event_type == "transcript_review":
                lines.append(
                    f"[{timestamp}] TRANSCRIPT {event.get('action')}: {event.get('transcript')}"
                )
            elif event_type == "intent":
                lines.append(
                    f"[{timestamp}] INTENT: transcript={event.get('transcript')} "
                    f"result={event.get('result')}"
                )
            elif event_type == "intent_classifier":
                result = event.get("result") or {}
                lines.append(
                    f"[{timestamp}] INTENT CLASSIFIER: transcript={event.get('transcript')} "
                    f"intent={result.get('intent')} field={result.get('field')} "
                    f"value={result.get('value')} confidence={result.get('confidence')}"
                )
            elif event_type == "action_handler":
                lines.append(
                    f"[{timestamp}] ACTION HANDLER: action={event.get('action')} "
                    f"handled={event.get('handled')} reason={event.get('reason')} "
                    f"tutorial_condition={event.get('tutorial_condition')} "
                    f"change={event.get('change')} "
                    f"memory_scope={event.get('memory_scope')} "
                    f"returned_fields={event.get('returned_fields')} "
                    f"nudge_level={event.get('nudge_level')} "
                    f"mistake_id={event.get('mistake_id')}"
                )
            elif event_type == "llm_decision":
                lines.append(
                    f"[{timestamp}] LLM DECISION ({event.get('mode')}): "
                    f"transcript={event.get('transcript')} "
                    f"decision={event.get('decision')} "
                    f"confidence={event.get('confidence')} "
                    f"leo_response={event.get('leo_response')} "
                    f"change={event.get('change')}"
                )
            elif event_type == "interpretation":
                lines.append(
                    f"[{timestamp}] INTERPRETATION ({event.get('mode')}): {event.get('result')}"
                )
            elif event_type == "um_write":
                lines.append(
                    f"[{timestamp}] UM WRITE: field={event.get('field')} "
                    f"action={event.get('action')} success={event.get('success')} "
                    f"status={event.get('status_code')}"
                )
            elif event_type == "tester_control":
                lines.append(f"[{timestamp}] TESTER: {event.get('action')}")
            elif event_type == "phase_skipped":
                lines.append(
                    f"[{timestamp}] Phase {event.get('phase')} skipped: "
                    f"{event.get('name')} ({event.get('condition')})"
                )
            elif event_type == "phase_end":
                lines.append(f"[{timestamp}] Phase {event.get('phase')} finished")
            else:
                lines.append(f"[{timestamp}] {event_type}: {event}")

        return "\n".join(lines) + "\n"

    def write_conversation_logs(self):
        if not getattr(self, "conversation_log", None):
            return
        try:
            self.sync_runtime_state_to_log()
            with open(self.conversation_log["json_path"], "w", encoding="utf-8") as json_file:
                json.dump(self.conversation_log, json_file, ensure_ascii=False, indent=2)
            with open(self.conversation_log["txt_path"], "w", encoding="utf-8") as txt_file:
                txt_file.write(self.render_conversation_text())
        except Exception as e:
            self.logger.error("Could not write conversation log: %s", e)

    def log_llm_decision(self, mode: str, transcript: str, result: dict, context: dict = None):
        """Log the context-aware LLM choice that decides Leo's next action."""
        if not getattr(self, "conversation_log", None):
            return

        change = result.get("change") if isinstance(result.get("change"), dict) else {}
        decision = result.get("response_type") or result.get("decision") or "unknown"
        self.log_conversation_event(
            "llm_decision",
            mode=mode,
            transcript=transcript or "(nothing)",
            decision=decision,
            confidence=result.get("confidence"),
            leo_response=result.get("leo_response"),
            change=change,
            proposes_change=bool(change),
            wrong_value_rejected=result.get("wrong_value_rejected"),
            wrong_value_accepted=result.get("wrong_value_accepted"),
            context=context or {},
            result=result,
        )

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
            phase=(self.current_turn_context or {}).get("phase"),
            name=(self.current_turn_context or {}).get("name"),
        )

    def log_action_handler_result(self, action_result: dict):
        self.log_conversation_event(
            "action_handler",
            phase=(self.current_turn_context or {}).get("phase"),
            name=(self.current_turn_context or {}).get("name"),
            **action_result,
        )

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

    # Helpers

    def say(self, text: str):
        """Speak text via NAO TTS and wait for it to finish before returning."""
        if not text or not text.strip():
            return
        text = self.strip_non_bmp(text)
        self.logger.info("LEO: %s", text)
        self.last_leo_utterance = text
        if self.simulation_mode:
            self.simulated_history.append({"speaker": "LEO", "text": text})
        self.log_conversation_event("utterance", speaker="LEO", text=text)
        if self.simulation_mode or self.USE_DESKTOP_MIC:
            print(f"\n[LEO]: {text}\n")
        else:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
            # Wait proportional to text length so Whisper doesn't start
            # listening while NAO is still speaking.
            # ~0.01s per character is a safe estimate for NAO's TTS speed.
            speaking_time = len(text) * 0.01
            time.sleep(speaking_time)

    def listen(self) -> str:
        if self.simulation_mode:
            self.logger.info("Simulating child response...")
            transcript = self.generate_simulated_child_response()
            self.logger.info("Simulated child: %s", transcript or "(nothing)")
            self.simulated_history.append({"speaker": "CHILD", "text": transcript or "(nothing)"})
            self.log_conversation_event(
                "utterance",
                speaker="CHILD",
                text=transcript or "(nothing)",
                simulated=True,
            )
            return transcript

        if self.use_keyboard_input():
            transcript = input("[CHILD]: ").strip()
            self.logger.info("Child typed: %s", transcript or "(nothing)")
            self.log_conversation_event(
                "utterance",
                speaker="CHILD",
                text=transcript or "(nothing)",
                input_mode="keyboard",
            )
            return transcript

        self.logger.info("Listening...")
        try:
            result = self.whisper.request(
                GetTranscript(
                    timeout=self.STT_TIMEOUT,
                    phrase_time_limit=self.STT_PHRASE_LIMIT,
                )
            )
            transcript = result.transcript.strip() if result and result.transcript else ""
            self.logger.info("Child: %s", transcript or "(nothing)")
            self.log_conversation_event(
                "utterance",
                speaker="CHILD",
                text=transcript or "(nothing)",
                input_mode="microphone",
            )
            return transcript
        except Exception as e:
            self.logger.error("STT error: %s", e)
            self.log_conversation_event("stt_error", error=str(e))
            return ""

    def review_transcript(self, transcript: str) -> str:
        """Let the tester approve Whisper text or listen again before continuing."""
        if not self.REVIEW_TRANSCRIPTS:
            return transcript

        while True:
            print("\n" + "-" * 72)
            print(f"[HEARD]: {transcript or '(nothing)'}")
            choice = input("Press Enter to continue, or R + Enter to listen again: ").strip().lower()
            print("-" * 72)

            if choice == "":
                self.log_conversation_event(
                    "transcript_review",
                    action="accepted",
                    transcript=transcript or "(nothing)",
                )
                return transcript
            if choice == "r":
                self.log_conversation_event(
                    "transcript_review",
                    action="retry_requested",
                    transcript=transcript or "(nothing)",
                )
                transcript = self.listen()
                continue

            print("Please press Enter to continue, or type R and press Enter to listen again.")

    def listen_with_review(self) -> str:
        """Listen once, then optionally let the tester approve the transcript."""
        if self.use_keyboard_input():
            return self.listen()
        return self.review_transcript(self.listen())

    def strip_non_bmp(self, text: str) -> str:
        """Remove emoji-style characters that NAO TTS can choke on."""
        safe_chars = []
        for char in str(text or ""):
            if ord(char) > 0xFFFF:
                continue
            if unicodedata.category(char) in ("So", "Sk"):
                continue
            safe_chars.append(char)
        return "".join(safe_chars)

    def classify_with_repeat(self, transcript: str):
        """Classify once, ask for repetition on low confidence, then retry."""
        result = self.clf.classify(transcript)
        if result.intent == REPEAT_SENTINEL:
            self.logger.info("Low confidence - asking to repeat.")
            self.say("Kun je dat nog een keer zeggen?")
            time.sleep(0.8)
            transcript = self.listen_with_review()
            result = self.clf.classify_retry(transcript)
        self.logger.info("Intent: %s", result.to_dict())
        self.log_intent_classifier_result(transcript, result)
        return result

    def llm_response(self, child_input: str) -> str:
        """L3: GPT generates a personalised Dutch follow-up."""
        if not child_input:
            return self.LLM_FALLBACK
        prompt = (
            f"Het kind zei: \"{child_input}\". "
            f"Reageer warm en enthousiast in een korte zin in het Nederlands."
        )
        try:
            if self.gpt is not None:
                reply = self.gpt.request(GPTRequest(prompt=prompt, stream=False))
                response = reply.response.strip() if reply and reply.response else ""
            else:
                reply = self.openai_client.chat.completions.create(
                    model=self.TOPIC_CHANGE_MODEL,
                    messages=[
                        {"role": "system", "content": self.LLM_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=80,
                    temperature=0.7,
                )
                response = reply.choices[0].message.content.strip()
            response = self.strip_non_bmp(response)
            return (response.split(".")[0].strip() + ".") if response else self.LLM_FALLBACK
        except Exception as e:
            self.logger.error("LLM error: %s", e)
            return self.LLM_FALLBACK

    def extract_json_object(self, raw: str) -> dict:
        """Parse a JSON object even if the model accidentally adds light wrapping."""
        if not raw:
            return {}
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}

    def clean_confirmation_question(self, question: str, change: dict) -> str:
        """Keep LLM-generated confirmation wording, but enforce style constraints."""
        if not self.is_known(question):
            return ""

        clean = str(question).strip()
        clean = clean.replace("Zeg ja of nee.", "").replace("Zeg ja of nee", "")
        clean = clean.replace("zeg ja of nee.", "").replace("zeg ja of nee", "")
        clean = " ".join(clean.split())

        old_value = change.get("old_value")
        if self.is_known(old_value) and str(old_value).strip().lower() in clean.lower():
            return ""

        return clean

    def confirmation_text(self, change: dict) -> str:
        llm_question = self.clean_confirmation_question(change.get("confirmation_question"), change)
        if llm_question:
            return llm_question

        if change["action"] == "delete":
            return f"Wil je dat ik {change['field_label']} vergeet?"

        new_value = change.get("new_value")
        return f"Wil je dat ik {change['field_label']} verander naar {new_value}?"

    # Action handler: embedded intent result + current Phase context -> next action.

    def turn_memory_context(self, turn: dict) -> dict:
        topic = turn.get("topic") or turn.get("mistake_topic") or {}
        return {
            "topic": topic,
            "fields": topic.get("fields", []) or [],
            "field_labels": topic.get("field_labels", {}) or {},
            "current_values": topic.get("current_values", {}) or {},
        }

    def allowed_change_fields(self, turn: dict) -> list:
        context = self.turn_memory_context(turn)
        fields = list(context.get("fields") or [])
        if turn.get("mistake_field") and turn["mistake_field"] not in fields:
            fields.append(turn["mistake_field"])
        return fields or list(self.UM_FIELDS)

    def action_result(
        self,
        action: str,
        handled: bool,
        reason: str = "",
        change: dict = None,
        leo_response: str = None,
        follow_up_needed: bool = False,
        **extra,
    ) -> dict:
        result = {
            "action": action,
            "handled": handled,
            "reason": reason,
            "change": change or {},
            "leo_response": leo_response,
            "follow_up_needed": follow_up_needed,
        }
        result.update(extra)
        return result

    def register_mistake_phase(self, turn: dict):
        """Remember that a deliberate mistake has been stated in this phase."""
        mistake_id = turn.get("mistake_id")
        if not mistake_id:
            return
        state = self.mistake_states.setdefault(mistake_id, {})
        state.update({
            "id": mistake_id,
            "mentioned": True,
            "corrected": state.get("corrected", False),
            "nudge_count": state.get("nudge_count", 0),
            "phase": turn.get("phase"),
            "field": turn.get("mistake_field"),
            "field_label": self.mistake_field_label(turn),
            "actual": turn.get("mistake_actual"),
            "wrong": turn.get("mistake_wrong"),
            "type": turn.get("mistake_type"),
        })

    def mark_current_mistake_corrected(self):
        """Mark the current deliberate mistake corrected after a confirmed UM change."""
        context = self.current_turn_context or {}
        mistake_id = context.get("mistake_id")
        if not mistake_id:
            return
        state = self.mistake_states.setdefault(mistake_id, {"id": mistake_id})
        state["corrected"] = True
        state["corrected_at_phase"] = context.get("phase")

    def mistake_field_label(self, turn: dict) -> str:
        field = turn.get("mistake_field")
        topic = turn.get("mistake_topic") or {}
        return (topic.get("field_labels") or {}).get(field) or self.field_label(field)

    def first_uncorrected_mistake_state(self) -> dict:
        for mistake_id in sorted(self.mistake_states):
            state = self.mistake_states[mistake_id]
            if state.get("mentioned") and not state.get("corrected"):
                return state
        return {}

    def nudge_state_for_turn(self, turn: dict) -> dict:
        mistake_id = turn.get("mistake_id")
        if mistake_id:
            self.register_mistake_phase(turn)
            return self.mistake_states.get(mistake_id, {})
        return self.first_uncorrected_mistake_state()

    def gentle_mistake_nudge_text(self, state: dict) -> str:
        field_label = state.get("field_label") or "iets uit mijn geheugen"
        return (
            f"Ik zei net iets uit mijn geheugen over {field_label}. "
            "Als daar iets niet helemaal klopt, mag je mij gewoon verbeteren. "
            "Wat moet ik hierover onthouden?"
        )

    def explicit_mistake_nudge_text(self, state: dict) -> str:
        field_label = state.get("field_label") or "iets uit mijn geheugen"
        wrong = state.get("wrong") or "dat"
        return (
            f"Ik zei net dat {wrong} jouw {field_label} is. "
            "Dat was misschien niet goed. Wat moet ik daarover onthouden?"
        )

    def mistake_nudge_action(self, turn: dict, force_explicit: bool = False) -> dict:
        """ActionHandler-owned nudge decision after no correction/memory access was detected."""
        state = self.nudge_state_for_turn(turn)
        if not state:
            return {}

        if force_explicit:
            level = 2
        else:
            level = int(state.get("nudge_count", 0)) + 1
            if level > 2:
                return {}

        state["nudge_count"] = max(level, int(state.get("nudge_count", 0)))
        response = (
            self.explicit_mistake_nudge_text(state)
            if level >= 2
            else self.gentle_mistake_nudge_text(state)
        )
        self.say(response)
        return self.action_result(
            "explicit_mistake_nudge" if level >= 2 else "gentle_mistake_nudge",
            True,
            "Action handler triggered nudge because no correction or memory access intent was detected.",
            leo_response=response,
            follow_up_needed=True,
            nudge_level=level,
            mistake_id=state.get("id"),
            mistake_field=state.get("field"),
            mistake_wrong=state.get("wrong"),
        )

    def change_from_intent_result(self, result, turn: dict, transcript: str) -> dict:
        intent = result.intent
        if intent not in ("um_add", "um_update", "dialogue_update", "um_delete"):
            return {}

        allowed = self.allowed_change_fields(turn)
        context = self.turn_memory_context(turn)
        field = result.field
        value = result.value

        if not field and turn.get("mistake_field") and intent in ("um_add", "um_update", "dialogue_update"):
            field = turn.get("mistake_field")
        if not field and len(allowed) == 1:
            field = allowed[0]
        if field not in allowed:
            return {}

        field_label = context["field_labels"].get(field) or self.field_label(field)
        old_value = (
            context["current_values"].get(field)
            or self.last_um_preview.get(field)
            or self.UNKNOWN_VALUE
        )

        if intent == "um_delete":
            return {
                "action": "delete",
                "field": field,
                "field_label": field_label,
                "old_value": str(old_value),
                "new_value": None,
                "confidence": result.confidence,
                "reason": f"Intent classifier returned {intent}.",
                "source_text": transcript,
                "confirmation_question": f"Wil je dat ik {field_label} vergeet?",
            }

        if not self.is_known(value):
            return {}
        if self.is_known(old_value) and str(old_value).strip().lower() == str(value).strip().lower():
            return {}

        return {
            "action": "update",
            "field": field,
            "field_label": field_label,
            "old_value": str(old_value),
            "new_value": str(value),
            "confidence": result.confidence,
            "reason": f"Intent classifier returned {intent}.",
            "source_text": transcript,
            "confirmation_question": f"Wil je dat ik {field_label} verander naar {value}?",
        }

    def is_rejection_without_value(self, result, transcript: str) -> bool:
        text = str(transcript or "").lower()
        rejection_words = ("klopt niet", "niet waar", "verkeerd", "helemaal niet", "nee")
        if result.intent in ("um_update", "dialogue_update") and not self.is_known(result.value):
            return True
        return any(word in text for word in rejection_words) and result.intent in (
            "dialogue_answer",
            "dialogue_social",
            "dialogue_none",
            "um_update",
        )

    def is_confirmation_yes(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        yes_phrases = (
            "ja", "jawel", "zeker", "klopt", "dat klopt", "goed",
            "is goed", "dat is goed", "doe dat maar", "alsjeblieft",
            "prima", "ok", "oke", "okÃ©", "mag",
        )
        return any(phrase in text for phrase in yes_phrases)

    def is_confirmation_no(self, result, transcript: str) -> bool:
        text = str(transcript or "").strip().lower()
        no_phrases = ("nee", "niet", "klopt niet", "laat maar", "verander niets")
        return any(phrase in text for phrase in no_phrases)

    def confirmation_decision_from_intent(self, result, transcript: str, change: dict) -> dict:
        refined_change = self.change_from_intent_result(
            result,
            {
                "topic": {
                    "fields": [change.get("field")],
                    "field_labels": {change.get("field"): change.get("field_label")},
                    "current_values": {change.get("field"): change.get("old_value")},
                }
            },
            transcript,
        )
        if refined_change and refined_change.get("new_value") != change.get("new_value"):
            return self.action_result(
                "refine_confirmation_change",
                True,
                "Child refined the value during confirmation.",
                change=refined_change,
            )
        if self.is_confirmation_no(result, transcript):
            return self.action_result(
                "reject_change",
                True,
                "Child rejected the proposed change.",
                change=change,
            )
        if self.is_confirmation_yes(result, transcript):
            return self.action_result(
                "confirm_change",
                True,
                "Child confirmed the proposed change.",
                change=change,
            )
        return self.action_result(
            "ask_confirmation_again",
            True,
            "Confirmation intent was unclear.",
            change=change,
            leo_response=self.confirmation_text(change),
        )

    def action_handler(self, result, transcript: str, turn: dict) -> dict:
        intent = result.intent
        mode = turn.get("response_mode")
        change = self.change_from_intent_result(result, turn, transcript)

        if change:
            self.confirm_topic_change(change)
            return self.action_result(
                f"confirm_{change['action']}",
                True,
                "UM-changing intent from classifier.",
                change=change,
            )

        if intent == "um_inspect":
            condition = self.tutorial_condition(self.last_um_preview)
            if condition == "C2":
                response = "Je kunt mijn geheugenboek op de tablet bekijken."
                self.say(response)
                return self.action_result(
                    "memory_access_tablet",
                    True,
                    "C2 memory access: child is redirected to the tablet memory book.",
                    leo_response=response,
                    tutorial_condition=condition,
                    requested_field=result.field,
                )

            response, memory_scope, returned_fields = self.memory_access_response(result, turn)
            self.say(response)
            return self.action_result(
                "memory_access",
                True,
                "Child requested memory access; returned scoped memory only.",
                leo_response=response,
                memory_scope=memory_scope,
                returned_fields=returned_fields,
                requested_field=result.field,
                tutorial_condition=condition,
            )

        if mode in ("mistake_interpretation", "topic_interpretation") and self.is_rejection_without_value(result, transcript):
            response = (
                self.mistake_correction_question(turn)
                if mode == "mistake_interpretation"
                else self.topic_correction_question(turn.get("topic", {}))
            )
            self.say(response)
            return self.action_result(
                "ask_correction_detail",
                True,
                "Child rejected remembered information without giving a new value.",
                leo_response=response,
                follow_up_needed=True,
            )

        if turn.get("condition") == "run_if_two_mistakes_no_corrections":
            nudge = self.mistake_nudge_action(turn, force_explicit=True)
            if nudge:
                return nudge

        if intent == "dialogue_question":
            response = self.llm_response(transcript)
            self.say(response)
            return self.action_result(
                "answer_dialogue_question",
                True,
                "Child asked a dialogue question.",
                leo_response=response,
            )

        if mode == "mistake_interpretation":
            nudge = self.mistake_nudge_action(turn)
            if nudge:
                return nudge

        if mode == "listen_only":
            return self.action_result(
                "listen_only",
                True,
                "Child response heard; no Leo response planned for this phase segment.",
            )

        if mode == "acknowledge":
            response = self.llm_response(transcript) if turn.get("llm_turn") and transcript else turn.get("follow_up")
            self.say(response or self.LLM_FALLBACK)
            return self.action_result(
                "acknowledge",
                True,
                "Acknowledgement phase.",
                leo_response=response,
            )

        if mode == "mistake_interpretation":
            response = turn.get("follow_up") or "Dankjewel, dan ga ik verder."
            self.say(response)
            return self.action_result(
                "no_memory_change",
                True,
                "No UM-changing correction detected after deliberate mistake.",
                leo_response=response,
            )

        if mode == "topic_interpretation":
            response = self.topic_no_update_response(turn.get("topic", {}))
            self.say(response)
            return self.action_result(
                "no_memory_change",
                True,
                "No UM-changing intent detected for topic phase.",
                leo_response=response,
            )

        if intent == "dialogue_social":
            response = "Haha ja! OkÃ©, verder!"
            self.say(response)
            return self.action_result("dialogue_social", True, "Social response.", leo_response=response)

        if intent == "dialogue_answer":
            response = turn.get("follow_up") or "OkÃ©, dankjewel."
            self.say(response)
            return self.action_result("dialogue_answer", True, "Direct answer.", leo_response=response)

        return self.action_result("unhandled", False, f"No route for intent {intent}.")

    def follow_up_action_handler(self, turn: dict, max_rounds: int = 3):
        """Listen after action-handler prompts such as correction questions or nudges."""
        action = {}
        for _ in range(max_rounds):
            time.sleep(0.5)
            transcript = self.listen_with_review()
            time.sleep(0.8)
            result = self.classify_with_repeat(transcript)
            action = self.action_handler(result, transcript, turn)
            self.log_action_handler_result(action)
            if not action.get("follow_up_needed"):
                return action
        return action

    def write_um_change(self, change: dict) -> bool:
        field = change["field"]
        if self.simulation_mode:
            try:
                if change["action"] == "delete":
                    self.simulated_persona[field] = self.UNKNOWN_VALUE
                    self.last_um_preview[field] = self.UNKNOWN_VALUE
                    new_value = None
                else:
                    new_value = change["new_value"]
                    self.simulated_persona[field] = new_value
                    self.last_um_preview[field] = new_value

                if self.SIMULATION_WRITE_PERSONA_FILE:
                    with open(self.simulated_persona_path, "w", encoding="utf-8") as persona_file:
                        json.dump(self.simulated_persona, persona_file, ensure_ascii=False, indent=2)

                self.log_conversation_event(
                    "um_write",
                    action=change.get("action"),
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=new_value,
                    success=True,
                    status_code="simulation",
                )
                return True
            except Exception as e:
                self.logger.error("Could not apply simulated UM change: %s", e)
                self.log_conversation_event(
                    "um_write",
                    action=change.get("action"),
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=change.get("new_value"),
                    success=False,
                    status_code="simulation",
                    error=str(e),
                )
                return False

        try:
            if change["action"] == "delete":
                url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/field/{field}"
                response = requests.delete(url, timeout=3)
                ok = response.status_code in (200, 202, 204, 404)
                if ok:
                    self.last_um_preview[field] = self.UNKNOWN_VALUE
                self.log_conversation_event(
                    "um_write",
                    action="delete",
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=None,
                    success=ok,
                    status_code=response.status_code,
                )
                return ok

            url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/fields"
            payload = {
                "fields": {field: change["new_value"]},
                "source": "cri_4_topic_confirmation",
            }
            response = requests.post(url, json=payload, timeout=3)
            ok = response.status_code in (200, 201, 202, 204)
            if ok:
                self.last_um_preview[field] = change["new_value"]
            self.log_conversation_event(
                "um_write",
                action="update",
                field=field,
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                success=ok,
                status_code=response.status_code,
            )
            return ok
        except Exception as e:
            self.logger.error("Could not write confirmed UM change: %s", e)
            self.log_conversation_event(
                "um_write",
                action=change.get("action"),
                field=field,
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                success=False,
                status_code=None,
                error=str(e),
            )
            return False

    def confirm_topic_change(self, change: dict) -> bool:
        self.pending_change = change

        while True:
            self.say(self.confirmation_text(change))
            time.sleep(0.5)

            confirmation = self.listen_with_review()
            time.sleep(0.8)

            result = self.classify_with_repeat(confirmation)
            decision = self.confirmation_decision_from_intent(result, confirmation, change)
            self.log_action_handler_result(decision)

            if decision["action"] == "refine_confirmation_change":
                change = decision["change"]
                self.pending_change = change
                continue

            if decision["action"] == "confirm_change":
                self.corrections_seen += 1
                self.mark_current_mistake_corrected()
                written = self.write_um_change(change)
                if self.current_turn_context:
                    self.phases_with_confirmed_change.add(self.current_turn_context.get("phase"))
                if written:
                    self.say("Dankjewel, ik heb dat aangepast.")
                else:
                    self.say("Dankjewel, ik heb dat genoteerd, maar opslaan lukte nu niet.")
                self.pending_change = None
                return True

            if decision["action"] == "reject_change":
                self.say("OkÃ©, dan verander ik niets.")
                self.pending_change = None
                return False

            self.say(decision.get("leo_response") or self.confirmation_text(change))

    def handle_intent(self, result, transcript: str) -> bool:
        """Old inactive route; active routing uses action_handler()."""
        return False
        intent = result.intent
        field = result.field

        if intent in ("um_update", "dialogue_update"):
            self.corrections_seen += 1
            self.logger.info("Correction count is now %d.", self.corrections_seen)

        if intent == "um_inspect":
            value = self.get_field(field)
            self.say(f"Ik weet dat jouw {field or 'antwoord'} {value} is.")
            return True

        elif intent == "um_update":
            self.say("Oh, je hebt gelijk! Ik pas het aan.")
            if field and result.value:
                old = self.get_field(field)
                self.say(f"Dus jouw {field} is {result.value}, niet {old}. Leuk!")
                # TODO: write correction to Eunike's API (Sherissa's task):
                # requests.post(f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/fields",
                #     json={"fields": {field: result.value}, "source": "child_correction"})
            return True

        elif intent == "dialogue_update":
            self.say("OkÃ©, ik hoorde dat. Klopt dat?")
            return True

        elif intent == "um_delete":
            self.say("OkÃ©, ik vergeet dat!")
            return True

        elif intent == "dialogue_question":
            self.say("Dat is een goede vraag! Ik vertel het je later.")
            return True

        elif intent == "dialogue_social":
            self.say("Haha ja! OkÃ©, verder!")
            return True

        # um_add, dialogue_answer, and dialogue_none fall through to L3.
        return False

    def phase_expects_response(self, turn: dict) -> bool:
        if turn.get("segments"):
            return any(segment.get("expects_response", True) for segment in turn["segments"])
        return turn.get("expects_response", True)

    def post_phase_control(self, turn: dict) -> str:
        """Testing checkpoint after a full phase finishes."""
        if not self.POST_PHASE_TEST_CONTROLS:
            return "continue"
        if not self.phase_expects_response(turn):
            return "continue"

        while True:
            print("\n" + "=" * 72)
            print(f"Phase {self.turn_phase(turn)} finished: {turn.get('name')}")
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

    def should_skip_phase(self, turn: dict) -> bool:
        condition = turn.get("condition")
        if condition == "skip_if_change_after_phase":
            return turn.get("condition_phase") in self.phases_with_confirmed_change
        if condition == "run_if_two_mistakes_no_corrections":
            return not (self.mistakes_mentioned >= 2 and self.corrections_seen == 0)
        return False

    def handle_interpreted_response(self, interpretation: dict, turn: dict, mode: str) -> bool:
        """
        Handle one LLM interpretation.

        Returns True when Leo asked a clarification question and should listen
        once more within the same phase.
        """
        change = interpretation.get("change", {})
        if change:
            self.confirm_topic_change(change)
            return False

        leo_response = interpretation.get("leo_response") or turn.get("follow_up") or self.LLM_FALLBACK
        self.say(leo_response)
        return interpretation.get("response_type") == "correction_unclear"

    def follow_up_interpretation(self, turn: dict, mode: str):
        """Listen once after Leo asks a clarification question, then interpret that answer."""
        time.sleep(0.5)
        transcript = self.listen_with_review()
        time.sleep(0.8)
        if not transcript.strip():
            self.say("OkÃ©, dan verander ik nu nog niets.")
            return

        if mode == "topic":
            interpretation = self.interpret_topic_response(transcript, turn.get("topic", {}))
        else:
            interpretation = self.interpret_mistake_response(transcript, turn)

        # One clarification round is enough for this phase. If it is still unclear,
        # keep the memory unchanged and move on.
        if interpretation.get("response_type") == "correction_unclear" and not interpretation.get("change"):
            self.say("Dankjewel. Ik weet het nog niet zeker, dus ik verander nu niets.")
            return

        self.handle_interpreted_response(interpretation, turn, mode)

    def segment_context(self, phase: dict, segment: dict, segment_index: int = None) -> dict:
        context = dict(phase)
        context.update(segment)
        context.pop("segments", None)
        context["phase"] = self.turn_phase(phase)
        if segment_index is not None:
            context["segment"] = segment_index
        return context

    def run_phase_segment(self, phase: dict, segment: dict, segment_index: int = None):
        context = self.segment_context(phase, segment, segment_index)
        self.current_turn_context = context

        if segment.get("skip_if_phase_confirmed_change") and self.turn_phase(phase) in self.phases_with_confirmed_change:
            return

        self.say(self.turn_text(context))
        self.register_mentioned_memory_fields(context)

        if not context.get("expects_response", True):
            self.logger.info("No child response expected for this phase segment.")
            return

        time.sleep(0.5)
        transcript = self.listen_with_review()
        time.sleep(0.8)

        result = self.classify_with_repeat(transcript)
        action = self.action_handler(result, transcript, context)
        self.log_action_handler_result(action)
        if action.get("follow_up_needed"):
            self.follow_up_action_handler(context)
            return

        if not action.get("handled"):
            if context.get("llm_turn") and transcript:
                self.say(self.llm_response(transcript))
            else:
                self.say(context.get("follow_up", ""))

    def run_phase(self, turn: dict, phase_index: int, total_phases: int):
        self.current_turn_context = turn
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
                self.mistakes_mentioned += 1
                self.register_mistake_phase(turn)
                self.logger.info(
                    "Mistake %s mentioned; count is now %d.",
                    turn["mistake_id"],
                    self.mistakes_mentioned,
                )

            segments = turn.get("segments")
            if segments:
                for index, segment in enumerate(segments, start=1):
                    if self.shutdown_event.is_set():
                        break
                    self.run_phase_segment(turn, segment, index)
                return

            self.run_phase_segment(turn, turn)
        finally:
            self.finish_turn_log()
            self.current_turn_context = None

    # Main loop

    def run(self):
        self.logger.info("Starting CRI 4.0 early interaction flow.")

        script = self.build_script()
        if not self.resume_from_log_path:
            self.memory_fields_mentioned_so_far = set()
        self.logger.info("Script ready - %d phases.", len(script))
        self.start_conversation_log(script)
        self.print_prestart_preview(script)

        try:
            if not self.simulation_mode and not self.USE_DESKTOP_MIC:
                self.nao.autonomous.request(NaoWakeUpRequest())

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
                if not self.simulation_mode and not self.USE_DESKTOP_MIC:
                    self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass
            self.finish_conversation_log()
            self.logger.info("Shutting down.")
            self.shutdown()


if __name__ == "__main__":
    dialogue_app = CRI_ScriptedDialogue(
        openai_env_path=os.path.abspath(os.path.join(_HERE, "..", "conf", ".env")),
        nao_ip="10.0.0.165",  # Replace with your NAO's IP.
    )
    dialogue_app.run()

