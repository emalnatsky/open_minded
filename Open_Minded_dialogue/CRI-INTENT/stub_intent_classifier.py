"""
Rule-based (stub) intent classifier for the CRI dialogue pipeline.

Loads valid field names directly from um_field_schema.json so there
is ONE source of truth. Adding a field to the schema = automatically
available here.

Returns the same JSON structure as Lena's real LLM classifier:
    {intent, field, value, confidence}

Intent naming convention (v3.0.0):
    um_*        — touches GraphDB (read or write via Eunike's API)
    dialogue_*  — conversation only, no database operation

Swap StubIntentClassifier for the real one via DialogueManager's
constructor (the interface is identical).

Language: Dutch (child utterances from Qualtrics)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Load valid fields from schema
# ---------------------------------------------------------------------------

def _load_valid_fields(schema_path: str = "um_field_schema.json") -> list:
    path = Path(schema_path)
    if not path.exists():
        return [
            "hobby_1", "hobby_2", "hobby_3", "hobby_4", "hobby_fav",
            "sports_enjoys", "sports_fav", "sports_plays", "sports_fav_play",
            "music_enjoys", "music_plays_instrument", "music_instrument",
            "books_enjoys", "books_fav_genre", "books_fav_title",
            "freetime_fav", "has_best_friend",
            "animals_enjoys", "animal_fav",
            "has_pet", "pet_type", "pet_name",
            "fav_food",
            "fav_subject", "school_strength", "school_difficulty",
            "interest", "aspiration", "role_model",
            "age",
        ]
    with open(path, encoding="utf-8") as f:
        return json.load(f)["valid_fields"]


# ---------------------------------------------------------------------------
# Field aliases
# ---------------------------------------------------------------------------

FIELD_ALIASES: dict = {
    # ── HOBBY ──────────────────────────────────────────────────────────────
    "hobby_fav":              ["lievelingshobby", "favoriete hobby", "liefste hobby",
                               "hobby vind je het leukst", "allerleukste hobby",
                               "mijn favoriete hobby", "leukste hobby"],
    "hobby_1":                ["hobby 1", "eerste hobby"],
    "hobby_2":                ["hobby 2", "tweede hobby"],
    "hobby_3":                ["hobby 3", "derde hobby"],
    "hobby_4":                ["hobby 4", "vierde hobby"],

    # ── SPORT ──────────────────────────────────────────────────────────────
    "sports_enjoys":          ["sport leuk", "van sport", "sportief",
                               "hou je van sport", "vind je sport"],
    "sports_fav":             ["lievelingssport", "favoriete sport", "leukste sport",
                               "sport vind je het leukst", "mijn lievelingssport"],
    "sports_plays":           ["zelf sport", "doe je een sport", "speel je sport",
                               "doe je zelf"],
    "sports_fav_play":        ["sport doe je zelf", "liefste sport om te doen",
                               "sport speel je zelf het liefst", "sport doe ik zelf"],

    # ── MUSIC ──────────────────────────────────────────────────────────────
    "music_enjoys":           ["muziek leuk", "van muziek", "muziek mooi",
                               "houd je van muziek", "ik hou van muziek"],
    "music_plays_instrument": ["instrument speel", "speel je een instrument",
                               "ik speel een instrument"],
    "music_instrument":       ["welk instrument", "instrument", "gitaar", "piano",
                               "viool", "drums", "fluit", "cello", "trompet",
                               "saxofoon", "ukelele", "keyboard"],

    # ── BOOKS ──────────────────────────────────────────────────────────────
    "books_enjoys":           ["lezen leuk", "graag lees", "lees je graag",
                               "boeken leuk", "ik lees graag"],
    "books_fav_genre":        ["soort boeken", "genre", "boeken genre",
                               "voor boeken lees je het liefst",
                               "soort boek", "avontuur boek", "fantasy boek"],
    "books_fav_title":        ["lievelingsboek", "favoriete boek", "boek heet",
                               "boek is", "welk boek", "mijn lievelingsboek"],

    # ── FREE TIME / SOCIAL ─────────────────────────────────────────────────
    "freetime_fav":           ["vrije tijd", "als ik vrij ben", "liefst doen",
                               "doe je liefst", "ik vrij ben", "ik doe het liefst",
                               "gamen", "buiten spelen", "knutselen", "dansen",
                               "puzzelen", "filmpjes kijken"],
    "has_best_friend":        ["beste vriend", "bff", "beste vriendin",
                               "iemand waarmee je speelt", "vrienden"],

    # ── ANIMALS ────────────────────────────────────────────────────────────
    "animals_enjoys":         ["dieren leuk", "van dieren", "houd je van dieren",
                               "ik hou van dieren"],
    "animal_fav":             ["lievelingsdier", "favoriete dier", "leukste dier",
                               "dier vind je het leukst", "mijn lievelingsdier"],

    # ── PETS ───────────────────────────────────────────────────────────────
    "pet_type":               ["soort huisdier", "wat voor huisdier",
                               "mijn huisdier is een", "mijn huisdier is",
                               "hond", "kat", "konijn", "hamster", "vogel", "vis", "reptiel"],
    "pet_name":               ["naam huisdier", "hoe heet je huisdier",
                               "huisdier heet", "naam van je huisdier",
                               "mijn huisdier heet"],
    "has_pet":                ["heb je een huisdier", "eigen dier",
                               "ik heb een huisdier", "huisdier"],

    # ── FOOD ───────────────────────────────────────────────────────────────
    "fav_food":               ["lievelingseten", "lievelings eten", "favoriete eten",
                               "lekkerste eten", "eten vind je het lekkerst",
                               "lievelingskostje", "het liefst eet",
                               "mijn lievelingseten", "lekkerste"],

    # ── SCHOOL ─────────────────────────────────────────────────────────────
    "fav_subject":            ["lievelingsvak", "favoriete vak", "leukste vak",
                               "vak vind je het leukst", "mijn lievelingsvak"],
    "school_strength":        ["goed in", "vakken ben je goed", "sterk in",
                               "makkelijk vak", "waar je goed in bent",
                               "ik ben goed in"],
    "school_difficulty":      ["moeilijk vak", "lastig vak", "vak dat je lastig vindt",
                               "vak vind je moeilijk", "school moeilijk",
                               "vind ik lastig", "vind ik moeilijk", "moeilijk",
                               "lastig vindt", "soms lastig"],

    # ── ASPIRATION ─────────────────────────────────────────────────────────
    "interest":               ["interesseert", "interessant", "meer over weten",
                               "nieuwsgierig naar", "vind je interessant",
                               "wil ik meer over weten", "ik wil meer over weten"],
    "aspiration":             ["later worden", "beroep", "droom", "wil je worden",
                               "werk je later", "wat wil je worden",
                               "ik wil later", "later wil ik"],
    "role_model":             ["voorbeeld", "kijk je op", "bewonder", "held",
                               "iemand die je cool vindt", "opkijkt naar",
                               "kijk ik op", "ik kijk op"],

    # ── PERSONAL ───────────────────────────────────────────────────────────
    "age":                    ["oud", "leeftijd", "jaar oud", "hoe oud",
                               "ik ben", "ik word"],
}


# ---------------------------------------------------------------------------
# Intent patterns — evaluated in order, first match wins.
# DO NOT reorder without understanding the priority logic.
# Naming convention: um_* = touches DB, dialogue_* = conversation only.
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list = [
    # UM_DELETE — check first
    (re.compile(
        r"\b(vergeet|verwijder|wis|schrap|gooi weg|dat klopt niet meer|niet meer waar)\b",
        re.IGNORECASE,
    ), "um_delete"),

    # UM_UPDATE — child corrects robot's stored knowledge, check before dialogue_update
    (re.compile(
        r"\b(eigenlijk niet meer|dat was fout|dat klopt niet|corrigeer|dat is veranderd|"
        r"niet meer|ik had het fout|was vroeger|nu is het anders)\b",
        re.IGNORECASE,
    ), "um_update"),

    # DIALOGUE_UPDATE — child corrects themselves mid-turn, check before um_add
    (re.compile(
        r"\b(nee wacht|ik bedoel|laat maar|nee toch|ik bedoelde|wacht nee|"
        r"eigenlijk bedoel ik|nee ik bedoel)\b",
        re.IGNORECASE,
    ), "dialogue_update"),

    # UM_INSPECT — child asks what robot knows about THEM, check before um_add
    (re.compile(
        r"\b(wat weet je|weet je nog|herinner je|wat heb je|vertel me|laat zien|"
        r"wat staat er|klopt het dat|weet je wat mijn|wat is mijn)\b",
        re.IGNORECASE,
    ), "um_inspect"),

    # DIALOGUE_QUESTION — child asks robot about itself or the world
    (re.compile(
        r"\b(waarom|hoe heet jij|wat kan jij|ben jij|hoe werkt|wat doe jij|"
        r"waarom wil je|wie ben jij|wat ben jij|kan jij)\b",
        re.IGNORECASE,
    ), "dialogue_question"),

    # DIALOGUE_SOCIAL — conversational filler, emotions, simple yes/no
    (re.compile(
        r"\b(haha|hehe|grappig|leuk|cool|wow|super|geweldig|oké|oke|"
        r"jaja|nee nee|echt waar|wauw|tof|nice)\b",
        re.IGNORECASE,
    ), "dialogue_social"),

    # UM_ADD — child provides new info about themselves
    (re.compile(
        r"\b(ik vind|mijn .{1,30} is|ik heet|ik ben|ik heb|ik doe|ik speel|ik lees|"
        r"ik wil|ik hou van|mijn lievelings|ik word later|later wil ik)\b",
        re.IGNORECASE,
    ), "um_add"),

    # DIALOGUE_NONE — filler sounds and hesitation (check BEFORE dialogue_answer)
    # Matches ONLY strings made entirely of filler words: "um", "um eh ja nee" etc.
    (re.compile(
        r"^\s*(um|uh|eh|hmm?|ahh?|uhh?)(\s+(um|uh|eh|hmm?|ahh?|uhh?|ja|nee))*\s*$",
        re.IGNORECASE,
    ), "dialogue_none"),

    # DIALOGUE_ANSWER — short responses, likely answering robot's question
    (re.compile(
        r"^[a-zàáâäèéêëìíîïòóôöùúûüA-Z0-9 '\-]{1,40}$",
        re.IGNORECASE,
    ), "dialogue_answer"),
]


# ---------------------------------------------------------------------------
# Return type — matches the JSON contract exactly
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent:     str
    field:      Optional[str]
    value:      Optional[str]
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "intent":     self.intent,
            "field":      self.field,
            "value":      self.value,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Stub classifier
# ---------------------------------------------------------------------------

class StubIntentClassifier:
    """
    Keyword / regex intent classifier for Dutch child utterances.

    Loads valid field names from um_field_schema.json at init.
    Falls back to hardcoded list if the file is not present.

    Intents (in priority order):
        um_delete        — child wants robot to forget something           → DB delete
        um_update        — child corrects robot's stored knowledge         → DB write
        dialogue_update  — child corrects themselves mid-turn              → no DB
        um_inspect       — child asks what robot knows about them          → DB read
        dialogue_question— child asks robot about itself or the world      → no DB
        dialogue_social  — filler / emotional reaction / simple yes/no     → no DB
        um_add           — child provides new info about themselves        → DB write
        dialogue_answer  — child answering robot's question (short)        → no DB
        dialogue_none    — silence, gibberish, or nothing recognised       → no DB

    Usage
    -----
    clf = StubIntentClassifier()
    result = clf.classify("Mijn lievelingseten is pizza")
    # IntentResult(intent='um_add', field='fav_food', value='pizza', confidence=1.0)

    Swap in real classifier
    -----------------------
    Any object with .classify(text: str) -> IntentResult is compatible.
    Pass it to DialogueManager(classifier=RealLLMClassifier()).
    """

    _VALUE_RE = re.compile(
        r"(?:is|ben|heet|zijn|wordt|vind ik|doe ik|speel ik|lees ik|hou ik van|heb ik)"
        r"\s+(?:een\s+)?([a-zàáâäèéêëìíîïòóôöùúûüA-Z0-9 '\-]{1,80}?)(?:[.,!?]|$)",
        re.IGNORECASE,
    )

    def __init__(self, schema_path: str = "um_field_schema.json"):
        self.valid_fields = _load_valid_fields(schema_path)

    def classify(self, text: str) -> IntentResult:
        """
        Classify a Dutch child utterance. Always returns a valid result, never raises.

        Parameters
        ----------
        text : str
            Raw transcript from Whisper STT.

        Returns
        -------
        IntentResult with intent, field, value, confidence.
        """
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None)

        text_clean = text.strip()
        intent = self._detect_intent(text_clean)
        field  = self._detect_field(text_clean)

        # only extract value for intents that write or correct a value
        value = (
            self._extract_value(text_clean)
            if intent in ("um_add", "um_update", "dialogue_update", "dialogue_answer")
            else None
        )

        return IntentResult(intent=intent, field=field, value=value, confidence=1.0)

    def _detect_intent(self, text: str) -> str:
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(text):
                return intent
        return "dialogue_none"

    def _detect_field(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for field, aliases in FIELD_ALIASES.items():
            if field not in self.valid_fields:
                continue
            for alias in aliases:
                if alias.lower() in text_lower:
                    return field
        return None

    # Secondary regex for "wil later X worden" / "wil X worden" patterns
    _WORDEN_RE = re.compile(
        r"(?:wil\s+(?:later\s+)?|word\s+later\s+)([a-zàáâäèéêëìíîïòóôöùúûüA-Z0-9\'\-][a-zàáâäèéêëìíîïòóôöùúûüA-Z0-9 \'\-]{0,40}?)\s+worden",
        re.IGNORECASE,
    )

    def _extract_value(self, text: str) -> Optional[str]:
        # Try primary regex first
        match = self._VALUE_RE.search(text)
        if match:
            val = match.group(1).strip()
            # Strip leading articles that slipped through
            for prefix in ("het ", "de ", "een "):
                if val.lower().startswith(prefix):
                    val = val[len(prefix):]
            return val if val else None
        # Try "wil ... worden" pattern
        match = self._WORDEN_RE.search(text)
        if match:
            return match.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# DialogueManager — plug-in point for CRI-BRANCH-BASIC.py
# ---------------------------------------------------------------------------

class DialogueManager:
    """
    Wires the classifier into the CRI dialogue loop.
    Swap stub for real classifier by passing classifier= argument.

    Example
    -------
    manager = DialogueManager()                        # stub
    manager = DialogueManager(classifier=RealLLM())    # real
    response = manager.handle("Mijn lievelingseten is pizza")
    """

    def __init__(self, classifier=None, schema_path: str = "um_field_schema.json"):
        self.classifier = classifier or StubIntentClassifier(schema_path)

    def handle(self, user_text: str) -> str:
        result = self.classifier.classify(user_text)
        return self._generate_response(result)

    @staticmethod
    def _generate_response(result: IntentResult) -> str:
        intent = result.intent
        field  = result.field or "dat"
        value  = result.value or "het"

        if intent == "um_add":
            return f"Oké, ik onthoud dat jouw {field} {value} is."
        if intent == "um_update":
            return f"Begrepen, ik pas jouw {field} aan naar {value}."
        if intent == "dialogue_update":
            return f"Oké, ik hoorde {value}. Klopt dat?"
        if intent == "um_delete":
            return f"Oké, ik vergeet jouw {field}."
        if intent == "um_inspect":
            return f"Je vroeg me naar jouw {field}."
        if intent == "dialogue_question":
            return "Dat is een goede vraag! Zal ik dat later uitleggen?"
        if intent == "dialogue_social":
            return "Haha, ja! Oké, verder!"
        if intent == "dialogue_answer":
            return "Oké, dankjewel!"
        return "Ik begreep dat niet helemaal. Kun je het nog een keer zeggen?"


# ---------------------------------------------------------------------------
# Smoke test — python intent_classifier.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clf = StubIntentClassifier()
    tests = [
        # um_add
        "Mijn lievelingseten is pizza",
        "Ik wil later dokter worden",
        "Ik heb een kat",
        # um_update
        "Eigenlijk niet meer pizza, nu is het sushi",
        "Dat was fout, ik vind eigenlijk gym het leukst",
        # dialogue_update
        "Pizza... nee wacht, ik bedoel sushi",
        "Nee toch, laat maar",
        # um_delete
        "Vergeet wat ik zei over mijn huisdier",
        # um_inspect
        "Wat weet je over mijn lievelingshobby?",
        "Wat is mijn lievelingseten?",
        # dialogue_question
        "Waarom wil je dat weten?",
        "Ben jij echt een robot?",
        # dialogue_social
        "Haha dat is grappig",
        "Oké!",
        # dialogue_answer
        "Pizza",
        "Voetbal",
        # dialogue_none
        "um eh ja nee",
        "",
    ]
    print(f"{'Utterance':<50} {'intent':<20} {'field':<22} value")
    print("-" * 115)
    for t in tests:
        r = clf.classify(t)
        print(f"{t!r:<50} {r.intent:<20}  {str(r.field):<20}  {r.value!r}")
