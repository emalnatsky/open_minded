"""
StubIntentClassifier — rule-based fallback classifier.

Uses regex patterns and field-alias matching.  No network calls, no API key
required.  Used directly during development (when GPT is unavailable) and as
the fallback inside GPTIntentClassifier when the GPT call fails or returns
low confidence.
"""

import re
import logging
from typing import Optional

from .intent_result import IntentResult, EMBEDDED_VALID_FIELDS

logger = logging.getLogger(__name__)

# ── Dutch field aliases ───────────────────────────────────────────────────────

FIELD_ALIASES = {
    "hobby_fav":        ["lievelingshobby", "favoriete hobby", "liefste hobby", "leukste hobby", "mijn favoriete hobby"],
    "hobbies":          ["hobby's", "hobbies", "wat vind je leuk"],
    "sports_enjoys":    ["sport leuk", "van sport", "sportief", "hou je van sport", "vind je sport"],
    "sports_fav_play":  ["sport doe je zelf", "liefste sport om te doen", "sport speel je zelf"],
    "music_enjoys":     ["muziek leuk", "van muziek", "muziek mooi", "houd je van muziek", "ik hou van muziek"],
    "books_enjoys":     ["lezen leuk", "graag lees", "lees je graag", "boeken leuk", "ik lees graag"],
    "books_fav_title":  ["lievelingsboek", "favoriete boek", "boek heet", "boek is", "welk boek", "mijn lievelingsboek"],
    "freetime_fav":     ["vrije tijd", "als ik vrij ben", "liefst doen", "doe je liefst", "gamen", "buiten spelen", "knutselen", "dansen", "puzzelen", "filmpjes kijken"],
    "has_best_friend":  ["beste vriend", "bff", "beste vriendin", "vrienden"],
    "animals_enjoys":   ["dieren leuk", "van dieren", "houd je van dieren", "ik hou van dieren"],
    "animal_fav":       ["lievelingsdier", "favoriete dier", "leukste dier", "mijn lievelingsdier"],
    "pet_type":         ["soort huisdier", "wat voor huisdier", "mijn huisdier is een", "mijn huisdier is", "hond", "kat", "konijn", "hamster", "vogel", "vis", "reptiel"],
    "pet_name":         ["naam huisdier", "hoe heet je huisdier", "huisdier heet", "naam van je huisdier", "mijn huisdier heet"],
    "has_pet":          ["heb je een huisdier", "eigen dier", "ik heb een huisdier", "huisdier"],
    "fav_food":         ["lievelingseten", "lievelings eten", "favoriete eten", "lekkerste eten", "het liefst eet", "mijn lievelingseten"],
    "fav_subject":      ["lievelingsvak", "favoriete vak", "leukste vak", "mijn lievelingsvak"],
    "school_strength":  ["goed in", "vakken ben je goed", "sterk in", "makkelijk vak", "waar je goed in bent", "ik ben goed in"],
    "school_difficulty":["moeilijk vak", "lastig vak", "vak dat je lastig vindt", "school moeilijk", "vind ik lastig", "vind ik moeilijk"],
    "interest":         ["interesseert", "interessant", "meer over weten", "nieuwsgierig naar", "vind je interessant"],
    "aspiration":       ["later worden", "beroep", "droom", "wil je worden", "wat wil je worden", "ik wil later", "later wil ik"],
    "role_model":       ["voorbeeld", "kijk je op", "bewonder", "held", "opkijkt naar", "ik kijk op"],
    "age":              ["leeftijd", "jaar oud", "hoe oud", "ik ben", "ik word"],
    "name":             ["naam", "hoe heet ik", "mijn naam"],
    "condition":        ["condition", "conditie"],
}

# ── Intent patterns (evaluated in order; first match wins) ───────────────────

INTENT_PATTERNS = [
    (re.compile(r"\b(vergeet|verwijder|wis|schrap|gooi weg|dat klopt niet meer|niet meer waar)\b", re.IGNORECASE), "um_delete"),
    (re.compile(r"\b(eigenlijk niet meer|dat was fout|dat klopt niet|corrigeer|dat is veranderd|niet meer|ik had het fout|was vroeger|nu is het anders)\b", re.IGNORECASE), "um_update"),
    (re.compile(r"\b(nee wacht|ik bedoel|laat maar|nee toch|ik bedoelde|wacht nee|eigenlijk bedoel ik|nee ik bedoel)\b", re.IGNORECASE), "dialogue_update"),
    (re.compile(
        r"\b(wat weet je|weet je nog|herinner je|wat heb je|vertel me|laat zien|"
        r"wat staat er|klopt het dat|weet je wat mijn|wat is mijn|"
        r"wat onthoud je|wat heb je onthouden|heb je iets onthouden|"
        r"geheugen zien|mijn geheugen zien|je geheugen zien|"
        r"wat zei je net over mij|wat weet je nog van mij|wat weet je over mij|"
        r"wat weet je allemaal over mij)\b",
        re.IGNORECASE,
    ), "um_inspect"),
    (re.compile(r"\b(waarom|hoe heet jij|wat kan jij|ben jij|hoe werkt|wat doe jij|waarom wil je|wie ben jij|wat ben jij|kan jij)\b", re.IGNORECASE), "dialogue_question"),
    (re.compile(r"\b(haha|hehe|grappig|leuk|cool|wow|super|geweldig|oké|oke|jaja|nee nee|echt waar|wauw|tof|nice)\b", re.IGNORECASE), "dialogue_social"),
    (re.compile(r"\b(ik vind|mijn .{1,30} is|ik heet|ik ben|ik heb|ik doe|ik speel|ik lees|ik wil|ik hou van|mijn lievelings|ik word later|later wil ik)\b", re.IGNORECASE), "um_add"),
    (re.compile(r"^\s*(um|uh|eh|hmm?|ahh?|uhh?)(\s+(um|uh|eh|hmm?|ahh?|uhh?|ja|nee))*\s*$", re.IGNORECASE), "dialogue_none"),
    (re.compile(r"^[a-z áâäèéêëìíîïòóôöùúûüA-Z0-9 '\-]{1,40}$", re.IGNORECASE), "dialogue_answer"),
]


class StubIntentClassifier:
    """Rule-based classifier — no network calls, used as fallback."""

    VALUE_RE = re.compile(
        r"(?:is|ben|heet|zijn|wordt|vind ik|doe ik|speel ik|lees ik|hou ik van|heb ik)"
        r"\s+(?:een\s+)?([a-z áâäèéêëìíîïòóôöùúûüA-Z0-9 '\-]{1,80}?)(?:[.,!?]|$)",
        re.IGNORECASE,
    )
    WORDEN_RE = re.compile(
        r"(?:wil\s+(?:later\s+)?|word\s+later\s+)([a-z áâäèéêëìíîïòóôöùúûüA-Z0-9'\-][a-z áâäèéêëìíîïòóôöùúûüA-Z0-9 '\-]{0,40}?)\s+worden",
        re.IGNORECASE,
    )

    def __init__(self, schema_path: str = None, valid_fields: list = None):
        self.valid_fields = set(valid_fields or EMBEDDED_VALID_FIELDS)

    def classify(self, text: str) -> IntentResult:
        if not text or not text.strip():
            return IntentResult(intent="dialogue_none", field=None, value=None)

        text_clean = text.strip()
        intent = self._detect_intent(text_clean)
        field = self._detect_field(text_clean)
        value = (
            self._extract_value(text_clean)
            if intent in ("um_add", "um_update", "dialogue_update", "dialogue_answer")
            else None
        )
        return IntentResult(intent=intent, field=field, value=value, confidence=1.0)

    def _detect_intent(self, text: str) -> str:
        for pattern, intent in INTENT_PATTERNS:
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

    def _extract_value(self, text: str) -> Optional[str]:
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
