"""
Intent result contract — shared by the stub and GPT classifiers.

Keep this file import-free of OpenAI / SIC so it can be loaded anywhere
(unit tests, the tablet server, the dialogue manager) without heavy deps.
"""

from dataclasses import dataclass
from typing import Optional

# Returned by classify() when the classifier isn't confident enough to
# commit to an intent.  The dialogue then asks the child to repeat.
REPEAT_SENTINEL = "dialogue_repeat"

# Every intent the classifiers may return.  The dialogue's action_handler
# maps these to behaviour; classifiers must never return anything outside
# this list.
VALID_INTENTS = [
    "um_add",
    "um_update",
    "um_delete",
    "um_inspect",
    "dialogue_update",
    "dialogue_answer",
    "dialogue_question",
    "dialogue_social",
    "dialogue_none",
]

# All UM fields the classifiers should recognise.  Overridden at runtime
# by passing valid_fields=list(self.UM_FIELDS) to the classifier constructor.
EMBEDDED_VALID_FIELDS = [
    "name", "exposure", "hobbies", "condition",
    "hobby_fav",
    "sports_enjoys", "sports_fav_play",
    "books_enjoys", "books_fav_title",
    "music_enjoys",
    "animals_enjoys", "animal_fav",
    "has_pet", "pet_type", "pet_name",
    "freetime_fav", "fav_food", "fav_subject",
    "school_strength", "school_difficulty",
    "interest", "aspiration", "role_model", "has_best_friend",
    "age",
]


@dataclass
class IntentResult:
    """Structured return value from any classifier."""
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
