"""
It's a phase-gated memory-category state for the tablet UI.

The tablet reads session_state.json to know which memory categories are
unlocked (i.e. Leo has mentioned them). This module owns:

  - FIELD_TO_CATEGORY: maps every UM field to its tablet category name
  - TabletStateWriter: writes/updates session_state.json each phase

Constructed once in CRI_ScriptedDialogue.setup() as self.tablet_state.
Called from start_turn_log() (via run_phase) each time a new phase starts.

File location:
  The tablet server (UM-TABLET/) reads ../session_state.json relative to
  itself, which resolves to Open_Minded_dialogue/session_state.json.
  The dialogue lives at Open_Minded_dialogue/CRI-DIALOGUE/, so it writes
  to ../session_state.json from its own folder — same file.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Field to the tablet category map ───────────────────────────────────────────────
# Every UM field the dialogue can mention, mapped to the tablet's category key.
# Categories shown as labelled sections in the tablet memory-book UI.

# Keys must match exactly what script.js CATEGORIES object uses.
FIELD_TO_CATEGORY = {
    # Hobbies / free time  → "hobby"
    "hobbies":           "hobby",
    "hobby_fav":         "hobby",
    "freetime_fav":      "hobby",

    # Sport                → "sport"
    "sports_enjoys":     "sport",
    "sports_fav_play":   "sport",

    # Music                → "muziek"
    "music_enjoys":      "muziek",

    # Books                → "boeken"
    "books_enjoys":      "boeken",
    "books_fav_title":   "boeken",

    # Animals + pets       → "dieren"
    "animals_enjoys":    "dieren",
    "animal_fav":        "dieren",
    "has_pet":           "dieren",
    "pet_type":          "dieren",
    "pet_name":          "dieren",

    # Food                 "eten"
    "fav_food":          "eten",

    # School                "school"
    "fav_subject":       "school",
    "school_strength":   "school",
    "school_difficulty": "school",

    # Future / aspirations  "aspiratie"
    "aspiration":        "aspiratie",
    "role_model":        "aspiratie",
    "interest":          "aspiratie",

    # Social               → "sociaal"
    "has_best_friend":   "sociaal",

    # These fields exist in UM but are never shown on the tablet:
    # name, age, exposure, condition → not in any CATEGORIES key
}


class TabletStateWriter:
    """
    Writes session_state.json for the tablet after every phase start.

    The file format the tablet expects:
    {
        "child_id": "614",
        "child_name": "Noor",
        "phase": 5,
        "unlocked_categories": ["hobbies", "sport"],
        "mistakes": {
            "M1": {"field": "hobby_fav", "wrong": "voetbal", "corrected": false},
            "M2": {"field": "fav_food",  "wrong": "pizza",   "corrected": false}
        }
    }
    """

    def __init__(
        self,
        *,
        state_path: str,
        get_child_id_fn=None,
        get_child_name_fn=None,
        get_tablet_name_fn=None,
        get_condition_fn=None,
        get_mistake_states_fn=None,
        enabled: bool = True,
    ):
        """
        Args:
            state_path:            Absolute path to session_state.json
            get_child_id_fn:       callable() → str
            get_child_name_fn:     callable() → str  (TTS/CRI name, fallback)
            get_tablet_name_fn:    callable() → str  (display name on tablet book cover)
            get_condition_fn:      callable() → str  (C1 or C2, written to session_state)
            get_mistake_states_fn: callable() → dict of mistake state dicts
            enabled:               False → no-op (e.g. when tablet isn't connected)
        """
        self.state_path = state_path
        self.enabled = enabled
        self._get_child_id = get_child_id_fn or (lambda: "")
        self._get_child_name = get_child_name_fn or (lambda: "")
        self._get_tablet_name = get_tablet_name_fn or get_child_name_fn or (lambda: "")
        self._get_condition = get_condition_fn or (lambda: "")
        self._get_mistake_states = get_mistake_states_fn or (lambda: {})

        # Tracks which categories have been unlocked so far this session.
        # Monotonically grows — categories are never re-locked.
        self._unlocked_categories: set = set()
        self._memory_access_active = False
        self._visible_fields = []

    def update(self, turn: dict):
        """
        Call this at the start of each phase (from start_turn_log / run_phase).

        Inspects the turn's used_fields and mistake fields to determine which
        tablet categories to unlock, then writes the JSON file.
        """
        if not self.enabled:
            return

        self._unlock_from_turn(turn)

        self._write_state(turn.get("phase"))

    def activate_memory_access(self, fields: list, phase=None):
        """Expose the exact mentioned-so-far fields for a tablet memory inspection."""
        if not self.enabled:
            return

        self._memory_access_active = True
        self._visible_fields = list(dict.fromkeys(fields or []))
        for field in self._visible_fields:
            category = FIELD_TO_CATEGORY.get(field)
            if category:
                self._unlocked_categories.add(category)
        self._write_state(phase)

    def _write_state(self, phase=None):
        mistake_summary = self._build_mistake_summary()
        state = {
            "child_id":            self._get_child_id(),
            "child_name":          self._get_tablet_name() or self._get_child_name(),
            "condition":           self._get_condition(),
            "phase":               phase,
            "unlocked_categories": sorted(self._unlocked_categories),
            "memory_access_active": self._memory_access_active,
            "visible_fields":      list(self._visible_fields),
            "mistakes":            mistake_summary,
        }

        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(
                "Tablet state written: phase=%s unlocked=%s",
                phase,
                sorted(self._unlocked_categories),
            )
        except Exception as e:
            logger.error("Could not write tablet state to %s: %s", self.state_path, e)

    def reset(self):
        """Clear unlocked categories at session start."""
        self._unlocked_categories.clear()
        self._memory_access_active = False
        self._visible_fields = []

    # ── internals ─────────────────────────────────────────────────────────────

    def _unlock_from_turn(self, turn: dict):
        """Add categories for all fields mentioned in this turn."""
        # Fields explicitly used in the scripted turn
        for field in (turn.get("used_fields") or {}).keys():
            category = FIELD_TO_CATEGORY.get(field)
            if category:
                self._unlocked_categories.add(category)

        # Mistake field (deliberate wrong value Leo stated)
        mistake_field = turn.get("mistake_field")
        if mistake_field:
            category = FIELD_TO_CATEGORY.get(mistake_field)
            if category:
                self._unlocked_categories.add(category)

        # Segments — each segment may have its own used_fields
        for segment in turn.get("segments") or []:
            for field in (segment.get("used_fields") or {}).keys():
                category = FIELD_TO_CATEGORY.get(field)
                if category:
                    self._unlocked_categories.add(category)

    def _build_mistake_summary(self) -> dict:
        """Compact view of mistake state for the tablet."""
        summary = {}
        for mistake_id, state in (self._get_mistake_states() or {}).items():
            summary[mistake_id] = {
                "field":     state.get("field"),
                "wrong":     state.get("wrong"),
                "corrected": bool(state.get("corrected", False)),
            }
        return summary
