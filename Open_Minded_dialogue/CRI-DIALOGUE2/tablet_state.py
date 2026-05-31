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
  to ../session_state.json from its own folder â€” same file.
"""

import os
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# â”€â”€ Field to the tablet category map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Every UM field the dialogue can mention, mapped to the tablet's category key.
# Categories shown as labelled sections in the tablet memory-book UI.

# Keys must match exactly what script.js CATEGORIES object uses.
FIELD_TO_CATEGORY = {
    # Hobbies / free time  â†’ "hobby"
    "hobbies":           "hobby",
    "hobby_fav":         "hobby",
    "freetime_fav":      "hobby",

    # Sport                â†’ "sport"
    "sports_enjoys":     "sport",
    "sports_fav_play":   "sport",

    # Music                â†’ "muziek"
    "music_enjoys":      "muziek",

    # Books                â†’ "boeken"
    "books_enjoys":      "boeken",
    "books_fav_title":   "boeken",

    # Animals + pets       â†’ "dieren"
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

    # Social               â†’ "sociaal"
    "has_best_friend":   "sociaal",

    # These fields exist in UM but are never shown on the tablet:
    # name, age, exposure, condition â†’ not in any CATEGORIES key
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
            get_child_id_fn:       callable() â†’ str
            get_child_name_fn:     callable() â†’ str  (TTS/CRI name, fallback)
            get_tablet_name_fn:    callable() â†’ str  (display name on tablet book cover)
            get_condition_fn:      callable() â†’ str  (C1 or C2, written to session_state)
            get_mistake_states_fn: callable() â†’ dict of mistake state dicts
            enabled:               False â†’ no-op (e.g. when tablet isn't connected)
        """
        self.state_path = state_path
        self.enabled = enabled
        self._get_child_id = get_child_id_fn or (lambda: "")
        self._get_child_name = get_child_name_fn or (lambda: "")
        self._get_tablet_name = get_tablet_name_fn or get_child_name_fn or (lambda: "")
        self._get_condition = get_condition_fn or (lambda: "")
        self._get_mistake_states = get_mistake_states_fn or (lambda: {})

        # Tracks which categories have been unlocked so far this session.
        # Monotonically grows â€” categories are never re-locked.
        self._unlocked_categories: set = set()
        self._memory_access_active = False
        self._visible_fields = []
        self._memory_access_prompt_id = 0
        self._tablet_reveal = None
        self._tablet_reveal_pending = None
        self._tablet_reveal_id = 0
        self._tablet_command = None
        self._tablet_command_counter = 0
        self._tablet_command_session_id = time.time_ns()

    def update(self, turn: dict):
        """
        Call this at the start of each phase (from start_turn_log / run_phase).

        Inspects the turn's used_fields and mistake fields to determine which
        tablet categories to unlock, then writes the JSON file.
        """
        if not self.enabled:
            return

        self._tablet_reveal = None
        self._tablet_reveal_pending = None
        self._unlock_from_turn(turn)

        self._write_state(turn.get("phase"))

    def activate_memory_access(self, fields: list, phase=None):
        """Expose all fields from the mentioned-so-far memory chapters."""
        if not self.enabled:
            return

        self._tablet_reveal = None
        self._tablet_reveal_pending = None
        self._memory_access_active = True
        self._memory_access_prompt_id += 1
        self._issue_tablet_command("memory_access_home")
        self._visible_fields = list(dict.fromkeys(fields or []))
        for field in self._visible_fields:
            category = FIELD_TO_CATEGORY.get(field)
            if category:
                self._unlocked_categories.add(category)
        self._write_state(phase)

    def refresh(self, phase=None):
        """Rewrite current tablet state after an in-phase memory change."""
        if not self.enabled:
            return
        self._write_state(phase)

    def reveal_change(self, *, field: str, old_value: str, new_value: str, phase=None):
        """Command the tablet to open the right page and animate one correction."""
        if not self.enabled:
            return

        category = FIELD_TO_CATEGORY.get(field)
        if category:
            self._unlocked_categories.add(category)

        self._tablet_reveal_id += 1
        self._tablet_reveal_pending = None
        self._tablet_reveal = {
            "id": self._tablet_reveal_id,
            "field": field,
            "category": category,
            "old_value": old_value,
            "new_value": new_value,
            "created_at": time.time(),
        }
        self._write_state(phase)

    def prepare_reveal_change(self, *, field: str, old_value: str, new_value: str, phase=None):
        """Tell the tablet a field is waiting for an operator-controlled reveal."""
        if not self.enabled:
            return

        category = FIELD_TO_CATEGORY.get(field)
        if category:
            self._unlocked_categories.add(category)

        self._tablet_reveal = None
        self._tablet_reveal_pending = {
            "field": field,
            "category": category,
            "old_value": old_value,
            "new_value": new_value,
            "created_at": time.time(),
        }
        self._write_state(phase)

    def clear_pending_reveal(self, phase=None):
        """Clear a reveal that was queued but should no longer be held back."""
        if not self.enabled:
            return
        self._tablet_reveal_pending = None
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
            "memory_access_prompt_id": self._memory_access_prompt_id,
            "visible_fields":      list(self._visible_fields),
            "mistakes":            mistake_summary,
            "tablet_reveal":       self._tablet_reveal,
            "tablet_reveal_pending": self._tablet_reveal_pending,
            "tablet_command":      self._tablet_command,
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
        self._memory_access_prompt_id = 0
        self._tablet_reveal = None
        self._tablet_reveal_pending = None
        self._tablet_reveal_id = 0
        self._tablet_command = None
        self._tablet_command_counter = 0
        self._tablet_command_session_id = time.time_ns()
        self._write_state(None)

    # â”€â”€ internals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _issue_tablet_command(self, command_type: str):
        """Create a one-shot command for the browser UI."""
        self._tablet_command_counter += 1
        self._tablet_command = {
            "id": f"{self._tablet_command_session_id}:{self._tablet_command_counter}",
            "type": command_type,
            "created_at": time.time(),
        }

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

    def _build_mistake_summary(self) -> dict:
        """Compact view of mistake state for the tablet."""
        summary = {}
        for mistake_id, state in (self._get_mistake_states() or {}).items():
            summary[mistake_id] = {
                "field":       state.get("field"),
                "field_label": state.get("field_label"),
                "actual":      state.get("actual"),
                "wrong":       state.get("wrong"),
                "corrected":   bool(state.get("corrected", False)),
            }
        return summary
