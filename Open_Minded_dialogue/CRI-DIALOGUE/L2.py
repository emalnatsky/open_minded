"""
SECOND LAYER 

No LLM needed. Used for: factual UM references, memory statements,summaries, and simple personalized recall.

TODO: Replace the stub strings below with validated Dutch lines.
      Keep the {field_name} placeholders exactly as they are — those are
      the UM field names that get filled at runtime.

STRUCTURE:
    L2 = {
        "phase_key": {
            "template_key": "Dutch string with {field_name} placeholders",
        }
    }
"""

from __future__ import annotations
import re
from typing import Optional


# --------------------------------------------------------------------------- L2 Templates---------------------------------------------------------------------------


L2_TEMPLATES: dict = {

    #  Phase 1: Greeting 
    "phase_1": {

    },

    #Phase 3: Interest exploration 
    "phase_3": {
        # Correct UM references


        # Deliberate mistakes


        # Correction reintegration after child corrects

    },

    #Phase 5: School bridge
    "phase_5": {
        # Correct UM reference


        # Deliberate mistake
 

        # Correction reintegration

    },

    #Phase 6: Aspiration
    "phase_6": {
        # Deliberate mistake 


        # Correct UM reference

    },

    #Phase 7: Memory inspection
    "phase_7": {
    },

    #Phase 8: Closing
    "phase_8": {

    },

    #Cross-phase: correction reintegration
    "correction": {

    },
}


# ---------------------------------------------------------------------------slot filler + Accessor--------------------------------------------------------------------------

class L2:
    """Accessor and slot-filler for L2 templates."""

    @staticmethod
    def fill(phase: str, key: str, um: dict, fallback: str = "") -> str:
        """
        Get a template and fill it with values from the UM dict.

        Parameters
        ----------
        phase    : str  e.g. "phase_3"
        key      : str  e.g. "state_animal_pet"
        um       : dict  user model values e.g. {"animal_fav": "honden", "pet_name": "Luna"}
        fallback : str  returned if template not found or fill fails

        Returns
        -------
        str — the filled Dutch utterance

        Example
        -------
        L2.fill("phase_3", "state_animal_pet", {"animal_fav": "honden", "pet_name": "Luna"})
        → "Jij vindt honden het leukste dier. En je hebt er zelf ook een — Luna!"
        """
        template = L2_TEMPLATES.get(phase, {}).get(key, "")
        if not template:
            return fallback

        try:
            return template.format_map(_SafeDict(um))
        except Exception:
            return fallback

    @staticmethod
    def get_raw(phase: str, key: str) -> str:
        """Get the raw template string without filling."""
        return L2_TEMPLATES.get(phase, {}).get(key, "")

    @staticmethod
    def missing_fields(phase: str, key: str, um: dict) -> list:
        """
        Return list of fields that are in the template but missing from um.
        Useful for debugging and testing.
        """
        template = L2_TEMPLATES.get(phase, {}).get(key, "")
        placeholders = re.findall(r"\{(\w+)\}", template)
        return [p for p in placeholders if p not in um or um[p] is None]


class _SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys instead of raising KeyError."""
    def __missing__(self, key):
        return f"{{{key}}}"
