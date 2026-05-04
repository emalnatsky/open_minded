"""
FIRST LAYER 

These are fixed strings with NO variable content and NO LLM involvement.
Used for: greetings, framing, nudges, correction acknowledgements,
          transitions, inspection prompts, and closing.

HOW TO USE:
    from l1_utterances import L1

    utterance = L1.get("phase_1", "greeting")
    # returns the fixed Dutch string

NEED TO complete the empty fields below with the validated Dutch lines
      from the interaction design plan.

STRUCTURE:
    L1_UTTERANCES = {
        "phase_key": {
            "utterance_key": "Dutch string",
        }
    }
"""

from __future__ import annotations

# --------------------------------------------------------------------------- L1 Utterances---------------------------------------------------------------------------

L1_UTTERANCES: dict = {

    #Phase 1: Greeting + framing 
    "phase_1": {
    },

    #Phase 2: Warm-up + Leo self-disclosure 
    "phase_2": {
    },

    #Phase 3: Interest exploration 
    "phase_3": {
    },

    #Phase 4: Nudge (conditional) 
    "phase_4": {
    },

    #Phase 5: School bridge
    "phase_5": {
    },

    #Phase 6: Aspiration
    "phase_6": {
    },

    #Phase 7: Memory inspection + co-construction
    "phase_7": {
    },

    #Phase 8: Closing
    "phase_8": {
    },

    # Cross-phase: correction acknowledgements
    "correction": {
    },

    #Cross-phase: repeat request
    "repeat": {
    },
}


# ---------------------------------------------------------------------------Accessor---------------------------------------------------------------------------

class L1:
    """Simple accessor for L1 utterances."""

    @staticmethod
    def get(phase: str, key: str, fallback: str = "") -> str:
        """
        Get a static utterance by phase and key.

        Parameters
        ----------
        phase : str   e.g. "phase_1", "correction"
        key   : str   e.g. "greeting", "acknowledge"
        fallback : str  returned if key not found

        Returns
        -------
        str — the Dutch utterance or fallback
        """
        return L1_UTTERANCES.get(phase, {}).get(key, fallback)

    @staticmethod
    def all_phases() -> list:
        return list(L1_UTTERANCES.keys())
