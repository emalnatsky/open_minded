"""
THIRD LAYER

Wraps GPT-4o-mini with a phase-aware system prompt so the LLM knows:
- Which SPT layer it is operating in (Orientation / Exploratory / Affective)
- Which UM fields are available for reference in this phase
- The robot's persona (Leo)
- A max token cap per generation

HOW TO USE:
    from L3 import L3Generator

    generator = L3Generator(openai_key="sk-...")
    response  = generator.generate(
        phase="phase_3",
        child_input="Ik ren ook in het park met mijn hond!",
        um={"animal_fav": "honden", "pet_name": "Luna"},
        context="Leo just mentioned the child has a dog named Luna.",
    )

SAFETY:
    L3 output is screened before being returned. If the safety check
    fails the fallback string is returned instead.
    Safety classification is a stub (SANDER needs to implements the real classifier).

TOD):
    - Fill in PHASE_ALLOWED_FIELDS with the right field lists per phase.
    - Fill in PHASE_PROMPTS with validated phase-specific instructions.
    - Implement the real _safety_check() method.
    - The generate() interface stays the same.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------config-------------------------------------------------------------------------

GPT_MODEL       = "gpt-4o-mini"
GPT_MAX_TOKENS  = 60    #changeable during testing
GPT_TEMPERATURE = 0.7   #still bit varied also changeable during testing

# Fallback returned when GPT fails or safety check rejects output
SAFETY_FALLBACK = "Wauw, dat klinkt interessant!"

# Base persona prepended to every system prompt
# TODO: expand with full persona constraints and field-access boundary
BASE_PERSONA = (
    "You are Leo, a friendly school robot talking to a Dutch child aged 8-11. "
    "Speak in Dutch. Reply in ONE short sentence (max 25 words). "
    "Be warm, enthusiastic, and age-appropriate. "
    "Do not ask a question unless explicitly instructed. "
    "Do not mention you are an AI or robot unless asked. "
    "Structure is pre-authored; you generate what Leo says within it."
)

# Fields allowed per phase = LLM may only reference these
# TODO : fill in the correct field lists per phase
PHASE_ALLOWED_FIELDS: dict = {
    "phase_1": [],
    "phase_2": [],
    "phase_3": [],
    "phase_4": [],
    "phase_5": [],
    "phase_6": [],
    "phase_7": [],
    "phase_8": [],
}

# Phase-specific instructions appended to BASE_PERSONA
# TODO: fill in validated instructions per phase
PHASE_PROMPTS: dict = {
    "phase_1": "",  # Greeting + framing
    "phase_2": "",  # Warm-up — Leo self-disclosure only
    "phase_3": "",  # Interest exploration (ORI + EXP)
    "phase_4": "",  # Nudge (conditional)
    "phase_5": "",  # School bridge (EXP)
    "phase_6": "",  # Aspiration (AFF — deepest layer)
    "phase_7": "",  # Memory inspection + co-construction
    "phase_8": "",  # Closing
}


# ---------------------------------------------------------------------------L3 Generator----------------------------------------------------------------------

class L3Generator:
    """
    Phase-aware GPT generator for Layer 3 personalised responses.

    Parameters
    ----------
    openai_key : str
        OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Usage
    -----
    generator = L3Generator(openai_key="sk-...")
    response  = generator.generate(
        phase="phase_3",
        child_input="Ik ren ook in het park!",
        um={"animal_fav": "honden", "pet_name": "Luna"},
        context="Leo just mentioned the child's dog.",
    )
    """

    def __init__(self, openai_key: Optional[str] = None):
        key = openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass openai_key= or set OPENAI_API_KEY."
            )
        self.client = OpenAI(api_key=key)
        logger.info("L3Generator ready. Model=%s", GPT_MODEL)

    def generate(
        self,
        phase:       str,
        child_input: str,
        um:          dict,
        context:     str = "",
        fallback:    str = SAFETY_FALLBACK,
    ) -> str:
        """
        Generate a phase-aware personalised Dutch response.

        Parameters
        ----------
        phase        : str   current phase key e.g. "phase_3"
        child_input  : str   what the child just said (raw Whisper transcript)
        um           : dict  current user model values (HARDCODED_UM for testing)
        context      : str   optional: what Leo just said before (for coherence)
        fallback     : str   returned if GPT fails or safety check fails

        Returns
        -------
        str — a short Dutch sentence for NAO to say. Never raises.
        """
        if not child_input or not child_input.strip():
            return fallback

        system_prompt = self._build_system_prompt(phase, um)
        user_message  = self._build_user_message(child_input, context, um, phase)

        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=GPT_MAX_TOKENS,
                temperature=GPT_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()

            # Trim to one sentence
            first_sentence = raw.split(".")[0].strip() + "."

            # Safety screen — stub for now, Lena implements real classifier
            if not self._safety_check(first_sentence):
                logger.warning("L3 safety check failed — using fallback.")
                return fallback

            logger.info("L3 generated for %s: %s", phase, first_sentence)
            return first_sentence

        except Exception as e:
            logger.error("L3 GPT error for %s: %s — using fallback.", phase, e)
            return fallback

    # ------------------------------------------------------------------internal-----------------------------------------------------------------

    def _build_system_prompt(self, phase: str, um: dict) -> str:
        """Build the full system prompt for this phase."""
        phase_instruction = PHASE_PROMPTS.get(phase, "")
        allowed_fields    = PHASE_ALLOWED_FIELDS.get(phase, [])

        # Build UM context — only include allowed fields
        um_context = "\n".join(
            f"  {field}: {um.get(field, 'unknown')}"
            for field in allowed_fields
            if field in um
        )
        um_section = (
            f"\nAvailable UM fields for this phase:\n{um_context}"
            if um_context else ""
        )

        prompt = BASE_PERSONA
        if phase_instruction:
            prompt += f"\n\nCurrent phase instruction:\n{phase_instruction}"
        prompt += um_section
        return prompt

    def _build_user_message(
        self,
        child_input: str,
        context:     str,
        um:          dict,
        phase:       str,
    ) -> str:
        """Build the user message sent to GPT."""
        parts = []
        if context:
            parts.append(f"Leo just said: \"{context}\"")
        parts.append(f"The child said: \"{child_input}\"")
        parts.append("Generate Leo's next response (one short Dutch sentence).")
        return "\n".join(parts)

    def _safety_check(self, text: str) -> bool:
        """
        Screen L3 output before NAO speaks it.

        STUB so always returns True for now.
        TODO: implement real safety classifier here.
        Should return False if content is: (STILL NEED DESIGN)
          - Inappropriate for a child aged 8-11
          - Contains oversharing prompts
          - Contains third-party personal information
          - Triggers distress signals
          - Is too long (> 40 words)
        """
        if not text:
            return False
        if len(text.split()) > 40:
            logger.warning("L3 output too long (%d words) — failing safety.", len(text.split()))
            return False
        return True  # STUB
