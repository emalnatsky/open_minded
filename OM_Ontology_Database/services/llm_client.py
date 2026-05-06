"""
Ollama LLM client for semantic validation.

Sends a structured prompt to the local Ollama server and parses
a JSON response indicating whether a field value is appropriate.

Privacy note: This never calls any cloud API. All inference is local.
If Ollama is not running, calls return a "skipped" result rather than
blocking writes — but SKIP_LLM_VALIDATION in config.py must be True
for this fallback to activate.
"""

import json
import logging

import httpx

from config import OLLAMA_URL, OLLAMA_MODEL, SKIP_LLM_VALIDATION

logger = logging.getLogger(__name__)


def validate_field_value(
    field_name: str,
    value: str,
    description: str,
) -> dict:
    """
    Ask the LLM whether this value is appropriate for this field.

    Returns a dict with:
      valid      : bool   — True if value is acceptable
      flag_type  : str    — "none" | "unexpected" | "malformed"
      reason     : str    — brief explanation in English (for logging/audit)
      skipped    : bool   — True if LLM was not called (Ollama unavailable)

    The prompt is intentionally structured to match the flagging categories
    used by Gheorghe (2025): missing (handled before this call), malformed,
    unexpected. We do not flag "missing" here — that is a structural check.
    """
    if SKIP_LLM_VALIDATION:
        return {"valid": True, "flag_type": "none", "reason": "LLM validation skipped (config)", "skipped": True}

    prompt = _build_prompt(field_name, value, description)

    try:
        raw_response = _call_ollama(prompt)
        return _parse_response(raw_response)
    except httpx.ConnectError:
        logger.warning(
            "Ollama not reachable at %s. LLM validation skipped for field '%s'.",
            OLLAMA_URL, field_name,
        )
        # Fail open: write goes through, but caller should log the skip.
        return {
            "valid": True,
            "flag_type": "none",
            "reason": "Ollama not available — semantic check skipped",
            "skipped": True,
        }
    except Exception as e:
        logger.error("LLM validation error for field '%s': %s", field_name, e)
        return {
            "valid": True,
            "flag_type": "none",
            "reason": f"LLM error — semantic check skipped: {e}",
            "skipped": True,
        }


def _build_prompt(field_name: str, value: str, description: str) -> str:
    return f"""You are a data quality validator for a social robot used with Dutch children aged 6 to 13.
Your job is to check whether a given field value is appropriate, plausible, and non-harmful.

Field name: {field_name}
Field description: {description}
Value provided: "{value}"

Check the following:
1. Is this a real, recognisable answer for this type of field?
2. Could this be a joke answer, gibberish, or random text?
3. Is it age-appropriate for a child aged 6-13?
4. Is it a plausible Dutch or English response for this field?
5. Is it harmful, offensive, or inappropriate for a child context?

Note: Dutch words and names are valid. Extinct animals (like dinosaurs) are valid for "favorite_animal".
Do NOT flag legitimate answers — only flag if the value is clearly wrong, nonsensical, or inappropriate.
Real movie titles, book titles, game names, and food names should NOT be flagged even if unusual.

Respond ONLY with a valid JSON object, no other text, no markdown:
{{"valid": true_or_false, "flag_type": "none_or_unexpected_or_malformed", "reason": "brief explanation in English"}}
"""


def _call_ollama(prompt: str) -> str:
    """POST to Ollama /api/generate and return the response text."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # low temperature for deterministic validation
                    "top_p": 0.9,
                },
            },
        )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _parse_response(raw: str) -> dict:
    """
    Parse the LLM JSON response.
    If parsing fails, fail open (valid=True) and log the raw output.
    """
    raw = raw.strip()
    # Strip markdown code fences if model adds them despite the prompt
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return {
            "valid":     bool(parsed.get("valid", True)),
            "flag_type": str(parsed.get("flag_type", "none")),
            "reason":    str(parsed.get("reason", "")),
            "skipped":   False,
        }
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response: %s", raw[:200])
        return {
            "valid":     True,
            "flag_type": "none",
            "reason":    "Could not parse LLM response — treated as valid",
            "skipped":   False,
        }