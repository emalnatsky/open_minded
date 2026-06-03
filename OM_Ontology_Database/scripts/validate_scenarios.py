"""
Validate CRI scenarios JSON before importing.

Checks each child's scenario for:
  - Exactly 4 mistakes (M1, M2, M3, M4)
  - All required mistake fields present and valid
  - All expected utterance step_ids present and non-empty
  - No unexpected or duplicate step_ids

USAGE:
    python scripts/validate_scenarios.py data/scenarios_from_xlsx.json

Run this BEFORE load_scenarios.py to catch problems early.
"""

import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_MISTAKES = {"M1", "M2", "M3", "M4"}

REQUIRED_MISTAKE_FIELDS = {"id", "target_field", "wrong_value", "mistake_type", "spt_level", "step"}

VALID_MISTAKE_TYPES = {"related-but-wrong", "completely-wrong"}
VALID_SPT_LEVELS = {"orientation", "exploratory affective", "affective exchange"}

# All 22 utterance step_ids + 2 topic metadata = 24 total
EXPECTED_UTTERANCES = {
    # Metadata
    "topic_1",
    "topic_2",
    # Part 1: Hobby bridge
    "p1_hobby_bridge_comment",
    # Part 1: Topic 1
    "p1_t1_recall",
    "p1_t1_open",
    "p1_t1_question",
    "p1_t1_followup",
    # Part 1: M1
    "p1_m1_wrong_hobby_opener",
    "p1_m1_followup_wrong_hobby",
    # Part 1: Topic 2
    "p1_t2_open",
    "p1_t2_followup",
    "p1_t2_close",
    # Part 1: M2
    "p1_m2_postcorrection_true_food",
    # Part 2: School
    "p2_fav_subject_comment_subject",
    "p2_subject_profile_link",
    # Part 2: M3
    "p2_m3_postcorrection_true_strength",
    "p2_school_wrap_after_difficulty",
    # Part 3: Aspiration
    "p3_future_theme_wrap",
    "p3_rolemodel_recall",
    "p3_rolemodel_ack",
    "p3_norolemodel_ack",
    # Part 3: M4
    "p3_m4_followup_wrong_aspiration",
    "p3_m4_postcorrection_reflection",
    # Closing
    "closing",
}


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_scenario(child_id: str, scenario: dict) -> list[str]:
    """Validate one child's scenario. Returns list of error strings."""
    errors = []

    # ── Mistakes ──────────────────────────────────────────────────────────
    mistakes = scenario.get("mistakes", [])
    mistake_ids = set()

    if len(mistakes) != 4:
        errors.append(f"Expected 4 mistakes, found {len(mistakes)}")

    for i, m in enumerate(mistakes):
        mid = m.get("id", f"(missing id at index {i})")

        # Check required fields
        for field in REQUIRED_MISTAKE_FIELDS:
            if field not in m or not str(m[field]).strip():
                errors.append(f"{mid}: missing or empty field '{field}'")

        # Check valid values
        if m.get("mistake_type") and m["mistake_type"] not in VALID_MISTAKE_TYPES:
            errors.append(f"{mid}: invalid mistake_type '{m['mistake_type']}' — expected {VALID_MISTAKE_TYPES}")

        if m.get("spt_level") and m["spt_level"] not in VALID_SPT_LEVELS:
            errors.append(f"{mid}: invalid spt_level '{m['spt_level']}' — expected {VALID_SPT_LEVELS}")

        # Track for completeness
        if m.get("id"):
            if m["id"] in mistake_ids:
                errors.append(f"{mid}: duplicate mistake id")
            mistake_ids.add(m["id"])

    missing_mistakes = EXPECTED_MISTAKES - mistake_ids
    if missing_mistakes:
        errors.append(f"Missing mistake(s): {sorted(missing_mistakes)}")

    extra_mistakes = mistake_ids - EXPECTED_MISTAKES
    if extra_mistakes:
        errors.append(f"Unexpected mistake id(s): {sorted(extra_mistakes)}")

    # ── Utterances ────────────────────────────────────────────────────────
    utterances = scenario.get("utterances", [])
    utterance_ids = set()

    for utt in utterances:
        step_id = utt.get("step_id", "")
        text = utt.get("text", "").strip()

        if not step_id:
            errors.append("Utterance with missing step_id")
            continue

        if step_id in utterance_ids:
            errors.append(f"Duplicate utterance: {step_id}")
        utterance_ids.add(step_id)

        if not text:
            errors.append(f"{step_id}: empty text")

    missing_utts = EXPECTED_UTTERANCES - utterance_ids
    if missing_utts:
        errors.append(f"Missing utterance(s): {sorted(missing_utts)}")

    extra_utts = utterance_ids - EXPECTED_UTTERANCES
    if extra_utts:
        errors.append(f"Unexpected utterance(s): {sorted(extra_utts)}")

    total_expected = len(EXPECTED_UTTERANCES)
    total_found = len(utterance_ids & EXPECTED_UTTERANCES)
    if total_found < total_expected:
        errors.append(f"Utterance coverage: {total_found}/{total_expected}")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_scenarios.py <scenarios.json>")
        sys.exit(1)

    filepath = sys.argv[1]

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        logger.error("No scenarios found in %s", filepath)
        sys.exit(1)

    logger.info("Validating %d scenario(s) from %s\n", len(scenarios), filepath)

    total_errors = 0
    total_ok = 0

    for scenario in scenarios:
        child_id = scenario.get("child_id", "(unknown)")
        errors = validate_scenario(child_id, scenario)

        mistakes = scenario.get("mistakes", [])
        utterances = scenario.get("utterances", [])
        utt_ids = {u.get("step_id") for u in utterances}
        utt_coverage = len(utt_ids & EXPECTED_UTTERANCES)

        if errors:
            print(f"✗ {child_id}: {len(mistakes)} mistakes, {utt_coverage}/{len(EXPECTED_UTTERANCES)} utterances — {len(errors)} error(s)")
            for err in errors:
                print(f"    → {err}")
            total_errors += len(errors)
        else:
            print(f"✓ {child_id}: {len(mistakes)} mistakes, {utt_coverage}/{len(EXPECTED_UTTERANCES)} utterances — OK")
            total_ok += 1

        print()

    # Summary
    print("=" * 60)
    print(f"  {total_ok}/{len(scenarios)} scenarios valid")
    if total_errors:
        print(f"  {total_errors} total error(s) found")
        print(f"\n  Fix errors, then re-run before loading.")
        sys.exit(1)
    else:
        print(f"\n  All clear! Run:")
        print(f"  python scripts/load_scenarios.py {filepath}")
        sys.exit(0)


if __name__ == "__main__":
    main()