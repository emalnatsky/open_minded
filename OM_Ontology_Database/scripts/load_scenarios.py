"""
Load manually curated CRI scenarios into GraphDB.

Reads a JSON file with mistakes and utterances per child, pushes them
through the API. Works the same for stubs and final utterances — just
run it again with updated text to overwrite.

USAGE:
    python scripts/load_scenarios.py data/scenarios.json

    # Load only specific children:
    python scripts/load_scenarios.py data/scenarios.json --child 701

    # Update just utterances for a child (skip re-creating mistakes):
    python scripts/load_scenarios.py data/scenarios.json --child 701 --utterances-only

JSON FORMAT:
    See scenario_example.json for the expected structure.

UPDATING UTTERANCES LATER:
    When real L2-pregen text replaces stubs, just update the "text" field
    in your JSON file and run this script again. The API overwrites existing
    utterances for the same step+branch automatically.

    You can also update a single utterance via curl:
    curl -X POST http://localhost:8000/api/um/701/scenario/utterance \\
         -H "Content-Type: application/json" \\
         -d '{"step_id": "m1_followup", "layer": "L2-pregen",
              "branch": "corrected", "text": "Real final text here"}'
"""

import json
import sys
import argparse
import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"


def load_scenarios(filepath: str, child_filter: str = None, utterances_only: bool = False):
    """Load scenarios from JSON file and push to API."""

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        logger.error("No scenarios found in %s", filepath)
        return

    logger.info("Loaded %d scenario(s) from %s", len(scenarios), filepath)

    success = 0
    failed = 0

    for entry in scenarios:
        child_id = entry.get("child_id")
        if not child_id:
            logger.warning("Skipping entry with no child_id")
            failed += 1
            continue

        if child_filter and child_id != child_filter:
            continue

        mistakes   = entry.get("mistakes", [])
        utterances = entry.get("utterances", [])

        if utterances_only:
            # Only update utterances, don't recreate the scenario
            logger.info("Updating %d utterance(s) for %s...", len(utterances), child_id)
            utt_success = 0
            for utt in utterances:
                resp = requests.post(
                    f"{API_URL}/api/um/{child_id}/scenario/utterance",
                    json=utt,
                    timeout=10,
                )
                if resp.status_code == 200:
                    utt_success += 1
                else:
                    logger.warning(
                        "  Failed to update %s/%s: %s",
                        utt["step_id"], utt["branch"], resp.text[:100]
                    )
            logger.info("  %s: %d/%d utterances updated.", child_id, utt_success, len(utterances))
            success += 1

        else:
            # Full scenario creation (mistakes + utterances)
            logger.info("Creating scenario for %s (%d mistakes, %d utterances)...",
                        child_id, len(mistakes), len(utterances))

            resp = requests.post(
                f"{API_URL}/api/um/{child_id}/scenario/generate",
                json={
                    "mistakes": mistakes,
                    "utterances": utterances,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                result = resp.json()["data"]
                logger.info(
                    "  %s: %d mistakes, %d utterances created.",
                    child_id, result["mistakes_created"], result["utterances_created"],
                )
                success += 1
            else:
                logger.error("  %s: FAILED — %s", child_id, resp.text[:200])
                failed += 1

    logger.info("\nDone. %d succeeded, %d failed.", success, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load CRI scenarios from JSON into GraphDB."
    )
    parser.add_argument(
        "filepath",
        help="Path to the scenarios JSON file"
    )
    parser.add_argument(
        "--child",
        help="Only load scenario for this child ID"
    )
    parser.add_argument(
        "--utterances-only",
        action="store_true",
        help="Only update utterances (don't recreate mistakes)"
    )
    args = parser.parse_args()

    load_scenarios(args.filepath, args.child, args.utterances_only)