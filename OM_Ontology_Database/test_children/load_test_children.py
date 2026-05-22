"""
Load test children into GraphDB through the API.

USAGE:
    python scripts/load_test_children.py data/test_children.json

Then load CRI scenarios separately:
    python scripts/load_scenarios.py data/test_scenarios.json

PREREQUISITES:
    - The FastAPI server must be running (python main.py)
    - GraphDB must be running
"""

import json
import sys
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
SOURCE  = "test_data"


def load_test_children(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    children = data.get("children", [])
    logger.info("Loading %d test children from %s", len(children), filepath)

    for child in children:
        child_id = child["child_id"]
        age = child.get("age")

        # Create child
        resp = requests.post(f"{API_URL}/api/um/", json={
            "child_id": child_id,
            "age": age,
            "session_id": SOURCE,
        })
        if resp.status_code == 409:
            logger.info("  %s already exists, updating fields.", child_id)
        elif resp.status_code not in (200, 201):
            logger.error("  Could not create %s: %s", child_id, resp.text[:100])
            continue
        else:
            logger.info("  Created %s (age=%s)", child_id, age)

        # Write scalar fields
        scalars = child.get("scalars", {})
        if scalars:
            resp = requests.post(
                f"{API_URL}/api/um/{child_id}/fields",
                json={"fields": scalars, "source": SOURCE, "session_id": SOURCE},
            )
            if resp.status_code == 200:
                result = resp.json().get("data", {})
                logger.info("    Scalars: %d written, %d skipped",
                            len(result.get("written", [])),
                            len(result.get("skipped", [])))
            else:
                logger.error("    Scalar write failed: %s", resp.text[:100])

        # Write node fields (each value individually to get separate nodes)
        for field_name, values in child.get("node_fields", {}).items():
            for val in values:
                resp = requests.post(
                    f"{API_URL}/api/um/{child_id}/fields",
                    json={
                        "fields": {field_name: val},
                        "source": SOURCE,
                        "session_id": SOURCE,
                    },
                )
                if resp.status_code == 200:
                    logger.info("    %s: %s ✓", field_name, val)
                else:
                    logger.warning("    %s: %s FAILED — %s",
                                   field_name, val, resp.text[:80])

        # Write pets (with extra_props)
        for pet in child.get("pets", []):
            extra = {}
            if pet.get("petName"):
                extra["petName"] = pet["petName"]
            resp = requests.post(
                f"{API_URL}/api/um/{child_id}/fields",
                json={
                    "fields": {"pets": pet["value"]},
                    "extra_props": extra,
                    "source": SOURCE,
                    "session_id": SOURCE,
                },
            )
            if resp.status_code == 200:
                logger.info("    pet: %s (name: %s) ✓",
                            pet["value"], pet.get("petName", "none"))
            else:
                logger.warning("    pet: %s FAILED — %s",
                               pet["value"], resp.text[:80])

        logger.info("  Done: %s\n", child_id)

    logger.info("All test children loaded.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_test_children.py <path_to_json>")
        sys.exit(1)
    load_test_children(sys.argv[1])