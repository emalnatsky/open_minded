"""
Qualtrics CSV ingestion script.

Reads the Qualtrics export CSV and populates GraphDB via the API.
Runs validation automatically (every write goes through main.py).

Usage:
    python scripts/load_qualtrics.py data/qualtrics_export.csv

Qualtrics CSV quirk: the export has TWO header rows.
  Row 1: Column IDs (Q1, Q2, etc.) — this is what we use.
  Row 2: Full question text — we skip this.
  Row 3+: Actual data.

Adjust QUALTRICS_COLUMN_MAP in models/um_fields.py to match your column names.
"""

import csv
import sys
import json
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
SOURCE  = "qualtrics_checkin"


def load_csv(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        logger.error("File not found: %s", filepath)
        sys.exit(1)

    # Import here to avoid circular dependency issues
    from models.um_fields import QUALTRICS_COLUMN_MAP

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Qualtrics exports two header rows. The DictReader uses the first as keys.
        # Skip the second header row (question text row).
        first_data_row = next(reader, None)
        if first_data_row is None:
            logger.error("CSV is empty.")
            return

        # Check if the first data row looks like headers (Qualtrics question text)
        # Heuristic: if 'ResponseId' column value contains non-ID text, it's the text row.
        if first_data_row.get("ResponseId", "").startswith("Response"):
            logger.info("Skipping Qualtrics question-text header row.")
            first_data_row = next(reader, None)

        rows = [first_data_row] + list(reader) if first_data_row else list(reader)

    logger.info("Found %d participant rows.", len(rows))
    success, failed = 0, 0

    for row in rows:
        child_id = _get_child_id(row)
        if not child_id:
            logger.warning("Skipping row with no ResponseId: %s", dict(row))
            failed += 1
            continue

        # ── Create child ──────────────────────────────────────────────────────
        age   = _parse_int(row.get("Q_age") or row.get("age"))
        grade = _clean(row.get("Q_grade") or row.get("class"))

        create_resp = requests.post(f"{API_URL}/api/um/", json={
            "child_id": child_id,
            "age": age,
            "grade": grade,
            "session_id": SOURCE,
        })

        if create_resp.status_code == 409:
            logger.info("Child %s already exists — skipping create, will update fields.", child_id)
        elif create_resp.status_code not in (200, 201):
            logger.error("Could not create child %s: %s", child_id, create_resp.text)
            failed += 1
            continue

        # ── Map Qualtrics columns to UM fields ────────────────────────────────
        fields: dict[str, object]     = {}
        extra_props: dict[str, str]   = {}

        for csv_col, um_field in QUALTRICS_COLUMN_MAP.items():
            raw_val = row.get(csv_col, "").strip()
            if not raw_val or raw_val.lower() in ("", "n/a", "nvt", "geen", "not specified"):
                continue  # skip empty / none-equivalent values

            # Companion props (e.g. hobbies_motivation) go to extra_props
            if "_" in um_field and um_field.split("_", 1)[0] in ("hobbies", "sports", "pets"):
                extra_props[um_field] = raw_val
            else:
                # Multi-value fields: split comma-separated answers into lists
                from models.um_fields import VALID_FIELDS as _VF
                is_multi = (
                    um_field in _VF
                    and _VF[um_field]["storage"] == "node"
                    and _VF[um_field].get("multi_value", False)
                )
                if is_multi:
                    vals = [v.strip() for v in raw_val.split(",") if v.strip()]
                    fields[um_field] = vals if len(vals) > 1 else vals[0]
                else:
                    fields[um_field] = raw_val

        # ── Write fields through API (validation runs automatically) ──────────
        if fields:
            update_resp = requests.post(
                f"{API_URL}/api/um/{child_id}/fields",
                json={
                    "fields": fields,
                    "extra_props": extra_props,
                    "source": SOURCE,
                    "session_id": SOURCE,
                },
            )
            if update_resp.status_code == 200:
                data = update_resp.json().get("data", {})
                written  = len(data.get("written", []))
                skipped  = len(data.get("skipped", []))
                logger.info(
                    "Child %s: %d fields written, %d skipped (validation failed).",
                    child_id, written, skipped,
                )
                if skipped:
                    for s in data["skipped"]:
                        logger.warning("  Skipped %s: %s", s["field"], s.get("errors"))
                success += 1
            else:
                logger.error("Update failed for %s: %s", child_id, update_resp.text)
                failed += 1
        else:
            logger.info("Child %s: no fields to write after mapping.", child_id)
            success += 1

    logger.info("Done. %d succeeded, %d failed.", success, failed)


def _get_child_id(row: dict) -> str | None:
    """Use ResponseId as the anonymous child identifier."""
    rid = row.get("ResponseId", "").strip()
    return rid if rid else None


def _clean(val) -> str | None:
    if not val:
        return None
    val = str(val).strip()
    return val if val else None


def _parse_int(val) -> int | None:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_qualtrics.py <path_to_csv>")
        sys.exit(1)
    load_csv(sys.argv[1])