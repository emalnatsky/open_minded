"""
Load a Qualtrics CSV export into the UM service via the API.

USAGE:
    python scripts/load_qualtrics_new.py path/to/export.csv

PREREQUISITES:
    - The FastAPI server must be running (python main.py)
    - GraphDB must be running with the open-memory-robots repository
    - SKIP_API_KEY_CHECK = True in config.py (or set X-API-Key header)

QUALTRICS CSV FORMAT:
    The Qualtrics export has 3 header rows:
      Row 1: short column IDs  (Q4_8, Q5_1, Q5_2, ...)
      Row 2: full question text (human-readable, used for verification only)
      Row 3: Qualtrics ImportId JSON (internal, ignored)
    Data rows start at row 4.

    This script reads Row 1 as the column key and skips rows 2-3.

COLUMN MAPPING:
    The mapping below was built from the actual Qualtrics survey shared by
    Sander/Sherissa (May 2026). If the survey changes, update this mapping.

    Multi-column fields (hobbies → Q5_1..Q5_4, pet names → Q29_1..Q29_8)
    are handled with special merge logic, not the simple 1:1 map.

    MC + "Other" fields (e.g. Q19 + Q19_11_TEXT for books genre) are merged:
    if the selected choice is "Anders, namelijk:" the _TEXT column is used.
"""

import csv
import logging
import sys

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
SOURCE  = "qualtrics_import"
CHILD_ID_COLUMN = "ExternalReference"


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING — matches the actual Qualtrics export headers (May 2026)
# ═══════════════════════════════════════════════════════════════════════════════

# Simple 1:1 columns → UM field
# Key = exact CSV column header, Value = UM field name
SIMPLE_MAP: dict[str, str] = {
    # ── Age ──
    "Q4_8":   "age",

    # ── Hobbies cluster ──
    "Q6":     "hobby_fav",
    "Q7":     "hobby_talk",

    # ── Sports cluster ──
    "Q8":     "sports_enjoys",
    "Q9_4":   "sports_fav",
    "Q10":    "sports_talk",
    "Q11":    "sports_plays",
    "Q12_4":  "sports_fav_play",
    "Q13":    "sports_play_talk",

    # ── Music cluster ──
    "Q14":    "music_enjoys",
    "Q15":    "music_plays_instrument",
    "Q16_4":  "music_instrument",
    "Q17":    "music_talk",

    # ── Books cluster ──
    "Q18":    "books_enjoys",
    "Q21":    "books_talk",

    # ── Sociaal ──
    "Q23":    "has_best_friend",

    # ── Dieren cluster ──
    "Q24":    "animals_enjoys",
    "Q25_4":  "animal_fav",
    "Q26":    "animal_talk",
    "Q27":    "has_pet",
    "Q30":    "pet_talk",

    # ── Eten ──
    "Q31_4":  "fav_food",

    # ── Aspiratie cluster ──
    "Q35_4":  "interest",
    "Q36_4":  "aspiration",
    "Q37":    "role_model",
}


# MC + "Other text" columns: (selected_choice_col, other_text_col) → UM field
# If the selected choice contains "Anders" or similar, use the text column.
MC_OTHER_MAP: list[tuple[str, str, str]] = [
    # (choice_col,   text_col,         um_field)
    ("Q19",          "Q19_11_TEXT",    "books_fav_genre"),
    ("Q20",          "Q20_1_TEXT",     "books_fav_title"),
    ("Q22",          "Q22_15_TEXT",    "freetime_fav"),
    # pet_type removed — handled by paired pet logic below
    ("Q32",          "Q32_12_TEXT",    "fav_subject"),
    ("Q33",          "Q33_12_TEXT",    "school_strength"),
    ("Q195",         "Q195_12_TEXT",   "school_difficulty"),
]


# Multi-column fields: multiple CSV columns merge into one UM field
HOBBIES_COLS = ["Q5_1", "Q5_2", "Q5_3", "Q5_4"]

# Pet name columns — maps each column to its animal type (Dutch)
# Q29_1=Hond, Q29_2=Kat, Q29_3=Konijn, Q29_4=Hamster,
# Q29_5=Vogel, Q29_6=Vis, Q29_7=Reptiel, Q29_8=Anders
PET_TYPE_MAP = {
    "Q29_1": "Hond",
    "Q29_2": "Kat",
    "Q29_3": "Konijn",
    "Q29_4": "Hamster",
    "Q29_5": "Vogel",
    "Q29_6": "Vis",
    "Q29_7": "Reptiel",
    "Q29_8": "Anders",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(filepath: str):
    """Read the Qualtrics CSV and push each respondent through the API."""

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Skip the two extra Qualtrics header rows (question text + ImportId)
        rows = list(reader)
        if len(rows) < 3:
            logger.error("CSV has fewer than 3 data rows — is it a valid Qualtrics export?")
            return
        data_rows = rows[2:]   # rows 0-1 are the extra headers

    logger.info("Loaded %d respondent rows from %s", len(data_rows), filepath)

    success = 0
    failed  = 0

    for row in data_rows:
        child_id = _get_child_id(row)
        if not child_id:
            logger.warning("Skipping row with no ResponseId")
            failed += 1
            continue

        # Skip incomplete responses
        progress = row.get("Progress", "0")
        if progress != "100":
            logger.info("Skipping %s (progress=%s%%, not complete)", child_id, progress)
            continue

        # ── Create child ──────────────────────────────────────────────────
        age = _parse_int(row.get("Q4_8"))

        create_resp = requests.post(f"{API_URL}/api/um/", json={
            "child_id": child_id,
            "age": age,
            "session_id": SOURCE,
        })

        if create_resp.status_code == 409:
            logger.info("Child %s already exists, updating fields.", child_id)
        elif create_resp.status_code not in (200, 201):
            logger.error("Could not create child %s: %s", child_id, create_resp.text)
            failed += 1
            continue

        # ── Map columns to UM fields ──────────────────────────────────────
        fields: dict[str, object] = {}

        # 1) Simple 1:1 mappings
        for csv_col, um_field in SIMPLE_MAP.items():
            raw = _clean(row.get(csv_col))
            if raw:
                # Age needs to stay as int
                if um_field == "age":
                    continue
                else:
                    fields[um_field] = raw

        # 2) MC + Other text merges
        for choice_col, text_col, um_field in MC_OTHER_MAP:
            val = _resolve_mc_other(row, choice_col, text_col)
            if val:
                fields[um_field] = val

        # 3) Hobbies — merge up to 4 columns into comma-separated string
        hobbies = []
        for col in HOBBIES_COLS:
            h = _clean(row.get(col))
            if h:
                hobbies.append(h)
        if hobbies:
            # Send as comma-separated; the API + graphdb_client will handle
            # splitting into individual hobby nodes if needed
            fields["hobbies"] = ", ".join(hobbies)

        # 4) Pets — create paired type+name entries
        # Each pet needs its own API call (separate extra_props per pet),
        # so we collect them here and write after the main batch.
        pet_entries = []  # list of {"value": "Hond", "extra_props": {"petName": "Toby"}}
        pet_type_val = _resolve_mc_other(row, "Q28", "Q28_8_TEXT")
        if pet_type_val:
            pet_types = [p.strip() for p in pet_type_val.split(",") if p.strip()]
            # Filter out "Anders" placeholder from the type list
            clean_types = []
            for pt in pet_types:
                if any(kw in pt.lower() for kw in ("anders", "namelijk", "other", "titel van het boek")):
                    # The "Anders" text was already resolved by _resolve_mc_other
                    continue
                clean_types.append(pt)
            # If _resolve_mc_other appended the custom text, it's already in the list
            for pt in (clean_types if clean_types else pet_types):
                # Find the name column for this pet type
                pet_name = None
                for col, animal in PET_TYPE_MAP.items():
                    if animal.lower() == pt.lower():
                        pet_name = _clean(row.get(col))
                        break
                entry = {"value": pt}
                if pet_name:
                    entry["extra_props"] = {"petName": pet_name}
                else:
                    entry["extra_props"] = {}
                pet_entries.append(entry)

        # 5) Fix Qualtrics piping for hobby_fav
        # Q6 asks "which hobby is your favourite?" and the choices are piped
        # from Q5_1..Q5_4. But Qualtrics exports the LABEL ("Hobby 1") not
        # the actual typed text. Resolve it here.
        if "hobby_fav" in fields:
            import re
            m = re.match(r"Hobby\s+(\d+)", str(fields["hobby_fav"]), re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1   # "Hobby 1" → index 0
                if 0 <= idx < len(HOBBIES_COLS):
                    actual = _clean(row.get(HOBBIES_COLS[idx]))
                    if actual:
                        fields["hobby_fav"] = actual
                    else:
                        del fields["hobby_fav"]

        # 6) Determine child type from source label
        if "new" in SOURCE.lower():
            fields["exposure"] = "new"
        elif "returning" in SOURCE.lower():
            fields["exposure"] = "returning"

        # ── Write fields through API ──────────────────────────────────────
        if fields:
            update_resp = requests.post(
                f"{API_URL}/api/um/{child_id}/fields",
                json={
                    "fields": fields,
                    "source": SOURCE,
                    "session_id": SOURCE,
                },
            )
            if update_resp.status_code == 200:
                data = update_resp.json().get("data", {})
                written = len(data.get("written", []))
                skipped = len(data.get("skipped", []))
                logger.info(
                    "Child %s: %d fields written, %d skipped.",
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

        # ── Write pets individually (each needs its own extra_props) ──────
        for pet in pet_entries:
            pet_resp = requests.post(
                f"{API_URL}/api/um/{child_id}/fields",
                json={
                    "fields": {"pets": pet["value"]},
                    "extra_props": pet.get("extra_props", {}),
                    "source": SOURCE,
                    "session_id": SOURCE,
                },
                timeout=10,
            )
            if pet_resp.status_code == 200:
                pet_data = pet_resp.json().get("data", {})
                pet_name_str = pet.get("extra_props", {}).get("petName", "")
                logger.info(
                    "  Pet written: %s (name: %s)",
                    pet["value"], pet_name_str or "none"
                )
            else:
                logger.warning(
                    "  Pet write failed for %s: %s",
                    pet["value"], pet_resp.text[:100]
                )

    logger.info("Done. %d succeeded, %d failed.", success, failed)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_child_id(row: dict) -> str | None:
    """Use assigned childID as the anonymous child identifier."""
    rid = row.get(CHILD_ID_COLUMN, "").strip()
    if not rid:
        # Fall back to ResponseId so the row isn't silently lost
        fallback = row.get("ResponseId", "").strip()
        if fallback:
            logger.warning("No Provided ChildID Number for %s — using ResponseId as fallback", fallback)
            return fallback
    return rid if rid else None


def _clean(val) -> str | None:
    """Return stripped string or None if empty/missing."""
    if not val:
        return None
    val = str(val).strip()
    if val.lower() in ("", "n/a", "nvt", "not specified"):
        return None
    return val if val else None


def _parse_int(val) -> int | None:
    """Try to parse an integer, return None on failure."""
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _resolve_mc_other(row: dict, choice_col: str, text_col: str) -> str | None:
    choice = _clean(row.get(choice_col))
    text   = _clean(row.get(text_col))
    if not choice:
        return text
    parts = [p.strip() for p in choice.split(",") if p.strip()]
    real_choices = []
    has_other = False
    for part in parts:
        if any(kw in part.lower() for kw in ("anders", "namelijk", "other", "titel van het boek")):
            has_other = True
        else:
            real_choices.append(part)
    if has_other and text:
        real_choices.append(text)
    if not real_choices:
        return text
    return ",".join(real_choices)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/load_qualtrics.py <path_to_csv> <source_label>")
        print("  e.g.: python scripts/load_qualtrics.py data/new.csv qualtrics_new")
        print("  e.g.: python scripts/load_qualtrics.py data/returning.csv qualtrics_returning")
        sys.exit(1)
    SOURCE = sys.argv[2]
    load_csv(sys.argv[1])