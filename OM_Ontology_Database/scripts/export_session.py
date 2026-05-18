"""
Export child UM session data to JSON or CSV.

USAGE:
    python scripts/export_session.py child_001              # single child → JSON
    python scripts/export_session.py --all                   # all children → JSON
    python scripts/export_session.py --all --format csv      # all children → one CSV
    python scripts/export_session.py --all --format both     # all children → JSON + CSV

The CSV output creates a flat table with one row per child and one column per
UM field, suitable for opening in Excel or importing into SPSS/R.

PREREQUISITES:
    - The FastAPI server must be running (python main.py)
    - GraphDB must be running with the open-memory-robots repository
"""

import sys
import csv
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone

API_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════════════════════
# JSON EXPORT (per-child, full detail)
# ═══════════════════════════════════════════════════════════════════════════════

def export_child_json(child_id: str, output_dir: str = "data/exports") -> Path | None:
    """Export a single child's full UM + history to a JSON file."""
    resp = requests.get(f"{API_URL}/api/um/{child_id}/export", timeout=30)
    if resp.status_code == 404:
        print(f"  Skipping '{child_id}' — not found.")
        return None
    resp.raise_for_status()
    data = resp.json()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"{child_id}_export_{ts}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    s = data["summary"]
    print(f"  {child_id}: {s['total_fields_populated']} fields, "
          f"{s['total_change_events']} changes → {filename.name}")
    return filename


# ═══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT (all children, flat table)
# ═══════════════════════════════════════════════════════════════════════════════

def export_all_csv(child_ids: list[str], output_dir: str = "data/exports") -> Path | None:
    """
    Export all children's current UM values into one flat CSV file.
    One row per child, one column per field. Multi-value node fields
    are joined with " | " separators.

    Does NOT include history — just current values. For history,
    use the JSON export.
    """
    if not child_ids:
        print("No children to export.")
        return None

    # Fetch schema to get all field names as column headers
    schema_resp = requests.get(f"{API_URL}/api/schema", timeout=10)
    schema_resp.raise_for_status()
    all_fields = list(schema_resp.json()["data"]["fields"].keys())

    # Fixed metadata columns + all UM fields
    meta_columns = ["child_id", "exposure", "age", "created_at"]
    # Remove fields already in meta_columns to avoid duplication
    field_columns = [f for f in all_fields if f not in ("age", "exposure")]
    headers = meta_columns + field_columns

    rows = []
    for child_id in child_ids:
        resp = requests.get(f"{API_URL}/api/um/{child_id}/export", timeout=30)
        if resp.status_code != 200:
            print(f"  Skipping '{child_id}' — {resp.status_code}")
            continue

        data = resp.json()
        current = data.get("current_profile", {})
        full = data.get("full_profile", {})

        row = {}
        row["child_id"] = child_id
        row["exposure"] = current.get("exposure", "")
        row["age"] = current.get("age", "")
        row["created_at"] = full.get("scalars", {}).get("createdAt", {}).get("value", "")

        # Fill in all UM fields from the flat current_profile
        for field in field_columns:
            row[field] = current.get(field, "")

        rows.append(row)
        print(f"  {child_id}: {len([v for v in row.values() if v])} columns populated")

    # Write CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"all_children_export_{ts}.csv"

    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV exported: {filename} ({len(rows)} children, {len(headers)} columns)")
    return filename


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_child_ids() -> list[str]:
    """Fetch all child IDs from the API."""
    resp = requests.get(f"{API_URL}/api/um/", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]["children"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export child UM session data to JSON and/or CSV."
    )
    parser.add_argument(
        "child_id", nargs="?",
        help="Child ID to export (omit if using --all)"
    )
    parser.add_argument(
        "--out", default="data/exports",
        help="Output directory (default: data/exports)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export every child in the database"
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "both"], default="json",
        help="Output format: json (detailed per-child), csv (flat table), both"
    )
    args = parser.parse_args()

    if args.all:
        ids = get_all_child_ids()
        if not ids:
            print("No children found in GraphDB.")
            sys.exit(0)

        if args.format in ("json", "both"):
            print(f"\nExporting {len(ids)} children as JSON to {args.out}/")
            for cid in ids:
                export_child_json(cid, args.out)

        if args.format in ("csv", "both"):
            print(f"\nExporting {len(ids)} children as CSV to {args.out}/")
            export_all_csv(ids, args.out)

        print("\nDone.")

    elif args.child_id:
        if args.format == "csv":
            print("CSV format requires --all (one row per child).")
            print("Use --format json for single child, or --all --format csv for all.")
            sys.exit(1)
        export_child_json(args.child_id, args.out)

    else:
        parser.print_help()
        sys.exit(1)