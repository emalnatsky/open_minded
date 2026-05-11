"""
Validation script: check which children have complete user models.

USAGE:
    python scripts/validate_completeness.py

Or with a specific Qualtrics CSV (to check only those children):
    python scripts/validate_completeness.py qualtrics_csv/Check_in_New.csv

Prints a summary per child: total fields filled, missing fields,
and an overall completeness percentage.
"""

import csv
import sys
import requests

API_URL = "http://localhost:8000"


def get_all_children_from_csv(filepath: str) -> list[str]:
    """Extract child IDs from a Qualtrics CSV."""
    ids = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[2:]  # skip Qualtrics extra header rows
        for row in rows:
            cid = row.get("ExternalReference", "").strip()
            if cid:
                ids.append(cid)
    return ids


def get_all_children_from_api() -> list[str]:
    """Get all child IDs from the API schema/listing."""
    # Use the list endpoint if it exists, otherwise this needs adjustment
    r = requests.get(f"{API_URL}/api/um/")
    if r.status_code == 200:
        return r.json().get("data", {}).get("children", [])
    return []


def check_child(child_id: str) -> dict:
    """Fetch a child's inspect view and check completeness."""
    r = requests.get(f"{API_URL}/api/um/{child_id}/inspect")
    if r.status_code == 404:
        return {"child_id": child_id, "status": "NOT FOUND", "filled": [], "missing": []}
    if r.status_code != 200:
        return {"child_id": child_id, "status": "ERROR", "filled": [], "missing": []}

    data = r.json()["data"]
    categories = data.get("categories", {})

    # Collect all fields that have values
    filled = []
    for cat_name, cat_data in categories.items():
        for field_name, field_data in cat_data.get("scalars", {}).items():
            if field_data.get("value"):
                filled.append(field_name)
        for field_name, nodes in cat_data.get("nodes", {}).items():
            if nodes:  # non-empty list
                filled.append(field_name)

    # Get expected fields from the schema
    schema_resp = requests.get(f"{API_URL}/api/schema")
    all_fields = list(schema_resp.json()["data"]["fields"].keys())

    missing = [f for f in all_fields if f not in filled]

    return {
        "child_id": child_id,
        "status": "OK",
        "filled": filled,
        "missing": missing,
        "total": len(all_fields),
    }


def main():
    # Get child IDs
    if len(sys.argv) > 1:
        children = get_all_children_from_csv(sys.argv[1])
        print(f"Checking {len(children)} children from CSV...\n")
    else:
        children = get_all_children_from_api()
        print(f"Checking {len(children)} children from API...\n")

    if not children:
        print("No children found.")
        return

    # Cache the schema
    schema_resp = requests.get(f"{API_URL}/api/schema")
    all_fields = list(schema_resp.json()["data"]["fields"].keys())
    total_fields = len(all_fields)

    print(f"Expected fields per child: {total_fields}\n")
    print("=" * 70)

    incomplete = []

    for child_id in children:
        result = check_child(child_id)

        filled_count = len(result["filled"])
        pct = (filled_count / total_fields * 100) if total_fields > 0 else 0
        status_icon = "✓" if pct == 100 else "○" if pct >= 70 else "✗"

        print(f"{status_icon} {child_id}: {filled_count}/{total_fields} ({pct:.0f}%)")

        if result["missing"]:
            print(f"    Missing: {', '.join(result['missing'])}")
            incomplete.append(result)

    # Summary
    print("=" * 70)
    complete = len(children) - len(incomplete)
    print(f"\n{complete}/{len(children)} children have complete UMs")
    if incomplete:
        print(f"{len(incomplete)} children have missing fields")


if __name__ == "__main__":
    main()