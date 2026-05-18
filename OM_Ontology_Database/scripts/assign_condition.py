"""
Randomly assign children to condition_1 or condition_2.

Balanced by gender AND exposure (new/returning), ensuring even distribution
across all 4 strata: (male,new), (male,returning), (female,new), (female,returning).

USAGE:
    python scripts/assign_conditions.py children.csv

INPUT CSV FORMAT (simple, you create this manually):
    child_id,gender,exposure
    701,m,new
    702,f,returning
    703,m,new
    ...

    gender: m / f
    exposure: new / returning  (should already be in the UM if you added the field)

OUTPUT:
    - Prints the assignment table
    - Calls the API to write 'condition' to each child's UM
    - Saves a copy to data/condition_assignments.csv
"""

import csv
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import requests

API_URL = "http://localhost:8000"
SOURCE  = "condition_assignment"


def load_children(filepath: str) -> list[dict]:
    """Load children from the manual CSV."""
    children = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("child_id", "").strip()
            gender = row.get("gender", "").strip().lower()
            exposure = row.get("exposure", "").strip().lower()
            if cid and gender and exposure:
                children.append({
                    "child_id": cid,
                    "gender": gender,
                    "exposure": exposure,
                })
            else:
                print(f"  WARNING: Skipping incomplete row: {row}")
    return children


def stratified_assign(children: list[dict], seed: int = None) -> list[dict]:
    """
    Stratified random assignment to condition_1 / condition_2.

    Groups children into 4 strata by (gender, exposure).
    Within each stratum, shuffles and assigns alternating conditions.
    If a stratum has odd count, the leftover is randomly assigned.
    """
    if seed is not None:
        random.seed(seed)

    # Group into strata
    strata = defaultdict(list)
    for child in children:
        key = (child["gender"], child["exposure"])
        strata[key].append(child)

    # Assign within each stratum
    for key, group in strata.items():
        random.shuffle(group)
        half = len(group) // 2
        for i, child in enumerate(group):
            child["condition"] = "condition_1" if i < half else "condition_2"

    # Flatten back
    all_assigned = []
    for group in strata.values():
        all_assigned.extend(group)

    return all_assigned


def write_to_api(assignments: list[dict]):
    """Write condition to each child's UM via the API."""
    success = 0
    failed = 0
    for child in assignments:
        resp = requests.post(
            f"{API_URL}/api/um/{child['child_id']}/fields",
            json={
                "fields": {"condition": child["condition"]},
                "source": SOURCE,
                "session_id": SOURCE,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            success += 1
        else:
            print(f"  FAILED {child['child_id']}: {resp.status_code} {resp.text[:100]}")
            failed += 1
    print(f"\n  Written to API: {success} succeeded, {failed} failed.")


def save_csv(assignments: list[dict], output_path: str = "data/condition_assignments.csv"):
    """Save assignments to CSV for records."""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["child_id", "gender", "exposure", "condition"])
        writer.writeheader()
        writer.writerows(assignments)
    print(f"  Saved to {output_path}")


def print_summary(assignments: list[dict]):
    """Print a summary table."""
    print(f"\n{'='*60}")
    print(f"  Condition Assignment Summary")
    print(f"{'='*60}")

    # Per-stratum counts
    counts = defaultdict(lambda: {"condition_1": 0, "condition_2": 0})
    for child in assignments:
        key = f"{child['gender']}, {child['exposure']}"
        counts[key][child["condition"]] += 1

    print(f"\n  {'Stratum':<25} {'C1':>5} {'C2':>5} {'Total':>7}")
    print(f"  {'-'*45}")
    for stratum in sorted(counts.keys()):
        c1 = counts[stratum]["condition_1"]
        c2 = counts[stratum]["condition_2"]
        print(f"  {stratum:<25} {c1:>5} {c2:>5} {c1+c2:>7}")

    total_c1 = sum(c["condition_1"] for c in counts.values())
    total_c2 = sum(c["condition_2"] for c in counts.values())
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<25} {total_c1:>5} {total_c2:>5} {total_c1+total_c2:>7}")

    # Full assignment list
    print(f"\n  {'Child ID':<15} {'Gender':<8} {'Exposure':<12} {'Condition'}")
    print(f"  {'-'*50}")
    for child in sorted(assignments, key=lambda c: c["child_id"]):
        print(f"  {child['child_id']:<15} {child['gender']:<8} {child['exposure']:<12} {child['condition']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/assign_conditions.py <children.csv>")
        print("\nCSV format:")
        print("  child_id,gender,exposure")
        print("  701,m,new")
        print("  702,f,returning")
        sys.exit(1)

    children = load_children(sys.argv[1])
    if not children:
        print("No valid children found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(children)} children.")

    # Use a fixed seed for reproducibility — change or remove for true random
    assignments = stratified_assign(children, seed=42)

    print_summary(assignments)

    # Ask before writing
    confirm = input("Write conditions to GraphDB? (y/n): ").strip().lower()
    if confirm == "y":
        write_to_api(assignments)
        save_csv(assignments)
        print("Done.")
    else:
        # Still save CSV even if not writing to API
        save_csv(assignments, "data/condition_assignments_preview.csv")
        print("Saved preview CSV (not written to API).")