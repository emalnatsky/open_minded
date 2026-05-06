import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone

API_URL = "http://localhost:8000"


def export_child(child_id: str, output_dir: str = "data/exports") -> Path:
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


def get_all_child_ids() -> list[str]:
    resp = requests.get(f"{API_URL}/api/um/", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]["children"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export child UM session data to JSON.")
    parser.add_argument("child_id", nargs="?", help="Child ID to export (omit if using --all)")
    parser.add_argument("--out", default="data/exports", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Export every child in the database")
    args = parser.parse_args()

    if args.all:
        ids = get_all_child_ids()
        if not ids:
            print("No children found in GraphDB.")
            sys.exit(0)
        print(f"Exporting {len(ids)} children to {args.out}/")
        for cid in ids:
            export_child(cid, args.out)
        print("Done.")

    elif args.child_id:
        export_child(args.child_id, args.out)

    else:
        parser.print_help()
        sys.exit(1)