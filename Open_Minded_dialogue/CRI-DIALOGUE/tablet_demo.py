"""
tablet_demo.py — standalone visual test for the UM tablet.

What this does:
    Walks through 6 fake "phases", each one writing session_state.json
    with a growing list of unlocked categories. Pauses between phases so
    you can look at the tablet and verify:
        - All categories locked at start
        - Categories unlock one by one as phases progress
        - Tappable pages show field values from the GraphDB
        - Erase + write animation fires when a UM value changes mid-session
        - Child name appears on the welcome book

What this does NOT need:
    - NAO, Whisper, GPT, SIC framework
    - The full CRI-BRANCH-BASIC4_0.py
    - Any other process besides the tablet server

What you must have running:
    Terminal 1:  python main.py                        (Eunike's API)
    Terminal 2:  cd UM-TABLET && python um_tablet_server.py
    Terminal 3:  cd CRI-DIALOGUE && python tablet_demo.py   (this file)

    Open the tablet URL in a browser side-by-side with this terminal.

Usage:
    python tablet_demo.py                          (uses CHILD_ID below)
    python tablet_demo.py 169                      (override child id)
    python tablet_demo.py 169 --no-um-writes       (skip the GraphDB pokes,
                                                    test phase-locking only)
"""

import os
import sys
import json
import time
from datetime import datetime

import requests


# ── Config ──────────────────────────────────────────────────────────────────

CHILD_ID = "3"                                  
UM_API_BASE = "http://localhost:8000"

# session_state.json lives in Open_Minded_dialogue/, one folder up from
# CRI-DIALOGUE/. The tablet server reads ../session_state.json relative
# to UM-TABLET/, which resolves to the same file.
_HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_STATE_PATH = os.path.abspath(os.path.join(_HERE, "..", "session_state.json"))


# ── The 6-phase script ──────────────────────────────────────────────────────

# Each phase is just a dict describing what to do. Keep it small and clear.
PHASES = [
    {
        "name":     "Welcome",
        "phase":    1,
        "unlocks":  [],
        "what":     "Session just started. ALL categories should be locked with 🔒.",
        "expect":   "Tablet: welcome screen with child's name on the book. "
                    "Tap the book — TOC opens — every category shows 🔒.",
    },
    {
        "name":     "Hobby talk",
        "phase":    2,
        "unlocks":  ["hobby"],
        "what":     "Leo just mentioned the child's favourite hobby.",
        "expect":   "Hobby is now tappable. Tap it — pills appear with hobby values "
                    "from the GraphDB. Other categories still 🔒.",
    },
    {
        "name":     "Sport talk",
        "phase":    3,
        "unlocks":  ["hobby", "sport"],
        "what":     "Leo just mentioned the child's sport.",
        "expect":   "Sport is now also tappable. Hobby is still tappable. "
                    "Remaining categories still 🔒.",
    },
    {
        "name":     "Mistake about food",
        "phase":    4,
        "unlocks":  ["hobby", "sport", "eten"],
        "what":     "Leo just said the wrong fav_food. We POST a fake-wrong "
                    "value to the DB so the tablet sees a value change.",
        "expect":   "Eten unlocks. Tap Eten — pill shows the (wrong) value.",
        "um_write": {"field": "fav_food", "value": "spruitjes"},
    },
    {
        "name":     "Mistake corrected",
        "phase":    4,
        "unlocks":  ["hobby", "sport", "eten"],
        "what":     "Child corrected Leo. We POST the corrected value to DB. "
                    "The tablet should erase the wrong value and write the new one.",
        "expect":   "If Eten page is open: watch the eraser swipe across the food pill "
                    "left-to-right, then the new value writes in.",
        "um_write": {"field": "fav_food", "value": "pannenkoeken"},
        "pause_before_write_s": 3,
    },
    {
        "name":     "Pet + Books + Music + School + Aspiratie",
        "phase":    5,
        "unlocks":  ["hobby", "sport", "eten", "dieren", "boeken", "muziek", "school", "sociaal", "aspiratie"],
        "what":     "Leo wrapped up all topics. Everything is unlocked.",
        "expect":   "Every category on the TOC is now tappable. No more 🔒.",
    },
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def write_session_state(phase_num: int, unlocked: list, child_id: str):
    """Mimic what TabletStateWriter.update() does on every turn."""
    state = {
        "child_id":            child_id,
        "child_name":          "",   # tablet uses roster lookup, not this field
        "phase":               phase_num,
        "unlocked_categories": list(unlocked),
        "mistakes":            {},
        "_demo":               True,
        "_written_at":         datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(SESSION_STATE_PATH), exist_ok=True)
    tmp = SESSION_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSION_STATE_PATH)


def write_um_field(child_id: str, field: str, value: str) -> bool:
    """POST one field update to Eunike's API. Returns True on success."""
    url = f"{UM_API_BASE}/api/um/{child_id}/fields"
    payload = {"fields": {field: value}, "source": "tablet_demo"}
    try:
        r = requests.post(url, json=payload, timeout=3)
        return r.status_code in (200, 201, 202, 204)
    except Exception as e:
        print(f"   ⚠  Could not write to UM API: {e}")
        return False


def check_um_api() -> bool:
    try:
        r = requests.get(f"{UM_API_BASE}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def header(text: str):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def divider():
    print("-" * 72)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    child_id = CHILD_ID
    do_um_writes = True
    for arg in sys.argv[1:]:
        if arg == "--no-um-writes":
            do_um_writes = False
        elif arg.isdigit():
            child_id = arg
        else:
            print(f"Ignoring unknown arg: {arg}")

    header(f"TABLET PHASE-LOCK DEMO")
    print(f"  Child ID:           {child_id}")
    print(f"  Session state file: {SESSION_STATE_PATH}")
    print(f"  UM API:             {UM_API_BASE}")
    print(f"  UM writes enabled:  {do_um_writes}")
    divider()

    # Sanity checks
    if not check_um_api():
        print("  ⚠  Eunike's UM API is NOT reachable at "
              f"{UM_API_BASE}. The tablet will still show locks/unlocks, "
              "but pills will be empty.")
        print("     Start Eunike's main.py to see real UM values.")
    else:
        print("  ✓ UM API reachable")

    # Before we start, wipe the session state so the tablet starts blank
    write_session_state(0, [], child_id)
    print(f"\n  Session state reset. Open the tablet now; you should see all categories locked.")

    input("\n  Press Enter to start phase 1 →  ")

    # Walk through phases
    for i, p in enumerate(PHASES, start=1):
        header(f"PHASE {i}/{len(PHASES)} — {p['name']}")
        print(f"  What just happened:  {p['what']}")
        print(f"  Tablet should show:  {p['expect']}")
        print(f"  Phase number:        {p['phase']}")
        print(f"  Unlocking:           {p['unlocks'] or '(nothing yet)'}")
        if p.get("um_write") and do_um_writes:
            uw = p["um_write"]
            print(f"  UM write:            {uw['field']} = '{uw['value']}'")
        divider()

        # Write session_state.json first so the tablet unlocks the right categories
        write_session_state(p["phase"], p["unlocks"], child_id)

        # Pause before the UM write so the tester can open the relevant category
        # page on the tablet before the value-change animation fires.
        pause = p.get("pause_before_write_s")
        if pause and p.get("um_write") and do_um_writes:
            print(f"  ⏱  Open '{p['um_write']['field']}'s' category page on the tablet "
                  f"now. Writing UM update in {pause}s ...")
            time.sleep(pause)

        # Now POST the UM value change (if any)
        if p.get("um_write") and do_um_writes:
            ok = write_um_field(child_id, p["um_write"]["field"], p["um_write"]["value"])
            print(f"  {'✓' if ok else '✗'} UM write to '{p['um_write']['field']}' → "
                  f"'{p['um_write']['value']}'")

        if i < len(PHASES):
            input("\n  Press Enter to advance to the next phase →  ")
        else:
            input("\n  Press Enter to finish the demo →  ")

    # Final
    header("DEMO COMPLETE")
    print("  All phases ran. The session_state.json file still has the last")
    print("  phase's state. To reset the tablet, restart this script or")
    print("  manually wipe the file.")
    print()


if __name__ == "__main__":
    main()
