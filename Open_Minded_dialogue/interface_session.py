"""
Our research oriented minimal terminal Interface to be connected to the CRU-BRANCH-FINAL (or whatever we name it)
BEFORE SESSION collects session config and the dialogue starts:
    - Child ID (GraphDB anonymous ID, used to pull UM from Eunike's API)
    - Child first name (confidential so only stays locally on our laptop)
    - Researcher name
    - Teacher name
    - Session number
    - Condition (1 = no tablet / 2 = with tablet)
    - Start phase (1-6)

DURING SESSION you can press these buttons for:
    B = repeat last turn (go back)
    P = pause / resume
    1-6 = jump to phase
    Q = quit
"""

import os
import sys
import json
import time
import select
import tty
import termios
import threading
import requests
from datetime import datetime

# Local config file (ONLY LOCAL NEVER ON GIT IMPORTANT)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_HERE, "session_config.json")


# ── Pre-session setup ─────────────────────────────────────────────────────────

def run_setup(um_api_base: str = "http://localhost:8000") -> dict:
    """
    Show pre-session terminal form. Returns config dict.

    config keys:
        child_id        str  ; GraphDB ID (anonymous)
        child_name      str   ; real first name (confidential, local only)
        researcher_name str
        teacher_name    str
        session_number  int
        condition       int   ; 1 or 2
        start_phase     int   ; 0-indexed
        timestamp       str   ; ISO format
    """
    _clear()

    print("CRI SESSION SETUP: Open-Minded Robots ")

    # Load previous config as defaults
    prev = {}

    # ── Fields ────────────────────────────────────────────────────────
    child_id = _ask(
        "Child ID ",
        prev.get("child_id", "Julianna_dutch")
    )

    child_name = _ask(
        "Child first name",
        prev.get("child_name", "")
    )

    researcher = _ask(
        "Researcher name",
        prev.get("researcher_name", "")
    )

    teacher = _ask(
        "Teacher name",
        prev.get("teacher_name", "")
    )

    session_nr_raw = _ask(
        "Session number",
        str(prev.get("session_number", 1))
    )
    session_nr = int(session_nr_raw) if session_nr_raw.isdigit() else 1

    cond_raw = _ask(
        "Condition  [1 = no tablet  /  2 = with tablet]",
        str(prev.get("condition", 1))
    )
    condition = int(cond_raw) if cond_raw in ("1", "2") else 1

    phase_raw = _ask(
        "Start from phase  [1-6]",
        str(prev.get("start_phase", 0) + 1)
    )
    start_phase = (int(phase_raw) - 1) if phase_raw.isdigit() and 1 <= int(phase_raw) <= 6 else 0

    # ── Verify child in Eunike's API ──────────────────────────────────
    print(f"\n  Checking '{child_id}' in UM API ({um_api_base})…")
    try:
        r = requests.get(f"{um_api_base}/api/um/{child_id}", timeout=3)
        if r.status_code == 200:
            print(" Child found")
        elif r.status_code == 404:
            print(" Child NOT found")
        else:
            print(f"  API returned {r.status_code}")
    except Exception:
        print(" Cannot reach UM API")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "─"*56)
    print(f"  Child ID:    {child_id}")
    print(f"  Child name:  {child_name or '(not set)'} ")
    print(f"  Researcher:  {researcher or '(not set)'}")
    print(f"  Teacher:     {teacher or '(not set)'}")
    print(f"  Session:     #{session_nr}")
    print(f"  Condition:   C{condition}  ({'no tablet' if condition == 1 else 'with tablet'})")
    print(f"  Start phase: {start_phase + 1}")
    print("─"*56)
    print("\n  Controls during session:")
    print("    P = pause / resume")
    print("    B = repeat last turn")
    print("    1-6 = jump to phase")
    print("    Q = quit")
    print()

    input("  Press Enter to start the session…")
    print()

    config = {
        "child_id":        child_id,
        "child_name":      child_name if child_name else "hoi",
        "researcher_name": researcher,
        "teacher_name":    teacher,
        "session_number":  session_nr,
        "condition":       condition,
        "start_phase":     start_phase,
        "timestamp":       datetime.now().isoformat(),
    }

    # Save locally for next session defaults
    _save_config(config)

    return config


# ── DURING session controls ───────────────────────────────────────────────────

class SessionControls:
    """
    Background keyboard for during-session controls.
    Runs in a daemon thread so does not block the dialogue loop!!!!!!!!!!!
    OR ELSE it will interupt your dialogue loop and the whisper will not be whispering anymore. 

    Usage:
        controls = SessionControls()
        controls.start()

        #in turn loop:
        i = controls.check(i, len(script))
        if controls.quit_requested:
            break
    """

    def __init__(self):
        self.paused         = False
        self.back_turn      = False
        self.jump_to_phase  = None   
        self.quit_requested = False
        self._thread        = None
        self._fd            = None
        self._old_term      = None
        self._running       = False

    def start(self):
        """Start background key listener. Call AFTER all setup is done."""
        self._fd       = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        self._running  = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Restore terminal"""
        self._running = False
        if self._old_term and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass

    def _loop(self):
        while self._running:
            try:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    key = sys.stdin.read(1).lower()
                    self._handle(key)
            except Exception:
                break

    def _handle(self, key: str):
        if key == 'p':
            self.paused = not self.paused
            print(f"\n  [{'PAUSED: press P to resume' if self.paused else 'RESUMED'}]",
                  flush=True)

        elif key == 'b':
            self.back_turn = True
            print("\n  [REPEAT LAST TURN]", flush=True)

        elif key in '123456':
            self.jump_to_phase = int(key) - 1
            print(f"\n  [JUMP TO PHASE {key}]", flush=True)

        elif key == 'q':
            self.quit_requested = True
            print("\n  [QUIT REQUESTED]", flush=True)

    def check(self, i: int, script_len: int) -> int:
        """
        Call this at the start and end of each turn.
        Handles pause (blocks until resumed), back, jump.
        Returns the turn index to go to next.
        """
        # Pause: block here until researcher presses P again
        while self.paused:
            time.sleep(0.3)

        if self.back_turn:
            self.back_turn = False
            return max(i - 1, 0)

        if self.jump_to_phase is not None:
            target = min(self.jump_to_phase, script_len - 1)
            self.jump_to_phase = None
            return target

        return i


# ── Helpers────────────────────────────────────────────────────

def _ask(label: str, default: str = "") -> str:
    """Print a prompt and return input, using default if empty."""
    default_hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{default_hint}: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def _clear():
    os.system("clear" if os.name != "nt" else "cls")


def _load_prev_config() -> dict:
    try:
        with open(_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(config: dict):
    try:
        with open(_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f" CAREFUL Could not save config: {e}")


# ── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    config = run_setup()
    print("\nConfig returned:")
    print(json.dumps(config, indent=2))
