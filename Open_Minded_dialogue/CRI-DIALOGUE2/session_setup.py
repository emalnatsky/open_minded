"""
SessionSetup — pre-session terminal interface for the CRI dialogue.

Constructed once in CRI_ScriptedDialogue.__init__:

    self.session_setup = SessionSetup(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.run_new_session_interface, self.apply_session_config, ...)
stay identical.

What lives here:
  - Roster loading (../conf/test_config.txt) — child name + condition by ID
  - Condition normalisation (E/C → C2/C1)
  - Local config file (session_config.local.json) — saved for resume support
  - Pre-session interactive prompt (Child ID, Researcher, Start phase)
  - Run-mode prompt (microphone vs keyboard)
  - Apply selected config to live dialogue state

This module READS class constants (ROSTER_PATH, SESSION_CONFIG_PATH,
TOTAL_SCRIPT_PHASES, UM_API_BASE, USE_FAKE_PERSONA_UM, ...) and other
methods (use_fake_persona_um, select_simulated_persona_by_child_id,
clean_pasted_path, run_resume_session_interface, ...) via self.d.

It WRITES many runtime state fields (CHILD_ID, local_child_name,
researcher_name, local_condition, start_phase_index, session_config,
child_input_mode, simulated_persona_path, simulation_mode) onto the
dialogue. Two-way binding is intentional — this is the entry hook
that bootstraps a session.
"""

import os
import csv
import json
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class SessionSetup:
    """Handles all pre-session bootstrap for one dialogue run."""

    def __init__(self, dialogue):
        self.d = dialogue

    # ── local config file & roster ───────────────────────────────────────────

    def load_local_session_config(self) -> dict:
        try:
            with open(self.d.SESSION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except Exception:
            return {}

    def load_roster(self) -> dict:
        """
        Read ../conf/test_config.txt and return a dict keyed by child ID:
            {
              "169": {"name": "Julianna", "gender": "girl",
                      "past_exposure": 0, "condition": "C1"},
              ...
            }
        Returns empty dict if file is missing or unreadable.
        """
        roster: dict = {}
        if not os.path.exists(self.d.ROSTER_PATH):
            print(f"  Roster file not found at {self.d.ROSTER_PATH}")
            return roster
        try:
            with open(self.d.ROSTER_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    cid = (row.get("id") or "").strip()
                    if not cid:
                        continue
                    past = (row.get("past_exposure") or "0").strip()
                    raw_condition = (row.get("condition") or "").strip()
                    # Normalise "C", "C1", "1", "C2", "2" to C1/C2
                    condition = self.normalize_condition_value(raw_condition, default="")
                    roster[cid] = {
                        "name":          (row.get("Naam") or "").strip(),
                        "gender":        (row.get("gender") or "").strip(),
                        "past_exposure": int(past) if past.isdigit() else 0,
                        "condition":     condition,
                        "raw_condition": raw_condition,
                    }
        except Exception as e:
            print(f"  Could not read roster: {e}")
        return roster

    def load_pl_config(self) -> dict:
        """
        Read test_config.pl and return session values.
        Looks for the .pl file in util/ (primary location), then conf/ as fallback.

        Returns dict with keys:
            child_id, first_name_cri, first_name_tablet,
            operator_name, condition, continue_session
        """
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.abspath(os.path.join(here, "..", "util", "test_config.pl")),  # primary
            os.path.abspath(os.path.join(here, "..", "conf", "test_config.pl")),  # fallback
            os.path.abspath(os.path.join(here, "test_config.pl")),                # same folder
        ]
        for path in candidates:
            if os.path.exists(path):
                return self._parse_pl_config(path)
        print(f"  test_config.pl not found. Expected at: {candidates[0]}")
        return {}

    def _parse_pl_config(self, path: str) -> dict:
        """Parse a Prolog-style config file and return a flat dict."""
        import re
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"  Could not read {path}: {e}")
            return {}

        # Strip comments
        text = re.sub(r"%[^\n]*", "", text)

        def get_fact(name):
            m = re.search(rf'{re.escape(name)}\(\s*[\'"]?([^\'"(),]+)[\'"]?\s*\)', text)
            return m.group(1).strip() if m else ""

        def get_local_var(var_name):
            m = re.search(
                rf'localVariable\(\s*{re.escape(var_name)}\s*,\s*["\']?([^"\'\\)]+)["\']?\s*\)',
                text,
            )
            return m.group(1).strip() if m else ""

        child_id        = get_fact("userId")
        first_name_cri  = get_local_var("first_name_cri")
        first_name_tablet = get_local_var("first_name_tablet") or first_name_cri
        operator_name   = get_local_var("operator_name")
        condition_raw   = get_fact("condition") or "experimental"
        continue_raw    = get_fact("continueSession") or "false"

        condition = self.normalize_condition_value(condition_raw, default="C1")
        continue_session = continue_raw.strip().lower() in ("true", "yes", "1")

        result = {
            "child_id":           child_id,
            "first_name_cri":     first_name_cri,
            "first_name_tablet":  first_name_tablet,
            "operator_name":      operator_name,
            "condition":          condition,
            "continue_session":   continue_session,
        }
        print(f"  Loaded {path}")
        print(f"  Child ID:       {child_id}")
        print(f"  CRI name:       {first_name_cri}  (NAO pronounces this)")
        print(f"  Tablet name:    {first_name_tablet}  (shown on book cover)")
        print(f"  Researcher:     {operator_name}")
        print(f"  Condition:      {condition}")
        return result

    def save_local_session_config(self, config: dict):
        try:
            os.makedirs(os.path.dirname(self.d.SESSION_CONFIG_PATH), exist_ok=True)
            with open(self.d.SESSION_CONFIG_PATH, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, ensure_ascii=False, indent=2)
        except Exception as e:
            self.d.logger.warning("Could not save local session config: %s", e)

    # ── small helpers (prompts, condition strings, phase parsing) ────────────

    def ask_session_value(self, label: str, default: str = "") -> str:
        default_hint = f" [{default}]" if default else ""
        try:
            value = input(f"  {label}{default_hint}: ").strip()
            return value if value else default
        except (EOFError, KeyboardInterrupt):
            return default

    def normalize_condition_value(self, value: str, default: str = "C1") -> str:
        """
        Normalise any condition representation to the internal "C1"/"C2".

        Roster convention (../conf/test_config.txt):
            "E" = experimental (with tablet) → internal "C2"
            "C" = control      (no tablet)   → internal "C1"

        Internal convention (used by all downstream tutorial/condition logic):
            "C1" = no tablet
            "C2" = with tablet
        """
        clean = str(value or "").strip().lower()
        if clean in ("e", "exp", "experimental", "tablet group"):
            return "C2"
        if clean in ("c", "ctrl", "control", "control group"):
            return "C1"
        if clean in ("2", "c2", "condition 2", "with tablet", "tablet"):
            return "C2"
        if clean in ("1", "c1", "condition 1", "no tablet", "without tablet", "geen tablet"):
            return "C1"
        return default

    def condition_display(self, condition: str) -> str:
        normalized = self.normalize_condition_value(condition)
        return f"{normalized} ({'with tablet' if normalized == 'C2' else 'no tablet'})"

    def parse_phase_index(self, value: str, default_index: int = 0) -> int:
        try:
            phase = int(str(value).strip())
            if 1 <= phase <= self.d.TOTAL_SCRIPT_PHASES:
                return phase - 1
        except (TypeError, ValueError):
            pass
        return default_index

    # ── apply config to live dialogue state ──────────────────────────────────

    def apply_session_config(self, config: dict):
        self.d.session_config = dict(config or {})
        child_id = str(self.d.session_config.get("child_id") or "").strip()
        if child_id:
            self.d.CHILD_ID = child_id

        # first_name_cri   → what NAO TTS pronounces in the dialogue
        # first_name_tablet → what appears on the tablet book cover
        # child_name is the legacy key — used as fallback for both
        legacy_name = str(self.d.session_config.get("child_name") or "").strip()
        cri_name    = str(self.d.session_config.get("first_name_cri") or legacy_name).strip()
        tablet_name = str(self.d.session_config.get("first_name_tablet") or legacy_name).strip()

        self.d.local_child_name        = cri_name     # used by TTS + script
        self.d.local_child_name_cri    = cri_name     # explicit TTS name
        self.d.local_child_name_tablet = tablet_name  # explicit tablet display name

        self.d.researcher_name = str(self.d.session_config.get("researcher_name") or "").strip()
        fake_persona_path = str(self.d.session_config.get("fake_persona_path") or "").strip()
        if fake_persona_path:
            self.d.simulated_persona_path = fake_persona_path
        self.d.local_condition = self.normalize_condition_value(
            self.d.session_config.get("condition"),
            default="",
        )
        self.d.start_phase_index = int(self.d.session_config.get("start_phase_index", 0) or 0)
        self.d.start_phase_index = max(0, min(self.d.start_phase_index, self.d.TOTAL_SCRIPT_PHASES - 1))

    # ── UM API checks during setup ───────────────────────────────────────────

    def check_child_in_um_api(self, child_id: str):
        if self.d.USE_FAKE_PERSONA_UM:
            print(f"\nUsing fake persona JSON for child '{child_id}'.")
            return

        print(f"\nChecking child '{child_id}' in UM API ({self.d.UM_API_BASE})...")
        try:
            response = requests.get(f"{self.d.UM_API_BASE}/api/um/{child_id}", timeout=3)
            if response.status_code == 200:
                print("  Child found.")
            elif response.status_code == 404:
                print("  Child not found.")
            else:
                print(f"  UM API returned {response.status_code}.")
        except Exception:
            print("  UM API is not reachable right now.")

    def get_um_field_for_child(self, child_id: str, field: str) -> str:
        """Read one UM field during pre-session setup, before CHILD_ID is applied."""
        if self.d.USE_FAKE_PERSONA_UM:
            wanted = str(child_id).strip()
            for persona in self.d.available_fake_personas():
                if persona["child_id"] == wanted:
                    try:
                        with open(persona["path"], "r", encoding="utf-8") as persona_file:
                            data = json.load(persona_file)
                        value = data.get(field)
                        return str(value).strip() if value else ""
                    except Exception:
                        return ""
            return ""

        try:
            response = requests.get(f"{self.d.UM_API_BASE}/api/um/{child_id}/field/{field}", timeout=3)
            if response.status_code == 200:
                value = response.json().get("data", {}).get("value")
                return str(value).strip() if value else ""
        except Exception:
            return ""
        return ""

    def session_condition_from_um(self, child_id: str) -> str:
        value = self.get_um_field_for_child(child_id, self.d.TUTORIAL_CONDITION_FIELD)
        return self.normalize_condition_value(value, default="C1")

    # ── interactive entry: new session ───────────────────────────────────────

    def _run_new_session_interface_roster_legacy(self):
        """
        Session setup using test_config.pl.

        Reads child_id, first_name_cri, first_name_tablet, operator_name,
        and condition directly from test_config.pl — no manual entry needed.
        The researcher just presses Enter to confirm and optionally picks
        the start phase.
        """
        print("\n" + "=" * 72)
        print("CRI SESSION SETUP")
        print("Reading from test_config.pl ...")
        print("=" * 72)

        pl = self.load_pl_config()

        # Pull values from the pl file
        child_id      = pl.get("child_id", "")
        first_name_cri    = pl.get("first_name_cri", "")
        first_name_tablet = pl.get("first_name_tablet", "")
        condition     = pl.get("condition", "C1")
        researcher    = pl.get("operator_name", "")

        # If pl file was empty or missing, fall back to asking
        if not child_id:
            child_id = self.ask_session_value("Child ID (not in pl file)", "")

        # 2. Researcher name — prefilled from pl, can override
        researcher = self.ask_session_value("Researcher name", researcher)

        # 3. Start phase
        total = self.d.TOTAL_SCRIPT_PHASES
        phase_raw = self.ask_session_value(f"Start phase (1-{total})", "1")
        try:
            phase_num = int(phase_raw)
        except (TypeError, ValueError):
            phase_num = 1
        phase_num = max(1, min(phase_num, total))
        start_phase_index = phase_num - 1

        fake_persona_path = self.d.simulated_persona_path
        if self.d.use_fake_persona_um():
            try:
                selected_persona = self.d.select_simulated_persona_by_child_id(child_id)
                self.d.load_simulated_persona()
                fake_persona_path = selected_persona.get("path", self.d.simulated_persona_path)
            except ValueError:
                print(f"  No fake persona JSON for child ID '{child_id}' — will read UM live.")

        # Verify the child exists in the UM source
        self.check_child_in_um_api(child_id)

        # Summary
        print("\n" + "-" * 56)
        print(f"  Child ID:      {child_id}")
        print(f"  CRI name:      {first_name_cri or '(not set)'}  ← NAO says this")
        print(f"  Tablet name:   {first_name_tablet or '(not set)'}  ← shown on book")
        print(f"  Researcher:    {researcher or '(not set)'}")
        print(f"  Condition:     {self.condition_display(condition) if condition else '(not set)'}")
        print(f"  Start phase:   {phase_num}")
        print("-" * 56)
        input("\nPress Enter to continue...")

        config = {
            "mode":                "new",
            "child_id":            child_id,
            "child_name":          first_name_cri,    # legacy key → TTS name
            "first_name_cri":      first_name_cri,    # explicit TTS name
            "first_name_tablet":   first_name_tablet, # explicit tablet display name
            "researcher_name":     researcher,
            "condition":           condition,
            "fake_persona_path":   fake_persona_path,
            "start_phase_index":   start_phase_index,
            "created_at":          datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.save_local_session_config(config)
        self.apply_session_config(config)

    # ── routing: new vs resume ───────────────────────────────────────────────

    def run_new_session_interface(self):
        if self.d.use_fake_persona_um():
            self._run_new_session_interface_fake_persona()
        else:
            self._run_new_session_interface_roster_legacy()

    def run_new_session_interface_fake_persona(self):
        """
        Minimal session setup for the current fake-persona workflow.

        New sessions ask for child ID, local child name, and researcher name.
        Condition/exposure are read from the selected fake persona or UM source,
        and new sessions always start at phase 1.
        """
        previous = self.load_local_session_config()
        print("\n" + "=" * 72)
        print("CRI SESSION SETUP")
        print("This local setup is for child ID, local first name, and researcher.")
        print("UM fields are read from fake persona JSON files. New sessions always start at phase 1.")

        personas = self.d.available_fake_personas()
        if personas:
            print("\nAVAILABLE FAKE PERSONAS")
            for persona in personas:
                print(
                    f"  {persona['child_id']}: {persona['name']} "
                    f"({persona['exposure']}, {persona['condition']})"
                )

        default_id = previous.get("child_id") or (personas[0]["child_id"] if personas else self.d.CHILD_ID)
        child_id = self.ask_session_value("Child ID", default_id)

        selected_persona = self.d.select_simulated_persona_by_child_id(child_id) if personas else {}
        if selected_persona:
            self.d.load_simulated_persona()

        persona_name = selected_persona.get("name") or ""
        child_name = self.ask_session_value(
            "Child first name (local only)",
            persona_name or previous.get("child_name", ""),
        )
        researcher = self.ask_session_value("Researcher name", previous.get("researcher_name", ""))

        self.check_child_in_um_api(child_id)
        condition = self.session_condition_from_um(child_id)
        start_phase_index = 0

        print("\n" + "-" * 56)
        print(f"  Child ID:    {child_id}")
        print(f"  Child name:  {child_name or '(not set)'}")
        print(f"  Researcher:  {researcher or '(not set)'}")
        if selected_persona:
            print(f"  Exposure:    {selected_persona['exposure']}")
        print(f"  Condition:   {self.condition_display(condition)}  [from fake persona]")
        print("  Start phase: 1  [new session]")
        print("-" * 56)
        input("\nPress Enter to continue...")

        config = {
            "mode": "new",
            "child_id": child_id,
            "child_name": child_name,
            "researcher_name": researcher,
            "condition": condition,
            "fake_persona_path": selected_persona.get("path", self.d.simulated_persona_path),
            "start_phase_index": start_phase_index,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.save_local_session_config(config)
        self.apply_session_config(config)

    def configure_session_interface(self):
        if not self.d.ASK_SESSION_INTERFACE_AT_START:
            return

        env_resume = self.d.clean_pasted_path(os.environ.get("CRI_RESUME_LOG_PATH", ""))
        if env_resume:
            self.d.run_resume_session_interface(env_resume)
            return

        env_mode = os.environ.get("CRI_SESSION_MODE", "").strip().lower()
        if env_mode in ("skip", "none", "off", "0"):
            return
        if env_mode in ("resume", "r"):
            self.d.run_resume_session_interface()
            return
        if env_mode in ("new", "n"):
            self.run_new_session_interface()
            return

        print("\n" + "=" * 72)
        print("CRI SESSION")
        print("Press Enter for a new session.")
        print("Type R + Enter to resume, or paste a previous conversation JSON path directly.")
        raw_choice = input("Session mode: ").strip()
        choice = raw_choice.lower()
        print("=" * 72)

        if choice in ("r", "resume"):
            self.d.run_resume_session_interface()
        elif self.d.clean_pasted_path(raw_choice).lower().endswith(".json"):
            self.d.run_resume_session_interface(raw_choice)
        else:
            self.run_new_session_interface()

    # ── run-mode prompt (mic vs keyboard) ────────────────────────────────────

    def configure_run_mode(self):
        """Ask at startup whether child responses should come from microphone or keyboard."""
        if not self.d.ASK_RUN_MODE_AT_START:
            return

        input_mode = os.environ.get("CRI_CHILD_INPUT_MODE", "").strip().lower()
        if input_mode in ("keyboard", "key", "k", "typed", "type"):
            self.d.child_input_mode = "keyboard"
            return
        if input_mode in ("microphone", "mic", "m", "whisper", "speech"):
            self.d.child_input_mode = "microphone"
            return

        env_choice = os.environ.get("CRI_SIMULATION_MODE", "").strip().lower()
        if env_choice in ("1", "true", "yes", "y", "sim", "simulation"):
            self.d.simulation_mode = True
            self.d.child_input_mode = "simulation"
            self.d.configure_simulated_persona()
            return
        if env_choice in ("0", "false", "no", "n", "real", "normal"):
            self.d.simulation_mode = False

        print("\n" + "=" * 72)
        print("CRI 4.0 CHILD INPUT MODE")
        print("Press Enter for microphone/Whisper input.")
        print("Type K + Enter for keyboard input.")
        choice = input("Child input mode: ").strip().lower()
        print("=" * 72)
        if choice in ("k", "key", "keyboard", "typed", "type"):
            self.d.child_input_mode = "keyboard"
        else:
            self.d.child_input_mode = "microphone"

    # ── input-mode predicates ────────────────────────────────────────────────

    def use_keyboard_input(self) -> bool:
        return self.d.child_input_mode == "keyboard"

    def use_microphone_input(self) -> bool:
        return self.d.child_input_mode == "microphone" and not self.d.simulation_mode
