"""
SessionSetup — pre-session terminal interface for the CRI dialogue.

Constructed once in CRI_ScriptedDialogue.__init__:

    self.session_setup = SessionSetup(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.run_new_session_interface, self.apply_session_config, ...)
stay identical.

What lives here:
  - Roster loading (../conf/test_config.txt) — child name + condition by ID
  - Condition normalisation to C/E
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
                      "past_exposure": 0, "condition": "C"},
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
                    # Normalise old/new condition labels to canonical C/E.
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

    def save_local_session_config(self, config: dict):
        try:
            os.makedirs(os.path.dirname(self.d.SESSION_CONFIG_PATH), exist_ok=True)
            with open(self.d.SESSION_CONFIG_PATH, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, ensure_ascii=False, indent=2)
        except Exception as e:
            self.d.logger.warning("Could not save local session config: %s", e)

    # --- daily session roster ----------------------------------------------

    def list_session_rosters(self) -> list:
        """Return available JSON roster files from _local/session_rosters."""
        roster_dir = getattr(self.d, "SESSION_ROSTER_DIR", "")
        if not roster_dir or not os.path.isdir(roster_dir):
            return []

        rosters = []
        for filename in sorted(os.listdir(roster_dir)):
            if filename.lower().endswith(".json"):
                rosters.append(os.path.join(roster_dir, filename))
        return rosters

    def load_session_roster(self, path: str) -> dict:
        """Load one daily roster JSON file."""
        with open(path, "r", encoding="utf-8-sig") as roster_file:
            roster = json.load(roster_file)

        if isinstance(roster, list):
            roster = {"children": roster}
        if not isinstance(roster, dict):
            raise ValueError("Roster must be a JSON object with a children list.")
        children = roster.get("children")
        if not isinstance(children, list) or not children:
            raise ValueError("Roster must contain a non-empty 'children' list.")
        return roster

    def select_session_roster_path(self) -> str:
        """Ask the researcher which daily roster file to use."""
        rosters = self.list_session_rosters()
        if not rosters:
            print(f"\nNo roster files found in {self.d.SESSION_ROSTER_DIR}")
            print("Falling back to manual setup.")
            return ""

        if len(rosters) == 1:
            print(f"\nUsing session roster: {os.path.basename(rosters[0])}")
            return rosters[0]

        print("\nAVAILABLE SESSION ROSTERS")
        for idx, path in enumerate(rosters, start=1):
            print(f"  {idx}. {os.path.basename(path)}")

        choice = self.ask_session_value("Choose roster", "1")
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(rosters):
                return rosters[index]

        cleaned = self.d.clean_pasted_path(choice)
        if cleaned:
            if os.path.exists(cleaned):
                return cleaned
            by_name = os.path.join(self.d.SESSION_ROSTER_DIR, cleaned)
            if os.path.exists(by_name):
                return by_name

        print("  Could not find that roster. Falling back to manual setup.")
        return ""

    def roster_child_display(self, entry: dict, index: int) -> str:
        child_id = str(entry.get("child_id") or "").strip()
        child_name = str(entry.get("child_name") or entry.get("name") or "").strip()
        label = f"{child_id} - {child_name}" if child_name else child_id
        return f"  {index}. {label}"

    def select_roster_child(self, roster: dict) -> dict:
        """Ask which child from a loaded roster should start now."""
        children = [
            child for child in roster.get("children", [])
            if isinstance(child, dict) and str(child.get("child_id") or "").strip()
        ]
        if not children:
            raise ValueError("Roster contains no child entries with child_id.")

        print("\nCHILDREN IN ROSTER")
        for idx, child in enumerate(children, start=1):
            print(self.roster_child_display(child, idx))

        choice = self.ask_session_value("Choose child", "1")
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(children):
                return children[index]

        wanted = choice.strip()
        for child in children:
            if str(child.get("child_id") or "").strip() == wanted:
                return child

        print("  Could not find that child. Using the first child in the roster.")
        return children[0]

    def session_config_from_roster_child(self, roster: dict, child: dict, roster_path: str) -> dict:
        """Build the normal session_config.local.json shape from roster data."""
        child_id = str(child.get("child_id") or "").strip()
        child_name = str(child.get("child_name") or child.get("name") or child_id).strip()
        researcher = str(
            child.get("researcher_name")
            or child.get("researcher")
            or roster.get("researcher_name")
            or roster.get("researcher")
            or ""
        ).strip()
        input_mode = str(
            child.get("child_input_mode")
            or child.get("input_mode")
            or roster.get("default_input_mode")
            or roster.get("child_input_mode")
            or getattr(self.d, "CHILD_INPUT_MODE", "keyboard")
        ).strip().lower()

        if "start_phase_index" in child:
            try:
                start_phase_index = int(child.get("start_phase_index") or 0)
            except (TypeError, ValueError):
                start_phase_index = 0
        else:
            start_phase_index = self.parse_phase_index(child.get("start_phase", "1"), default_index=0)

        condition = str(child.get("condition") or "").strip()
        if condition:
            condition = self.normalize_condition_value(condition, default="")
        if not condition:
            condition = self.session_condition_from_um(child_id)

        return {
            "mode": "roster",
            "child_id": child_id,
            "child_name": child_name,
            "researcher_name": researcher,
            "condition": condition,
            "child_input_mode": input_mode,
            "fake_persona_path": str(child.get("fake_persona_path") or self.d.simulated_persona_path),
            "start_phase_index": start_phase_index,
            "roster_path": roster_path,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    # ── small helpers (prompts, condition strings, phase parsing) ────────────

    def ask_session_value(self, label: str, default: str = "") -> str:
        default_hint = f" [{default}]" if default else ""
        try:
            value = input(f"  {label}{default_hint}: ").strip()
            return value if value else default
        except (EOFError, KeyboardInterrupt):
            return default

    def normalize_condition_value(self, value: str, default: str = "C") -> str:
        """
        Normalise any condition representation to the internal "C"/"E".

        Canonical convention:
            "C" = Control: conversational-only memory access
            "E" = Experiment: transmedial metaphor-supported memory access

        Legacy C1/C2 values are still accepted as input:
            "C1" → C
            "C2" → E
        """
        clean = str(value or "").strip().lower()
        if clean in (
            "e", "exp", "experiment", "experimental", "experimental group",
            "metaphor", "metaphor-supported", "transmedial",
            "tablet group", "with tablet", "tablet", "2", "c2", "condition 2",
            "condition_2",
        ):
            return self.d.CONDITION_EXPERIMENT
        if clean in (
            "c", "ctrl", "control", "control group",
            "conversation", "conversational", "conversational-only",
            "verbal", "spoken", "no tablet", "without tablet", "geen tablet",
            "1", "c1", "condition 1",
            "condition_1",
        ):
            return self.d.CONDITION_CONTROL
        return default

    def condition_display(self, condition: str) -> str:
        normalized = self.normalize_condition_value(condition)
        label = self.d.CONDITION_LABELS.get(normalized, "unknown condition")
        return f"{normalized} ({label})"

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
            self.d.last_cri_scenario = {}
            self.d.last_cri_scenario_loaded = False
        self.d.local_child_name = str(self.d.session_config.get("child_name") or "").strip()
        self.d.researcher_name = str(self.d.session_config.get("researcher_name") or "").strip()
        fake_persona_path = str(self.d.session_config.get("fake_persona_path") or "").strip()
        if fake_persona_path:
            self.d.simulated_persona_path = fake_persona_path
        self.d.local_condition = self.normalize_condition_value(
            self.d.session_config.get("condition"),
            default="",
        )
        configured_input = str(
            self.d.session_config.get("child_input_mode")
            or self.d.session_config.get("input_mode")
            or getattr(self.d, "child_input_mode", "")
            or getattr(self.d, "CHILD_INPUT_MODE", "keyboard")
        ).strip().lower()
        if configured_input in ("keyboard", "key", "k", "typed", "type"):
            self.d.child_input_mode = "keyboard"
        elif configured_input in ("microphone", "mic", "m", "whisper", "speech"):
            self.d.child_input_mode = "microphone"
        elif configured_input in ("simulation", "sim"):
            self.d.child_input_mode = "simulation"
            self.d.simulation_mode = True
        self.d.start_phase_index = int(self.d.session_config.get("start_phase_index", 0) or 0)
        self.d.start_phase_index = max(0, min(self.d.start_phase_index, self.d.TOTAL_SCRIPT_PHASES - 1))

    # ── UM API checks during setup ───────────────────────────────────────────

    def check_child_in_um_api(self, child_id: str):
        if self.d.use_fake_persona_um():
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

    def available_um_children(self) -> list:
        """Return child IDs from the live UM API, or [] if it is unavailable."""
        try:
            response = requests.get(f"{self.d.UM_API_BASE}/api/um/", timeout=3)
            if response.status_code == 200:
                children = response.json().get("data", {}).get("children", [])
                return [str(child_id) for child_id in children]
        except Exception:
            return []
        return []

    def get_um_field_for_child(self, child_id: str, field: str) -> str:
        """Read one UM field during pre-session setup, before CHILD_ID is applied."""
        if self.d.use_fake_persona_um():
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
        return self.normalize_condition_value(value, default=self.d.CONDITION_CONTROL)

    # ── interactive entry: new session ───────────────────────────────────────

    def _run_new_session_interface_roster_legacy(self):
        """
        Minimal session setup.

        Asks only:
            1. Child ID
            2. Researcher name
            3. Start phase (1-N)

        Child name + condition are looked up from ../conf/test_config.txt
        by the entered Child ID.  No defaults from previous sessions.
        """
        print("\n" + "=" * 72)
        print("CRI SESSION SETUP")
        print(f"Roster source: {self.d.ROSTER_PATH}")
        print("=" * 72)

        roster = self.load_roster()
        if roster:
            print(f"\n  {len(roster)} children in roster: {', '.join(sorted(roster.keys()))}")
        else:
            print("\n  WARNING: roster is empty. Child name + condition will be blank.")

        # 1. Child ID
        child_id = self.ask_session_value("Child ID", "")
        while not child_id:
            print("  Child ID is required.")
            child_id = self.ask_session_value("Child ID", "")

        # Roster lookup (silent fallback if not present)
        roster_entry = roster.get(child_id, {})
        child_name = roster_entry.get("name", "")
        condition  = roster_entry.get("condition", "")
        if not roster_entry:
            print(f"  Child ID '{child_id}' not in roster — name + condition will be blank.")

        # 2. Researcher name (always blank — no prefill)
        researcher = self.ask_session_value("Researcher name", "")

        # 3. Start phase (1-indexed for the human)
        total = self.d.TOTAL_SCRIPT_PHASES
        phase_raw = self.ask_session_value(f"Start phase (1-{total})", "1")
        try:
            phase_num = int(phase_raw)
        except (TypeError, ValueError):
            phase_num = 1
        phase_num = max(1, min(phase_num, total))
        start_phase_index = phase_num - 1

        # Persona JSON lookup is only relevant when running with fake personas.
        # In live mode (USE_FAKE_PERSONA_UM=False) we read UM straight from
        # Eunike's API and skip this entirely.
        fake_persona_path = self.d.simulated_persona_path
        if self.d.use_fake_persona_um():
            try:
                selected_persona = self.d.select_simulated_persona_by_child_id(child_id)
                self.d.load_simulated_persona()
                fake_persona_path = selected_persona.get("path", self.d.simulated_persona_path)
            except ValueError:
                print(f"  No fake persona JSON for child ID '{child_id}' — will read UM live.")

        # Verify the child exists in the UM source (live API or fake JSON)
        self.check_child_in_um_api(child_id)

        # Summary
        print("\n" + "-" * 56)
        print(f"  Child ID:    {child_id}")
        print(f"  Child name:  {child_name or '(not in roster)'}")
        print(f"  Researcher:  {researcher or '(not set)'}")
        print(f"  Condition:   {self.condition_display(condition) if condition else '(not in roster)'}")
        print(f"  Start phase: {phase_num}")
        print("-" * 56)
        input("\nPress Enter to continue...")

        config = {
            "mode":              "new",
            "child_id":          child_id,
            "child_name":        child_name,
            "researcher_name":   researcher,
            "condition":         condition,
            "fake_persona_path": fake_persona_path,
            "start_phase_index": start_phase_index,
            "created_at":        datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        # Saved for resume support only — NOT read back as defaults next run
        self.save_local_session_config(config)
        self.apply_session_config(config)

    # ── routing: new vs resume ───────────────────────────────────────────────

    def run_new_session_interface(self):
        """
        Minimal session setup for the current workflow.

        New sessions ask for child ID, local child name, and researcher name.
        Condition/exposure are read from the selected fake persona or live UM
        source, and new sessions always start at phase 1.
        """
        previous = self.load_local_session_config()
        print("\n" + "=" * 72)
        print("CRI SESSION SETUP")
        print("This local setup is for child ID, local first name, and researcher.")

        use_fake = self.d.use_fake_persona_um()
        personas = self.d.available_fake_personas() if use_fake else []
        if use_fake:
            print("UM fields are read from fake persona JSON files. New sessions always start at phase 1.")
        else:
            print(f"UM fields are read live from GraphDB through {self.d.UM_API_BASE}.")
            print("New sessions always start at phase 1.")
            children = self.available_um_children()
            if children:
                print(f"\nAVAILABLE GRAPHDB CHILDREN: {', '.join(children)}")
            else:
                print("\nNo GraphDB children listed yet, or the UM API is not reachable.")

        if personas:
            print("\nAVAILABLE FAKE PERSONAS")
            for persona in personas:
                print(
                    f"  {persona['child_id']}: {persona['name']} "
                    f"({persona['exposure']}, {persona['condition']})"
                )

        default_id = previous.get("child_id") or (personas[0]["child_id"] if personas else "")
        child_id = self.ask_session_value("Child ID", default_id)
        while not child_id:
            print("  Child ID is required.")
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
        condition_source = "fake persona" if selected_persona else "GraphDB/UM API"
        print(f"  Condition:   {self.condition_display(condition)}  [from {condition_source}]")
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

    def run_roster_session_interface(self):
        """
        Start a session by choosing from a prepared daily roster JSON.

        The roster fills child_id, child_name, researcher and input mode.
        Condition/profile/scenario content still come from GraphDB.
        """
        print("\n" + "=" * 72)
        print("CRI ROSTER SESSION")
        print(f"Roster folder: {self.d.SESSION_ROSTER_DIR}")
        print("=" * 72)

        roster_path = self.select_session_roster_path()
        if not roster_path:
            self.run_new_session_interface()
            return

        try:
            roster = self.load_session_roster(roster_path)
            child = self.select_roster_child(roster)
            config = self.session_config_from_roster_child(roster, child, roster_path)
        except Exception as e:
            print(f"  Could not use roster: {e}")
            print("  Falling back to manual setup.")
            self.run_new_session_interface()
            return

        self.check_child_in_um_api(config["child_id"])

        print("\n" + "-" * 56)
        print(f"  Roster:      {os.path.basename(roster_path)}")
        print(f"  Child ID:    {config['child_id']}")
        print(f"  Child name:  {config['child_name'] or '(not set)'}")
        print(f"  Researcher:  {config['researcher_name'] or '(not set)'}")
        print(f"  Input mode:  {config['child_input_mode'] or '(ask at startup)'}")
        print(f"  Condition:   {self.condition_display(config['condition'])}  [from GraphDB/roster]")
        print(f"  Start phase: {int(config['start_phase_index']) + 1}")
        print("-" * 56)
        input("\nPress Enter to continue...")

        self.save_local_session_config(config)
        self.apply_session_config(config)

    def configure_session_interface(self):
        if not self.d.ASK_SESSION_INTERFACE_AT_START:
            return

        # Crash recovery remains available through environment variables, but
        # the normal researcher-facing startup flow is roster-first.
        env_resume = self.d.clean_pasted_path(os.environ.get("CRI_RESUME_LOG_PATH", ""))
        if env_resume:
            self.d.run_resume_session_interface(env_resume)
            return

        env_mode = os.environ.get("CRI_SESSION_MODE", "").strip().lower()
        if env_mode in ("skip", "none", "off", "0"):
            return
        if env_mode in ("roster", "daily", "day"):
            self.run_roster_session_interface()
            return
        if env_mode in ("resume", "r"):
            self.d.run_resume_session_interface()
            return
        if env_mode in ("new", "n", "manual", "m"):
            self.run_new_session_interface()
            return

        self.run_roster_session_interface()

    # ── run-mode prompt (mic vs keyboard) ────────────────────────────────────

    def configure_run_mode(self):
        """Ask at startup whether child responses should come from microphone or keyboard."""
        if not self.d.ASK_RUN_MODE_AT_START:
            return

        configured_input = str(
            self.d.session_config.get("child_input_mode")
            or self.d.session_config.get("input_mode")
            or getattr(self.d, "child_input_mode", "")
            or getattr(self.d, "CHILD_INPUT_MODE", "keyboard")
        ).strip().lower()
        if configured_input in ("keyboard", "key", "k", "typed", "type"):
            self.d.child_input_mode = "keyboard"
            return
        if configured_input in ("microphone", "mic", "m", "whisper", "speech"):
            self.d.child_input_mode = "microphone"
            return
        if configured_input in ("simulation", "sim"):
            self.d.simulation_mode = True
            self.d.child_input_mode = "simulation"
            self.d.configure_simulated_persona()
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
        print("Press Enter for keyboard input.")
        print("Type M + Enter for microphone/Whisper input.")
        choice = input("Child input mode: ").strip().lower()
        print("=" * 72)
        if choice in ("m", "mic", "microphone", "whisper", "speech"):
            self.d.child_input_mode = "microphone"
        else:
            self.d.child_input_mode = "keyboard"

    # ── input-mode predicates ────────────────────────────────────────────────

    def use_keyboard_input(self) -> bool:
        return self.d.child_input_mode == "keyboard"

    def use_microphone_input(self) -> bool:
        return self.d.child_input_mode == "microphone" and not self.d.simulation_mode
