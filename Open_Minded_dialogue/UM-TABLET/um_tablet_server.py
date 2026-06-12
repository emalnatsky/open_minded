"""
SIC webserver that serves a live User Model dashboard to a tablet.
NO REDIS 
Architecture:
    Eunike's FastAPI + GraphDB (UM store)
        ↓  poll every 2s via REST API
    UMTabletServer (this file)
        ↓  WebInfoMessage broadcast via Socket.IO
    webfiles/index.html  (tablet browser)

The tablet just opens http://<laptop-ip>:8080 on WiFi.

BUT we need for it: 
    1. Eunike's UM API running:  python main.py
    2. run-webserver in separate terminal
    3. This script:        python um_tablet_server.py
"""

import os
import json
import re
import socket
import threading
import time
import webbrowser
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests as http

from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication
from sic_framework.services.webserver.webserver_service import (
    WebInfoMessage,
    Webserver,
    WebserverConf,
)

# ---------------------------------------------------------------------------config-------------------------------------------------------------------------

CHILD_ID        = "3"                          # fallback — overridden by session_state.json once dialogue starts
UM_API_BASE     = "http://localhost:8000"
POLL_INTERVAL_S = 2.0
WEB_PORT        = 8080
TABLET_EVENT_PORT = 8081
API_TIMEOUT_S   = 3.0
SESSION_STATE_PATH = "../_local/session_state.json"  # written by CRI-DIALOGUE/tablet_state.py
TABLET_EVENTS_LOG_PATH = "../_local/tablet_events.jsonl"
TEST_CONFIG_PATH = "../util/test_config.pl"
# Note: child name + ID now come from session_state.json (written by the dialogue).
# No separate roster file needed — the dialogue reads util/test_config.pl and
# writes the resolved names into session_state.json.


class UMTabletServer(SICApplication):
    """
    Serves a live User Model dashboard to a tablet browser over WiFi.
    Polls Eunike's FastAPI every POLL_INTERVAL_S seconds.
    Always broadcasts every poll so late-connecting browsers get data.
    """

    def __init__(self):
        super(UMTabletServer, self).__init__()
        self.webserver = None
        self._tablet_event_server = None
        self._child_id = CHILD_ID
        self._last_missing_child_warning_id = None
        self.set_log_level(sic_logging.WARNING)
        self.load_local_env()
        self.setup()

    # ------------------------------------------------------------------setup---------------------------------------------------------------

    def setup(self):
        self.logger.debug("Setting up UM Tablet Server …")

        current_dir  = os.path.dirname(os.path.abspath(__file__))
        webfiles_dir = os.path.join(current_dir, "webfiles")

        web_conf = WebserverConf(
            host="0.0.0.0",
            port=WEB_PORT,
            templates_dir=webfiles_dir,
            static_dir=webfiles_dir,
            cors_allowed_origins="*",
        )
        self.webserver = Webserver(conf=web_conf)

        threading.Thread(target=self._check_api,      daemon=True).start()
        threading.Thread(target=self._start_tablet_event_receiver, daemon=True).start()
        threading.Thread(target=lambda: self._open_when_ready(f"http://localhost:{WEB_PORT}"), daemon=True).start()
        threading.Thread(target=self._log_tablet_url, daemon=True).start()

        self.logger.debug("Setup complete.")

    # ------------------------------------------------------------------helpers----------------------------------------------------------------

    def load_local_env(self):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = (
            os.path.join(here, "..", "_local", "config", ".env"),
            os.path.join(here, "..", "conf", ".env"),
        )
        for path in candidates:
            if os.path.exists(path):
                self.load_env(path)
                return

    def _normalize_condition_value(self, value: str) -> str:
        clean = " ".join(str(value or "").strip().lower().replace("_", " ").split())
        if clean in ("e", "c2", "2", "condition 2", "experimental", "experiment", "tablet", "with tablet"):
            return "E"
        if clean in ("c", "c1", "1", "condition 1", "control", "no tablet", "without tablet"):
            return "C"
        return str(value or "").strip()

    def _read_test_config_state(self) -> dict:
        """Read current child ID/name/condition from util/test_config.pl."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, TEST_CONFIG_PATH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return {}

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

        child_id = get_fact("userId")
        child_name = get_local_var("first_name_tablet") or get_local_var("first_name_cri")
        condition = self._normalize_condition_value(get_fact("condition"))

        state = {}
        if child_id:
            state["child_id"] = child_id
        if child_name:
            state["child_name"] = child_name
        if condition:
            state["condition"] = condition
        return state

    def _read_session_state(self) -> dict:
        """
        Read session_state.json, then overlay current util/test_config.pl identity.

        The overlay prevents an old local session_state.json from keeping a
        previous child's name on the tablet before the new CRI session writes
        its first state update.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, SESSION_STATE_PATH)
        state = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)
            if isinstance(loaded_state, dict):
                state.update(loaded_state)
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning("Could not read session_state.json: %s", e)

        config_state = self._read_test_config_state()
        if config_state:
            state.update(config_state)

        session_child_id = str(state.get("child_id") or "").strip()
        if session_child_id and session_child_id != self._child_id:
            self.logger.debug(
                "Child ID updated from local session config: %s -> %s",
                self._child_id, session_child_id,
            )
            self._child_id = session_child_id
        return state

    def _check_api(self):
        time.sleep(1.0)
        try:
            r = http.get(f"{UM_API_BASE}/", timeout=API_TIMEOUT_S)
            if r.status_code == 200:
                self.logger.debug("Eunike's UM API reachable at %s ✓", UM_API_BASE)
            else:
                self.logger.warning("UM API returned status %s.", r.status_code)
        except Exception:
            self.logger.warning("Cannot reach UM API at %s. Is main.py running?", UM_API_BASE)

    def _open_when_ready(self, url: str):
        ready_url = url.rstrip("/") + "/readyz"
        deadline  = time.time() + 10.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(ready_url, timeout=0.5) as r:
                    if r.status == 200:
                        webbrowser.open(url, new=2)
                        return
            except Exception:
                pass
            time.sleep(0.2)
        webbrowser.open(url, new=2)

    def _get_lan_ips(self) -> list:
        candidates = []

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                candidates.append(sock.getsockname()[0])
        except Exception:
            pass

        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                candidates.append(info[4][0])
        except Exception:
            pass

        usable = []
        for ip_address in candidates:
            if (
                ip_address
                and not ip_address.startswith(("127.", "169.254."))
                and ip_address not in usable
            ):
                usable.append(ip_address)
        return usable or ["<YOUR_LAPTOP_IP>"]

    def _log_tablet_url(self):
        tablet_urls = [f"http://{lan_ip}:{WEB_PORT}" for lan_ip in self._get_lan_ips()]
        time.sleep(1.5)
        print("")
        print("=" * 55)
        print("Open this on the tablet (same WiFi):")
        for tablet_url in tablet_urls:
            print(tablet_url)
        if len(tablet_urls) > 1:
            print("Use the URL with the same IP range as the tablet.")
        print("=" * 55)
        print("")

    def _tablet_events_log_path(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, TABLET_EVENTS_LOG_PATH))

    def _append_tablet_event(self, event: dict):
        if not isinstance(event, dict):
            return
        payload = dict(event)
        payload["server_wall_time"] = time.time()
        payload.setdefault("child_id", self._child_id)
        path = self._tablet_events_log_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning("Could not write tablet event log: %s", e)

    def _start_tablet_event_receiver(self):
        outer = self

        class TabletEventHandler(BaseHTTPRequestHandler):
            def _send_cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                self.send_response(204)
                self._send_cors()
                self.end_headers()

            def do_POST(self):
                if self.path != "/tablet-event":
                    self.send_response(404)
                    self._send_cors()
                    self.end_headers()
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", "0") or 0), 65536)
                    raw = self.rfile.read(length) if length else b"{}"
                    event = json.loads(raw.decode("utf-8"))
                    outer._append_tablet_event(event)
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}')
                except Exception:
                    self.send_response(400)
                    self._send_cors()
                    self.end_headers()

            def log_message(self, format, *args):
                return

        try:
            self._tablet_event_server = ThreadingHTTPServer(("0.0.0.0", TABLET_EVENT_PORT), TabletEventHandler)
            self.logger.debug("Tablet event receiver listening on port %s", TABLET_EVENT_PORT)
            self._tablet_event_server.serve_forever()
        except Exception as e:
            self.logger.warning("Tablet event receiver unavailable on port %s: %s", TABLET_EVENT_PORT, e)

    def _fetch_um(self) -> dict:
        try:
            url      = f"{UM_API_BASE}/api/um/{self._child_id}/inspect"
            response = http.get(url, timeout=API_TIMEOUT_S)

            if response.status_code == 404:
                if self._child_id != self._last_missing_child_warning_id:
                    self.logger.warning(
                        "Child '%s' not found. Has Eunike loaded the data?", self._child_id
                    )
                    self._last_missing_child_warning_id = self._child_id
                else:
                    self.logger.debug("Child '%s' still not found.", self._child_id)
                return {}

            if response.status_code != 200:
                self.logger.warning("UM API returned %s.", response.status_code)
                return {}

            self._last_missing_child_warning_id = None

            data          = response.json().get("data", {})
            categories    = data.get("categories", {})
            change_counts = data.get("change_counts", {})

            flat: dict = {}
            for cat_key, cat_data in categories.items():
                category_id = cat_key

                for field, meta in cat_data.get("scalars", {}).items():
                    flat[field] = {
                        "value":    meta.get("value") if isinstance(meta, dict) else meta,
                        "category": category_id,
                        "changes":  change_counts.get(field, 0),
                    }

                for field, entries in cat_data.get("nodes", {}).items():
                    if isinstance(entries, list):
                        if field == "pets":
                            pet_types = []
                            pet_names = []
                            for entry in entries:
                                if not isinstance(entry, dict):
                                    continue
                                pet_type = str(entry.get("value") or "").strip()
                                extra_props = entry.get("extra_props") or {}
                                pet_name = str(extra_props.get("petName") or "").strip()
                                if pet_type:
                                    pet_types.append(pet_type)
                                if pet_name:
                                    pet_names.append(pet_name)
                            if pet_types:
                                flat["pet_type"] = {
                                    "value":    ", ".join(pet_types),
                                    "category": category_id,
                                    "changes":  change_counts.get(field, 0),
                                }
                            if pet_names:
                                flat["pet_name"] = {
                                    "value":    ", ".join(pet_names),
                                    "category": category_id,
                                    "changes":  change_counts.get(field, 0),
                                }
                            continue

                        values = [e.get("value", "") for e in entries if e.get("value")]
                        flat[field] = {
                            "value":    ", ".join(values) if values else None,
                            "category": category_id,
                            "changes":  change_counts.get(field, 0),
                        }
                    else:
                        flat[field] = {
                            "value":    entries,
                            "category": category_id,
                            "changes":  change_counts.get(field, 0),
                        }

            # self.logger.debug(
            #     "Fetched %d fields across %d categories for child '%s'.",
            #     len(flat), len(categories), self._child_id
            # )
            return flat

        except http.exceptions.ConnectionError:
            self.logger.warning("Cannot connect to UM API — is main.py running?")
            return {}
        except Exception as e:
            self.logger.warning("UM API fetch failed: %s", e)
            return {}

    def _broadcast_um(self, um: dict):
        try:
            state = self._read_session_state()
            # child_name comes from session_state.json (written by tablet_state.py
            # using first_name_tablet from util/test_config.pl)
            child_name = state.get("child_name") or str(self._child_id)
            condition  = state.get("condition", "")
            self.webserver.send_message(
                WebInfoMessage("um_update", {
                    "session_id":          state.get("session_id"),
                    "child_id":            self._child_id,
                    "child_name":          child_name,
                    "condition":           condition,
                    "fields":              um,
                    "unlocked_categories": state.get("unlocked_categories", []),
                    "memory_access_active": bool(state.get("memory_access_active", False)),
                    "memory_access_prompt_id": state.get("memory_access_prompt_id"),
                    "visible_fields":       state.get("visible_fields", []),
                    "mistakes":             state.get("mistakes", {}),
                    "tablet_reveal":        state.get("tablet_reveal"),
                    "tablet_reveal_pending": state.get("tablet_reveal_pending"),
                    "tablet_command":       state.get("tablet_command"),
                    "current_phase":       state.get("phase"),
                    "current_turn_name":   state.get("current_turn_name"),
                    "timestamp":           time.strftime("%H:%M:%S"),
                })
            )
        except Exception as e:
            self.logger.warning("Broadcast failed: %s", e)

    # -----------------------------------------------------------------Main loop---------------------------------------------------------------

    def run(self):
        self.logger.debug(
            "Starting UM Tablet Server — polling every %.1fs …", POLL_INTERVAL_S
        )
        last_um = None
        try:
            while not self.shutdown_event.is_set():
                self._read_session_state()
                um = self._fetch_um()
                self._broadcast_um(um)

                if um != last_um:
                    self.logger.debug("UM changed — broadcasting to tablet.")
                    last_um = um

                time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            self.logger.debug("Interrupted.")
        finally:
            self.logger.debug("Shutting down UM Tablet Server.")
            self.shutdown()


if __name__ == "__main__":
    server = UMTabletServer()
    server.run()
