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
import socket
import threading
import time
import webbrowser
import urllib.request
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
API_TIMEOUT_S   = 3.0
SESSION_STATE_PATH = "../_local/session_state.json"  # written by CRI-DIALOGUE/tablet_state.py
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
        self._child_id = CHILD_ID
        self._last_missing_child_warning_id = None
        self.set_log_level(sic_logging.WARNING)
        self.load_env("../conf/.env")
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
        threading.Thread(target=lambda: self._open_when_ready(f"http://localhost:{WEB_PORT}"), daemon=True).start()
        threading.Thread(target=self._log_tablet_url, daemon=True).start()

        self.logger.debug("Setup complete.")

    # ------------------------------------------------------------------helpers----------------------------------------------------------------

    def _read_session_state(self) -> dict:
        """
        Read session_state.json written by the dialogue's TabletStateWriter.
        Also updates self._child_id automatically from the file so the tablet
        server tracks whichever child the dialogue is talking to.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, SESSION_STATE_PATH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                session_child_id = str(state.get("child_id") or "").strip()
                if session_child_id and session_child_id != self._child_id:
                    self.logger.debug(
                        "Child ID updated from session_state: %s → %s",
                        self._child_id, session_child_id,
                    )
                    self._child_id = session_child_id
                return state
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning("Could not read session_state.json: %s", e)
        return {}

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

    def _log_tablet_url(self):
        try:
            hostname = socket.gethostname()
            lan_ip   = socket.gethostbyname(hostname)
        except Exception:
            lan_ip = "YOUR_LAPTOP_IP"
        time.sleep(1.5)
        self.logger.debug("=" * 55)
        self.logger.debug("  Open this on the tablet (same WiFi):")
        self.logger.debug("  http://%s:%s", lan_ip, WEB_PORT)
        self.logger.debug("=" * 55)

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

            self.logger.debug(
                "Fetched %d fields across %d categories for child '%s'.",
                len(flat), len(categories), self._child_id
            )
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
