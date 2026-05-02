"""
um_tablet_server.py
--------------------
SIC webserver that serves a live User Model dashboard to a tablet.
NO REDIS 
Architecture:
    Eunike's FastAPI + GraphDB (UM store)
        ↓  poll every 2s via REST API
    UMTabletServer (this file)
        ↓  WebInfoMessage broadcast via Socket.IO
    webfiles/index.html  (STUB tablet browser)

The tablet just opens http://<laptop-ip>:8080 on WiFi.

Prerequisites:
    1. Eunike's UM API running:  uvicorn main:app --port 8000
    2. This script:              python um_tablet_server.py

NOTE: There is no stub child data yet + Eunike API is not connected. I need info abt where server runs and from where/how to load child
"""

import os
import socket
import threading
import time
import webbrowser
import urllib.request

import requests as http   # plain HTTP calls to Eunike's FastAPI

from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication
from sic_framework.services.webserver.webserver_service import (
    WebInfoMessage,
    Webserver,
    WebserverConf,
)

# --------------------------------------------------------------------------- Config:  edit these to match your setup ---------------------------------------------------------------------------

CHILD_ID        = "test_child_001"       # need to match wiith what Eunike loaded from Qualtrics
UM_API_BASE     = "http://localhost:8000"  # Eunike's FastAPI base URL
POLL_INTERVAL_S = 2.0               # how often to fetch UM (seconds)
WEB_PORT        = 8080              # port the tablet browser connects to
API_TIMEOUT_S   = 3.0               # seconds before HTTP request gives up


class UMTabletServer(SICApplication):
    """
    Serves a live User Model dashboard to a tablet browser over WiFi.

    Polls Eunike's FastAPI every POLL_INTERVAL_S seconds.
    Pushes any changes to connected tablet browsers via Socket.IO.
    No page refresh needed on the tablet.
    """

    def __init__(self):
        super(UMTabletServer, self).__init__()

        self.webserver  = None
        self._child_id  = CHILD_ID

        self.set_log_level(sic_logging.INFO)
        self.load_env("../conf/.env")
        self.setup()

    # ------------------------------------------------------------------ Setup ------------------------------------------------------------------

    def setup(self):
        self.logger.info("Setting up UM Tablet Server …")

        # ---------------------------------------------------------Webserver --------------------------------------------------
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        webfiles_dir = os.path.join(current_dir, "webfiles")

        web_conf = WebserverConf(
            host="0.0.0.0",   # accept connections from any device on WiFi
            port=WEB_PORT,
            templates_dir=webfiles_dir,
            static_dir=webfiles_dir,
        )
        self.webserver = Webserver(conf=web_conf)

        # ---------------------------------------------------------Check Eunike's API is reachable ---------------------------
        threading.Thread(target=self._check_api, daemon=True).start()

        # ---------------------------------------------------------Open browser on laptop (testing for julianna) -------------------------
        url = f"http://localhost:{WEB_PORT}"
        threading.Thread(
            target=lambda: self._open_when_ready(url), daemon=True
        ).start()

        # --------------------------------------------------------- Log tablet URL --------------------------------------------
        threading.Thread(
            target=self._log_tablet_url, daemon=True
        ).start()

        self.logger.info("Setup complete.")

    # ------------------------------------------------------------------ Helpers ------------------------------------------------------------------

    def _check_api(self):
        """Warn if Eunike's API is not reachable at startup."""
        time.sleep(1.0)
        try:
            r = http.get(f"{UM_API_BASE}/", timeout=API_TIMEOUT_S)
            if r.status_code == 200:
                self.logger.info("Eunike's UM API reachable at %s ✓", UM_API_BASE)
            else:
                self.logger.warning(
                    "UM API returned status %s — check if uvicorn is running.", r.status_code
                )
        except Exception:
            self.logger.warning(
                "Cannot reach UM API at %s. "
                "Make sure Eunike's server is running: uvicorn main:app --port 8000",
                UM_API_BASE,
            )

    def _open_when_ready(self, url: str):
        """Poll /readyz then open browser on this machine."""
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
        """Print the WiFi URL the tablet should open."""
        try:
            hostname = socket.gethostname()
            lan_ip   = socket.gethostbyname(hostname)
        except Exception:
            lan_ip = "YOUR_LAPTOP_IP"

        time.sleep(1.5)
        self.logger.info("=" * 55)
        self.logger.info("  Open this on the tablet (same WiFi):")
        self.logger.info("  http://%s:%s", lan_ip, WEB_PORT)
        self.logger.info("=" * 55)

    def _fetch_um(self) -> dict:
        """
        Fetch the full UM for the current child from Eunike's API.

        Uses the /inspect endpoint which returns fields grouped by category
        for the tablet GUI needs. Eunike's main.py documents this
        endpoint as: 'Called by Julianna's GUI for the main memory overview.'

        Returns a flat dict of {field_name: value} for easy comparison
        and broadcast. Returns empty dict on any error.
        """
        try:
            url = f"{UM_API_BASE}/api/um/{self._child_id}/inspect"
            response = http.get(url, timeout=API_TIMEOUT_S)

            if response.status_code == 404:
                self.logger.warning(
                    "Child '%s' not found in UM API. "
                    "Has Eunike loaded the Qualtrics data yet?", self._child_id
                )
                return {}

            if response.status_code != 200:
                self.logger.warning(
                    "UM API returned status %s for child %s.",
                    response.status_code, self._child_id
                )
                return {}

            data = response.json().get("data", {})
            categories = data.get("categories", {})
            change_counts = data.get("change_counts", {})

            # Flatten scalars and nodes across all categories into one dict
            # so the tablet can render them and the polling loop can do
            # simple equality comparison with last_um.
            flat: dict = {}
            for cat_key, cat_data in categories.items():
                cat_label = cat_data.get("label", cat_key)

                for field, meta in cat_data.get("scalars", {}).items():
                    # Scalar values come as {"value": ..., "source": ..., ...}
                    flat[field] = {
                        "value":    meta.get("value") if isinstance(meta, dict) else meta,
                        "category": cat_label,
                        "changes":  change_counts.get(field, 0),
                    }

                for field, entries in cat_data.get("nodes", {}).items():
                    # Node fields are lists of {"value": ..., "source": ...}
                    if isinstance(entries, list):
                        values = [e.get("value", "") for e in entries if e.get("value")]
                        flat[field] = {
                            "value":    ", ".join(values) if values else None,
                            "category": cat_label,
                            "changes":  change_counts.get(field, 0),
                        }
                    else:
                        flat[field] = {
                            "value":    entries,
                            "category": cat_label,
                            "changes":  change_counts.get(field, 0),
                        }

            return flat

        except http.exceptions.ConnectionError:
            self.logger.warning(
                "Cannot connect to UM API — is uvicorn running on port 8000?"
            )
            return {}
        except Exception as e:
            self.logger.warning("UM API fetch failed: %s", e)
            return {}

    def _broadcast_um(self, um: dict):
        """Push UM data to all connected tablet browsers via Socket.IO."""
        try:
            self.webserver.send_message(
                WebInfoMessage("um_update", {
                    "child_id":  self._child_id,
                    "fields":    um,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            )
        except Exception as e:
            self.logger.warning("Broadcast failed: %s", e)

    # ------------------------------------------------------------------Main loop ------------------------------------------------------------------

    def run(self):
        self.logger.info(
            "Starting UM Tablet Server — polling Eunike's API every %.1fs …",
            POLL_INTERVAL_S
        )

        last_um = None

        try:
            while not self.shutdown_event.is_set():
                um = self._fetch_um()

                # only broadcast if something changed (or first run)
                if um != last_um:
                    self.logger.info("UM changed — broadcasting to tablet.")
                    self._broadcast_um(um)
                    last_um = um

                time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            self.logger.info("Interrupted.")
        finally:
            self.logger.info("Shutting down UM Tablet Server.")
            self.shutdown()


if __name__ == "__main__":
    server = UMTabletServer()
    server.run()
