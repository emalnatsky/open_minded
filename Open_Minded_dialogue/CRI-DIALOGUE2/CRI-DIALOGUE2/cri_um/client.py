"""
UMClient — User Model I/O against Eunike's FastAPI (or fake-persona JSONs).

Constructed once in CRI_ScriptedDialogue.__init__:

    self.um = UMClient(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.get_field, self.pull_um, self.write_um_change, ...) stay identical.

What lives here:
  - Per-field GET: get_field
  - Bulk profile GET + parsing: pull_um, pull_um_bulk, field_value_from_profile
  - Pregenerated utterance fields: is_pregenerated_utterance_field,
    pregenerated_fields_from_profile
  - Cached cache-first reads: memory_value (uses self.d.last_um_preview)
  - Quick predicates: is_known, known, first_known, yesish
  - Writes / deletes: write_um_change

This module READS class constants (UM_API_BASE, CHILD_ID, UM_FIELDS,
UNKNOWN_VALUE, PREGENERATED_UTTERANCE_PREFIXES, SIMULATION_WRITE_PERSONA_FILE),
dialogue state (last_um_preview, simulated_persona, simulated_persona_path),
methods (use_fake_persona_um, simulated_um_profile, format_dutch_list,
log_conversation_event), and self.d.logger.

WRITES are to self.d.last_um_preview (cache), self.d.simulated_persona
(fake mode), and via self.d.log_conversation_event for audit trail.

When the persona module moves out (stage 6) the persona-related calls
(use_fake_persona_um, simulated_um_profile) will stay on self.d as
pass-throughs — no code change needed here.
"""

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class UMClient:
    """All UM reads + writes against Eunike's API or fake-persona JSON."""

    def __init__(self, dialogue):
        self.d = dialogue

    # ── single-field read ────────────────────────────────────────────────────

    def get_field(self, field: str) -> str:
        """
        Pull a single UM field from Eunike's API.
        GET /api/um/{child_id}/field/{field_name}; no API key needed.

        Returns the value as a Dutch string.
        Returns 'dat weet ik nog niet' if field not set or API unreachable.
        """
        if not field:
            return self.d.UNKNOWN_VALUE
        if self.d.use_fake_persona_um():
            return self.d.simulated_um_profile().get(field, self.d.UNKNOWN_VALUE)

        url = f"{self.d.UM_API_BASE}/api/um/{self.d.CHILD_ID}/field/{field}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                value = resp.json().get("data", {}).get("value")
                if value:
                    self.d.logger.info("UM[%s] = %s", field, value)
                    return str(value)
                return self.d.UNKNOWN_VALUE
            elif resp.status_code == 404:
                self.d.logger.info("UM field '%s' not set for child '%s'.", field, self.d.CHILD_ID)
                return self.d.UNKNOWN_VALUE
            else:
                self.d.logger.warning("UM API returned %d for field '%s'.", resp.status_code, field)
                return self.d.UNKNOWN_VALUE
        except requests.exceptions.ConnectionError:
            self.d.logger.error("UM API not reachable at %s - is Eunike's main.py running?", self.d.UM_API_BASE)
            return self.d.UNKNOWN_VALUE
        except Exception as e:
            self.d.logger.error("UM error for field '%s': %s", field, e)
            return self.d.UNKNOWN_VALUE

    # ── bulk profile read & parsing ──────────────────────────────────────────

    def field_value_from_profile(self, profile: dict, field: str) -> str:
        """Extract one field from a bulk UM profile response."""
        scalar_entry = profile.get("scalars", {}).get(field)
        if isinstance(scalar_entry, dict):
            value = scalar_entry.get("value")
            return str(value) if value else self.d.UNKNOWN_VALUE

        node_entries = profile.get("nodes", {}).get(field)
        if isinstance(node_entries, list) and node_entries:
            values = [
                str(entry.get("value"))
                for entry in node_entries
                if isinstance(entry, dict) and entry.get("value")
            ]
            if values:
                return self.d.format_dutch_list(values)

        return self.d.UNKNOWN_VALUE

    def is_pregenerated_utterance_field(self, field: str) -> bool:
        """Fields with these prefixes are L2-pregen utterances stored in UM/GraphDB."""
        if str(field) == "script_plan":
            return False
        return any(str(field).startswith(prefix) for prefix in self.d.PREGENERATED_UTTERANCE_PREFIXES)

    def pregenerated_fields_from_profile(self, profile: dict) -> dict:
        """Pull L2-pregen utterance fields even when they are not core UM fields."""
        extra = {}
        for container_name in ("scalars", "nodes"):
            container = profile.get(container_name, {})
            if not isinstance(container, dict):
                continue
            for field in container:
                if self.is_pregenerated_utterance_field(field):
                    extra[field] = self.field_value_from_profile(profile, field)
        return extra

    def pull_um_bulk(self) -> dict:
        """Fetch all UM fields in one request using GET /api/um/{child_id}."""
        url = f"{self.d.UM_API_BASE}/api/um/{self.d.CHILD_ID}"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(f"UM profile returned {resp.status_code}")

        profile = resp.json().get("data", {}).get("profile", {})
        if not profile:
            raise RuntimeError("UM profile response did not contain data.profile")

        um = {
            field: self.field_value_from_profile(profile, field)
            for field in self.d.UM_FIELDS
        }
        um.update(self.pregenerated_fields_from_profile(profile))

        known_count = sum(1 for value in um.values() if self.is_known(value))
        self.d.logger.info("UM bulk profile pulled: %d/%d fields set.", known_count, len(self.d.UM_FIELDS))
        for field, value in um.items():
            if self.is_known(value):
                self.d.logger.info("UM[%s] = %s", field, value)
        return um

    def pull_um(self) -> dict:
        """Fetch all UM fields used by the 4.0 early interaction flow."""
        if self.d.use_fake_persona_um():
            um = self.d.simulated_um_profile()
            known_count = sum(1 for value in um.values() if self.is_known(value))
            self.d.logger.info("Fake persona UM profile loaded: %d/%d fields set.", known_count, len(self.d.UM_FIELDS))
            for field, value in um.items():
                if self.is_known(value):
                    self.d.logger.info("FAKE UM[%s] = %s", field, value)
            self.d.last_um_preview = um
            return um

        try:
            um = self.pull_um_bulk()
        except Exception as e:
            self.d.logger.warning("UM bulk profile pull failed (%s); falling back to per-field reads.", e)
            um = {field: self.get_field(field) for field in self.d.UM_FIELDS}

        self.d.last_um_preview = um
        return um

    # ── predicates ───────────────────────────────────────────────────────────

    def is_known(self, value: str) -> bool:
        """Return True when a UM value is present enough to safely mention."""
        if value is None:
            return False
        clean = str(value).strip()
        return bool(clean) and clean.lower() != self.d.UNKNOWN_VALUE

    def known(self, um: dict, field: str, fallback: str = "") -> str:
        value = um.get(field, self.d.UNKNOWN_VALUE)
        return value if self.is_known(value) else fallback

    def first_known(self, um: dict, fields: list, fallback: str = "") -> tuple:
        for field in fields:
            value = self.known(um, field)
            if value:
                return field, value
        return "", fallback

    def yesish(self, value: str) -> bool:
        return self.is_known(value) and str(value).strip().lower() in ("ja", "yes", "true")

    # ── cache-first read ─────────────────────────────────────────────────────

    def memory_value(self, field: str) -> str:
        """Read from the already-pulled UM snapshot first, then fall back to the API."""
        value = self.d.last_um_preview.get(field)
        if self.is_known(value):
            return str(value)
        return self.get_field(field)

    # ── writes / deletes ─────────────────────────────────────────────────────

    def write_um_change(self, change: dict) -> bool:
        field = change["field"]
        if self.d.use_fake_persona_um():
            try:
                if change["action"] == "delete":
                    self.d.simulated_persona[field] = self.d.UNKNOWN_VALUE
                    self.d.last_um_preview[field] = self.d.UNKNOWN_VALUE
                    new_value = None
                else:
                    new_value = change["new_value"]
                    self.d.simulated_persona[field] = new_value
                    self.d.last_um_preview[field] = new_value

                if self.d.SIMULATION_WRITE_PERSONA_FILE:
                    with open(self.d.simulated_persona_path, "w", encoding="utf-8") as persona_file:
                        json.dump(self.d.simulated_persona, persona_file, ensure_ascii=False, indent=2)

                self.d.log_conversation_event(
                    "um_write",
                    action=change.get("action"),
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=new_value,
                    success=True,
                    status_code="fake_persona",
                )
                return True
            except Exception as e:
                self.d.logger.error("Could not apply simulated UM change: %s", e)
                self.d.log_conversation_event(
                    "um_write",
                    action=change.get("action"),
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=change.get("new_value"),
                    success=False,
                    status_code="simulation",
                    error=str(e),
                )
                return False

        try:
            if change["action"] == "delete":
                url = f"{self.d.UM_API_BASE}/api/um/{self.d.CHILD_ID}/field/{field}"
                response = requests.delete(url, timeout=3)
                ok = response.status_code in (200, 202, 204, 404)
                if ok:
                    self.d.last_um_preview[field] = self.d.UNKNOWN_VALUE
                self.d.log_conversation_event(
                    "um_write",
                    action="delete",
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=None,
                    success=ok,
                    status_code=response.status_code,
                )
                return ok

            url = f"{self.d.UM_API_BASE}/api/um/{self.d.CHILD_ID}/fields"
            payload = {
                "fields": {field: change["new_value"]},
                "source": "cri_4_topic_confirmation",
            }
            response = requests.post(url, json=payload, timeout=3)
            ok = response.status_code in (200, 201, 202, 204)
            if ok:
                self.d.last_um_preview[field] = change["new_value"]
            self.d.log_conversation_event(
                "um_write",
                action="update",
                field=field,
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                success=ok,
                status_code=response.status_code,
            )
            return ok
        except Exception as e:
            self.d.logger.error("Could not write confirmed UM change: %s", e)
            self.d.log_conversation_event(
                "um_write",
                action=change.get("action"),
                field=field,
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                success=False,
                status_code=None,
                error=str(e),
            )
            return False
