"""
UMClient — User Model I/O against Eunike's FastAPI (or fake-persona JSONs).

Constructed once in CRI_ScriptedDialogue.__init__:

    self.um = UMClient(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.get_field, self.pull_um, self.write_um_change, ...) stay identical.

What lives here:
  - Per-field GET: get_field
  - Bulk profile GET + parsing: pull_um, pull_um_bulk, field_value_from_profile
  - CRI scenario GET + parsing: pull_cri_scenario, scenario_utterance,
    scenario_mistake
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

    PROFILE_NODE_FIELD_ALIASES = {
        "hobbies": ("hobbies", "hasHobby", "LIKES_HOBBY"),
        "hobby_fav": ("hobby_fav", "hasFavouriteHobby"),
        "sports_fav_play": ("sports_fav_play", "playsSport", "PLAYS_SPORT"),
        "books_fav_title": ("books_fav_title", "hasFavouriteBook", "LIKES_BOOK"),
        "animal_fav": ("animal_fav", "hasFavouriteAnimal", "LIKES_ANIMAL"),
        "pets": ("pets", "hasPetInstance", "HAS_PET"),
        "pet_type": ("pets", "hasPetInstance", "HAS_PET"),
        "pet_name": ("pets", "hasPetInstance", "HAS_PET_NAME"),
        "freetime_fav": ("freetime_fav", "likesFreeTimeActivity"),
        "fav_food": ("fav_food", "hasFavouriteFood"),
        "fav_subject": ("fav_subject", "likesSubject", "LIKES_SUBJECT"),
        "school_strength": ("school_strength", "strongAtSubject", "STRONG_AT_SUBJECT"),
        "school_difficulty": ("school_difficulty", "findsDifficult", "FINDS_DIFFICULT"),
        "interest": ("interest", "hasInterest"),
        "aspiration": ("aspiration", "hasAspiration"),
        "role_model": ("role_model", "hasRoleModel"),
    }

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
        if field == "name":
            child_name = self.field_value_from_profile(profile, "child_name")
            if self.is_known(child_name):
                return child_name

        if field in ("pets", "pet_type", "pet_name"):
            return self.pet_value_from_profile(profile, field)

        scalar_entry = profile.get("scalars", {}).get(field)
        if isinstance(scalar_entry, dict):
            value = scalar_entry.get("value")
            return str(value) if value else self.d.UNKNOWN_VALUE

        node_entries = self.node_entries_from_profile(profile, field)
        if isinstance(node_entries, list) and node_entries:
            values = [
                str(entry.get("value"))
                for entry in node_entries
                if isinstance(entry, dict) and entry.get("value")
            ]
            if values:
                return self.d.format_dutch_list(values)

        return self.d.UNKNOWN_VALUE

    def node_entries_from_profile(self, profile: dict, field: str) -> list:
        """Return node entries for both field-name and ontology-relation profile keys."""
        nodes = profile.get("nodes", {})
        if not isinstance(nodes, dict):
            return []
        for key in self.PROFILE_NODE_FIELD_ALIASES.get(field, (field,)):
            entries = nodes.get(key)
            if isinstance(entries, list) and entries:
                return entries
        return []

    def pet_value_from_profile(self, profile: dict, field: str) -> str:
        """Expose GraphDB's consolidated pets nodes as Dialogue 2 read fields.

        The current ontology stores one `pets` node per pet, with the animal
        type as the primary value and `petName` in `extra_props`. Dialogue 2's
        script still speaks in terms of pet type and pet name, so this method
        derives those read-only aliases from the richer node shape.
        """
        node_entries = self.node_entries_from_profile(profile, field)
        if not isinstance(node_entries, list) or not node_entries:
            return self.d.UNKNOWN_VALUE

        values = []
        for entry in node_entries:
            if not isinstance(entry, dict):
                continue
            pet_type = str(entry.get("value") or "").strip()
            extra_props = entry.get("extra_props") or {}
            pet_name = str(extra_props.get("petName") or entry.get("value") or "").strip()

            if field == "pet_type" and pet_type:
                values.append(pet_type)
            elif field == "pet_name" and pet_name:
                values.append(pet_name)
            elif field == "pets":
                if pet_type and pet_name:
                    values.append(f"{pet_name} ({pet_type})")
                elif pet_name:
                    values.append(pet_name)
                elif pet_type:
                    values.append(pet_type)

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
        local_name = str(getattr(self.d, "local_child_name", "") or "").strip()
        if local_name and not self.is_known(um.get("name")):
            um["name"] = local_name
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

    # --- CRI scenario reads -------------------------------------------------

    def pull_cri_scenario(self) -> dict:
        """
        Fetch the child's curated CRI scenario from GraphDB.

        This is separate from the UM profile: the profile is what the child
        told us, while the scenario is experiment apparatus: planned mistakes
        and pregenerated Leo utterances.
        """
        if self.d.use_fake_persona_um():
            self.d.last_cri_scenario = {}
            self.d.last_cri_scenario_loaded = True
            return {}

        url = f"{self.d.UM_API_BASE}/api/um/{self.d.CHILD_ID}/scenario"
        try:
            response = requests.get(url, timeout=5)
            self.d.last_cri_scenario_loaded = True
            if response.status_code == 200:
                scenario = response.json().get("data", {}).get("scenario", {})
                self.d.last_cri_scenario = scenario if isinstance(scenario, dict) else {}
                utterances = self.d.last_cri_scenario.get("utterances") or {}
                mistakes = self.d.last_cri_scenario.get("mistakes") or []
                self.d.logger.info(
                    "CRI scenario pulled: %d mistakes, %d utterance steps.",
                    len(mistakes),
                    len(utterances),
                )
                return self.d.last_cri_scenario
            if response.status_code == 404:
                self.d.logger.info("No CRI scenario found for child '%s'.", self.d.CHILD_ID)
                self.d.last_cri_scenario = {}
                return {}
            self.d.logger.warning("CRI scenario API returned %d.", response.status_code)
        except requests.exceptions.ConnectionError:
            self.d.logger.warning("UM API not reachable while pulling CRI scenario at %s.", self.d.UM_API_BASE)
        except Exception as e:
            self.d.logger.warning("Could not pull CRI scenario: %s", e)

        self.d.last_cri_scenario = {}
        self.d.last_cri_scenario_loaded = True
        return {}

    def cri_scenario(self) -> dict:
        if getattr(self.d, "last_cri_scenario_loaded", False):
            return getattr(self.d, "last_cri_scenario", {}) or {}
        return self.pull_cri_scenario()

    def scenario_keys_for(self, key: str) -> list:
        keys = [str(key or "").strip()]
        aliases = getattr(self.d, "SCENARIO_UTTERANCE_ALIASES", {}).get(keys[0], ())
        keys.extend(str(alias).strip() for alias in aliases if alias)
        return [candidate for candidate in keys if candidate]

    def clean_scenario_text(self, text: str) -> str:
        """Remove authoring-only markers before text reaches Leo."""
        clean = str(text or "").strip()
        for marker in ("[STUB]", "[stub]"):
            if clean.startswith(marker):
                return clean[len(marker):].strip()
        return clean

    def scenario_utterance(self, key: str, branch: str = "default", fallback: str = "") -> str:
        """
        Read one pregenerated utterance from the CRI scenario.

        The GraphDB shape is:
            utterances[step_id][branch] = text
        """
        utterances = self.cri_scenario().get("utterances") or {}
        wanted_branch = str(branch or "default").strip() or "default"

        for scenario_key in self.scenario_keys_for(key):
            branches = utterances.get(scenario_key)
            if not isinstance(branches, dict) or not branches:
                continue
            if wanted_branch in branches:
                return self.clean_scenario_text(branches[wanted_branch])
            if "default" in branches:
                return self.clean_scenario_text(branches["default"])
            if len(branches) == 1:
                return self.clean_scenario_text(next(iter(branches.values())))

        return fallback

    def scenario_mistake(self, mistake_id: str) -> dict:
        """Return a mistake definition, normalized to Dialogue 2's old keys."""
        wanted = str(mistake_id or "").strip().upper()
        for mistake in self.cri_scenario().get("mistakes") or []:
            if not isinstance(mistake, dict):
                continue
            if str(mistake.get("id", "")).strip().upper() != wanted:
                continue
            normalized = dict(mistake)
            if "field" not in normalized and normalized.get("target_field"):
                normalized["field"] = normalized["target_field"]
            if "type" not in normalized and normalized.get("mistake_type"):
                normalized["type"] = normalized["mistake_type"]
            return normalized
        return {}

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
