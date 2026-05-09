import sys
import os
import time
import json
import random
import requests
from datetime import datetime

from dotenv import load_dotenv
from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoWakeUpRequest,
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)
from sic_framework.services.openai_whisper_stt.whisper_stt import (
    GetTranscript,
    SICWhisper,
    WhisperConf,
)
from sic_framework.services.llm import GPT, GPTConf, GPTRequest
from openai import OpenAI

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_INTENT = os.path.join(_HERE, "CRI-INTENT")
sys.path.append(_INTENT)

from stub_intent_classifier import StubIntentClassifier
from gpt_intent_classifier import GPTIntentClassifier, REPEAT_SENTINEL


class CRI_ScriptedDialogue(SICApplication):
    """
    CRI 4.0 early interaction flow.

    The script pulls known child-memory fields from the UM API before starting,
    prints a preview, then runs a five-step dialogue with correct memories,
    one deliberate mistake, and a friendly goodbye.

    UM connection:
        GET http://localhost:8000/api/um/{child_id}/field/{field_name}
        No API key needed for reads.
        If a field is not set, robot says "dat weet ik nog niet".
    """

    # UM connection
    UM_API_BASE = "http://localhost:8000"
    CHILD_ID = "Julianna"
    UNKNOWN_VALUE = "dat weet ik nog niet"
    UM_FIELDS = (
        "child_name", "name", "age", "hobbies", "hobby_fav",
        "sports_enjoys", "sports_fav", "sports_plays", "sports_fav_play",
        "books_enjoys", "books_fav_genre", "books_fav_title",
        "music_enjoys", "music_talk", "music_plays_instrument", "music_instrument",
        "animals_enjoys", "animal_talk", "animal_fav",
        "has_pet", "pet_type", "pet_name",
        "freetime_fav", "fav_food", "aspiration",
    )

    # Whisper
    STT_TIMEOUT = 20
    STT_PHRASE_LIMIT = 18

    # LLM
    LLM_FALLBACK = "Wauw, dat klinkt heel leuk!"
    LLM_SYSTEM = (
        "Jij bent een vriendelijke robot genaamd Leo en je praat tegen een Nederlands kind van 8 tot 11 jaar oud. "
        "Geef antwoord in een korte zin (maximaal 25 woorden). "
        "Wees warm, enthousiast en geschikt voor de leeftijden tussen 8 en 11. "
        "Vraag geen vragen. Praat in het Nederlands."
    )

    TOPIC_DOMAIN_ORDER = ("pet", "sports", "books", "music", "animals", "hobby", "freetime")
    BOOLEANISH_FIELDS = (
        "has_pet", "sports_enjoys", "sports_plays", "books_enjoys",
        "music_enjoys", "music_talk", "music_plays_instrument", "animals_enjoys",
    )
    OPPOSITE_VALUE_FALLBACKS = {
        "huisdier": ("een robotdinosaurus", "een steen", "Rover"),
        "sport": ("schaken op de bank", "stilzitten", "ballet"),
        "boeken": ("geen boeken lezen", "een kookboek", "een boek over vrachtwagens"),
        "muziek": ("helemaal geen muziek", "drummen", "opera zingen"),
        "hobby": ("stilzitten", "postzegels sorteren", "breien"),
        "eten": ("spruitjes", "broccoli", "zoute drop"),
        "droom": ("bankdirecteur worden", "op kantoor wonen", "robots repareren"),
    }
    TOPIC_CHANGE_MODEL = "gpt-4o-mini"
    TOPIC_RESPONSE_TYPES = (
        "no_change", "story", "possible_update",
        "possible_delete", "correction_unclear",
        "wants_other_topic", "question", "unclear",
    )

    # Desktop mic flag
    USE_DESKTOP_MIC = False
    WAIT_FOR_PREVIEW_CONFIRMATION = True
    REVIEW_TRANSCRIPTS = True
    POST_STEP_TEST_CONTROLS = True
    SCRIPT_VERSION = "CRI-BRANCH-BASIC4.0"

    # Conversation logging
    CONVERSATION_LOG_ENABLED = True
    CONVERSATION_LOG_ROOT = os.path.abspath(os.path.join(_HERE, "conversations"))

    def __init__(self, openai_env_path=None, nao_ip="10.0.0.165"):
        super(CRI_ScriptedDialogue, self).__init__()
        self.nao_ip = nao_ip
        self.openai_env_path = openai_env_path
        self.nao = None
        self.whisper = None
        self.gpt = None
        self.clf = None
        self.desktop = None
        self.openai_client = None
        self.mistakes_mentioned = 0
        self.corrections_seen = 0
        self.last_um_preview = {}
        self.pending_change = None
        self.conversation_log = None
        self.current_turn_log = None
        self.set_log_level(sic_logging.INFO)
        self.setup()

    # Setup

    def setup(self):
        self.logger.info("Setting up CRI pipeline...")

        if self.openai_env_path:
            load_dotenv(self.openai_env_path)

        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Intent classifier: GPT with stub fallback
        try:
            self.clf = GPTIntentClassifier(
                openai_key=os.environ["OPENAI_API_KEY"],
                schema_path=os.path.join(_INTENT, "um_field_schema.json"),
                contract_path=os.path.join(_INTENT, "intent_classification_contract.json"),
            )
            self.logger.info("GPTIntentClassifier ready.")
        except Exception as e:
            self.logger.warning("GPTIntentClassifier failed (%s) - using stub.", e)
            self.clf = StubIntentClassifier(
                schema_path=os.path.join(_INTENT, "um_field_schema.json")
            )

        self.logger.info("UM: LIVE - %s, child=%s", self.UM_API_BASE, self.CHILD_ID)

        # NAO
        if not self.USE_DESKTOP_MIC:
            self.logger.info("Connecting to NAO at %s...", self.nao_ip)
            self.nao = Nao(ip=self.nao_ip)
            self.logger.info("NAO connected.")

        # Whisper
        if self.USE_DESKTOP_MIC:
            from sic_framework.devices.desktop import Desktop
            self.desktop = Desktop()
            self.whisper = SICWhisper(
                input_source=self.desktop.mic,
                conf=WhisperConf(openai_key=os.environ["OPENAI_API_KEY"])
            )
        else:
            self.whisper = SICWhisper(
                input_source=self.nao.mic,
                conf=WhisperConf(openai_key=os.environ["OPENAI_API_KEY"])
            )
        time.sleep(1.0)

        # GPT for L3 responses
        self.gpt = GPT(conf=GPTConf(
            openai_key=os.environ["OPENAI_API_KEY"],
            system_message=self.LLM_SYSTEM,
            model="gpt-4o-mini",
            max_tokens=140,
            temp=0.7,
        ))
        self.logger.info("Setup complete.")

    # UM pulling

    def get_field(self, field: str) -> str:
        """
        Pull a single UM field from Eunike's API.
        GET /api/um/{child_id}/field/{field_name}; no API key needed.

        Returns the value as a Dutch string.
        Returns 'dat weet ik nog niet' if field not set or API unreachable.
        """
        if not field:
            return self.UNKNOWN_VALUE

        url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/field/{field}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                value = resp.json().get("data", {}).get("value")
                if value:
                    self.logger.info("UM[%s] = %s", field, value)
                    return str(value)
                return self.UNKNOWN_VALUE
            elif resp.status_code == 404:
                self.logger.info("UM field '%s' not set for child '%s'.", field, self.CHILD_ID)
                return self.UNKNOWN_VALUE
            else:
                self.logger.warning("UM API returned %d for field '%s'.", resp.status_code, field)
                return self.UNKNOWN_VALUE
        except requests.exceptions.ConnectionError:
            self.logger.error("UM API not reachable at %s - is Eunike's main.py running?", self.UM_API_BASE)
            return self.UNKNOWN_VALUE
        except Exception as e:
            self.logger.error("UM error for field '%s': %s", field, e)
            return self.UNKNOWN_VALUE

    def field_value_from_profile(self, profile: dict, field: str) -> str:
        """Extract one field from a bulk UM profile response."""
        scalar_entry = profile.get("scalars", {}).get(field)
        if isinstance(scalar_entry, dict):
            value = scalar_entry.get("value")
            return str(value) if value else self.UNKNOWN_VALUE

        node_entries = profile.get("nodes", {}).get(field)
        if isinstance(node_entries, list) and node_entries:
            values = [
                str(entry.get("value"))
                for entry in node_entries
                if isinstance(entry, dict) and entry.get("value")
            ]
            if values:
                return self.format_dutch_list(values)

        return self.UNKNOWN_VALUE

    def pull_um_bulk(self) -> dict:
        """Fetch all UM fields in one request using GET /api/um/{child_id}."""
        url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(f"UM profile returned {resp.status_code}")

        profile = resp.json().get("data", {}).get("profile", {})
        if not profile:
            raise RuntimeError("UM profile response did not contain data.profile")

        um = {
            field: self.field_value_from_profile(profile, field)
            for field in self.UM_FIELDS
        }

        known_count = sum(1 for value in um.values() if self.is_known(value))
        self.logger.info("UM bulk profile pulled: %d/%d fields set.", known_count, len(self.UM_FIELDS))
        for field, value in um.items():
            if self.is_known(value):
                self.logger.info("UM[%s] = %s", field, value)
        return um

    def is_known(self, value: str) -> bool:
        """Return True when a UM value is present enough to safely mention."""
        if value is None:
            return False
        clean = str(value).strip()
        return bool(clean) and clean.lower() != self.UNKNOWN_VALUE

    def pull_um(self) -> dict:
        """Fetch all UM fields used by the 4.0 early interaction flow."""
        try:
            um = self.pull_um_bulk()
        except Exception as e:
            self.logger.warning("UM bulk profile pull failed (%s); falling back to per-field reads.", e)
            um = {field: self.get_field(field) for field in self.UM_FIELDS}

        self.last_um_preview = um
        return um

    def known(self, um: dict, field: str, fallback: str = "") -> str:
        value = um.get(field, self.UNKNOWN_VALUE)
        return value if self.is_known(value) else fallback

    def first_known(self, um: dict, fields: list, fallback: str = "") -> tuple:
        for field in fields:
            value = self.known(um, field)
            if value:
                return field, value
        return "", fallback

    def yesish(self, value: str) -> bool:
        return self.is_known(value) and str(value).strip().lower() in ("ja", "yes", "true")

    def pick_wrong_value(self, actual: str, candidates: list) -> str:
        actual_clean = str(actual or "").strip().lower()
        for candidate in candidates:
            if candidate.lower() != actual_clean:
                return candidate
        return candidates[0]

    def split_memory_values(self, value: str) -> list:
        """Split simple comma/and-separated UM strings into speakable values."""
        if not self.is_known(value):
            return []

        cleaned = str(value).replace(";", ",")
        parts = []
        for chunk in cleaned.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if " en " in chunk:
                parts.extend(part.strip() for part in chunk.split(" en ") if part.strip())
            else:
                parts.append(chunk)
        return parts

    def unique_values(self, values: list, limit: int = None) -> list:
        seen = set()
        unique = []
        for value in values:
            clean = str(value).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                unique.append(clean)
            if limit and len(unique) >= limit:
                break
        return unique

    def format_dutch_list(self, values: list, fallback: str = "") -> str:
        values = self.unique_values(values)
        if not values:
            return fallback
        if len(values) == 1:
            return values[0]
        return ", ".join(values[:-1]) + " en " + values[-1]

    def broad_clusters(self, um: dict) -> list:
        clusters = []
        if self.yesish(um.get("sports_enjoys")) or self.known(um, "sports_fav") or self.known(um, "sports_fav_play"):
            clusters.append("sport")
        if self.yesish(um.get("books_enjoys")) or self.known(um, "books_fav_title") or self.known(um, "books_fav_genre"):
            clusters.append("boeken")
        if self.yesish(um.get("music_enjoys")) or self.known(um, "music_instrument"):
            clusters.append("muziek")
        if self.yesish(um.get("animals_enjoys")) or self.known(um, "animal_fav") or self.known(um, "pet_name"):
            clusters.append("dieren")
        if self.known(um, "hobby_fav") or self.known(um, "hobbies"):
            clusters.append("hobby's")
        return self.unique_values(clusters, limit=3)

    def known_hobbies(self, um: dict) -> list:
        values = []
        values.extend(self.split_memory_values(um.get("hobby_fav")))
        values.extend(self.split_memory_values(um.get("hobbies")))
        values.extend(self.split_memory_values(um.get("freetime_fav")))
        return self.unique_values(values, limit=2)

    def opening_summary(self, um: dict) -> str:
        """Step 2: correct opening summary with no child response yet."""
        age = self.known(um, "age")
        hobbies = self.format_dutch_list(self.known_hobbies(um), "dingen die jij leuk vindt")
        clusters = self.format_dutch_list(self.broad_clusters(um), "wat jij leuk vindt")

        if age:
            opening = f"Je bent {age} jaar, je houdt van {hobbies}, en je hebt eerder verteld over {clusters}."
        else:
            opening = f"Je houdt van {hobbies}, en je hebt eerder verteld over {clusters}."

        specific_memory = self.specific_correct_memory(um)

        return (
            f"Ik weet nog een paar dingen over jou. {opening} "
            f"Ik weet ook nog dat {specific_memory}. "
            "Vandaag kunnen we praten over iets wat jij kiest, of ik kan beginnen met iets waarvan ik denk dat je het leuk vindt."
        )

    def specific_correct_memory(self, um: dict) -> str:
        pet = self.known(um, "pet_name")
        animal = self.known(um, "animal_fav")
        food = self.known(um, "fav_food")
        aspiration = self.known(um, "aspiration")
        hobby = self.known(um, "hobby_fav") or self.known(um, "hobbies")

        if pet:
            return f"{pet} belangrijk voor je is"
        if animal:
            return f"je {animal} leuk vindt"
        if food:
            return f"je lievelingseten {food} is"
        if aspiration:
            return f"je later {aspiration} wilt"
        if hobby:
            return f"je graag iets doet met {hobby}"
        return "ik nog niet alles zeker weet, maar wel goed wil luisteren"

    def topic_candidate(
        self,
        domain: str,
        label: str,
        fields: list,
        field_labels: dict,
        current_values: dict,
        correct_values: list,
        memory_link: str,
        options: list,
        reground: str,
    ) -> dict:
        return {
            "domain": domain,
            "label": label,
            "fields": fields,
            "field_labels": field_labels,
            "current_values": current_values,
            "correct_values": self.unique_values(correct_values, limit=2),
            "memory_link": memory_link,
            "options": self.unique_values(options, limit=2),
            "reground": reground,
        }

    def topic_candidates(self, um: dict) -> list:
        """Build all usable Step 3 topic candidates, then Step 3 picks one at random."""
        candidates = []

        pet = self.known(um, "pet_name")
        pet_type = self.known(um, "pet_type")
        animal = self.known(um, "animal_fav")
        if pet or pet_type or animal:
            subject = pet or animal or f"je {pet_type}"
            current = {
                field: self.known(um, field)
                for field in ("pet_name", "pet_type", "animal_fav", "has_pet")
                if self.known(um, field)
            }
            correct_values = [f"{subject} bij jou hoort"]
            if animal:
                correct_values.append(f"je {animal} leuk vindt")
            candidates.append(self.topic_candidate(
                domain="huisdier",
                label=subject,
                fields=["pet_name", "pet_type", "animal_fav", "has_pet"],
                field_labels={
                    "pet_name": "de naam van je huisdier",
                    "pet_type": "het soort huisdier",
                    "animal_fav": "je lievelingsdier",
                    "has_pet": "of je een huisdier hebt",
                },
                current_values=current,
                correct_values=correct_values,
                memory_link=f"{subject} belangrijk voor je is",
                options=[subject, animal or pet_type or "dieren"],
                reground=f"Wat ik zeker wil onthouden, is dat {subject} belangrijk voor je is.",
            ))

        sport = self.known(um, "sports_fav_play") or self.known(um, "sports_fav")
        if sport or self.yesish(um.get("sports_enjoys")):
            label = sport or "sport"
            current = {
                field: self.known(um, field)
                for field in ("sports_enjoys", "sports_fav", "sports_plays", "sports_fav_play")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="sport",
                label=label,
                fields=["sports_enjoys", "sports_fav", "sports_plays", "sports_fav_play"],
                field_labels={
                    "sports_enjoys": "of je sport leuk vindt",
                    "sports_fav": "je lievelingssport",
                    "sports_plays": "of je een sport doet",
                    "sports_fav_play": "de sport die je graag doet",
                },
                current_values=current,
                correct_values=[f"je iets met {label} hebt", "sport eerder in jouw geheugen stond"],
                memory_link=f"{label} iets is waar jij iets mee hebt",
                options=[label, "sport"],
                reground=f"Ik houd goed vast dat {label} iets is waar jij iets mee hebt.",
            ))

        book = self.known(um, "books_fav_title") or self.known(um, "books_fav_genre")
        if book or self.yesish(um.get("books_enjoys")):
            label = book or "boeken"
            current = {
                field: self.known(um, field)
                for field in ("books_enjoys", "books_fav_genre", "books_fav_title")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="boeken",
                label=label,
                fields=["books_enjoys", "books_fav_genre", "books_fav_title"],
                field_labels={
                    "books_enjoys": "of je boeken leuk vindt",
                    "books_fav_genre": "je favoriete soort boeken",
                    "books_fav_title": "je favoriete boek",
                },
                current_values=current,
                correct_values=[f"{label} bij jouw boekenwereld hoort", "je eerder iets over boeken vertelde"],
                memory_link=f"{label} bij jouw boekenwereld hoort",
                options=[label, "boeken"],
                reground=f"Ik weet in elk geval dat {label} bij jouw boekenwereld hoort.",
            ))

        music = self.known(um, "music_instrument")
        if music or self.yesish(um.get("music_enjoys")):
            label = music or "muziek"
            current = {
                field: self.known(um, field)
                for field in ("music_enjoys", "music_talk", "music_plays_instrument", "music_instrument")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="muziek",
                label=label,
                fields=["music_enjoys", "music_talk", "music_plays_instrument", "music_instrument"],
                field_labels={
                    "music_enjoys": "of je muziek leuk vindt",
                    "music_talk": "of je over muziek wilt praten",
                    "music_plays_instrument": "of je een instrument speelt",
                    "music_instrument": "welk instrument je speelt",
                },
                current_values=current,
                correct_values=[f"{label} bij jou en muziek hoort", "je eerder iets over muziek vertelde"],
                memory_link=f"{label} iets met jou en muziek te maken heeft",
                options=[label, "muziek"],
                reground=f"Ik onthoud goed dat {label} iets met jou en muziek te maken heeft.",
            ))

        hobby = self.known(um, "hobby_fav") or self.known(um, "hobbies")
        if hobby:
            current = {
                field: self.known(um, field)
                for field in ("hobby_fav", "hobbies", "freetime_fav")
                if self.known(um, field)
            }
            candidates.append(self.topic_candidate(
                domain="hobby",
                label=hobby,
                fields=["hobby_fav", "hobbies", "freetime_fav"],
                field_labels={
                    "hobby_fav": "je favoriete hobby",
                    "hobbies": "je hobby's",
                    "freetime_fav": "wat je graag in je vrije tijd doet",
                },
                current_values=current,
                correct_values=[f"je graag iets doet met {hobby}", f"{hobby} bij jouw interesses hoort"],
                memory_link=f"{hobby} bij jouw interesses hoort",
                options=[hobby, "je hobby's"],
                reground=f"Ik weet zeker dat {hobby} bij jouw interesses hoort.",
            ))

        food = self.known(um, "fav_food")
        if food:
            food_display = self.format_dutch_list(self.split_memory_values(food), food)
            candidates.append(self.topic_candidate(
                domain="eten",
                label="je lievelingseten",
                fields=["fav_food"],
                field_labels={"fav_food": "je lievelingseten"},
                current_values={"fav_food": food},
                correct_values=[f"je lievelingseten {food_display} is"],
                memory_link=f"je lievelingseten {food_display} is",
                options=[food_display, "iets anders dat je lekker vindt"],
                reground=f"Ik weet zeker dat {food_display} met jouw lievelingseten te maken heeft.",
            ))

        aspiration = self.known(um, "aspiration")
        if aspiration:
            candidates.append(self.topic_candidate(
                domain="droom",
                label=aspiration,
                fields=["aspiration"],
                field_labels={"aspiration": "wat je later wilt doen of worden"},
                current_values={"aspiration": aspiration},
                correct_values=[f"je later {aspiration} wilt"],
                memory_link=f"je later {aspiration} wilt",
                options=[aspiration, "je dromen"],
                reground=f"Ik onthoud dat {aspiration} iets is waar je later iets mee wilt.",
            ))

        return candidates

    def select_topic_domain(self, um: dict) -> dict:
        candidates = self.topic_candidates(um)
        if candidates:
            topic = random.choice(candidates)
            self.logger.info(
                "Random Step 3 topic picked: %s (%s).",
                topic["label"],
                topic["domain"],
            )
            return topic

        return self.topic_candidate(
            domain="kennismaken",
            label="iets wat jij leuk vindt",
            fields=[],
            field_labels={},
            current_values={},
            correct_values=["ik nog niet alles zeker weet"],
            memory_link="ik graag wil leren wat jij belangrijk vindt",
            options=["je hobby's", "iets nieuws"],
            reground="Ik wil vooral goed onthouden wat jij belangrijk vindt.",
        )

    def topic_key(self, topic: dict) -> tuple:
        return (topic.get("domain"), topic.get("label"))

    def preferred_memory_item(self, topic: dict) -> tuple:
        """
        Pick one concrete remembered value from a topic.

        Content fields are better for deliberate mistakes than yes/no fields,
        because Leo can make a clearer wrong statement about them.
        """
        current_values = topic.get("current_values", {}) or {}
        fields = topic.get("fields", []) or list(current_values.keys())
        ordered_fields = [
            field for field in fields
            if field in current_values and field not in self.BOOLEANISH_FIELDS
        ]
        ordered_fields.extend(
            field for field in fields
            if field in current_values and field not in ordered_fields
        )

        for field in ordered_fields:
            value = current_values.get(field)
            if self.is_known(value):
                return field, value
        return "", ""

    def select_deliberate_mistake_topic(self, um: dict, discussed_topic: dict) -> dict:
        """Pick a random UM topic for Step 4 that is not the Step 3 topic."""
        discussed_key = self.topic_key(discussed_topic)
        candidates = []
        for candidate in self.topic_candidates(um):
            if self.topic_key(candidate) == discussed_key:
                continue
            field, actual = self.preferred_memory_item(candidate)
            if field and self.is_known(actual):
                candidates.append(candidate)

        if candidates:
            topic = random.choice(candidates)
            self.logger.info(
                "Random Step 4 mistake topic picked: %s (%s).",
                topic["label"],
                topic["domain"],
            )
            return topic

        self.logger.warning(
            "No alternate UM topic available for Step 4; using a generic fallback mistake topic."
        )
        return self.topic_candidate(
            domain="fallback",
            label="iets anders",
            fields=["hobby_fav"],
            field_labels={"hobby_fav": "je favoriete hobby"},
            current_values={"hobby_fav": "iets wat je leuk vindt"},
            correct_values=["je ergens enthousiast over bent"],
            memory_link="ik nog beter wil leren wat jij leuk vindt",
            options=["iets leuks", "iets anders"],
            reground="Ik wil vooral goed onthouden wat jij belangrijk vindt.",
        )

    def fallback_opposite_value(self, topic: dict, actual: str) -> str:
        candidates = self.OPPOSITE_VALUE_FALLBACKS.get(
            topic.get("domain"),
            ("iets helemaal anders", "iets dat niet klopt"),
        )
        actual_clean = str(actual or "").strip().lower()
        for candidate in candidates:
            if candidate.lower() != actual_clean:
                return candidate
        return candidates[0]

    def opposite_memory_value(self, topic: dict, field: str, actual: str) -> str:
        """Ask the LLM for a child-friendly wrong value for the deliberate mistake."""
        fallback = self.fallback_opposite_value(topic, actual)
        if not self.openai_client or not self.is_known(actual):
            return fallback

        prompt = {
            "task": (
                "Choose one deliberately wrong or opposite memory value for Leo to say. "
                "It must be child-friendly, short, and clearly different from the actual value."
            ),
            "topic": {
                "domain": topic.get("domain"),
                "label": topic.get("label"),
                "field": field,
                "field_label": topic.get("field_labels", {}).get(field, field),
                "actual_value": actual,
            },
            "rules": [
                "Return Dutch.",
                "Return only a short noun phrase or short activity phrase.",
                "Do not insult the child.",
                "Do not reuse the actual value.",
                "Make it clearly wrong or opposite enough that the child can correct Leo.",
            ],
            "output_schema": {
                "wrong_value": "short Dutch phrase",
                "reason": "short reason",
            },
        }
        messages = [
            {
                "role": "system",
                "content": "Return ONLY valid JSON for a child-facing Dutch robot dialogue.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=120,
                temperature=0.8,
            )
            parsed = self.extract_json_object(response.choices[0].message.content)
            wrong_value = str(parsed.get("wrong_value") or "").strip()
            if self.is_known(wrong_value) and wrong_value.lower() != str(actual).strip().lower():
                self.logger.info(
                    "LLM picked deliberate wrong value for %s: actual=%s wrong=%s",
                    field,
                    actual,
                    wrong_value,
                )
                return wrong_value
        except Exception as e:
            self.logger.error("Could not generate opposite memory value: %s", e)

        return fallback

    def deliberate_mistake_text(self, topic: dict, field: str, wrong_value: str) -> str:
        field_label = topic.get("field_labels", {}).get(field, "iets over jou")
        return (
            f"Ik dacht dat ik ook had onthouden dat {wrong_value} iets te maken heeft met {field_label}. "
            "Kan je mij hier wat meer over vertellen?"
        )

    def topic_examples_sentence(self, topic: dict) -> str:
        values = self.unique_values(topic.get("correct_values", []), limit=2)
        if len(values) >= 2:
            return f"Bijvoorbeeld: je zei dat {values[0]} en dat {values[1]}."
        if len(values) == 1:
            return f"Bijvoorbeeld: je zei dat {values[0]}."
        return "Bijvoorbeeld: er staat al iets over dat onderwerp in mijn geheugen."

    def first_topic_question(self, topic: dict) -> str:
        return (
            f"Ik denk dat {topic['label']} een goed onderwerp is om mee te beginnen, "
            "want daar heb je eerder best veel over verteld. "
            f"{self.topic_examples_sentence(topic)} "
            "Dat vond ik leuk om te onthouden, omdat het iets zegt over wat jij belangrijk of grappig vindt. "
            "Ik ben benieuwd: is daar sinds de vorige keer nog iets nieuws over gebeurd?"
        )

    def easier_topic_question(self, topic: dict) -> str:
        options = self.unique_values(topic.get("options", []), limit=2)
        while len(options) < 2:
            options.append("iets anders")
        return (
            "Dat is ook goed. Dan maak ik het makkelijker: "
            f"wil je liever vertellen over {options[0]}, {options[1]} of iets heel anders?"
        )

    def is_low_content_response(self, transcript: str) -> bool:
        clean = str(transcript or "").strip().lower()
        if not clean:
            return True
        low_content = {
            "ja", "nee", "ok", "oke", "oké", "inderdaad",
            "geen idee", "weet ik niet", "dat weet ik niet",
        }
        return clean in low_content or len(clean.split()) <= 2

    def topic_correction_question(self, topic: dict) -> str:
        current_values = topic.get("current_values", {}) or {}
        value_parts = []
        for field in topic.get("fields", []):
            value = current_values.get(field)
            if self.is_known(value) and field not in self.BOOLEANISH_FIELDS:
                label = topic.get("field_labels", {}).get(field, field)
                value_parts.append(f"{label} is {value}")

        if value_parts:
            remembered = self.format_dutch_list(value_parts[:2])
            return (
                f"Oké, dan wil ik het goed maken. Wat klopt er niet precies: {remembered}? "
                "Moet ik iets veranderen of vergeten?"
            )

        return (
            "Oké, dan wil ik het goed maken. Wat klopt er niet precies? "
            "Moet ik iets veranderen of vergeten?"
        )

    def mistake_correction_question(self, turn: dict) -> str:
        topic = turn.get("mistake_topic", {})
        field = turn.get("mistake_field")
        field_label = topic.get("field_labels", {}).get(field, "dit onderwerp")
        field_questions = {
            "pet_name": "Hoe heet je huisdier?",
            "pet_type": "Wat voor huisdier heb je?",
            "animal_fav": "Wat is je lievelingsdier?",
            "fav_food": "Wat is je lievelingseten?",
            "hobby_fav": "Wat is je favoriete hobby?",
            "hobbies": "Welke hobby's wil je dat ik onthoud?",
            "freetime_fav": "Wat doe je graag in je vrije tijd?",
            "sports_fav": "Wat is je lievelingssport?",
            "sports_fav_play": "Welke sport doe je graag?",
            "books_fav_title": "Wat is je favoriete boek?",
            "books_fav_genre": "Welke soort boeken vind je leuk?",
            "music_instrument": "Welk instrument speel je?",
            "aspiration": "Wat wil je later doen of worden?",
        }
        question = field_questions.get(
            field,
            f"Wat moet ik over {field_label} onthouden?",
        )
        return (
            "Oeps, dan had ik dat verkeerd. "
            f"{question}"
        )

    def mistake_acceptance_change(self, turn: dict, confirmation_question: str = "") -> dict:
        """Build a focused confirmation change when the child accepts Leo's deliberate mistake."""
        topic = turn.get("mistake_topic", {})
        field = turn.get("mistake_field")
        field_label = topic.get("field_labels", {}).get(field, field or "dit onderwerp")
        old_value = turn.get("mistake_actual") or self.UNKNOWN_VALUE
        new_value = turn.get("mistake_wrong")
        fallback_question = f"Zal ik onthouden dat {new_value} bij jou past?"
        if self.is_known(confirmation_question) and str(new_value).lower() not in str(confirmation_question).lower():
            confirmation_question = f"{confirmation_question.rstrip(' ?')}: {new_value}?"

        return {
            "action": "update",
            "field": field,
            "field_label": field_label,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "confidence": 1.0,
            "reason": "Child appeared to accept Leo's deliberate wrong value; asking focused confirmation.",
            "source_text": "",
            "confirmation_question": confirmation_question or fallback_question,
        }

    def topic_no_update_response(self, topic: dict) -> str:
        return (
            f"Oké, dan is er niets nieuws over {topic['label']}. "
            "Dan onthoud ik wat ik al wist."
        )

    def fallback_topic_response(self, response_type: str, topic: dict, transcript: str = "") -> str:
        if response_type == "no_change":
            return self.topic_no_update_response(topic)
        if response_type == "wants_other_topic":
            return "Dat is goed. Dan hoeven we hier niet over te praten."
        if response_type == "question":
            return "Dat is een goede vraag. Ik vertel kort wat ik weet, en dan gaan we verder."
        if response_type == "correction_unclear":
            return self.topic_correction_question(topic)
        if response_type == "unclear":
            return "Ik weet niet helemaal zeker wat je bedoelt, dus ik verander nu niets."
        if transcript:
            return (
                f"Ah, dus {transcript}. "
                f"Dat past wel bij wat ik al van je weet, want {topic.get('memory_link', 'dit belangrijk voor je is')}."
            )
        return self.LLM_FALLBACK

    def turn_text(self, turn: dict) -> str:
        """Single source of truth for the scripted text Leo says for a turn."""
        return turn["leo_text"]

    def print_prestart_preview(self, script: list):
        """Print pulled UM values and planned 4.0 steps before interaction starts."""
        if not self.WAIT_FOR_PREVIEW_CONFIRMATION:
            return

        print("\n" + "=" * 72)
        print("CRI 4.0 PRE-START CHECK")
        print(f"Child id: {self.CHILD_ID}")
        print(f"UM API:   {self.UM_API_BASE}")
        print("\nPulled UM fields:")

        if not self.last_um_preview:
            print("  (no UM fields were captured)")
        else:
            for field in sorted(self.last_um_preview):
                value = self.last_um_preview[field]
                marker = "" if self.is_known(value) else "  [missing]"
                print(f"  {field}: {value}{marker}")

        print("\nPlanned interaction:")
        for turn in script:
            flags = []
            if turn.get("mistake_id"):
                flags.append(turn["mistake_id"])
            if turn.get("conditional"):
                flags.append("conditional")
            suffix = f" ({', '.join(flags)})" if flags else ""
            print(f"  {turn.get('step')}. {turn.get('name')} [{turn.get('layer')}]{suffix}")
            print(f"     {self.turn_text(turn)}")

        print("=" * 72)
        input("Press Enter to start the interaction...")
        print()

    def build_script(self) -> list:
        """
        5-step early interaction flow:
        greeting, correct summary, richer topic, one memory error, and goodbye.
        """
        um = self.pull_um()
        name = self.known(um, "child_name") or self.known(um, "name")
        greeting = f"Hoi {name}, leuk je weer te zien. Leo hier." if name else "Hoi, leuk je weer te zien. Leo hier."
        summary = self.opening_summary(um)
        topic = self.select_topic_domain(um)

        mistake_topic = self.select_deliberate_mistake_topic(um, topic)
        mistake_field, mistake_actual = self.preferred_memory_item(mistake_topic)
        mistake_wrong = self.opposite_memory_value(mistake_topic, mistake_field, mistake_actual)
        goodbye = (
            f"Dankjewel, {name}. " if name else "Dankjewel. "
        ) + (
            "Ik vond het heel fijn om met je te praten. "
            "Ik vind het leuk dat ik je weer een beetje beter heb leren kennen. "
            "Tot de volgende keer!"
        )

        self.logger.info(
            "UM pulled for 4.0 flow - child:%s topic:%s mistake_topic:%s mistake_field:%s",
            name or "unknown", topic["domain"], mistake_topic["domain"], mistake_field or "fallback"
        )

        return [
            {
                "step": 1,
                "name": "Greeting",
                "layer": "L1",
                "leo_text": (
                    f"{greeting} "
                    "Ik probeer dingen over jou te onthouden, zoals wat je leuk vindt, "
                    "waar je graag over praat en wat je eerder hebt verteld. "
                    "Soms onthoud ik iets verkeerd. Dan mag je me gewoon verbeteren. "
                    "Je kunt ook altijd zeggen: Dat klopt niet, Ik wil iets anders vertellen, "
                    "of Laten we over iets anders praten."
                ),
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
            {
                "step": 2,
                "name": "Opening summary",
                "layer": "L2",
                "leo_text": summary,
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
            {
                "step": 3,
                "name": "First richer topic domain",
                "layer": "L2+L3",
                "leo_text": self.first_topic_question(topic),
                "follow_up": self.easier_topic_question(topic),
                "response_mode": "topic_interpretation",
                "topic": topic,
                "memory_link": topic["memory_link"],
                "llm_turn": True,
            },
            {
                "step": 4,
                "name": "Deliberate memory mistake",
                "layer": "L2",
                "mistake_id": "M1",
                "mistake_field": mistake_field,
                "mistake_actual": mistake_actual,
                "mistake_wrong": mistake_wrong,
                "mistake_topic": mistake_topic,
                "leo_text": self.deliberate_mistake_text(mistake_topic, mistake_field, mistake_wrong),
                "follow_up": "Dankjewel, dat helpt mij om mijn geheugen beter te maken.",
                "response_mode": "mistake_interpretation",
                "llm_turn": True,
            },
            {
                "step": 5,
                "name": "Goodbye",
                "layer": "L1",
                "leo_text": goodbye,
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
        ]

    # Conversation logging

    def log_timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def safe_filename_part(self, value: str) -> str:
        clean = str(value or "").strip()
        safe = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in clean
        )
        return safe.strip("_") or "child"

    def conversation_child_name(self) -> str:
        return (
            self.known(self.last_um_preview, "child_name")
            or self.known(self.last_um_preview, "name")
            or self.CHILD_ID
        )

    def planned_turn_log(self, turn: dict) -> dict:
        entry = {
            "step": turn.get("step"),
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "leo_text": self.turn_text(turn),
            "expects_response": turn.get("expects_response", True),
            "response_mode": turn.get("response_mode"),
        }
        if turn.get("topic"):
            entry["topic"] = turn["topic"]
        if turn.get("mistake_topic"):
            entry["mistake"] = {
                "topic": turn.get("mistake_topic"),
                "field": turn.get("mistake_field"),
                "actual": turn.get("mistake_actual"),
                "wrong": turn.get("mistake_wrong"),
            }
        return entry

    def start_conversation_log(self, script: list):
        if not self.CONVERSATION_LOG_ENABLED:
            return

        started = datetime.now().astimezone()
        session_id = started.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.CONVERSATION_LOG_ROOT, session_id)
        counter = 2
        while os.path.exists(session_dir):
            session_dir = os.path.join(self.CONVERSATION_LOG_ROOT, f"{session_id}_{counter}")
            counter += 1

        child_name = self.conversation_child_name()
        file_base = self.safe_filename_part(child_name)
        os.makedirs(session_dir, exist_ok=True)

        self.conversation_log = {
            "session_id": os.path.basename(session_dir),
            "script_version": self.SCRIPT_VERSION,
            "child_id": self.CHILD_ID,
            "child_name": child_name,
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": None,
            "folder": session_dir,
            "txt_path": os.path.join(session_dir, f"{file_base}.txt"),
            "json_path": os.path.join(session_dir, f"{file_base}.json"),
            "um_snapshot_start": dict(self.last_um_preview),
            "planned_steps": [self.planned_turn_log(turn) for turn in script],
            "turns": [],
            "events": [],
        }
        self.current_turn_log = None
        self.write_conversation_logs()
        self.logger.info("Conversation log folder: %s", session_dir)

    def finish_conversation_log(self):
        if not getattr(self, "conversation_log", None):
            return
        if self.current_turn_log:
            self.finish_turn_log()
        self.conversation_log["ended_at"] = self.log_timestamp()
        self.write_conversation_logs()

    def start_turn_log(self, turn: dict):
        if not getattr(self, "conversation_log", None):
            return
        self.current_turn_log = {
            "step": turn.get("step"),
            "name": turn.get("name"),
            "layer": turn.get("layer"),
            "started_at": self.log_timestamp(),
            "ended_at": None,
            "leo_text": self.turn_text(turn),
            "events": [],
        }
        self.conversation_log["turns"].append(self.current_turn_log)
        self.log_conversation_event(
            "step_start",
            step=turn.get("step"),
            name=turn.get("name"),
            layer=turn.get("layer"),
        )

    def finish_turn_log(self):
        if not getattr(self, "conversation_log", None) or not self.current_turn_log:
            return
        step = self.current_turn_log.get("step")
        name = self.current_turn_log.get("name")
        self.current_turn_log["ended_at"] = self.log_timestamp()
        self.log_conversation_event("step_end", step=step, name=name)
        self.current_turn_log = None

    def log_conversation_event(self, event_type: str, **data):
        if not getattr(self, "conversation_log", None):
            return

        event = {
            "timestamp": self.log_timestamp(),
            "type": event_type,
        }
        event.update(data)
        self.conversation_log["events"].append(event)
        if self.current_turn_log is not None:
            self.current_turn_log.setdefault("events", []).append(event)
        self.write_conversation_logs()

    def render_conversation_text(self) -> str:
        log = self.conversation_log or {}
        lines = [
            f"Session: {log.get('session_id', '')}",
            f"Script: {log.get('script_version', '')}",
            f"Child id: {log.get('child_id', '')}",
            f"Child name: {log.get('child_name', '')}",
            f"Started: {log.get('started_at', '')}",
            f"Ended: {log.get('ended_at') or ''}",
            "",
            "UM snapshot at start:",
        ]

        for field, value in sorted((log.get("um_snapshot_start") or {}).items()):
            lines.append(f"  {field}: {value}")

        lines.extend(["", "Planned steps:"])
        for turn in log.get("planned_steps", []):
            lines.append(f"  {turn.get('step')}. {turn.get('name')} [{turn.get('layer')}]")
            lines.append(f"     {turn.get('leo_text')}")

        lines.extend(["", "Conversation:"])
        for event in log.get("events", []):
            timestamp = event.get("timestamp", "")
            event_type = event.get("type")
            if event_type == "step_start":
                lines.append("")
                lines.append(
                    f"[{timestamp}] STEP {event.get('step')}: {event.get('name')} [{event.get('layer')}]"
                )
            elif event_type == "utterance":
                lines.append(f"[{timestamp}] {event.get('speaker')}: {event.get('text')}")
            elif event_type == "transcript_review":
                lines.append(
                    f"[{timestamp}] TRANSCRIPT {event.get('action')}: {event.get('transcript')}"
                )
            elif event_type == "intent":
                lines.append(f"[{timestamp}] INTENT: {event.get('result')}")
            elif event_type == "interpretation":
                lines.append(
                    f"[{timestamp}] INTERPRETATION ({event.get('mode')}): {event.get('result')}"
                )
            elif event_type == "um_write":
                lines.append(
                    f"[{timestamp}] UM WRITE: field={event.get('field')} "
                    f"action={event.get('action')} success={event.get('success')} "
                    f"status={event.get('status_code')}"
                )
            elif event_type == "tester_control":
                lines.append(f"[{timestamp}] TESTER: {event.get('action')}")
            elif event_type == "step_end":
                lines.append(f"[{timestamp}] STEP {event.get('step')} finished")
            else:
                lines.append(f"[{timestamp}] {event_type}: {event}")

        return "\n".join(lines) + "\n"

    def write_conversation_logs(self):
        if not getattr(self, "conversation_log", None):
            return
        try:
            with open(self.conversation_log["json_path"], "w", encoding="utf-8") as json_file:
                json.dump(self.conversation_log, json_file, ensure_ascii=False, indent=2)
            with open(self.conversation_log["txt_path"], "w", encoding="utf-8") as txt_file:
                txt_file.write(self.render_conversation_text())
        except Exception as e:
            self.logger.error("Could not write conversation log: %s", e)

    # Helpers

    def say(self, text: str):
        """Speak text via NAO TTS and wait for it to finish before returning."""
        if not text or not text.strip():
            return
        self.logger.info("LEO: %s", text)
        self.log_conversation_event("utterance", speaker="LEO", text=text)
        if self.USE_DESKTOP_MIC:
            print(f"\n[LEO]: {text}\n")
        else:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
            # Wait proportional to text length so Whisper doesn't start
            # listening while NAO is still speaking.
            # ~0.01s per character is a safe estimate for NAO's TTS speed.
            speaking_time = len(text) * 0.01
            time.sleep(speaking_time)

    def listen(self) -> str:
        self.logger.info("Listening...")
        try:
            result = self.whisper.request(
                GetTranscript(
                    timeout=self.STT_TIMEOUT,
                    phrase_time_limit=self.STT_PHRASE_LIMIT,
                )
            )
            transcript = result.transcript.strip() if result and result.transcript else ""
            self.logger.info("Child: %s", transcript or "(nothing)")
            self.log_conversation_event("utterance", speaker="CHILD", text=transcript or "(nothing)")
            return transcript
        except Exception as e:
            self.logger.error("STT error: %s", e)
            self.log_conversation_event("stt_error", error=str(e))
            return ""

    def review_transcript(self, transcript: str) -> str:
        """Let the tester approve Whisper text or listen again before continuing."""
        if not self.REVIEW_TRANSCRIPTS:
            return transcript

        while True:
            print("\n" + "-" * 72)
            print(f"[HEARD]: {transcript or '(nothing)'}")
            choice = input("Press Enter to continue, or R + Enter to listen again: ").strip().lower()
            print("-" * 72)

            if choice == "":
                self.log_conversation_event(
                    "transcript_review",
                    action="accepted",
                    transcript=transcript or "(nothing)",
                )
                return transcript
            if choice == "r":
                self.log_conversation_event(
                    "transcript_review",
                    action="retry_requested",
                    transcript=transcript or "(nothing)",
                )
                transcript = self.listen()
                continue

            print("Please press Enter to continue, or type R and press Enter to listen again.")

    def listen_with_review(self) -> str:
        """Listen once, then optionally let the tester approve the transcript."""
        return self.review_transcript(self.listen())

    def classify_with_repeat(self, transcript: str):
        """Classify once, ask for repetition on low confidence, then retry."""
        result = self.clf.classify(transcript)
        if result.intent == REPEAT_SENTINEL:
            self.logger.info("Low confidence - asking to repeat.")
            self.log_conversation_event("intent", result=result.to_dict(), retry_requested=True)
            self.say("Kun je dat nog een keer zeggen?")
            time.sleep(0.8)
            transcript = self.listen_with_review()
            result = self.clf.classify_retry(transcript)
        self.logger.info("Intent: %s", result.to_dict())
        self.log_conversation_event("intent", result=result.to_dict(), retry_requested=False)
        return result

    def llm_response(self, child_input: str) -> str:
        """L3: GPT generates a personalised Dutch follow-up."""
        if not child_input:
            return self.LLM_FALLBACK
        prompt = (
            f"Het kind zei: \"{child_input}\". "
            f"Reageer warm en enthousiast in een korte zin in het Nederlands."
        )
        try:
            reply = self.gpt.request(GPTRequest(prompt=prompt, stream=False))
            response = reply.response.strip() if reply and reply.response else ""
            return (response.split(".")[0].strip() + ".") if response else self.LLM_FALLBACK
        except Exception as e:
            self.logger.error("LLM error: %s", e)
            return self.LLM_FALLBACK

    def extract_json_object(self, raw: str) -> dict:
        """Parse a JSON object even if the model accidentally adds light wrapping."""
        if not raw:
            return {}
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}

    def normalize_topic_change(self, parsed: dict, transcript: str, topic: dict) -> dict:
        """Validate an LLM-proposed memory change before it can become pending."""
        change = parsed.get("change") if isinstance(parsed.get("change"), dict) else {}
        action = str(change.get("action") or parsed.get("action") or "none").lower()
        field = change.get("field") or parsed.get("field")
        confidence = float(change.get("confidence") or parsed.get("change_confidence") or parsed.get("confidence") or 0.0)

        if action not in ("update", "delete"):
            return {}
        if field not in topic.get("fields", []):
            self.logger.info("Topic response change ignored: field %s not allowed for topic.", field)
            return {}

        old_value = change.get("old_value") or topic.get("current_values", {}).get(field) or self.UNKNOWN_VALUE
        new_value = change.get("new_value")
        if action == "update" and not self.is_known(new_value):
            return {}
        if (
            action == "update"
            and self.is_known(old_value)
            and self.is_known(new_value)
            and str(old_value).strip().lower() == str(new_value).strip().lower()
        ):
            self.logger.info("Topic response change ignored: new value equals old value.")
            return {}

        normalized = {
            "action": action,
            "field": field,
            "field_label": topic.get("field_labels", {}).get(field, field),
            "old_value": str(old_value),
            "new_value": str(new_value) if new_value is not None else None,
            "confidence": confidence,
            "reason": change.get("reason") or parsed.get("reason", ""),
            "source_text": transcript,
            "confirmation_question": change.get("confirmation_question") or parsed.get("confirmation_question"),
        }
        self.logger.info("Pending topic change proposed by interpreter: %s", normalized)
        return normalized

    def interpret_topic_response(self, transcript: str, topic: dict) -> dict:
        """
        Let the LLM decide what the child's Step 3 answer means and what Leo should say.

        The LLM may propose a UM update/delete, but only for fields connected to
        the chosen random topic. The code still asks the child for confirmation
        before writing anything to GraphDB.
        """
        if not transcript or not transcript.strip():
            return {
                "response_type": "unclear",
                "leo_response": self.fallback_topic_response("unclear", topic, transcript),
                "change": {},
                "confidence": 1.0,
            }

        prompt = {
            "task": (
                "Interpret the child's answer after Leo mentioned remembered information "
                "and asked: 'Is daar sinds de vorige keer nog iets nieuws over gebeurd?' "
                "First decide whether the child says Leo's remembered information is correct "
                "or incorrect. Then return a natural short response for Leo, and optionally "
                "a proposed UM change."
            ),
            "topic": {
                "domain": topic.get("domain"),
                "label": topic.get("label"),
                "remembered_examples": topic.get("correct_values", []),
                "memory_link": topic.get("memory_link"),
                "allowed_fields": topic.get("fields", []),
                "field_labels": topic.get("field_labels", {}),
                "current_values": topic.get("current_values", {}),
            },
            "child_utterance": transcript,
            "rules": [
                "You decide from the full child utterance whether Leo's remembered information is still correct or should change.",
                "Use response_type no_change when no UM change is needed.",
                "Use response_type story when the child tells a normal new story/detail but does not correct memory.",
                "Use response_type question when the child asks Leo a question.",
                "Use possible_update/delete when the child clearly corrects, replaces, renames, or asks to forget a remembered value.",
                "Use correction_unclear when the child says the remembered information is wrong but does not provide the correct value or deletion target.",
                "For possible_update/delete, choose only one field from allowed_fields.",
                "Do not propose a memory change for ordinary storytelling.",
                "leo_response must be in Dutch, warm, and short.",
                "For correction_unclear, leo_response should ask what exactly Leo should change or forget.",
                "For no_change/story/question, leo_response should not ask another question.",
                "If proposing a change, include a Dutch confirmation_question Leo can ask before writing.",
                "confirmation_question must repeat the UM field/value Leo would store, must not mention the old value, and must not say 'zeg ja of nee'.",
            ],
            "output_schema": {
                "response_type": list(self.TOPIC_RESPONSE_TYPES),
                "leo_response": "Dutch sentence(s), or null if a confirmation question should be asked",
                "confidence": "number from 0 to 1",
                "change": {
                    "action": "none | update | delete",
                    "field": "one allowed field or null",
                    "old_value": "string or null",
                    "new_value": "string or null",
                    "confidence": "number from 0 to 1",
                    "reason": "short reason",
                    "confirmation_question": "Dutch confirmation question that repeats the proposed new value, does not mention old_value, and does not say yes/no; or null",
                },
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious dialogue interpreter for Leo, a Dutch child-facing robot. "
                    "Return ONLY valid JSON. You can propose memory changes, but you must be conservative."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=360,
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            parsed = self.extract_json_object(raw)
        except Exception as e:
            self.logger.error("Topic response interpretation error: %s", e)
            parsed = {}

        response_type = str(parsed.get("response_type") or "").lower()
        if response_type not in self.TOPIC_RESPONSE_TYPES:
            response_type = "unclear"

        change = self.normalize_topic_change(parsed, transcript, topic)
        if response_type in ("possible_update", "possible_delete") and not change:
            response_type = "correction_unclear"
            parsed["leo_response"] = None

        leo_response = parsed.get("leo_response")
        if not self.is_known(leo_response):
            leo_response = self.fallback_topic_response(response_type, topic, transcript)

        interpretation = {
            "response_type": response_type,
            "leo_response": leo_response,
            "change": change,
            "confidence": float(parsed.get("confidence") or 0.0),
        }
        self.logger.info("Topic response interpretation: %s", interpretation)
        self.log_conversation_event("interpretation", mode="topic", result=interpretation)
        return interpretation

    def interpret_mistake_response(self, transcript: str, turn: dict) -> dict:
        """
        Let the LLM interpret the child's answer to Leo's deliberate Step 4 mistake.

        This uses the same pattern as Step 3: the LLM decides whether no change,
        update, delete, or clarification is needed. The code asks for confirmation
        before writing any proposed update/delete.
        """
        topic = turn.get("mistake_topic", {})
        if not transcript or not transcript.strip():
            return {
                "response_type": "unclear",
                "leo_response": "Ik hoorde het niet goed, dus ik verander nu niets.",
                "change": {},
                "confidence": 1.0,
            }

        prompt = {
            "task": (
                "Leo deliberately said a wrong remembered value and asked the child to tell more. "
                "Interpret the child's answer and decide whether any UM change is needed."
            ),
            "mistake_context": {
                "topic_domain": topic.get("domain"),
                "topic_label": topic.get("label"),
                "field": turn.get("mistake_field"),
                "field_label": topic.get("field_labels", {}).get(turn.get("mistake_field"), turn.get("mistake_field")),
                "current_um_value": turn.get("mistake_actual"),
                "wrong_value_leo_said": turn.get("mistake_wrong"),
                "allowed_fields": topic.get("fields", []),
                "current_values": topic.get("current_values", {}),
            },
            "child_utterance": transcript,
            "rules": [
                "Use the same decision logic as Step 3.",
                "You decide from the full child utterance whether the UM should stay the same, update, or delete something.",
                "Use response_type no_change only when no UM change is needed and the child is not merely rejecting Leo's wrong value.",
                "If the child agrees that Leo's deliberately wrong value is correct, set wrong_value_accepted true.",
                "When wrong_value_accepted is true and the child gives no other correction, do not propose a change yourself; provide accepted_wrong_value_confirmation.",
                "accepted_wrong_value_confirmation must sound natural and subtle. It must repeat the proposed new value so the child knows exactly what Leo would remember.",
                "Do not say Leo made a mistake, do not mention deliberate mistake, do not contrast with the old UM value, and do not say 'zeg ja of nee'.",
                "Use possible_update/delete when the child clearly corrects, replaces, renames, or asks to forget a remembered value.",
                "Use correction_unclear when the child rejects Leo's wrong value but does not provide the correct value or deletion target.",
                "Only propose a change for one allowed field.",
                "Do not write or confirm anything yourself; only propose the change.",
                "leo_response must be Dutch, warm, and short.",
                "For correction_unclear, leo_response should ask directly for the exact missing field, not a broad topic question.",
                "For example: if the field is pet_name, ask what the pet is called; if the field is fav_food, ask what the favorite food is.",
                "If proposing a change, include a Dutch confirmation_question Leo can ask before writing.",
                "confirmation_question must repeat the UM field/value Leo would store, must not mention the old value, and must not say 'zeg ja of nee'.",
            ],
            "output_schema": {
                "response_type": list(self.TOPIC_RESPONSE_TYPES),
                "wrong_value_rejected": "boolean; true when the child says Leo's deliberately wrong value is wrong",
                "wrong_value_accepted": "boolean; true when the child says Leo's deliberately wrong value is correct",
                "accepted_wrong_value_confirmation": "natural Dutch confirmation question that repeats wrong_value_leo_said, does not mention the old value, and does not say yes/no when wrong_value_accepted is true, otherwise null",
                "leo_response": "Dutch sentence(s), or null if a confirmation question should be asked",
                "confidence": "number from 0 to 1",
                "change": {
                    "action": "none | update | delete",
                    "field": "one allowed field or null",
                    "old_value": "string or null",
                    "new_value": "string or null",
                    "confidence": "number from 0 to 1",
                    "reason": "short reason",
                    "confirmation_question": "Dutch confirmation question that repeats the proposed new value, does not mention old_value, and does not say yes/no; or null",
                },
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious dialogue interpreter for Leo, a Dutch child-facing robot. "
                    "Return ONLY valid JSON. The code will always ask the child to confirm before writing any proposed memory change."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=360,
                temperature=0.2,
            )
            parsed = self.extract_json_object(response.choices[0].message.content)
        except Exception as e:
            self.logger.error("Mistake response interpretation error: %s", e)
            parsed = {}

        response_type = str(parsed.get("response_type") or "").lower()
        if response_type not in self.TOPIC_RESPONSE_TYPES:
            response_type = "unclear"

        change = self.normalize_topic_change(parsed, transcript, topic)
        if response_type in ("possible_update", "possible_delete") and not change:
            response_type = "correction_unclear"
            parsed["leo_response"] = None

        wrong_value_rejected = bool(parsed.get("wrong_value_rejected"))
        wrong_value_accepted = bool(parsed.get("wrong_value_accepted"))
        if wrong_value_accepted and not change:
            change = self.mistake_acceptance_change(
                turn,
                confirmation_question=parsed.get("accepted_wrong_value_confirmation") or "",
            )
            response_type = "possible_update"
            parsed["leo_response"] = None

        if wrong_value_rejected and not change:
            if response_type == "no_change":
                response_type = "correction_unclear"
            parsed["leo_response"] = None

        leo_response = parsed.get("leo_response")
        if not self.is_known(leo_response):
            if change:
                leo_response = self.confirmation_text(change)
            elif response_type == "correction_unclear":
                leo_response = self.mistake_correction_question(turn)
            else:
                leo_response = self.fallback_topic_response(response_type, topic, transcript)

        interpretation = {
            "response_type": response_type,
            "leo_response": leo_response,
            "change": change,
            "confidence": float(parsed.get("confidence") or 0.0),
            "wrong_value_rejected": wrong_value_rejected,
            "wrong_value_accepted": wrong_value_accepted,
        }
        self.logger.info("Mistake response interpretation: %s", interpretation)
        self.log_conversation_event("interpretation", mode="mistake", result=interpretation)
        return interpretation

    def clean_confirmation_question(self, question: str, change: dict) -> str:
        """Keep LLM-generated confirmation wording, but enforce style constraints."""
        if not self.is_known(question):
            return ""

        clean = str(question).strip()
        clean = clean.replace("Zeg ja of nee.", "").replace("Zeg ja of nee", "")
        clean = clean.replace("zeg ja of nee.", "").replace("zeg ja of nee", "")
        clean = " ".join(clean.split())

        old_value = change.get("old_value")
        if self.is_known(old_value) and str(old_value).strip().lower() in clean.lower():
            return ""

        return clean

    def confirmation_text(self, change: dict) -> str:
        llm_question = self.clean_confirmation_question(change.get("confirmation_question"), change)
        if llm_question:
            return llm_question

        if change["action"] == "delete":
            return f"Wil je dat ik {change['field_label']} vergeet?"

        new_value = change.get("new_value")
        return f"Wil je dat ik {change['field_label']} verander naar {new_value}?"

    def interpret_confirmation_response(self, transcript: str, change: dict) -> dict:
        """
        LLM classifier for confirmation answers.

        This helper cannot write to GraphDB directly; it only returns a decision.
        The caller writes only when the LLM returns confirm_yes.
        """
        if not transcript or not transcript.strip():
            return {
                "decision": "unclear",
                "leo_response": "Ik hoorde niets duidelijks, dus ik verander nu niets.",
                "confidence": 1.0,
            }

        prompt = {
            "task": (
                "Leo asked the child to confirm a proposed memory change. "
                "Classify the child's answer. Be conservative: only confirm_yes "
                "when the child clearly agrees with this exact change."
            ),
            "proposed_change": {
                "action": change.get("action"),
                "field": change.get("field"),
                "field_label": change.get("field_label"),
                "old_value": change.get("old_value"),
                "new_value": change.get("new_value"),
                "confirmation_question": self.confirmation_text(change),
            },
            "child_answer": transcript,
            "decisions": {
                "confirm_yes": "child clearly agrees to the proposed change",
                "confirm_no": "child rejects the change or says it is wrong",
                "clarify": "child says something that changes or refines the proposed change",
                "unclear": "not enough information to safely write",
            },
            "rules": [
                "Use the meaning of the child's full answer, not keyword matching.",
                "Ordinary Dutch agreement or permission responses can be confirm_yes if they clearly answer Leo's confirmation question.",
                "If the child changes or refines the value, use clarify instead of confirm_yes.",
                "For unclear or clarify, leo_response should ask again naturally whether Leo may store the proposed new value.",
                "leo_response must not mention old_value and must not say 'zeg ja of nee'.",
            ],
            "output_schema": {
                "decision": "confirm_yes | confirm_no | clarify | unclear",
                "confidence": "number from 0 to 1",
                "leo_response": "short Dutch response if not confirm_yes, otherwise null",
                "reason": "short reason",
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious confirmation classifier for a child-facing robot. "
                    "Return ONLY valid JSON. Never approve a memory write unless the child clearly agrees."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.TOPIC_CHANGE_MODEL,
                messages=messages,
                max_tokens=180,
                temperature=0.0,
            )
            parsed = self.extract_json_object(response.choices[0].message.content)
        except Exception as e:
            self.logger.error("Confirmation interpretation error: %s", e)
            parsed = {}

        decision = str(parsed.get("decision") or "unclear").lower()
        if decision not in ("confirm_yes", "confirm_no", "clarify", "unclear"):
            decision = "unclear"

        confidence = float(parsed.get("confidence") or 0.0)

        result = {
            "decision": decision,
            "confidence": confidence,
            "leo_response": parsed.get("leo_response"),
            "reason": parsed.get("reason", ""),
        }
        self.logger.info("Confirmation interpretation: %s", result)
        self.log_conversation_event("interpretation", mode="confirmation", result=result)
        return result

    def write_um_change(self, change: dict) -> bool:
        field = change["field"]
        try:
            if change["action"] == "delete":
                url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/field/{field}"
                response = requests.delete(url, timeout=3)
                ok = response.status_code in (200, 202, 204, 404)
                if ok:
                    self.last_um_preview[field] = self.UNKNOWN_VALUE
                self.log_conversation_event(
                    "um_write",
                    action="delete",
                    field=field,
                    old_value=change.get("old_value"),
                    new_value=None,
                    success=ok,
                    status_code=response.status_code,
                )
                return ok

            url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/fields"
            payload = {
                "fields": {field: change["new_value"]},
                "source": "cri_4_topic_confirmation",
            }
            response = requests.post(url, json=payload, timeout=3)
            ok = response.status_code in (200, 201, 202, 204)
            if ok:
                self.last_um_preview[field] = change["new_value"]
            self.log_conversation_event(
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
            self.logger.error("Could not write confirmed UM change: %s", e)
            self.log_conversation_event(
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

    def confirm_topic_change(self, change: dict) -> bool:
        self.pending_change = change

        while True:
            self.say(self.confirmation_text(change))
            time.sleep(0.5)

            confirmation = self.listen_with_review()
            time.sleep(0.8)

            interpretation = self.interpret_confirmation_response(confirmation, change)
            if interpretation["decision"] == "confirm_yes":
                self.corrections_seen += 1
                written = self.write_um_change(change)
                if written:
                    self.say("Dankjewel, ik heb dat aangepast.")
                else:
                    self.say("Dankjewel, ik heb dat genoteerd, maar opslaan lukte nu niet.")
                self.pending_change = None
                return True

            if interpretation["decision"] == "confirm_no":
                self.say(interpretation.get("leo_response") or "Oké, dan verander ik niets.")
                self.pending_change = None
                return False

            self.say(
                interpretation.get("leo_response")
                or "Ik weet het nog niet helemaal zeker."
            )

    def handle_intent(self, result, transcript: str) -> bool:
        """
        Route classified intent to correct response.
        Returns True if handled (skip L3), False to fall through to L3.
        """
        intent = result.intent
        field = result.field

        if intent in ("um_update", "dialogue_update"):
            self.corrections_seen += 1
            self.logger.info("Correction count is now %d.", self.corrections_seen)

        if intent == "um_inspect":
            value = self.get_field(field)
            self.say(f"Ik weet dat jouw {field or 'antwoord'} {value} is.")
            return True

        elif intent == "um_update":
            self.say("Oh, je hebt gelijk! Ik pas het aan.")
            if field and result.value:
                old = self.get_field(field)
                self.say(f"Dus jouw {field} is {result.value}, niet {old}. Leuk!")
                # TODO: write correction to Eunike's API (Sherissa's task):
                # requests.post(f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/fields",
                #     json={"fields": {field: result.value}, "source": "child_correction"})
            return True

        elif intent == "dialogue_update":
            self.say("Oké, ik hoorde dat. Klopt dat?")
            return True

        elif intent == "um_delete":
            self.say("Oké, ik vergeet dat!")
            return True

        elif intent == "dialogue_question":
            self.say("Dat is een goede vraag! Ik vertel het je later.")
            return True

        elif intent == "dialogue_social":
            self.say("Haha ja! Oké, verder!")
            return True

        # um_add, dialogue_answer, and dialogue_none fall through to L3.
        return False

    def post_step_control(self, turn: dict) -> str:
        """Testing checkpoint after a full step finishes."""
        if not self.POST_STEP_TEST_CONTROLS:
            return "continue"
        if not turn.get("expects_response", True):
            return "continue"

        while True:
            print("\n" + "=" * 72)
            print(f"Step {turn.get('step')} finished: {turn.get('name')}")
            choice = input("Press Enter for next step, T + Enter to repeat this step, or Q + Enter to quit: ")
            choice = choice.strip().lower()
            print("=" * 72)

            if choice == "":
                self.log_conversation_event("tester_control", action="continue")
                return "continue"
            if choice in ("t", "r", "repeat", "again"):
                self.log_conversation_event("tester_control", action="repeat_step")
                return "repeat"
            if choice == "q":
                self.log_conversation_event("tester_control", action="quit")
                return "quit"

            print("Please press Enter, or type T to repeat, or Q to quit.")

    def handle_interpreted_response(self, interpretation: dict, turn: dict, mode: str) -> bool:
        """
        Handle one LLM interpretation.

        Returns True when Leo asked a clarification question and should listen
        once more within the same step.
        """
        change = interpretation.get("change", {})
        if change:
            self.confirm_topic_change(change)
            return False

        leo_response = interpretation.get("leo_response") or turn.get("follow_up") or self.LLM_FALLBACK
        self.say(leo_response)
        return interpretation.get("response_type") == "correction_unclear"

    def follow_up_interpretation(self, turn: dict, mode: str):
        """Listen once after Leo asks a clarification question, then interpret that answer."""
        time.sleep(0.5)
        transcript = self.listen_with_review()
        time.sleep(0.8)
        if not transcript.strip():
            self.say("Oké, dan verander ik nu nog niets.")
            return

        if mode == "topic":
            interpretation = self.interpret_topic_response(transcript, turn.get("topic", {}))
        else:
            interpretation = self.interpret_mistake_response(transcript, turn)

        # One clarification round is enough for this step. If it is still unclear,
        # keep the memory unchanged and move on.
        if interpretation.get("response_type") == "correction_unclear" and not interpretation.get("change"):
            self.say("Dankjewel. Ik weet het nog niet zeker, dus ik verander nu niets.")
            return

        self.handle_interpreted_response(interpretation, turn, mode)

    def run_turn(self, turn: dict, step_index: int, total_steps: int):
        self.start_turn_log(turn)
        try:
            self.logger.info(
                "=== Step %s/%d [%s: %s] ===",
                turn.get("step", step_index + 1),
                total_steps,
                turn["layer"],
                turn.get("name", ""),
            )

            # say() waits for NAO to finish before Whisper starts.
            self.say(self.turn_text(turn))

            if turn.get("mistake_id"):
                self.mistakes_mentioned += 1
                self.logger.info(
                    "Mistake %s mentioned; count is now %d.",
                    turn["mistake_id"],
                    self.mistakes_mentioned,
                )

            if not turn.get("expects_response", True):
                self.logger.info("No child response expected for this step.")
                return

            # Extra buffer before Whisper starts listening.
            time.sleep(0.5)

            transcript = self.listen_with_review()
            time.sleep(0.8)

            if turn.get("response_mode") == "topic_interpretation":
                if not transcript.strip():
                    self.say(turn["follow_up"])
                    time.sleep(0.5)
                    transcript = self.listen_with_review()
                    time.sleep(0.8)
                    if not transcript.strip():
                        return

                interpretation = self.interpret_topic_response(transcript, turn.get("topic", {}))
                needs_follow_up = self.handle_interpreted_response(interpretation, turn, "topic")
                if needs_follow_up:
                    self.follow_up_interpretation(turn, "topic")
                return

            if turn.get("response_mode") == "mistake_interpretation":
                if not transcript.strip():
                    self.say(turn["follow_up"])
                    return

                interpretation = self.interpret_mistake_response(transcript, turn)
                needs_follow_up = self.handle_interpreted_response(interpretation, turn, "mistake")
                if needs_follow_up:
                    self.follow_up_interpretation(turn, "mistake")
                return

            result = self.classify_with_repeat(transcript)
            handled = self.handle_intent(result, transcript)

            if not handled:
                if turn["llm_turn"] and transcript:
                    self.say(self.llm_response(transcript))
                else:
                    self.say(turn["follow_up"])
        finally:
            self.finish_turn_log()

    # Main loop

    def run(self):
        self.logger.info("Starting CRI 4.0 early interaction flow.")

        script = self.build_script()
        self.logger.info("Script ready - %d turns.", len(script))
        self.start_conversation_log(script)
        self.print_prestart_preview(script)

        try:
            if not self.USE_DESKTOP_MIC:
                self.nao.autonomous.request(NaoWakeUpRequest())

            i = 0
            while i < len(script):
                if self.shutdown_event.is_set():
                    break

                turn = script[i]

                repeat_step = True
                while repeat_step and not self.shutdown_event.is_set():
                    repeat_step = False
                    self.run_turn(turn, i, len(script))

                    action = self.post_step_control(turn)
                    if action == "repeat":
                        self.logger.info("Repeating step %s on tester request.", turn.get("step", i + 1))
                        repeat_step = True
                    elif action == "quit":
                        self.logger.info("Tester requested quit after step %s.", turn.get("step", i + 1))
                        self.shutdown_event.set()

                if not self.shutdown_event.is_set() and i < len(script) - 1:
                    time.sleep(1.0)

                i += 1

            self.logger.info("Dialogue completed.")

        except KeyboardInterrupt:
            self.logger.info("Interrupted.")
        except Exception as e:
            self.logger.error("Error: %s", e)
        finally:
            try:
                if not self.USE_DESKTOP_MIC:
                    self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass
            self.finish_conversation_log()
            self.logger.info("Shutting down.")
            self.shutdown()


if __name__ == "__main__":
    dialogue_app = CRI_ScriptedDialogue(
        openai_env_path=os.path.abspath(os.path.join(_HERE, "..", "conf", ".env")),
        nao_ip="10.0.0.165",  # Replace with your NAO's IP.
    )
    dialogue_app.run()
