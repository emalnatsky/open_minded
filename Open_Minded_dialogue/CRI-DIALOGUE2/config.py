"""
Configuration constants for the CRI dialogue.

All values here are static — they are loaded once at import time and never
change at runtime.  Runtime state (current child ID, current condition,
session config) lives on the CRI_ScriptedDialogue instance, not here.

To add a new tuning knob:
    1. Define it here with a sensible default + a comment.
    2. Mirror it on the class in CRI-BRANCH-BASIC4_0.py:
           UM_API_BASE = config.UM_API_BASE
       That keeps every existing `self.UM_API_BASE` lookup working.
"""

import os

# Where this config file lives — used to resolve roster/persona paths
# relative to the dialogue project, not the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))


# ── UM connection ────────────────────────────────────────────────────────────

UM_API_BASE = "http://localhost:8000"

# Default sentinel — overridden by apply_session_config() at session start.
# If you see this string in logs, the session interface didn't run.
CHILD_ID = "_UNSET_"

UNKNOWN_VALUE = "dat weet ik nog niet"

# Every UM field Leo can read or write
UM_FIELDS = (
    "name", "exposure", "condition",
    "age", "hobbies", "hobby_fav",
    "sports_enjoys", "sports_fav_play",
    "books_enjoys", "books_fav_title",
    "music_enjoys",
    "animals_enjoys", "animal_fav",
    "has_pet", "pet_type", "pet_name",
    "freetime_fav", "fav_food", "fav_subject",
    "school_strength", "school_difficulty",
    "aspiration", "role_model", "interest", "has_best_friend",
)

# Subset of UM_FIELDS shown in the startup memory-table preview
SCRIPT_TABLE_FIELDS = (
    "name", "age", "hobbies", "hobby_fav",
    "sports_enjoys", "sports_fav_play",
    "music_enjoys",
    "books_enjoys", "books_fav_title",
    "freetime_fav",
    "animals_enjoys", "animal_fav",
    "has_pet", "pet_type", "pet_name",
    "fav_food",
    "fav_subject", "school_strength", "school_difficulty",
    "aspiration", "role_model", "interest", "has_best_friend",
)

# Human-readable Dutch label for each UM field
FIELD_LABELS = {
    "name":              "je naam",
    "exposure":          "of we elkaar al eerder hebben gezien",
    "condition":         "de conditie",
    "age":               "je leeftijd",
    "hobbies":           "je hobby's",
    "hobby_fav":         "je favoriete hobby",
    "sports_enjoys":     "of je sport leuk vindt",
    "sports_fav_play":   "de sport die je graag doet",
    "books_enjoys":      "of je boeken leuk vindt",
    "books_fav_title":   "je favoriete boek",
    "music_enjoys":      "of je muziek leuk vindt",
    "animals_enjoys":    "of je dieren leuk vindt",
    "animal_fav":        "je lievelingsdier",
    "has_pet":           "of je een huisdier hebt",
    "pet_type":          "het soort huisdier",
    "pet_name":          "de naam van je huisdier",
    "freetime_fav":      "wat je graag in je vrije tijd doet",
    "fav_food":          "je lievelingseten",
    "fav_subject":       "je lievelingsvak",
    "school_strength":   "waar je goed in bent op school",
    "school_difficulty": "wat je moeilijk vindt op school",
    "aspiration":        "wat je later wilt doen of worden",
    "role_model":        "naar wie je opkijkt",
    "interest":          "je interesses",
    "has_best_friend":   "of je een beste vriend hebt",
}


# ── Whisper (speech-to-text) ─────────────────────────────────────────────────

STT_TIMEOUT      = 20    # seconds Whisper waits for any speech
STT_PHRASE_LIMIT = 18    # seconds max for a single phrase


# ── LLM ──────────────────────────────────────────────────────────────────────

LLM_FALLBACK = "Wauw, dat klinkt heel leuk!"
LLM_SYSTEM = (
    "Jij bent een vriendelijke robot genaamd Leo en je praat tegen een Nederlands kind van 8 tot 11 jaar oud. "
    "Geef antwoord in een korte zin (maximaal 25 woorden). "
    "Wees warm, enthousiast en geschikt voor de leeftijden tussen 8 en 11. "
    "Vraag geen vragen. Praat in het Nederlands. Gebruik geen emoji's."
)

TOPIC_CHANGE_MODEL = "gpt-4o-mini"


# ── Field categories used for mistake / memory-access logic ──────────────────

# Fields whose values are yes/no-ish rather than free text
BOOLEANISH_FIELDS = (
    "has_pet", "sports_enjoys", "books_enjoys",
    "music_enjoys", "animals_enjoys",
)

# Fields Leo skips when narrating "what I remember about you" — they exist
# to gate other questions, not to be repeated back to the child
MEMORY_ACCESS_CONTROL_FIELDS = (
    "has_pet",
    "sports_enjoys",
    "books_enjoys",
    "music_enjoys",
    "animals_enjoys",
)

TUTORIAL_CONDITION_FIELD = "condition"

CONDITION_CONTROL    = "C1"
CONDITION_EXPERIMENT = "C2"
CONDITION_LABELS = {
    "C1": "Control (no tablet)",
    "C2": "Experimental (with tablet)",
}

# Scenario utterance aliases — maps short pregen keys to UM field names.
# Empty dict by default; populated if the scenario DB uses aliases.
SCENARIO_UTTERANCE_ALIASES = {}

# Roster paths — now reads from util/test_config.pl (gitignored)
SESSION_ROSTER_DIR  = os.path.abspath(os.path.join(_HERE, "..", "util"))
SESSION_ROSTER_PATH = os.path.abspath(os.path.join(_HERE, "..", "util", "test_config.pl"))

# Fields that should never appear when Leo reads memory back to the child
MEMORY_ACCESS_EXCLUDED_FIELDS = (
    "exposure",
    "condition",
)

# When the LLM can't pick a deliberately-wrong value, fall back to these
OPPOSITE_VALUE_FALLBACKS = {
    "huisdier": ("een robotdinosaurus", "een steen", "Rover"),
    "sport":    ("schaken op de bank", "stilzitten", "ballet"),
    "boeken":   ("Harry Potter", "een kookboek", "een boek over vrachtwagens"),
    "muziek":   ("helemaal geen muziek", "drummen", "opera zingen"),
    "hobby":    ("stilzitten", "postzegels sorteren", "breien"),
    "eten":     ("pizza", "spruitjes", "broccoli"),
    "school":   ("rekenen", "aardrijkskunde", "gym"),
    "droom":    ("juf", "bankdirecteur worden", "robots repareren"),
}


# ── Content-plan layer tags ──────────────────────────────────────────────────
# These label the source of each utterance (scripted, UM-templated, etc.)

CONTENT_PLAN_SEQUENCE       = "sequence"
CASE_FULLY_SCRIPTED         = "fully_scripted"
CASE_UM_TEMPLATE            = "um_template"
CASE_PREAUTHORED_POOL       = "preauthored_pool"
CASE_LLM_PREGENERATED       = "llm_pregenerated"
CASE_RUNTIME_LLM_BRANCH     = "runtime_llm_branch"
CASE_MIXED_SEQUENCE         = "mixed_sequence"

# Prefixes that mark a field as containing a pre-generated utterance
PREGENERATED_UTTERANCE_PREFIXES = ("pregen_", "script_", "llm_pregen_")


# ── Run-mode switches ────────────────────────────────────────────────────────

# True when NAO is not connected (use laptop mic + print Leo's lines).
# Flip to False for production sessions with the robot.
USE_DESKTOP_MIC = False

ASK_RUN_MODE_AT_START         = True
SIMULATION_MODE               = False

# Where fake-persona JSONs live (used when USE_FAKE_PERSONA_UM=True)
SIMULATED_PERSONA_DIR  = os.path.abspath(os.path.join(_HERE, "fake_personas"))
SIMULATED_PERSONA_PATH = os.path.join(SIMULATED_PERSONA_DIR, "noor_1001.json")

# True  → read UM from local fake_personas/*.json files (offline dev)
# False → pull UM live from Eunike's API (real GraphDB)
USE_FAKE_PERSONA_UM           = False
SIMULATION_WRITE_PERSONA_FILE = False
WAIT_FOR_PREVIEW_CONFIRMATION = True
REVIEW_TRANSCRIPTS            = True
POST_PHASE_TEST_CONTROLS      = True

CHILD_INPUT_MODE = "microphone"   # or "keyboard"
SCRIPT_VERSION   = "CRI-BRANCH-BASIC4.0"
TOTAL_SCRIPT_PHASES = 9

ASK_SESSION_INTERFACE_AT_START = True


# ── File paths (resolved to absolute paths relative to this config file) ─────

SESSION_CONFIG_PATH = os.path.abspath(os.path.join(_HERE, "session_config.local.json"))
SESSION_STATE_PATH  = os.path.abspath(os.path.join(_HERE, "..", "_local", "session_state.json"))
LOCAL_ENV_PATH      = os.path.abspath(os.path.join(_HERE, "..", "conf", ".env"))

# Config file for session — Prolog-style, lives in util/ (gitignored).
# Contains child ID, CRI name (for NAO TTS), tablet name (for book cover),
# researcher name, and condition. Edit this before each session.
ROSTER_PATH = os.path.abspath(os.path.join(_HERE, "..", "util", "test_config.pl"))


# ── Conversation logging ─────────────────────────────────────────────────────

CONVERSATION_LOG_ENABLED = True
CONVERSATION_LOG_ROOT    = os.path.abspath(os.path.join(_HERE, "conversations"))
