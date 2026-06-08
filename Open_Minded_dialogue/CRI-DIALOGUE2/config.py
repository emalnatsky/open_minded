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
PACKAGE_ROOT = _HERE
OUTER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
LOCAL_ROOT = os.path.abspath(os.path.join(OUTER_ROOT, "_local"))


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

# Fixed startup DB retrieval checklist.
STARTUP_OVERVIEW_UM_FIELDS = SCRIPT_TABLE_FIELDS

STARTUP_OVERVIEW_SCENARIO_FIELDS = (
    ("topic_1", "default"),
    ("topic_2", "default"),
    ("p1_hobby_bridge_comment", "default"),
    ("p1_t1_recall", "default"),
    ("p1_t1_open", "default"),
    ("p1_t1_question", "default"),
    ("p1_t1_followup", "default"),
    ("p1_t2_open", "default"),
    ("p1_t2_followup", "default"),
    ("p1_t2_close", "default"),
    ("p1_m1_wrong_hobby_opener", "default"),
    ("p1_m1_followup_wrong_hobby", "default"),
    ("p1_followup_postcorrection_true_hobby", "default"),
    ("p1_m2_followup_wrong_food", "default"),
    ("p1_m2_postcorrection_true_food", "default"),
    ("p2_fav_subject_comment_subject", "default"),
    ("p2_subject_profile_link", "default"),
    ("p2_m3_postcorrection_true_strength", "default"),
    ("p2_school_wrap_after_difficulty", "default"),
    ("p3_future_theme_wrap", "default"),
    ("p3_rolemodel_recall", "default"),
    ("p3_rolemodel_ack", "default"),
    ("p3_norolemodel_ack", "default"),
    ("p3_m4_followup_wrong_aspiration", "default"),
    ("p3_m4_postcorrection_reflection", "default"),
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

CONDITION_CONTROL    = "C"
CONDITION_EXPERIMENT = "E"
CONDITION_LABELS = {
    "C": "Control: conversational-only memory access",
    "E": "Experiment: transmedial metaphor-supported memory access",
}
CONDITION_ALIASES = {
    "c": CONDITION_CONTROL,
    "c1": CONDITION_CONTROL,
    "1": CONDITION_CONTROL,
    "condition 1": CONDITION_CONTROL,
    "condition_1": CONDITION_CONTROL,
    "control": CONDITION_CONTROL,
    "control group": CONDITION_CONTROL,
    "ctrl": CONDITION_CONTROL,
    "no tablet": CONDITION_CONTROL,
    "without tablet": CONDITION_CONTROL,
    "geen tablet": CONDITION_CONTROL,
    "e": CONDITION_EXPERIMENT,
    "c2": CONDITION_EXPERIMENT,
    "2": CONDITION_EXPERIMENT,
    "condition 2": CONDITION_EXPERIMENT,
    "condition_2": CONDITION_EXPERIMENT,
    "experimental": CONDITION_EXPERIMENT,
    "experiment": CONDITION_EXPERIMENT,
    "exp": CONDITION_EXPERIMENT,
    "tablet": CONDITION_EXPERIMENT,
    "tablet group": CONDITION_EXPERIMENT,
    "with tablet": CONDITION_EXPERIMENT,
}
TABLET_REVEAL_WAIT_SECONDS = 5.0

# Scenario utterance aliases — maps short pregen keys to UM field names.
# Empty dict by default; populated if the scenario DB uses aliases.
SCENARIO_UTTERANCE_ALIASES = {
    "leo_ministory_opening": ("p1_leo_ministory_opening",),
    "leo_ministory_followup": ("p1_leo_ministory_followup",),
    "leo_ministory_wrap": ("p1_leo_ministory_wrap",),
    "hobbies_bridge": ("p1_hobby_bridge_comment",),
    "sport_recall": ("p1_sport_recall",),
    "sport_open": ("p1_sport_open",),
    "sport_followup": ("p1_sport_followup",),
    "sport_followup_choice": ("p1_sport_followup",),
    "music_open": ("p1_music_open",),
    "music_ack": ("p1_music_ack",),
    "music_followup": ("p1_music_followup",),
    "animals_open": ("p1_animals_open",),
    "animals_followup": ("p1_animals_followup",),
    "books_open": ("p1_books_open",),
    "books_ack": ("p1_books_ack",),
    "books_followup": ("p1_books_followup",),
    "m1_wrong_opener": ("p1_m1_wrong_hobby_opener",),
    "m1_wrong_followup": ("p1_m1_followup_wrong_hobby",),
    "m1_corrected_followup": ("p1_followup_postcorrection_true_hobby",),
    "m2_wrong_followup": ("p1_m2_followup_wrong_food",),
    "m2_corrected_followup": ("p1_m2_postcorrection_true_food",),
    "p2_fav_subject_comment": ("p2_fav_subject_comment_subject",),
    "p2_subject_profile_link": ("p2_subject_profile_link",),
    "m3_corrected_followup": ("p2_m3_postcorrection_true_strength",),
    "school_difficulty_wrap": ("p2_school_wrap_after_difficulty",),
    "p3_rolemodel_recall": ("p3_rolemodel_recall",),
    "p3_rolemodel_ack": ("p3_rolemodel_ack",),
    "p3_norolemodel_ack": ("p3_norolemodel_ack",),
    "p3_m4_followup_wrong_aspiration": ("p3_m4_followup_wrong_aspiration",),
    "p3_m4_postcorrection_reflection": ("p3_m4_postcorrection_reflection",),
    "pet_kind_question": ("p1_animals_followup",),
}

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

# True = use the Windows/default desktop microphone input (laptop, DJI, USB mic).
# False = use NAO's microphone input. If False, CONNECT_NAO must be True.
USE_DESKTOP_MIC = True

# True = connect NAO for speech output, LEDs, wake/rest, and optional NAO mic input.
# Launchers override this per run mode; direct script runs use this default.
CONNECT_NAO = True

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
REVIEW_TRANSCRIPTS            = False
POST_PHASE_TEST_CONTROLS      = False

CHILD_INPUT_MODE = "microphone"   # or "keyboard"
SCRIPT_VERSION   = "CRI-BRANCH-BASIC4.0"
TOTAL_SCRIPT_PHASES = 19

ASK_SESSION_INTERFACE_AT_START = True


# ── File paths (resolved to absolute paths relative to this config file) ─────

SESSION_CONFIG_PATH = os.path.abspath(os.path.join(LOCAL_ROOT, "session_config.local.json"))
SESSION_STATE_PATH  = os.path.abspath(os.path.join(LOCAL_ROOT, "session_state.json"))
LOCAL_ENV_PATH      = os.path.abspath(os.path.join(LOCAL_ROOT, "config", ".env"))

# Config file for session — Prolog-style, lives in util/ (gitignored).
# Contains child ID, CRI name (for NAO TTS), tablet name (for book cover),
# researcher name, and condition. Edit this before each session.
ROSTER_PATH = os.path.abspath(os.path.join(_HERE, "..", "util", "test_config.pl"))


# ── Conversation logging ─────────────────────────────────────────────────────

CONVERSATION_LOG_ENABLED = True
CONVERSATION_LOG_ROOT    = os.path.abspath(os.path.join(LOCAL_ROOT, "conversations"))
