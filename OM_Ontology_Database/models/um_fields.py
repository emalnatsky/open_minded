"""
models/um_fields.py

Fixed schema for the child User Model — OMR UM v5.1.
Source: OMR UM v5.1 Summary for Qualtrics Check-in (PDF, April 2026).

This dict is the SINGLE SOURCE OF TRUTH for:
  - Which fields are allowed (allowlist against SPARQL injection)
  - Field types and valid value ranges (used for type-casting + SHACL)
  - Storage type: scalar (RDF datatype property on Child) or node (named RDF node)
  - Qualtrics display logic: gate_by, gate_condition, gate_value
  - Whether Qualtrics stores the answer cleanly (stored_cleanly=False → llm_validate=True)
  - Which fields are targeted in the memory-probe / DECRI demo (mistake_priority)
  - Sensitivity tier using Julianna's privacy onion layer classification
  - Category labels for GUI display
  - is_talk_preference: whether this is a consent/preference field (not factual knowledge)

DO NOT add new fields at runtime. Schema changes require:
  1. Update this file
  2. Update um_schema.ttl and um_shapes.ttl in Protégé
  3. Reload ontology files into GraphDB
  4. Update QUALTRICS_COLUMN_MAP below

GATE LOGIC ENCODING:
  gate_by       : JSON key of the gate question this field depends on (or None)
  gate_condition: "equals"     → field visible ONLY IF gate_by field = gate_value
                  "not_equals" → field visible IF gate_by field ≠ gate_value
                                 (i.e. visible for both "ja" and "weet niet")
  gate_value    : the value being compared against (typically "ja" or "nee")

  Example: sports_fav (Q04) is "Visible if Q03 ≠ Nee"
    → gate_by="sports_enjoys", gate_condition="not_equals", gate_value="nee"
  Example: sports_fav_play (Q06) is "Visible ONLY if Q05 = Ja"
    → gate_by="sports_plays", gate_condition="equals", gate_value="ja"

  When a gate field = the blocking value:
    - If declined_sentinel is set: store that value with source="gate_not_met"
    - If declined_sentinel is None: field is simply absent (no triple written)
    - Either way, the API must NOT write user-provided data for a blocked field.

MISTAKE PRIORITY:
  Indicates whether this field is used in the memory-probe phase of the interaction,
  where Leo deliberately states an incorrect value to test the child's correction response.
  Values: None, "primary", "secondary", "decri"
  "decri" marks aspiration as the field specifically designed to demo the open-memory
  Delete/Update flow during the user study.

RELATIONSHIP NAMES:
  Must match the object property names in um_schema.ttl exactly.
  e.g. "hasHobby" here → um:hasHobby in the ontology.

"""

from typing import Literal, Optional

StorageType = Literal["scalar", "node"]
FieldType   = Literal["string", "boolean", "enum", "integer"]
GateCond    = Literal["equals", "not_equals"]


VALID_FIELDS: dict[str, dict] = {

    "hobbies": {
        # Q01 — Welke hobby's heb je?
        # Free text, raw/mixed → always ask LLM. Multi-value (child may list several).
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "hobby",
        "relationship": "hasHobby",             # ← was LIKES_HOBBY
        "target_class": "Hobby",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's hobbies (can be multiple)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    "hobby_fav": {
        # Q02 — Welke hobby vind je het allerleukst?
        # CONVERTED TO NODE: children sometimes name multiple favourites.
        # Reuses Hobby class — hasFavouriteHobby differentiates from hasHobby.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "hobby",
        "relationship": "hasFavouriteHobby",     # ← new relationship
        "target_class": "Hobby",                  # ← reuses Hobby class
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite hobby(s)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "hobby_talk": {
        # MC. Always visible (hobbies are always visible). Informs whether
        # the robot should bring up hobbies as a conversation topic.
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "hobby",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about hobbies",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,              # ← consent/preference field
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: SPORT  (Q03–Q06)
    # ══════════════════════════════════════════════════════════════

    "sports_enjoys": {
        # Q03 — Vind je sport leuk? GATE QUESTION.
        # MC. Unlocks Q04 and Q05 if answer ≠ "nee".
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child enjoys sport in general",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "sports_fav": {
        # Q04 — Welke sport vind je het leukst?
        # CONVERTED TO NODE: children sometimes name multiple favourites.
        # Reuses Sport class — hasFavouriteSport differentiates from playsSport.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "relationship": "hasFavouriteSport",      # ← new relationship
        "target_class": "Sport",                   # ← reuses Sport class
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite sport(s) to watch or do",
        "gate_by": None,  #"sports_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "sports_plays": {
        # Q05 — Doe je zelf een sport? MC.
        # Visible if Q03 ≠ Nee. Gates Q06.
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether the child actively plays a sport",
        "gate_by": None, #"sports_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "sports_fav_play": {
        # Q06 — Welke sport doe je zelf het liefst?
        # Free text, raw/mixed. STRICTLY only if Q05 = Ja (not "weet niet").
        # This is the sport they actively play, not just enjoy watching.
        # Primary mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "relationship": "playsSport",            # ← was PLAYS_SPORT
        "target_class": "Sport",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Sport(s) the child actively plays",
        "gate_by": None, #"sports_plays",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    "sports_talk": {
        # Q10 — Zou je het leuk vinden om met mij te kletsen over sport?
        # MC. Visible if sports_enjoys ≠ nee.
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about sports",
        "gate_by": "sports_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": "nee",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,              # ← consent/preference field
    },

    "sports_play_talk": {
        # Q13 — Zou je het leuk vinden om met mij te kletsen over welke sport jij speelt?
        # MC. Visible only if sports_plays = ja.
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about the sport they play",
        "gate_by": "sports_plays",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": "nee",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,              # ← consent/preference field
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: MUZIEK  (Q07–Q10)
    # ══════════════════════════════════════════════════════════════

    "music_enjoys": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "muziek",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child enjoys music",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "music_talk": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "muziek",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about music",
        "gate_by": "music_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": "nee",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": True,              # ← consent/preference field
    },

    "music_plays_instrument": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "muziek",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child plays a musical instrument",
        "gate_by": None, #"music_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "music_instrument": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "muziek",
        "relationship": "playsInstrument",       # ← was PLAYS_INSTRUMENT
        "target_class": "Instrument",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Instrument(s) the child plays",
        "gate_by": None, #"music_plays_instrument",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: BOEKEN  (Q11–Q13)
    # ══════════════════════════════════════════════════════════════

    "books_enjoys": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child enjoys reading",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "books_fav_genre": {
        # Q12 — Wat voor boeken lees je het liefst? MC.
        # ← CONVERTED TO NODE: children can select multiple genres.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "relationship": "likesBookGenre",        # ← new node relationship
        "target_class": "BookGenre",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Book genres the child likes",
        "gate_by": None, #"books_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "books_fav_title": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "relationship": "hasFavouriteBook",      # ← was LIKES_BOOK
        "target_class": "Book",
        "node_property": "title",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite book title",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    "books_talk": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about books",
        "gate_by": "books_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": "nee",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": True,              # ← consent/preference field
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: VRIJE TIJD  (Q14)
    # ══════════════════════════════════════════════════════════════

    "freetime_fav": {
        # Q14 — Wat doe je het liefst als je vrij bent? MC, always visible.
        # ← CONVERTED TO NODE: children can select multiple activities.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "sociaal",
        "relationship": "likesFreeTimeActivity", # ← new node relationship
        "target_class": "FreetimeActivity",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "What child likes doing in free time",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: SOCIAAL  (Q15)
    # ══════════════════════════════════════════════════════════════

    "has_best_friend": {
        "storage": "scalar",
        "type": "enum",
        "allowed_values": ["ja", "nee", "wil ik liever niet zeggen", "liever niet zeggen"],
        "required": False,
        "sensitivity_tier": 2,
        "category": "sociaal",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child has a best friend",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: DIEREN  (Q16–Q21)
    # ══════════════════════════════════════════════════════════════

    "animals_enjoys": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child likes animals in general",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "animal_fav": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "relationship": "hasFavouriteAnimal",    # ← was LIKES_ANIMAL
        "target_class": "Animal",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite animal",
        "gate_by": None, #"animals_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "animal_talk": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about animals",
        "gate_by": "animals_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": "nee",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": True,              # ← consent/preference field
    },

    "has_pet": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether the child has a pet",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    # ── CONSOLIDATED PET FIELD ────────────────────────────────────
    # Replaces the old separate pet_type + pet_name fields.
    # Each Pet node now has both petType and petName on the same instance,
    # so you always know which name belongs to which animal type.

    "pets": {
        # Q20+Q29 — Pet type + pet name paired on same node.
        # Multi-value: child can have multiple pets.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "relationship": "hasPetInstance",        # ← replaces HAS_PET + HAS_PET_NAME
        "target_class": "Pet",
        "node_property": "petType",              # primary value (e.g. "Hond")
        "extra_node_props": ["petName"],          # name lives alongside type
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's pet (type and name on same node)",
        "gate_by": None, #"has_pet",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    "pet_talk": {
        "storage": "scalar",
        "type": "boolean",
        "allowed_values": ["ja", "nee", "misschien", "een beetje", "weet niet", "Geen huisdier"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether child wants to talk to Leo about their pet",
        "gate_by": "has_pet",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": "Geen huisdier",
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": True,              # ← consent/preference field
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: ETEN  (Q22)
    # ══════════════════════════════════════════════════════════════

    "fav_food": {
        # Q22 — Wat vind je het lekkerste om te eten?
        # CONVERTED TO NODE: children sometimes name multiple foods.
        # Primary CRI mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "eten",
        "relationship": "hasFavouriteFood",       # ← new relationship
        "target_class": "Food",                    # ← new class
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite food(s)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "primary",
        "stored_cleanly": True,
        "is_talk_preference": False,
    },


    # ══════════════════════════════════════════════════════════════
    # CLUSTER: SCHOOL  (Q23–Q25)
    # ══════════════════════════════════════════════════════════════

    "fav_subject": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "school",
        "relationship": "likesSubject",          # ← was LIKES_SUBJECT
        "target_class": "Subject",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "School subject(s) child likes most",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "school_strength": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "school",
        "relationship": "strongAtSubject",       # ← was STRONG_AT_SUBJECT
        "target_class": "Subject",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "School subject(s) child feels strong in",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "school_difficulty": {
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 2,
        "category": "school",
        "relationship": "findsDifficult",        # ← was FINDS_DIFFICULT
        "target_class": "Subject",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "School subject(s) child finds difficult",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: ASPIRATIE  (Q24–Q27)
    # ══════════════════════════════════════════════════════════════

    "interest": {
        # Q24 — Welke onderwerpen of dingen vind jij heel interessant?
        # Free long text. Always visible, required, child-authored.
        # Kept as scalar for now. Interest class exists in ontology for future use.
        "storage": "node",
        "multi_value": True,
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "aspiratie",
        "relationship": "hasInterest",  # ← add
        "target_class": "Interest",  # ← add
        "node_property": "name",  # ← add
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "What the child finds interesting (free text)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    "aspiration": {
        # Q26 — Wat wil jij later worden? Free text.
        # CONVERTED TO NODE: children sometimes give multiple answers.
        # DECRI demo field for open-memory Delete/Update interaction.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "aspiratie",
        "relationship": "hasAspiration",           # ← new relationship
        "target_class": "Aspiration",              # ← new class
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "What the child wants to be when they grow up",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "decri",
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "role_model": {
        # Q27 — Is er iemand naar wie jij echt opkijkt?
        # CONVERTED TO NODE: children sometimes name multiple people.
        # Sensitivity tier 2.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 2,
        "category": "aspiratie",
        "relationship": "hasRoleModel",            # ← new relationship
        "target_class": "RoleModel",               # ← new class
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Person(s) the child looks up to (public figure or personal)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
        "is_talk_preference": False,
    },

    # ══════════════════════════════════════════════════════════════
    # AGE + SYSTEM FIELDS
    # ══════════════════════════════════════════════════════════════

    "age": {
        "storage": "scalar",
        "type": "integer",
        "min": 6,
        "max": 14,
        "required": False,
        "sensitivity_tier": 1,
        "category": "sociaal",
        "llm_validate": False,
        "xsd_type": "xsd:integer",
        "description": "Child's age in years",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "exposure": {
        "storage": "scalar",
        "type": "enum",
        "allowed_values": ["new", "returning"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sociaal",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether the child is new to the project or returning from a previous year",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

    "condition": {
        "storage": "scalar",
        "type": "enum",
        "allowed_values": ["condition_1", "condition_2"],
        "required": False,
        "sensitivity_tier": 1,
        "category": "sociaal",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Whether the child is using the tablet or verbal to correct during CRI",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
        "is_talk_preference": False,
    },

}



# ── Helpers: gate logic ───────────────────────────────────────────────────────

def is_field_gated(field_name: str) -> bool:
    """Return True if this field depends on a gate question."""
    return VALID_FIELDS.get(field_name, {}).get("gate_by") is not None


def check_gate(field_name: str, current_values: dict[str, str]) -> bool:
    """
    Given the current UM values (a flat dict of field→value), check whether
    the gate condition for this field is satisfied.

    Returns True if the field should be populated (gate is open).
    Returns False if the gate blocks this field.

    If field has no gate, always returns True.
    """
    fdef = VALID_FIELDS.get(field_name, {})
    gate_by = fdef.get("gate_by")
    if gate_by is None:
        return True

    gate_value    = fdef.get("gate_value", "ja")
    gate_condition = fdef.get("gate_condition", "equals")
    actual_value  = current_values.get(gate_by, "").lower().strip()

    if gate_condition == "equals":
        return actual_value == gate_value.lower()
    else:  # "not_equals"
        return actual_value != gate_value.lower()


def get_declined_sentinel(field_name: str) -> str | None:
    """
    Return the value to store when a gate blocks this field, or None
    if the field should simply be absent when the gate is closed.
    """
    return VALID_FIELDS.get(field_name, {}).get("declined_sentinel")


# ── GUI display labels ────────────────────────────────────────────────────────

CATEGORY_LABELS: dict[str, str] = {
    "hobby":      "Hobby's",
    "sport":      "Sport",
    "muziek":     "Muziek",
    "boeken":     "Boeken",
    "sociaal":    "Sociaal",
    "dieren":     "Dieren",
    "eten":       "Eten",
    "school":     "School",
    "aspiratie":  "Dromen",
}

# ── Sensitivity tier descriptions ─────────────────────────────────────────────

SENSITIVITY_TIERS: dict[int, str] = {
    1: "Orientation — freely shareable, low sensitivity",
    2: "Exploratory Affective Exchange — share with care, medium sensitivity",
    3: "Sensitive — requires explicit opt-in",
}

# ── Mistake priority description ──────────────────────────────────────────────

MISTAKE_PRIORITY_LABELS: dict[str, str] = {
    "primary":   "Primary — first field probed in memory check phase",
    "secondary": "Secondary — probed after primary if time allows",
    "decri":     "DECRI demo — field used to demonstrate open-memory update/delete",
}

# ── Recommended probe order (for interaction script) ─────────────────────────
# From the UM v5.1 document: fav_food → strongest interest field → aspiration

PROBE_ORDER = [
    "fav_food",           # primary
    "hobby_fav",          # secondary — most likely to be filled
    "sports_fav_play",    # secondary — if sports_plays = ja
    "books_fav_title",    # secondary — if books_enjoys ≠ nee
    "animal_fav",         # secondary — if animals_enjoys ≠ nee
    "music_instrument",   # secondary — if music_plays_instrument = ja
    "freetime_fav",       # secondary — always visible
    "aspiration",         # decri demo
]

# ── Qualtrics CSV column → UM field name mapping ─────────────────────────────
# Keys are the exact column headers in the Qualtrics export CSV.
# Adjust Q-numbers to match your actual Qualtrics survey IDs.

QUALTRICS_COLUMN_MAP: dict[str, str] = {
    # ── Simple 1:1 mappings ──────────────────────────────────────────────
    # Key = exact CSV column header from Qualtrics export (May 2026)
    "Q4_8":   "age",
    "Q6":     "hobby_fav",
    "Q7":     "hobby_talk",
    "Q8":     "sports_enjoys",
    "Q9_4":   "sports_fav",
    "Q10":    "sports_talk",
    "Q11":    "sports_plays",
    "Q12_4":  "sports_fav_play",
    "Q13":    "sports_play_talk",
    "Q14":    "music_enjoys",
    "Q15":    "music_plays_instrument",
    "Q16_4":  "music_instrument",
    "Q17":    "music_talk",
    "Q18":    "books_enjoys",
    "Q21":    "books_talk",
    "Q23":    "has_best_friend",
    "Q24":    "animals_enjoys",
    "Q25_4":  "animal_fav",
    "Q26":    "animal_talk",
    "Q27":    "has_pet",
    "Q30":    "pet_talk",
    "Q31_4":  "fav_food",
    "Q35_4":  "interest",
    "Q36_4":  "aspiration",
    "Q37":    "role_model",
    # ── Multi-column and MC+Other fields are handled separately ──────────
    # in load_qualtrics_new.py (HOBBIES_COLS, PET_TYPE_MAP, MC_OTHER_MAP).
    # See that file for: Q5_1..Q5_4 (hobbies), Q19+Q19_11_TEXT (books genre),
    # Q20+Q20_1_TEXT (books title), Q22+Q22_15_TEXT (freetime),
    # Q28+Q28_8_TEXT (pet type), Q29_1..Q29_8 (pet names),
    # Q32+Q32_12_TEXT (fav subject), Q33+Q33_12_TEXT (school strength),
    # Q195+Q195_12_TEXT (school difficulty).
}