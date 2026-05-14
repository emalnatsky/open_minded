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
        "relationship": "LIKES_HOBBY",
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
    },

    "hobby_fav": {
        # Q02 — Welke hobby vind je het allerleukst?
        # Single favourite hobby. Stored cleanly. Primary mistake field.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "hobby",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's single favourite hobby",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": True,
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
    },

    "sports_fav": {
        # Q04 — Welke sport vind je het leukst?
        # Free text. Visible if Q03 ≠ Nee (includes "weet niet").
        # Favourite sport even if not personally playing.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "sport",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite sport to watch or do",
        "gate_by": None, #"sports_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
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
        "relationship": "PLAYS_SPORT",
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: MUZIEK  (Q07–Q10)
    # ══════════════════════════════════════════════════════════════

    "music_enjoys": {
        # Q07 — Hou je echt van muziek? GATE QUESTION.
        # MC. Unlocks Q08 and Q09 if ≠ Nee.
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
    },

    "music_talk": {
        # Q08 — Zou je het leuk vinden om daar later met Leo over te praten?
        # MC. Opt-in gate for music as conversation topic. Visible if Q07 ≠ Nee.
        # This is a consent/preference field, not a factual one.
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
        "declined_sentinel": "nee",   # if music_enjoys=nee, store "nee" — robot won't probe music
        "mistake_priority": None,
        "stored_cleanly": True,
    },

    "music_plays_instrument": {
        # Q09 — Speel je een instrument? MC. Visible if Q07 ≠ Nee. Gates Q10.
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
    },

    "music_instrument": {
        # Q10 — Welk instrument speel je?
        # Free text, raw/mixed. STRICTLY only if Q09 = Ja.
        # Secondary mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": True,       # can play more than one instrument
        "required": False,
        "sensitivity_tier": 1,
        "category": "muziek",
        "relationship": "PLAYS_INSTRUMENT",
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: BOEKEN  (Q11–Q13)
    # ══════════════════════════════════════════════════════════════

    "books_enjoys": {
        # Q11 — Lees je graag? GATE QUESTION.
        # MC. Unlocks Q12 and Q13 if ≠ Nee.
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
    },

    "books_fav_genre": {
        # Q12 — Wat voor boeken lees je het liefst? MC.
        # Visible if Q11 ≠ Nee. Multiple choice → no LLM needed.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "Child's favourite book genre",
        "gate_by": None, #"books_enjoys",
        "gate_condition": "not_equals",
        "gate_value": "nee",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
    },

    "books_fav_title": {
        # Q13 — Wat is jouw lievelingsboek?
        # Free text, raw/mixed. Visible if Q11 ≠ Nee.
        # LLM validates but should NOT flag real book titles — prompt must account for this.
        # Primary mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": False,
        "required": False,
        "sensitivity_tier": 1,
        "category": "boeken",
        "relationship": "LIKES_BOOK",
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
    },

    "books_talk": {
        # Q21 — Zou je het leuk vinden om met mij te kletsen over boeken en lezen?
        # MC. Visible if books_enjoys ≠ nee.
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: VRIJE TIJD  (Q14)
    # ══════════════════════════════════════════════════════════════

    "freetime_fav": {
        # Q14 — Wat doe je het liefst als je vrij bent? MC, always visible.
        # Mistake priority YES. MC → no LLM needed.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "sociaal",
        "llm_validate": False,
        "xsd_type": "xsd:string",
        "description": "What child most likes to do in free time (MC options)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": True,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: SOCIAAL  (Q15)
    # ══════════════════════════════════════════════════════════════

    "has_best_friend": {
        # Q15 — Heb je een beste vriend?
        # MC. Always visible but skippable. No opt-in gate needed.
        # "Liever niet zeggen" is a sensitivity-aware opt-out.
        # If chosen: robot knows not to push on social topics.
        "storage": "scalar",
        "type": "enum",
        "allowed_values": ["ja", "nee", "wil ik liever niet zeggen", "liever niet zeggen"],
        "required": False,
        "sensitivity_tier": 2,       # social relationships are tier 2
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: DIEREN  (Q16–Q21)
    # ══════════════════════════════════════════════════════════════

    "animals_enjoys": {
        # Q16 — Hou je van dieren? GATE QUESTION.
        # MC. Unlocks Q17 and Q18 if ≠ Nee.
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
    },

    "animal_fav": {
        # Q17 — Welk dier vind jij het leukst?
        # Free text. Stored cleanly = True (but still LLM validate for plausibility).
        # Visible if Q16 ≠ Nee. Secondary mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": False,
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "relationship": "LIKES_ANIMAL",
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
    },

    "animal_talk": {
        # Q18 — Zou je daar later met Leo over willen praten?
        # MC. Opt-in for animals as conversation topic. Visible if Q16 ≠ Nee.
        # Same pattern as music_talk — consent/preference field.
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
    },

    "has_pet": {
        # Q19 — Heb je een huisdier? MC. Always visible. Gates Q20 and Q21.
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
    },

    "pet_type": {
        # Q20 — Wat voor huisdier heb je? MC, raw/mixed (can have multiple types).
        # STRICTLY only if Q19 = Ja. Secondary mistake field.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "relationship": "HAS_PET",
        "target_class": "Pet",
        "node_property": "petType",
        "extra_node_props": ["petName"],
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Type(s) of pet the child has",
        "gate_by": None, #"has_pet",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": None,
        "mistake_priority": "secondary",
        "stored_cleanly": False,
    },

    "pet_name": {
        # Q21 — Hoe heet jouw huisdier? Free text, raw/mixed.
        # STRICTLY only if Q19 = Ja. NOT a mistake field — names are personal.
        # No LLM validation: pet names are arbitrary strings, LLM would over-flag.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "dieren",
        "relationship": "HAS_PET_NAME",
        "target_class": "PetName",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,       # names: do not LLM-validate, would over-flag
        "xsd_type": "xsd:string",
        "description": "Name(s) of child's pet(s)",
        "gate_by": None, #"has_pet",
        "gate_condition": "equals",
        "gate_value": "ja",
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
    },

    "pet_talk": {
        # Q30 — Zou je het leuk vinden om met mij te kletsen over een huisdier?
        # MC. Visible only if has_pet = ja.
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: ETEN  (Q22)
    # ══════════════════════════════════════════════════════════════

    "fav_food": {
        # Q22 — Wat is jouw lievelingseten? Free text. Always visible.
        # PRIMARY mistake field — highest priority for the memory probe phase.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "eten",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Child's favourite food",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "primary",
        "stored_cleanly": True,
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: SCHOOL  (Q23–Q25)
    # ══════════════════════════════════════════════════════════════

    "fav_subject": {
        # Q23 — Wat is je lievelingsvak op school? MC. Always visible.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "school",
        "relationship": "LIKES_SUBJECT",
        "target_class": "Subject",
        "node_property": "name",
        "extra_node_props": [],
        "llm_validate": False,       # MC — controlled values
        "xsd_type": "xsd:string",
        "description": "School subject(s) child likes most",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": True,
    },

    "school_strength": {
        # Q24 — In welke vakken ben jij goed? MC. Always visible.
        # New in v5.1. Multi-value (can be good at multiple subjects).
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 1,
        "category": "school",
        "relationship": "STRONG_AT_SUBJECT",
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
    },

    "school_difficulty": {
        # Q25 — Is er een vak dat je soms lastig vindt? MC. Always visible but optional.
        # Tier 2: sharing academic difficulty is more sensitive than liking a subject.
        "storage": "node",
        "type": "string",
        "multi_value": True,
        "required": False,
        "sensitivity_tier": 2,
        "category": "school",
        "relationship": "FINDS_DIFFICULT",
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
    },

    # ══════════════════════════════════════════════════════════════
    # CLUSTER: ASPIRATIE  (Q24–Q27)
    # ══════════════════════════════════════════════════════════════

    "interest": {
        # Q24 — Welke onderwerpen of dingen vind jij heel interessant?
        # Free long text. Always visible, required, child-authored.
        # Future: LLM could extract specific interests into node entities.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "aspiratie",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "What the child finds interesting (free text)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
    },

    "aspiration": {
        # Q26 — Wat wil jij later worden? Free text. Optional.
        # "DECRI" mistake priority: specifically designed as the demo field for
        # the open-memory Delete/Update interaction — robot states a wrong
        # aspiration, child corrects it, and can see the change in the GUI.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 1,
        "category": "aspiratie",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "What the child wants to be when they grow up",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": "decri",
        "stored_cleanly": True,
    },

    "role_model": {
        # Q27 — Is er iemand naar wie jij echt opkijkt? Free text, raw/mixed.
        # Always visible. No mistake priority — personal/sensitive enough not to probe.
        # Tier 2: naming a person (celebrity or personal) is more sensitive.
        "storage": "scalar",
        "type": "string",
        "required": False,
        "sensitivity_tier": 2,
        "category": "aspiratie",
        "llm_validate": True,
        "xsd_type": "xsd:string",
        "description": "Person the child looks up to (public figure or personal)",
        "gate_by": None,
        "gate_condition": None,
        "gate_value": None,
        "declined_sentinel": None,
        "mistake_priority": None,
        "stored_cleanly": False,
    },

    # ══════════════════════════════════════════════════════════════
    # AGE  (Q27 in spreadsheet — added back in v5.1 update)
    # ══════════════════════════════════════════════════════════════

    "age": {
        # ID 27 — Hoe oud ben je?
        # Integer, range 6-13. Always visible. Asked in Qualtrics.
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
    },

    "exposure": {
        # Set during import — "new" or "returning"
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
    },

    "condition": {
        # For tagging whether using the tablet or not
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
    # in load_qualtrics.py (HOBBIES_COLS, PET_NAME_COLS, MC_OTHER_MAP).
    # See that file for: Q5_1..Q5_4 (hobbies), Q19+Q19_11_TEXT (books genre),
    # Q20+Q20_1_TEXT (books title), Q22+Q22_15_TEXT (freetime),
    # Q28+Q28_8_TEXT (pet type), Q29_1..Q29_8 (pet names),
    # Q32+Q32_12_TEXT (fav subject), Q33+Q33_12_TEXT (school strength),
    # Q195+Q195_12_TEXT (school difficulty).
}