"""
MemoryAccess — phase-scoped UM read responses for the CRI dialogue.

Constructed once in CRI_ScriptedDialogue.__init__:

    self.mem = MemoryAccess(self)

The dialogue keeps thin pass-through wrappers so existing call sites
(self.current_phase_memory_fields, self.memory_access_response, ...)
stay identical.

What lives here:
  - Filtering: which UM fields are child-facing vs. internal control
  - Phase-scoped sets: what Leo has spoken in the current turn, and
    cumulatively across the session
  - Memory access response: when the child asks "wat weet je over mij",
    return only the scope Leo has actually brought up so far

This module READS:
  - Class constants: MEMORY_ACCESS_EXCLUDED_FIELDS,
    MEMORY_ACCESS_CONTROL_FIELDS, UM_FIELDS
  - Dialogue state: last_um_preview, memory_fields_mentioned_so_far
  - Methods on self.d: yesish, is_known, known, field_label,
    memory_value, unique_values

It WRITES self.d.memory_fields_mentioned_so_far via
register_mentioned_memory_fields (called when Leo finishes a turn).
"""

import logging

logger = logging.getLogger(__name__)


class MemoryAccess:
    """Phase-scoped memory read helpers."""

    def __init__(self, dialogue):
        self.d = dialogue

    # ── filtering helpers ────────────────────────────────────────────────────

    def public_memory_fields(self, fields) -> list:
        """Keep memory access away from internal control fields such as exposure."""
        excluded = set(self.d.MEMORY_ACCESS_EXCLUDED_FIELDS)
        return [
            field for field in fields
            if field and field in self.d.UM_FIELDS and field not in excluded
        ]

    def is_child_facing_memory_field(self, field: str) -> bool:
        """True for UM fields that are meaningful to say back to the child."""
        if field not in self.d.UM_FIELDS:
            return False
        if field in self.d.MEMORY_ACCESS_EXCLUDED_FIELDS:
            return False
        if field in self.d.MEMORY_ACCESS_CONTROL_FIELDS:
            return False
        value = self.d.last_um_preview.get(field)
        if self.d.yesish(value) and self.d.field_label(field).startswith("of je"):
            return False
        return True

    def child_facing_memory_fields(self, fields) -> list:
        return [
            field for field in self.public_memory_fields(self.d.unique_values(list(fields or [])))
            if self.is_child_facing_memory_field(field)
        ]

    # ── phase-scoped sets ────────────────────────────────────────────────────

    def current_phase_memory_fields(self, turn: dict) -> list:
        """UM fields Leo actually said in the currently active phase segment."""
        fields = []
        fields.extend(turn.get("spoken_fields") or [])
        fields.extend((turn.get("used_fields") or {}).keys())

        if turn.get("mistake_field"):
            fields.append(turn["mistake_field"])

        return self.child_facing_memory_fields(fields)

    def register_mentioned_memory_fields(self, turn: dict):
        """Remember which UM fields Leo has already brought into this conversation."""
        mentioned = getattr(self.d, "memory_fields_mentioned_so_far", None)
        if mentioned is None:
            self.d.memory_fields_mentioned_so_far = set()
            mentioned = self.d.memory_fields_mentioned_so_far

        for field in self.current_phase_memory_fields(turn):
            mentioned.add(field)

    def memory_access_scope(self, turn: dict) -> list:
        """Memory Leo may reveal now: previous mentions plus current phase fields."""
        mentioned = getattr(self.d, "memory_fields_mentioned_so_far", set())
        fields = list(mentioned) + self.current_phase_memory_fields(turn)
        return self.child_facing_memory_fields(fields)

    # ── response generation ──────────────────────────────────────────────────

    def memory_access_summary(self, fields: list, limit: int = None) -> str:
        parts = []
        for field in fields:
            value = self.d.memory_value(field)
            if self.d.is_known(value):
                parts.append(f"{self.d.field_label(field)}: {value}")
            if limit and len(parts) >= limit:
                break
        return "; ".join(parts)

    def memory_review_lines(self, fields: list) -> list:
        """Grouped, child-facing review of all memory fields Leo has already used."""
        fields = self.child_facing_memory_fields(fields)
        field_set = set(fields)
        used = set()

        def value(field: str) -> str:
            if field not in field_set:
                return ""
            memory_value = self.d.memory_value(field)
            return str(memory_value) if self.d.is_known(memory_value) else ""

        lines = []

        name = value("name")
        age = value("age")
        if name or age:
            if name and age:
                lines.append(f"Ik weet nog dat je {name} heet en {age} jaar bent.")
            elif name:
                lines.append(f"Ik weet nog dat je {name} heet.")
            else:
                lines.append(f"Ik weet nog dat je {age} jaar bent.")
            used.update({"name", "age"})

        hobby_bits = []
        hobbies = value("hobbies")
        hobby_fav = value("hobby_fav")
        freetime = value("freetime_fav")
        if hobbies:
            hobby_bits.append(f"je houdt van {hobbies}")
        if hobby_fav:
            hobby_bits.append(f"{hobby_fav} jouw favoriete hobby is")
        if freetime:
            hobby_bits.append(f"je in je vrije tijd graag {freetime} doet")
        if hobby_bits:
            lines.append(f"Over wat jij leuk vindt weet ik dat {self.d.format_dutch_list(hobby_bits)}.")
            used.update({"hobbies", "hobby_fav", "freetime_fav"})

        sport = value("sports_fav_play")
        if sport:
            lines.append(f"Over sport weet ik dat je {sport} doet.")
            used.add("sports_fav_play")

        music = value("music_enjoys")
        if music:
            lines.append(f"Over muziek weet ik dat je daar {music} over zei.")
            used.add("music_enjoys")

        book = value("books_fav_title")
        if book:
            lines.append(f"Over boeken weet ik dat {book} bij jouw boekenwereld hoort.")
            used.add("books_fav_title")

        animal_bits = []
        animal = value("animal_fav")
        pets = value("pets")
        pet_type = value("pet_type")
        pet_name = value("pet_name")
        if animal:
            animal_bits.append(f"je lievelingsdier {animal} is")
        if pets:
            animal_bits.append(f"je huisdieren {pets} zijn")
        elif pet_name and pet_type:
            animal_bits.append(f"je huisdier {pet_name} een {pet_type} is")
        elif pet_name:
            animal_bits.append(f"je huisdier {pet_name} heet")
        elif pet_type:
            animal_bits.append(f"je een {pet_type} als huisdier hebt")
        if animal_bits:
            lines.append(f"Over dieren weet ik dat {self.d.format_dutch_list(animal_bits)}.")
            used.update({"animal_fav", "pets", "pet_type", "pet_name"})

        food = value("fav_food")
        if food:
            lines.append(f"Ik weet ook nog dat je lievelingseten {food} is.")
            used.add("fav_food")

        school_bits = []
        fav_subject = value("fav_subject")
        school_strength = value("school_strength")
        school_difficulty = value("school_difficulty")
        if fav_subject:
            school_bits.append(f"{fav_subject} jouw lievelingsvak is")
        if school_strength:
            school_bits.append(f"{school_strength} iets is waar je goed in bent")
        if school_difficulty:
            school_bits.append(f"{school_difficulty} soms wat lastiger voelt")
        if school_bits:
            lines.append(f"Over school weet ik dat {self.d.format_dutch_list(school_bits)}.")
            used.update({"fav_subject", "school_strength", "school_difficulty"})

        future_bits = []
        interest = value("interest")
        role_model = value("role_model")
        aspiration = value("aspiration")
        if interest:
            future_bits.append(f"je interesse hebt in {interest}")
        if role_model:
            future_bits.append(f"{role_model} iemand is naar wie je opkijkt")
        if aspiration:
            profession = aspiration[:-len("worden")].strip() if aspiration.lower().endswith("worden") else aspiration
            future_bits.append(f"je later {profession} wilt worden")
        if future_bits:
            lines.append(f"En over later weet ik dat {self.d.format_dutch_list(future_bits)}.")
            used.update({"interest", "role_model", "aspiration"})

        for field in fields:
            if field in used:
                continue
            field_value = value(field)
            if field_value:
                lines.append(f"Verder weet ik dat {self.d.field_label(field)} {field_value} is.")

        return lines or ["Ik heb vandaag nog niet zoveel uit mijn geheugen gebruikt."]

    def memory_review_text(self, fields: list) -> str:
        return " ".join(self.memory_review_lines(fields))

    def memory_access_response(self, result, turn: dict) -> tuple:
        """Return a phase-aware memory answer for a um_inspect intent.

        Returns (leo_text, scope, returned_fields) as a 3-tuple.
        """
        scope = self.memory_access_scope(turn)
        requested_field = getattr(result, "field", None)

        if requested_field and requested_field in scope:
            value = self.d.memory_value(requested_field)
            returned = [requested_field] if self.d.is_known(value) else []
            if returned:
                return f"Ik weet dat {self.d.field_label(requested_field)} {value} is.", scope, returned
            return f"Over {self.d.field_label(requested_field)} weet ik nu nog niets zeker.", scope, returned

        summary = self.memory_review_text(scope)
        if requested_field and requested_field not in scope:
            if summary:
                return (
                    "Daar hebben we het vandaag nog niet over gehad. "
                    f"Ik kan wel vertellen wat ik al genoemd heb: {summary}."
                ), scope, []
            return (
                "Daar hebben we het vandaag nog niet over gehad, "
                "en ik heb vandaag nog niet zoveel uit mijn geheugen genoemd."
            ), scope, []

        if summary:
            returned = [
                field for field in scope
                if self.d.is_known(self.d.memory_value(field))
            ]
            return f"Ik heb vandaag al gebruikt: {summary}", scope, returned

        return "Ik heb vandaag nog niet zoveel uit mijn geheugen genoemd.", scope, []
