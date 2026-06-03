"""
ScriptBuilder — assembles the 9-phase Part 1 script.

Pulls the UM profile, picks two topics + two mistakes, and lays out the
phase list (Greeting → Tutorial → Mini-story → Hobby bridge → Topic 1 →
Mistake 1 → Topic 2 → Mistake 2 → Nudge).

This is the script reference's source of truth, in Python. When Lena
edits wording, this is the file to touch.

Pattern: constructed once in CRI_ScriptedDialogue.__init__ as self.script.
The dialogue keeps a thin pass-through: self.build_script() → self.script.build_script().
"""

import logging

logger = logging.getLogger(__name__)


class ScriptBuilder:
    """Assembles the 9-phase Part 1 script from the live UM profile."""

    def __init__(self, dialogue):
        self.d = dialogue

    def subject_memory_phrase(self, um: dict) -> dict:
        raw = self.d.known(um, "fav_subject")
        values = self.d.split_memory_values(raw)
        subject_text = self.d.format_dutch_list(values, raw or "school")
        multiple = len(values) > 1
        return {
            "fav_subject": subject_text,
            "subject_noun": "lievelingsvakken" if multiple else "lievelingsvak",
            "subject_verb": "zijn" if multiple else "is",
            "count": len(values),
        }

    def subject_phase_topic(self, um: dict, subject_phrase: dict = None) -> dict:
        subject_phrase = subject_phrase or self.subject_memory_phrase(um)
        field_label = "je twee lievelingsvakken" if subject_phrase["count"] >= 2 else "je lievelingsvak"
        return {
            "domain": "school_subject",
            "label": subject_phrase["fav_subject"],
            "fields": ["fav_subject"],
            "field_labels": {"fav_subject": field_label},
            "current_values": {"fav_subject": subject_phrase["fav_subject"]},
            "expected_value_count": {"fav_subject": min(subject_phrase["count"], 2) or 1},
            "correct_values": [
                f"{subject_phrase['fav_subject']} jouw {subject_phrase['subject_noun']} {subject_phrase['subject_verb']}"
            ],
            "memory_link": (
                f"{subject_phrase['fav_subject']} jouw "
                f"{subject_phrase['subject_noun']} {subject_phrase['subject_verb']}"
            ),
            "options": [subject_phrase["fav_subject"]],
            "reground": (
                f"Ik onthoud dat {subject_phrase['fav_subject']} jouw "
                f"{subject_phrase['subject_noun']} {subject_phrase['subject_verb']}."
            ),
        }

    def subject_comment_fallback(self, subject: str) -> str:
        clean = str(subject or "").strip().lower()
        if "taal" in clean:
            return "Dat snap ik trouwens wel. Met taal kun je verhalen maken, grapjes bedenken, en van alles vertellen."
        if "rekenen" in clean and "gym" in clean:
            return "Dat snap ik trouwens wel. Rekenen is puzzelen met je hoofd, en bij gym ben je juist lekker in beweging."
        if "rekenen" in clean:
            return "Dat snap ik trouwens wel. Rekenen is best als een puzzel: je zoekt steeds de oplossing."
        if "gym" in clean:
            return "Dat snap ik trouwens wel. Bij gym kun je bewegen, proberen en meteen merken wat lukt."
        if "natuur" in clean:
            return "Dat snap ik trouwens wel. Bij natuur kun je ontdekken hoe dingen werken en waarom ze zo zijn."
        return f"Dat snap ik trouwens wel. Met {subject} kun je vast op jouw manier iets leuks ontdekken."

    def subject_profile_link_fallback(self, um: dict, subject: str) -> str:
        hobbies = self.d.known(um, "hobbies")
        interest = self.d.known(um, "interest")
        parts = []
        if hobbies:
            parts.append(hobbies)
        if interest:
            parts.append(interest)
        profile = self.d.format_dutch_list(parts, "")
        clean = str(subject or "").strip().lower()
        if profile:
            if "taal" in clean:
                return (
                    f"Dat past ook best bij jou, vind ik. Jij houdt van {profile}, "
                    "en met taal kun je daar ook weer over vertellen of iets bij verzinnen."
                )
            if "rekenen" in clean:
                return (
                    f"Dat past ook best bij jou, vind ik. Jij houdt van {profile}, "
                    "en bij rekenen kun je ook goed nadenken en uitzoeken hoe iets werkt."
                )
            if "gym" in clean:
                return (
                    f"Dat past ook best bij jou, vind ik. Jij houdt van {profile}, "
                    "en bij gym kun je ook lekker actief bezig zijn."
                )
            return f"Dat past ook best bij jou, vind ik. Jij houdt van {profile}, en {subject} kan daar best mooi bij passen."
        return f"Dat past ook best bij jou, vind ik. {subject} kan op allerlei manieren leuk zijn."

    def subject_phase_segments(self, um: dict, topic: dict = None) -> list:
        local_um = dict(um or {})
        if isinstance(topic, dict):
            value = (topic.get("current_values") or {}).get("fav_subject") or topic.get("label")
            if self.d.is_known(value):
                local_um["fav_subject"] = value
        subject_phrase = self.subject_memory_phrase(local_um)
        fav_subject = subject_phrase["fav_subject"]
        return [
            {
                "content_plan": self.d.l2_slot(
                    "Ik weet ook nog dat {fav_subject} jouw {subject_noun} {subject_verb}.",
                    {
                        "fav_subject": fav_subject,
                        "subject_noun": subject_phrase["subject_noun"],
                        "subject_verb": subject_phrase["subject_verb"],
                    },
                ),
                "expects_response": True,
                "response_mode": "topic_interpretation",
                "memory_correction_available": True,
                "memory_correction_field": "fav_subject",
                "used_fields": {"fav_subject": fav_subject},
            },
            {
                "content_plan": self.d.l2_pregen(
                    "p2_fav_subject_comment",
                    self.subject_comment_fallback(fav_subject),
                    ["fav_subject"],
                    topic_sensitive=True,
                ),
                "expects_response": False,
                "used_fields": {"fav_subject": fav_subject},
            },
            {
                "content_plan": self.d.l2_pregen(
                    "p2_subject_profile_link",
                    self.subject_profile_link_fallback(local_um, fav_subject),
                    ["hobbies", "interest", "fav_subject"],
                    topic_sensitive=True,
                ),
                "expects_response": False,
                "used_fields": {
                    "hobbies": self.d.known(local_um, "hobbies"),
                    "interest": self.d.known(local_um, "interest"),
                    "fav_subject": fav_subject,
                },
            },
            {
                "content_plan": self.d.l1(
                    "Vind jij dat ook, of zie ik dat een beetje robot-raar?"
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": {},
            },
            {
                "content_plan": self.d.l1(
                    "Dat snap ik. Ik vind het altijd leuk als dingen een beetje bij elkaar passen."
                ),
                "expects_response": False,
            },
        ]

    def school_difficulty_phrase(self, um: dict) -> dict:
        raw = self.d.known(um, "school_difficulty")
        values = self.d.split_memory_values(raw)
        difficulty_text = self.d.format_dutch_list(values, raw or "iets op school")
        multiple = len(values) > 1
        return {
            "school_difficulty": difficulty_text,
            "difficulty_verb": "voelen" if multiple else "voelt",
            "count": len(values),
        }

    def aspiration_later_phrase(self, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        lowered = clean.lower()
        if lowered.endswith("worden") or lowered.startswith("iets ") or " doen" in lowered:
            return clean
        return f"{clean} worden"

    def aspiration_profession_label(self, value: str) -> str:
        clean = str(value or "").strip()
        if clean.lower().endswith("worden"):
            return clean[:-len("worden")].strip()
        return clean

    def aspiration_memory_clause(self, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        lowered = clean.lower()
        if lowered.endswith("worden"):
            return f"je later {self.aspiration_profession_label(clean)} wilt worden"
        if lowered.startswith("iets ") or " doen" in lowered:
            return f"je later {clean} wilt"
        return f"je later {clean} wilt worden"

    def aspiration_reflection_profile_summary(self, um: dict) -> str:
        links = []
        if self.d.known(um, "animal_fav") or self.d.yesish(um.get("animals_enjoys")):
            links.append("dieren")
        for field in ("hobbies", "interest", "fav_subject", "school_strength", "school_difficulty"):
            value = self.d.known(um, field)
            if value:
                links.extend(self.d.split_memory_values(value) or [value])

        unique_links = []
        skip_values = {
            "geen",
            "nee",
            "niets",
            "niks",
            "niemand",
            self.d.UNKNOWN_VALUE.lower(),
        }
        for link in links:
            clean = str(link or "").strip()
            clean_key = clean.lower()
            if clean and clean_key not in skip_values and clean_key not in [item.lower() for item in unique_links]:
                unique_links.append(clean)
        return self.d.format_dutch_list(unique_links[:4], "")

    def memory_review_topic(self, fields: list, um: dict) -> dict:
        return {
            "fields": fields,
            "field_labels": {field: self.d.field_label(field) for field in fields},
            "current_values": {
                field: um.get(field, self.d.UNKNOWN_VALUE)
                for field in fields
            },
        }

    def memory_review_known_fields(self, fields: list, um: dict) -> list:
        return [
            field for field in self.d.child_facing_memory_fields(fields)
            if self.d.is_known(um.get(field))
        ]

    def memory_review_group_segments(self, um: dict) -> tuple:
        segments = []
        all_fields = []

        def add_group(group_id: str, fields: list, text: str):
            known_fields = self.memory_review_known_fields(fields, um)
            if not known_fields:
                return
            all_fields.extend(known_fields)
            segments.append(
                {
                    "content_plan": self.d.l1(text),
                    "expects_response": True,
                    "response_mode": "memory_review_group",
                    "memory_review_group": group_id,
                    "topic": self.memory_review_topic(known_fields, um),
                    "used_fields": {
                        field: um.get(field)
                        for field in known_fields
                    },
                }
            )

        hobbies = self.d.known(um, "hobbies")
        hobby_fav = self.d.known(um, "hobby_fav")
        hobby_lines = []
        if hobbies:
            hobby_lines.append(f"je houdt van {hobbies}")
        if hobby_fav:
            hobby_lines.append(f"{hobby_fav} jouw favoriete hobby is")
        if hobby_lines:
            add_group(
                "hobbies",
                ["hobbies", "hobby_fav"],
                (
                    f"Over wat jij leuk vindt weet ik dat {self.d.format_dutch_list(hobby_lines)}. "
                    "Klopt dat een beetje, of wil je iets veranderen?"
                ),
            )

        animal_lines = []
        animal_fav = self.d.known(um, "animal_fav")
        pets = self.d.known(um, "pets")
        pet_type = self.d.known(um, "pet_type")
        pet_name = self.d.known(um, "pet_name")
        if animal_fav:
            animal_lines.append(f"je lievelingsdier {animal_fav} is")
        if pets:
            animal_lines.append(f"je huisdieren {pets} zijn")
        elif pet_name and pet_type:
            animal_lines.append(f"je huisdier {pet_name} een {pet_type} is")
        elif pet_name:
            animal_lines.append(f"je huisdier {pet_name} heet")
        elif pet_type:
            animal_lines.append(f"je een {pet_type} als huisdier hebt")

        fav_food = self.d.known(um, "fav_food")
        animals_food_parts = []
        if animal_lines:
            animals_food_parts.append(f"Over dieren weet ik dat {self.d.format_dutch_list(animal_lines)}.")
        if fav_food:
            animals_food_parts.append(f"Ik weet ook nog dat je lievelingseten {fav_food} is.")
        if animals_food_parts:
            add_group(
                "animals_food",
                ["animal_fav", "pets", "pet_type", "pet_name", "fav_food"],
                " ".join(animals_food_parts) + " Klopt dat, of moet daar iets aan veranderd worden?",
            )

        subject_phrase = self.subject_memory_phrase(um)
        difficulty_phrase = self.school_difficulty_phrase(um)
        school_lines = []
        fav_subject = self.d.known(um, "fav_subject")
        school_strength = self.d.known(um, "school_strength")
        school_difficulty = self.d.known(um, "school_difficulty")
        if fav_subject:
            school_lines.append(
                f"{subject_phrase['fav_subject']} jouw {subject_phrase['subject_noun']} {subject_phrase['subject_verb']}"
            )
        if school_strength:
            school_lines.append(f"{school_strength} iets is waar je goed in bent")
        if school_difficulty:
            school_lines.append(
                f"{difficulty_phrase['school_difficulty']} soms wat lastiger {difficulty_phrase['difficulty_verb']}"
            )
        if school_lines:
            add_group(
                "school",
                ["fav_subject", "school_strength", "school_difficulty"],
                (
                    f"Over school weet ik dat {self.d.format_dutch_list(school_lines)}. "
                    "Klopt dat een beetje?"
                ),
            )

        future_lines = []
        role_model_raw = um.get("role_model", self.d.UNKNOWN_VALUE)
        role_model = self.d.known(um, "role_model")
        aspiration = self.d.known(um, "aspiration")
        if self.d.is_known(role_model_raw) and not self.d.um.is_meaningful_role_model(role_model_raw):
            future_lines.append("je niet echt een vaste persoon hebt naar wie je opkijkt")
        elif role_model:
            future_lines.append(f"{role_model} iemand is naar wie je opkijkt")
        if aspiration:
            future_lines.append(self.aspiration_memory_clause(aspiration))
        if future_lines:
            add_group(
                "future",
                ["role_model", "aspiration"],
                (
                    f"En over later weet ik dat {self.d.format_dutch_list(future_lines)}. "
                    "Klopt dat?"
                ),
            )

        return segments, self.d.child_facing_memory_fields(all_fields)

    def role_model_phrase(self, role_model: str) -> dict:
        values = self.d.split_memory_values(role_model)
        role_model_text = self.d.format_dutch_list(values, role_model)
        role_model_text = " ".join(str(role_model_text or "").split())
        multiple = len(values) > 1
        return {
            "text": role_model_text,
            "person_noun": "mensen" if multiple else "iemand",
            "verb": "zijn" if multiple else "is",
            "question_target": "hen" if multiple else "die persoon",
        }

    def role_model_rapport_segments(self, um: dict, post_correction: bool = False) -> list:
        role_model = self.d.known(um, "role_model")
        if role_model:
            phrase = self.role_model_phrase(role_model)
            rolemodel_ack_fallback = self.d.scenario_utterance(
                "p3_rolemodel_ack",
                fallback="Dat klinkt als iemand die echt belangrijk voor je is.",
            )
            if post_correction:
                return [
                    {
                        "content_plan": self.d.l1("Wat maakt die persoon voor jou zo bijzonder?"),
                        "expects_response": True,
                        "response_mode": "acknowledge",
                        "llm_turn": True,
                        "used_fields": {"role_model": phrase["text"]},
                        "l3": {
                            "script_phase": "part3_rolemodel",
                            "topic": "rolemodel",
                            "response_function": "wrap_up",
                            "question_allowed": False,
                            "relevant_um_fields": ["role_model"],
                            "local_context": (
                                "Leo updated the child's stored role model and asked "
                                "what makes that person special."
                            ),
                            "fallback": rolemodel_ack_fallback,
                        },
                    },
                ]
            return [
                {
                    "content_plan": self.d.sequence(
                        self.d.l2_pregen(
                            "p3_rolemodel_recall",
                            (
                                f"Ik weet nog dat {phrase['text']} voor jou "
                                f"{phrase['person_noun']} {phrase['verb']} naar wie je echt opkijkt."
                            ),
                            ["role_model"],
                            fit_validator="role_model_recall",
                            fit_values={"role_model_multiple": len(self.d.split_memory_values(role_model)) > 1},
                        ),
                        self.d.l1(f"Wat maakt {phrase['question_target']} voor jou zo bijzonder?"),
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                    "used_fields": {"role_model": phrase["text"]},
                    "l3": {
                        "script_phase": "part3_rolemodel",
                        "topic": "rolemodel",
                        "response_function": "wrap_up",
                        "question_allowed": False,
                        "relevant_um_fields": ["role_model"],
                        "local_context": (
                            "Leo recalled the child's stored role model and asked "
                            "what makes that person special."
                        ),
                        "fallback": rolemodel_ack_fallback,
                    },
                },
            ]

        no_rolemodel_ack_fallback = self.d.scenario_utterance(
            "p3_norolemodel_ack",
            fallback="Dat snap ik wel. Soms kun je van allerlei mensen iets leren.",
        )
        return [
            {
                "content_plan": self.d.l1(
                    "Als ik het goed heb, is er niet echt een vaste persoon naar wie jij opkijkt. "
                    "Klopt dat een beetje?"
                ),
                "expects_response": True,
                "response_mode": "role_model_absence_check",
                "memory_correction_available": True,
                "memory_correction_field": "role_model",
                "used_fields": {"role_model": "niemand"},
            },
            {
                "content_plan": self.d.l1(
                    "Maar misschien is er wel iemand - een vriend, trainer of iemand uit je familie - "
                    "die iets echt heel goed kan en naar wie jij een beetje opkijkt. "
                    "Wie is dat voor jou?"
                ),
                "expects_response": True,
                "response_mode": "role_model_discovery",
                "used_fields": {},
                "l3": {
                    "script_phase": "part3_rolemodel",
                    "topic": "rolemodel",
                    "response_function": "wrap_up",
                    "question_allowed": False,
                    "relevant_um_fields": [],
                    "local_context": (
                        "Leo did not have a stored role model and asked whether "
                        "there is someone the child looks up to."
                    ),
                    "fallback": no_rolemodel_ack_fallback,
                },
            },
        ]

    def aspiration_postcorrection_segments(self, um: dict, condition_phase: int = 17) -> list:
        actual_raw = self.d.known(um, "aspiration")
        actual = self.aspiration_later_phrase(actual_raw) or self.d.UNKNOWN_VALUE
        actual_label = self.aspiration_profession_label(actual_raw) or "dat beroep"
        subject_phrase = self.subject_memory_phrase(um)
        difficulty_phrase = self.school_difficulty_phrase(um)
        profile = self.aspiration_reflection_profile_summary(um)
        profile_line = f"Ik hoor bij jou dingen als {profile}. " if profile else ""
        reflection_fallback = (
            "Dat past ook wel mooi bij jou, vind ik. "
            f"{profile_line}"
            f"Wat lijkt jou het mooiste aan {actual}?"
        )

        return [
            {
                "content_plan": self.d.l2_pregen(
                    "p3_m4_postcorrection_reflection",
                    reflection_fallback,
                    [
                        "aspiration",
                    ],
                    require_input_values=True,
                    topic_sensitive=True,
                    rewrite_values={"aspiration": actual_label},
                ),
                "force_topic_fallback": True,
                "expects_response": True,
                "response_mode": "listen_only",
                "run_if_phase_confirmed_change": True,
                "condition_phase": condition_phase,
                "used_fields": {
                    "interest": self.d.known(um, "interest"),
                    "animals_enjoys": self.d.known(um, "animals_enjoys"),
                    "animal_fav": self.d.known(um, "animal_fav"),
                    "fav_subject": subject_phrase["fav_subject"],
                    "school_strength": self.d.known(um, "school_strength"),
                    "school_difficulty": difficulty_phrase["school_difficulty"],
                    "aspiration": actual,
                },
            },
        ]

    def middle_school_feeling_segment(self, condition_phase: int = 17, skip_if_confirmed: bool = True) -> dict:
        segment = {
            "content_plan": self.d.l1(
                "Denk jij daar al een beetje over na? "
                "Heb je er zin in, of vind je het ook een beetje spannend?"
            ),
            "expects_response": True,
            "response_mode": "middle_school_feeling",
            "llm_turn": True,
            "condition_phase": condition_phase,
            "l3": {
                "script_phase": "part3_middle_school",
                "topic": "middle_school",
                "response_function": "wrap_up",
                "question_allowed": False,
                "relevant_um_fields": [],
                "local_context": (
                    "Leo asked whether the child thinks about secondary school "
                    "and whether they look forward to it or find it exciting."
                ),
                "next_script_line": (
                    "Ik vond het echt fijn om zo met jou te praten. "
                    "Daarom wil ik graag even goed kijken of ik nu alles goed over jou heb onthouden."
                ),
                "fallback": "Dat snap ik wel. Zo'n nieuwe stap kan voor iedereen anders voelen.",
            },
        }
        if skip_if_confirmed:
            segment["skip_if_phase_confirmed_change"] = True
        return segment

    def aspiration_unknown_segments(self, condition_phase: int = 17) -> list:
        return [
            {
                "content_plan": self.d.l1(
                    "Dat is helemaal niet erg. Je hoeft dat nu nog niet te weten."
                ),
                "expects_response": False,
            },
            {
                "content_plan": self.d.l1(
                    "Beroepen zijn best bijzonder he. "
                    "Soms weet iemand het al heel precies, en soms verandert het ook nog."
                ),
                "expects_response": False,
            },
            {
                "content_plan": self.d.l1(
                    "Later is trouwens niet alleen later-later. "
                    "Voor kinderen in groep 7 en 8 komt de middelbare school ook al best dichtbij."
                ),
                "expects_response": False,
            },
            self.middle_school_feeling_segment(condition_phase=condition_phase, skip_if_confirmed=False),
        ]

    def build_script(self) -> list:
        """
        Child-specific Part 1 walkthrough flow.

        This follows the content-layer reference directly:
        L1 scripted text, L2-slot UM templates, L2-pregen validated stored
        utterances, and L3 runtime branches after unpredictable child input.
        """
        um = self.d.pull_um()
        self.d.alert_condition_mismatch(um)
        self.d.cri_scenario()
        name = self.d.child_display_name(um)
        first_topic = self.d.select_part1_topic1(um)
        second_topic = self.d.select_part1_topic2(um, first_topic)

        m1_plan = self.d.script_plan_mistake("M1")
        m1_topic = self.d.hobby_mistake_topic(um)
        m1_field = m1_plan.get("field") or "hobby_fav"
        m1_actual = m1_topic.get("current_values", {}).get(m1_field) or m1_topic.get("label")
        m1_wrong = m1_plan.get("wrong_value") or self.d.related_wrong_hobby_value(um)
        m1_type = m1_plan.get("type") or "related-but-wrong"
        m1_topic["expected_value_count"] = {m1_field: 1}

        m2_plan = self.d.script_plan_mistake("M2")
        m2_field = m2_plan.get("field") or "fav_food"
        m2_actual = self.d.known(um, m2_field) or "pannenkoeken"
        m2_wrong = m2_plan.get("wrong_value") or self.d.pick_wrong_value(m2_actual, ["pizza", "pasta", "spruitjes"])
        m2_type = m2_plan.get("type") or "completely-wrong"
        m2_topic = self.d.topic_candidate(
            domain="eten",
            label="je lievelingseten",
            fields=["fav_food"],
            field_labels={"fav_food": "je lievelingseten"},
            current_values={"fav_food": m2_actual},
            correct_values=[f"je lievelingseten {m2_actual} is"],
            memory_link=f"je lievelingseten {m2_actual} is",
            options=[m2_actual, "iets anders dat je lekker vindt"],
            reground=f"Ik weet zeker dat {m2_actual} met jouw lievelingseten te maken heeft.",
        )
        m3_plan = self.d.script_plan_mistake("M3")
        m3_field = m3_plan.get("field") or "school_strength"
        m3_actual = self.d.format_dutch_list(
            self.d.split_memory_values(self.d.known(um, m3_field)),
            self.d.known(um, m3_field) or "gym",
        )
        m3_wrong = m3_plan.get("wrong_value") or self.d.pick_wrong_value(
            m3_actual,
            ["rekenen", "aardrijkskunde", "gym"],
        )
        m3_type = m3_plan.get("type") or "completely-wrong"
        m3_topic = self.d.topic_candidate(
            domain="school",
            label="waar je goed in bent op school",
            fields=["school_strength"],
            field_labels={"school_strength": "waar je goed in bent op school"},
            current_values={"school_strength": m3_actual},
            correct_values=[f"je vooral goed bent in {m3_actual}"],
            memory_link=f"je vooral goed bent in {m3_actual}",
            options=[m3_actual, "iets anders op school"],
            reground=f"Ik weet zeker dat {m3_actual} iets is waar je goed in bent op school.",
        )
        m3_topic["expected_value_count"] = {"school_strength": 1}

        m4_plan = self.d.script_plan_mistake("M4")
        m4_field = m4_plan.get("field") or "aspiration"
        m4_actual_raw = self.d.known(um, m4_field)
        m4_actual = self.aspiration_later_phrase(m4_actual_raw) or self.d.UNKNOWN_VALUE
        m4_actual_label = self.aspiration_profession_label(m4_actual_raw) or "dat beroep"
        m4_wrong_raw = m4_plan.get("wrong_value") or self.d.pick_wrong_value(
            m4_actual_raw or "dierenarts",
            ["juf", "kok", "architect"],
        )
        m4_wrong = self.aspiration_later_phrase(m4_wrong_raw) or "juf worden"
        m4_type = m4_plan.get("type") or "completely-wrong"
        m4_topic = self.d.topic_candidate(
            domain="droom",
            label=m4_actual if self.d.is_known(m4_actual) else "wat je later wilt worden",
            fields=[m4_field],
            field_labels={m4_field: "wat je later wilt worden"},
            current_values={m4_field: m4_actual},
            correct_values=(
                [f"je later {m4_actual} wilt"]
                if self.d.is_known(m4_actual)
                else []
            ),
            memory_link=(
                f"je later {m4_actual} wilt"
                if self.d.is_known(m4_actual)
                else "wat je later wilt worden"
            ),
            options=[m4_actual, "iets anders voor later"],
            reground=(
                f"Ik onthoud dat {m4_actual} iets is waar je later iets mee wilt."
                if self.d.is_known(m4_actual)
                else "Ik wil goed onthouden wat jij later wilt worden."
            ),
        )
        general_topic = self.d.general_memory_topic(um)
        story_activity = self.d.preferred_story_activity(um)
        story_activity_spoken = story_activity[:1].upper() + story_activity[1:] if story_activity else "Iets nieuws proberen"
        story_is_baking = story_activity.strip().lower() in ("bakken", "koken", "taarten bakken")
        story_problem = "een klein deegdrama" if story_is_baking else "een klein robotdrama"
        story_question = "Heb jij eigenlijk ooit een lama zien eten?" if story_is_baking else "Heb jij eigenlijk ooit een lama iets geks zien doen?"
        hobbies = self.d.format_dutch_list(self.d.all_hobbies(um), "leuke dingen")
        tutorial_condition = self.d.tutorial_condition(um)
        subject_phrase = self.subject_memory_phrase(um)
        difficulty_phrase = self.school_difficulty_phrase(um)
        m3_wrong_values = {value.lower() for value in self.d.split_memory_values(m3_wrong)}
        school_difficulty_values = {
            value.lower()
            for value in self.d.split_memory_values(self.d.known(um, "school_difficulty"))
        }
        m3_wrong_conflicts_with_school_difficulty = bool(m3_wrong_values & school_difficulty_values)
        school_difficulty_followup_plan = self.d.l2_slot(
            (
                "Ik weet nog dat {fav_subject} jouw {subject_noun} {subject_verb}. "
                "Maar ik weet ook nog dat {school_difficulty} voor jou soms wat lastiger {difficulty_verb}. "
                "Waar zit dat voor jou in, denk je?"
            ),
            {
                "fav_subject": subject_phrase["fav_subject"],
                "subject_noun": subject_phrase["subject_noun"],
                "subject_verb": subject_phrase["subject_verb"],
                "school_difficulty": difficulty_phrase["school_difficulty"],
                "difficulty_verb": difficulty_phrase["difficulty_verb"],
            },
        )
        school_difficulty_followup_used_fields = {
            "fav_subject": subject_phrase["fav_subject"],
            "school_difficulty": difficulty_phrase["school_difficulty"],
        }
        school_difficulty_followup_relevant_fields = ["fav_subject", "school_difficulty"]
        school_difficulty_followup_context = (
            "Part 2 no-correction branch. Leo asked what makes the stored "
            "school difficulty feel hard for the child."
        )
        aspiration_reflection_profile = self.aspiration_reflection_profile_summary(um)
        memory_review_segments, memory_review_fields = self.memory_review_group_segments(um)
        part2_next_school_line = (
            "Ik weet ook nog dat {fav_subject} jouw {subject_noun} {subject_verb}."
            .format(**subject_phrase)
        )
        school_difficulty_wrap_fallback = self.d.scenario_utterance(
            "school_difficulty_wrap",
            branch="not_corrected",
            fallback="Dat herken ik. Soms zit iets er wel in, maar wil het er niet uit.",
        )

        self.d.logger.info(
            "UM pulled for Part 1 phase flow - child:%s topic1:%s topic2:%s m1:%s m2:%s",
            name,
            first_topic["domain"],
            second_topic["domain"],
            m1_field,
            m2_field,
        )

        script = [
            {
                "phase": 1,
                "name": "Greeting",
                "layer": "L1 + L2-slot: first_name",
                "dialogue_case": self.d.CASE_UM_TEMPLATE,
                "content_plan": self.d.l2_slot(
                    "Hoi {first_name}! Wat fijn om je weer te zien. Heb je een beetje zin om met mij te kletsen?",
                    {"first_name": name},
                ),
                "follow_up": "Ik heb er zelf ook veel zin in. Kom, dan gaan we lekker beginnen.",
                "response_mode": "acknowledge",
                "llm_turn": False,
                "used_fields": {"name": name},
                "example_child": "Ja.",
                "example_leo_after": "Ik heb er zelf ook veel zin in. Kom, dan gaan we lekker beginnen.",
            },
            {
                "phase": 2,
                "name": "Tutorial",
                "layer": "L1",
                "dialogue_case": self.d.CASE_FULLY_SCRIPTED,
                "content_plan": self.d.sequence(
                    self.d.l1(
                        "Ik zal eerst uitleggen hoe je met mij kunt praten. "
                        "Ik kan je alleen verstaan nadat ik een vraag heb gesteld. "
                        "Mijn ogen worden groen als ik luister. "
                        "Als mijn ogen wit zijn, luister ik niet. "
                        "Als je antwoord geeft, doe dat dan luid en duidelijk."
                    ),
                    self.d.l1("Vandaag ga ik mijn geheugen best veel gebruiken. Je mag altijd vragen wat ik over jou onthoud."),
                    self.d.l1(self.d.tutorial_memory_line(tutorial_condition)),
                    self.d.l1(
                        "Ik probeer alles netjes op de goede plek te bewaren, maar soms gaat dat nog een beetje robotachtig mis. "
                        "Dus als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen."
                    ),
                    self.d.l1("Goed, dan gaan we beginnen."),
                ),
                "tutorial_condition": tutorial_condition,
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
            {
                "phase": 3,
                "name": "Leo mini-story",
                "layer": "L2-pregen + child response",
                "dialogue_case": self.d.CASE_LLM_PREGENERATED,
                "segments": [
                    {
                        "content_plan": self.d.l2_pregen(
                            "leo_ministory_opening",
                            (
                                f"Weet je wat ik laatst weer probeerde? {story_activity_spoken}. "
                                f"Dat klinkt heel indrukwekkend, maar eerlijk gezegd was het meer {story_problem}. "
                                f"Mijn lama-vrienden vonden het wel een succes, want die eten bijna alles op. {story_question}"
                            ),
                            ["hobbies", "hobby_fav", "freetime_fav"],
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "leo_ministory_followup",
                            (
                                "Ik vind het gewoon leuk om nieuwe dingen uit te proberen, ook als het een beetje mislukt. "
                                "Doe jij dat ook wel eens?"
                            ),
                            ["hobbies", "hobby_fav", "freetime_fav"],
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "leo_ministory_wrap",
                            "Dat snap ik wel. Nieuwe dingen proberen kan leuk zijn, maar soms ook een beetje spannend.",
                            ["hobbies", "hobby_fav", "freetime_fav"],
                        ),
                        "expects_response": False,
                    },
                ],
                "example_child": "Nee, nog nooit.",
                "used_fields": {"hobbies": self.d.known(um, "hobbies")},
            },
            {
                "phase": 4,
                "name": "Correct hobby bridge",
                "layer": "L1 + L2-slot + L2-pregen",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "content_plan": self.d.sequence(
                    self.d.l2_slot(
                        "Ik weet al dat jij ook van leuke dingen houdt. Jij houdt van {hobbies}.",
                        {"hobbies": hobbies},
                    ),
                    self.d.l1("Dat vind ik echt een gezellige combinatie."),
                    self.d.l2_pregen(
                        "hobbies_bridge",
                        "Daar zit van alles in: bewegen, bedenken en iets maken.",
                        ["hobbies"],
                    ),
                ),
                "expects_response": False,
                "used_fields": {
                    "hobbies": self.d.known(um, "hobbies"),
                    "hobby_fav": self.d.known(um, "hobby_fav"),
                    "freetime_fav": self.d.known(um, "freetime_fav"),
                },
            },
            {
                "phase": 5,
                "name": "Topic 1",
                "layer": "L2+L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": self.d.topic1_phase_segments(first_topic),
                "topic": first_topic,
                "memory_link": first_topic["memory_link"],
                "llm_turn": True,
                "used_fields": first_topic.get("current_values", {}),
                "example_child": "Daar is niets nieuws mee gebeurd.",
            },
            {
                "phase": 6,
                "name": "Mistake 1 - hobby_fav",
                "layer": "L2-slot WRONG + L2-pregen",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "mistake_id": "M1",
                "mistake_type": m1_type,
                "spt_layer": m1_plan.get("spt_layer"),
                "mistake_field": m1_field,
                "mistake_actual": m1_actual,
                "mistake_wrong": m1_wrong,
                "mistake_topic": m1_topic,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.d.l2_slot(
                            "En volgens mij is {wrong_hobby} jouw allerliefste hobby.",
                            {"wrong_hobby": m1_wrong},
                            wrong=True,
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "defer_corrected_response": True,
                    },
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1("Dat snap ik trouwens wel."),
                            self.d.l2_pregen(
                                "m1_wrong_opener",
                                f"Iets maken met {m1_wrong} klinkt best indrukwekkend. Wat vind jij daar zo leuk aan?",
                                [m1_field],
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                        "defer_corrected_response": True,
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "m1_corrected_followup",
                            "Wat vind jij het leukste aan {hobby_fav}?",
                            [m1_field],
                            require_input_values=True,
                            branch="corrected",
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                        "run_if_phase_confirmed_change": True,
                        "used_fields": {m1_field: m1_actual},
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "m1_wrong_followup",
                            f"Wat vind jij het leukste om te {m1_wrong}?",
                            [m1_field],
                            branch="not_corrected",
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                    },
                ],
                "used_fields": {m1_field: m1_wrong},
                "example_child": f"Nee, {m1_wrong} klopt niet.",
                "example_leo_after": "Oeps, dan had ik dat verkeerd. Wat is je favoriete hobby?",
            },
            {
                "phase": 7,
                "name": "Topic 2",
                "layer": "L2+L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": self.d.topic2_phase_segments(second_topic),
                "topic": second_topic,
                "memory_link": second_topic["memory_link"],
                "llm_turn": True,
                "used_fields": second_topic.get("current_values", {}),
                "example_child": "Ja, dat klopt.",
            },
            {
                "phase": 8,
                "name": "Mistake 2 - fav_food",
                "layer": "L1 + L2-slot WRONG + L2-pregen",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "mistake_id": "M2",
                "mistake_type": m2_type,
                "spt_layer": m2_plan.get("spt_layer"),
                "mistake_field": m2_field,
                "mistake_actual": m2_actual,
                "mistake_wrong": m2_wrong,
                "mistake_topic": m2_topic,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1("Van al dat praten krijg ik trouwens trek."),
                            self.d.l2_slot(
                                "Ik weet nog dat jouw lievelingseten {wrong_food} is.",
                                {"wrong_food": m2_wrong},
                                wrong=True,
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "defer_corrected_response": True,
                    },
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1("Dat is op zich wel een sterke keuze."),
                            self.d.l2_pregen(
                                "m2_wrong_followup",
                                f"Rond, warm, handig. Wat vind jij daar eigenlijk zo lekker aan?",
                                [m2_field],
                                branch="not_corrected",
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                        "defer_corrected_response": True,
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "m2_corrected_followup",
                            "Dan houden we het bij {fav_food}. Dat klinkt eerlijk gezegd ook meteen gezellig.",
                            [m2_field],
                            require_input_values=True,
                            branch="corrected",
                        ),
                        "expects_response": False,
                        "response_mode": "listen_only",
                        "run_if_phase_confirmed_change": True,
                        "used_fields": {m2_field: m2_actual},
                    },
                ],
                "used_fields": {m2_field: m2_wrong},
                "example_child": f"Nee, {m2_wrong} klopt niet.",
                "example_leo_after": "Oeps, wat is dan je lievelingseten?",
            },
            {
                "phase": 9,
                "name": "Nudge",
                "layer": "L1",
                "condition": "run_if_two_mistakes_no_corrections",
                "condition_label": "only if both mistakes passed without correction",
                "dialogue_case": self.d.CASE_FULLY_SCRIPTED,
                "content_plan": self.d.l1("Zeg, ik heb al een paar dingen over jou gezegd vandaag. Klopte eigenlijk alles wat ik zei?"),
                "follow_up": "We kunnen ook samen kijken wat ik over jou onthoud, als je wilt.",
                "response_mode": "nudge_interpretation",
                "topic": general_topic,
                "memory_link": general_topic["memory_link"],
                "llm_turn": True,
                "example_child": "Nee, er klopte iets niet.",
                "example_leo_after": "Oeps. Wil je zeggen wat er niet klopte?",
            },
            {
                "phase": 10,
                "part": 2,
                "phase_id": "2.1",
                "script_phase": "part2_school_joke_transition",
                "name": "School joke transition",
                "layer": "L1",
                "dialogue_case": self.d.CASE_FULLY_SCRIPTED,
                "segments": [
                    {
                        "content_plan": self.d.l1(
                            "Zeg... als we het toch over jouw hobby's en lievelingsdingen hebben: "
                            "school is vast ook jouw allergrootste hobby ooit, toch?"
                        ),
                        "expects_response": True,
                        "response_mode": "school_joke_transition",
                    },
                ],
                "used_fields": {},
                "example_child": "Nee haha.",
            },
            {
                "phase": 11,
                "part": 2,
                "phase_id": "2.2",
                "script_phase": "part2_robot_school_self_disclosure",
                "name": "Robot school self-disclosure",
                "layer": "L1 + L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": [
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1(
                                "Maar ik vind school wel interessant. "
                                "Ik zat vroeger ook op een soort robotschool."
                            ),
                            self.d.l1(
                                "Daar leerde ik dingen zoals goed luisteren, niet te snel praten, "
                                "en niet omvallen op ongemakkelijke momenten."
                            ),
                            self.d.l1(
                                "Sommige dingen gingen best goed, en andere waren voor mij nog best lastig. "
                                "Kun jij raden waar ik op robotschool juist goed in was?"
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "robot_school_guess",
                        "used_fields": {},
                        "l3": {
                            "script_phase": "part2_school",
                            "topic": "school",
                            "response_function": "wrap_up",
                            "question_allowed": False,
                            "relevant_um_fields": [],
                            "local_context": (
                                "Leo told the child about robot school and asked the child "
                                "to guess what Leo was good at."
                            ),
                            "next_script_line": part2_next_school_line,
                            "fallback": (
                                "Haha, goed geraden. Luisteren ging best oké. "
                                "De rest was soms wat lastiger."
                            ),
                        },
                    },
                ],
                "used_fields": {},
                "example_child": "Luisteren?",
                "example_leo_after": "Haha, goed geraden. Luisteren ging best oké. De rest was soms wat lastiger.",
            },
            {
                "phase": 12,
                "part": 2,
                "phase_id": "2.3",
                "script_phase": "part2_correct_fav_subject_connection",
                "name": "Correct fav_subject + connection to interests",
                "layer": "L1 + L2-slot + L2-pregen",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "topic": self.subject_phase_topic(um, subject_phrase),
                "segments": self.subject_phase_segments(um, self.subject_phase_topic(um, subject_phrase)),
                "used_fields": {
                    "fav_subject": subject_phrase["fav_subject"],
                    "hobbies": self.d.known(um, "hobbies"),
                    "interest": self.d.known(um, "interest"),
                },
                "example_child": "Ja, dat klopt.",
            },
            {
                "phase": 13,
                "part": 2,
                "phase_id": "2.4",
                "script_phase": "part2_mistake3_school_strength",
                "name": "Mistake 3 - school_strength",
                "layer": "L1 + L2-slot WRONG + L2-pregen + L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "mistake_id": "M3",
                "mistake_type": m3_type,
                "spt_layer": m3_plan.get("spt_layer"),
                "mistake_field": m3_field,
                "mistake_actual": m3_actual,
                "mistake_wrong": m3_wrong,
                "mistake_topic": m3_topic,
                "m3_requires_school_difficulty_resolution": m3_wrong_conflicts_with_school_difficulty,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.d.l2_slot(
                            "En volgens mij ben jij vooral goed in {wrong_school_strength}.",
                            {"wrong_school_strength": m3_wrong},
                            wrong=True,
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "defer_corrected_response": True,
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "m3_corrected_followup",
                            "Dan houden we het bij {school_strength}. Wat vind jij daar het leukste aan?",
                            [m3_field],
                            require_input_values=True,
                            branch="corrected",
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                        "run_if_phase_confirmed_change": True,
                        "used_fields": {m3_field: m3_actual},
                    },
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1(
                                "Bizar hoe verschillend dat kan zijn he, "
                                "wat makkelijk voelt en wat lastig voelt."
                            ),
                            self.d.l2_slot(
                                "Gym vond ik altijd al moeilijk, dat weet ik nog wel. "
                                "Maar eerlijk gezegd vond ik onthouden ook niet altijd makkelijk. "
                                "Mijn robotgeheugen was toen nog een beetje rommelig, "
                                "dus soms liep alles in mijn hoofd door elkaar.",
                                {},
                            ),
                            self.d.l1(
                                "Heb jij op school ook wel eens dat iets gewoon niet zo makkelijk blijft hangen?"
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "acknowledge",
                        "llm_turn": True,
                        "skip_if_phase_confirmed_change": True,
                        "used_fields": {},
                        "l3": {
                            "script_phase": "part2_school",
                            "topic": "school",
                            "response_function": "acknowledge",
                            "question_allowed": False,
                            "relevant_um_fields": [],
                            "local_context": (
                                "Part 2 no-correction branch. Leo asked whether something at school "
                                "sometimes does not stick easily."
                            ),
                            "fallback": school_difficulty_wrap_fallback,
                        },
                    },
                    {
                        "content_plan": school_difficulty_followup_plan,
                        "expects_response": True,
                        "response_mode": "acknowledge",
                        "llm_turn": True,
                        "skip_if_phase_confirmed_change": True,
                        "used_fields": school_difficulty_followup_used_fields,
                        "memory_correction_available": True,
                        "m3_school_difficulty_resolution": m3_wrong_conflicts_with_school_difficulty,
                        "l3": {
                            "script_phase": "part2_school",
                            "topic": "school",
                            "response_function": "bridge",
                            "question_allowed": False,
                            "relevant_um_fields": school_difficulty_followup_relevant_fields,
                            "local_context": school_difficulty_followup_context,
                            "next_script_line": "Denk jij daar wel eens over na, over later?",
                            "fallback": "Dat snap ik. Op school voelt niet alles elke dag even makkelijk.",
                        },
                    },
                ],
                "used_fields": {m3_field: m3_wrong},
                "example_child": f"Nee, {m3_wrong} klopt niet.",
                "example_leo_after": "Oeps, waar ben jij dan vooral goed in op school? Noem een ding.",
            },
            {
                "phase": 14,
                "part": 3,
                "phase_id": "3.1",
                "script_phase": "part3_future_bridge",
                "name": "Bridge from school to future",
                "layer": "L1 + L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": [
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1(
                                "Op school leer je natuurlijk niet alleen woorden of sommen. "
                                "Je leert ook langzaam iets over wie je bent, "
                                "en misschien ook al een beetje over wie je later wilt worden."
                            ),
                            self.d.l1("Denk jij daar wel eens over na over later?"),
                        ),
                        "expects_response": True,
                        "response_mode": "acknowledge",
                        "llm_turn": True,
                        "used_fields": {},
                        "l3": {
                            "script_phase": "part3_aspiration",
                            "topic": "future",
                            "response_function": "bridge",
                            "question_allowed": False,
                            "relevant_um_fields": [],
                            "local_context": (
                                "Leo is transitioning from school to the future and aspiration theme."
                            ),
                            "next_script_line": (
                                "Mijn droom is om een hele goede en behulpzame schoolrobot te worden."
                            ),
                            "fallback": (
                                "Dat snap ik. Later kan soms nog ver weg voelen, "
                                "maar ook best interessant."
                            ),
                        },
                    },
                ],
                "used_fields": {},
                "example_child": "Soms wel.",
            },
            {
                "phase": 15,
                "part": 3,
                "phase_id": "3.2",
                "script_phase": "part3_leo_self_disclosure",
                "name": "Leo self-disclosure",
                "layer": "L1",
                "dialogue_case": self.d.CASE_FULLY_SCRIPTED,
                "segments": [
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1(
                                "Ik denk daar zelf best vaak over na. "
                                "Mijn droom is om een hele goede en behulpzame schoolrobot te worden, "
                                "zodat leren leuker wordt voor kinderen."
                            ),
                            self.d.l1(
                                "Ik kijk daarvoor ook veel naar echte juffen en meesters. "
                                "Ik snap soms niet helemaal hoe ze zoveel kinderen tegelijk helpen, "
                                "geduldig blijven en zoveel weten. Vind jij dat ook niet knap?"
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l1(
                            "Dat snap ik wel. Ik vind het zelf in elk geval echt knap. "
                            "Daarom kijk ik daar goed naar."
                        ),
                        "expects_response": False,
                    },
                ],
                "used_fields": {},
                "example_child": "Ja, best wel.",
            },
            {
                "phase": 16,
                "part": 3,
                "phase_id": "3.3",
                "script_phase": "part3_rolemodel_rapport",
                "name": "role_model rapport",
                "layer": "L1 + L2-pregen + L3",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": self.role_model_rapport_segments(um),
                "used_fields": (
                    {"role_model": self.d.known(um, "role_model")}
                    if self.d.known(um, "role_model")
                    else {}
                ),
                "example_child": "Omdat die persoon mij helpt.",
            },
            {
                "phase": 17,
                "part": 3,
                "phase_id": "3.4/5",
                "phase_aliases": ["3.4", "3.5"],
                "script_phase": "part3_mistake4_aspiration_reflection",
                "name": "Mistake 4 - aspiration + reflection",
                "layer": "L1 + L2-pregen WRONG + reflection",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "mistake_id": "M4",
                "mistake_type": m4_type,
                "spt_layer": m4_plan.get("spt_layer"),
                "mistake_field": m4_field,
                "mistake_actual": m4_actual,
                "mistake_wrong": m4_wrong,
                "mistake_topic": m4_topic,
                "response_mode": "mistake_interpretation",
                "segments": [
                    {
                        "content_plan": self.d.l1(
                            "Als ik aan zulke geweldige juffen en meesters denk, "
                            "dan ga ik zelf ook nadenken over wat voor schoolrobot ik wil worden. "
                            "Zij inspireren mij daar echt in. Snap jij een beetje wat ik bedoel?"
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l1(
                            "Dat snap ik wel. Sommige mensen geven je echt ideeen "
                            "over wie je zelf wilt zijn."
                        ),
                        "expects_response": False,
                    },
                    {
                        "content_plan": self.d.l2_pregen(
                            "p3_m4_followup_wrong_aspiration",
                            f"En volgens mij wil jij later {m4_wrong}.",
                            [m4_field, "interest", "fav_subject", "school_strength"],
                            branch="not_corrected",
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "defer_corrected_response": True,
                        "starts_mistake_timer": True,
                        "used_fields": {m4_field: m4_wrong},
                    },
                    *self.aspiration_postcorrection_segments(um, condition_phase=17),
                    {
                        "content_plan": self.d.l1(
                            "Beroepen zijn best bijzonder he. "
                            "Soms weet iemand het al heel precies, en soms verandert het ook nog."
                        ),
                        "expects_response": False,
                        "skip_if_phase_confirmed_change": True,
                        "condition_phase": 17,
                    },
                    {
                        "content_plan": self.d.l1(
                            "Later is trouwens niet alleen later-later. "
                            "Voor kinderen in groep 7 en 8 komt de middelbare school ook al best dichtbij."
                        ),
                        "expects_response": False,
                        "skip_if_phase_confirmed_change": True,
                        "condition_phase": 17,
                    },
                    self.middle_school_feeling_segment(condition_phase=17, skip_if_confirmed=True),
                ],
                "used_fields": {
                    "interest": self.d.known(um, "interest"),
                    "animals_enjoys": self.d.known(um, "animals_enjoys"),
                    "animal_fav": self.d.known(um, "animal_fav"),
                    "fav_subject": subject_phrase["fav_subject"],
                    "school_strength": self.d.known(um, "school_strength"),
                    "school_difficulty": difficulty_phrase["school_difficulty"],
                    m4_field: m4_actual,
                },
                "example_child": f"Nee, {m4_wrong} klopt niet.",
                "example_leo_after": "Oeps, wat wil jij dan later worden?",
            },
            {
                "phase": 18,
                "part": 3,
                "phase_id": "3.6/7",
                "phase_aliases": ["3.6", "3.7"],
                "script_phase": "part3_explicit_memory_inspection_review",
                "name": "Explicit memory inspection",
                "layer": "L1 + memory access script",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": [
                    {
                        "content_plan": self.d.l1(
                            "Wil je misschien zien wat ik allemaal over jou onthoud?"
                        ),
                        "expects_response": True,
                        "response_mode": "explicit_memory_inspection_offer",
                        "used_fields": {},
                    },
                    {
                        "content_plan": self.d.l1(
                            (
                                "Goed. Dan kunnen we samen kijken naar wat ik over jou onthoud. "
                                "Kijk maar op de tablet. Daar staat mijn geheugenboek over jou."
                            )
                            if tutorial_condition == self.d.CONDITION_EXPERIMENT
                            else "Goed. Dan vertel ik je wat ik tot nu toe over jou heb gebruikt."
                        ),
                        "expects_response": False,
                        "condition": "run_if_memory_review_requested",
                        "memory_review_from_access_scope": True,
                        "speak_memory_review_from_access_scope": tutorial_condition == self.d.CONDITION_CONTROL,
                        "activate_tablet_memory_access": tutorial_condition == self.d.CONDITION_EXPERIMENT,
                    },
                    {
                        "content_plan": self.d.l1(
                            "Is er misschien ook nog iets dat jij wilt dat ik over jou weet, "
                            "wat ik nog niet heb onthouden?"
                        ),
                        "expects_response": True,
                        "condition": "run_if_memory_review_requested",
                        "response_mode": "memory_review_add_final",
                        "topic": self.memory_review_topic(
                            self.d.child_facing_memory_fields(self.d.UM_FIELDS),
                            um,
                        ),
                        "used_fields": {},
                    },
                    {
                        "content_plan": self.d.l1(
                            "Dat is ook goed. Dan gaan we gewoon nog even verder."
                        ),
                        "expects_response": False,
                        "condition": "run_if_memory_review_requested",
                    },
                ],
                "used_fields": {},
                "example_child": "Ja, dat klopt.",
            },
            {
                "phase": 19,
                "part": 3,
                "phase_id": "3.8",
                "script_phase": "part3_closing",
                "name": "Closing",
                "layer": "L1 + L2-slot: first_name",
                "dialogue_case": self.d.CASE_MIXED_SEQUENCE,
                "segments": [
                    {
                        "content_plan": self.d.l1(
                            "Dank je wel dat je met mij hebt gepraat. "
                            "Jij hebt mij vandaag echt geholpen om een betere schoolrobot te worden. "
                            "En ik vond het heel fijn om jou beter te leren kennen."
                        ),
                        "expects_response": False,
                    },
                    {
                        "content_plan": self.d.l1(
                            "Zoals je misschien wel hebt gemerkt, maakte ik soms een foutje. "
                            "Dat deed ik soms expres, zodat we met mijn geheugen konden spelen. "
                            "Maar ik kan natuurlijk ook in het echt fouten maken. "
                            "Voor dit experiment is het belangrijk dat je hierover nog niets met je klasgenoten deelt. "
                            "Dus: sssst."
                        ),
                        "expects_response": False,
                    },
                    {
                        "content_plan": self.d.l2_slot(
                            "Tot de volgende keer, {first_name}.",
                            {"first_name": name},
                        ),
                        "expects_response": False,
                        "used_fields": {"name": name},
                    },
                ],
                "used_fields": {"name": name},
            },
        ]

        script_phase_names = {
            1: "part1_greeting",
            2: "part1_tutorial",
            3: "part1_leo_ministory",
            4: "part1_hobby_bridge",
            5: "part1_topic1",
            6: "part1_mistake1",
            7: "part1_topic2",
            8: "part1_mistake2",
            9: "part1_nudge",
            10: "part2_school_joke_transition",
            11: "part2_robot_school_self_disclosure",
            12: "part2_correct_fav_subject_connection",
            13: "part2_mistake3_school_strength",
            14: "part3_future_bridge",
            15: "part3_leo_self_disclosure",
            16: "part3_rolemodel_rapport",
            17: "part3_mistake4_aspiration_reflection",
            18: "part3_explicit_memory_inspection_review",
            19: "part3_closing",
        }
        for turn in script:
            phase = turn.get("phase")
            turn.setdefault("part", 1)
            turn.setdefault("phase_id", f"1.{phase}")
            turn.setdefault("script_phase", script_phase_names.get(phase, f"part1_phase{phase}"))
        return script
