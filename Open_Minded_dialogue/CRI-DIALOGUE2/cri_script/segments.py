"""
Segments — topic 1 and topic 2 phase segments for the CRI script.

These are the multi-turn phase builders for the topics the script picks
based on the child's UM profile (sport/music/animals/books for Topic 1;
pet/food/general for Topic 2). Big blocks of structured content plans —
mostly L2-slot templates and L2-pregen utterances.

Pattern: constructed once in CRI_ScriptedDialogue.__init__ as self.segments.
The dialogue keeps thin pass-through wrappers so existing call sites
(self.topic1_phase_segments(topic), etc.) stay identical.
"""

import logging

logger = logging.getLogger(__name__)


class Segments:
    """Topic 1 and Topic 2 phase segment builders."""

    def __init__(self, dialogue):
        self.d = dialogue

    def topic_step_keys(self, generic_prefix: str | None, suffix: str, legacy_key: str) -> list:
        keys = []
        if generic_prefix:
            keys.append(f"{generic_prefix}_{suffix}")
        keys.append(legacy_key)
        return keys

    def topic_l2_pregen(
        self,
        generic_prefix: str | None,
        suffix: str,
        legacy_key: str,
        fallback: str,
        input_fields: list = None,
        require_input_values: bool = False,
    ) -> dict:
        return self.d.l2_pregen(
            self.topic_step_keys(generic_prefix, suffix, legacy_key),
            fallback,
            input_fields or [],
            require_input_values=require_input_values,
            topic_sensitive=True,
        )

    def topic_label(self, topic: dict) -> str:
        domain = topic.get("domain")
        current = topic.get("current_values") or {}
        if domain == "sport":
            return current.get("sports_fav_play") or topic.get("label") or "sport"
        if domain == "boeken":
            return current.get("books_fav_title") or topic.get("label") or "boeken"
        if domain == "muziek":
            return topic.get("label") or "muziek"
        if domain == "huisdier":
            return (
                current.get("pet_name")
                or current.get("animal_fav")
                or current.get("pet_type")
                or topic.get("label")
                or "dieren"
            )
        return topic.get("label") or "dat"

    def pet_subject_parts(self, topic: dict) -> tuple[str, str, str]:
        current = topic.get("current_values") or {}
        pet_name = str(current.get("pet_name") or "").strip()
        pet_type = str(current.get("pet_type") or "").strip().lower()
        animal = str(current.get("animal_fav") or "").strip().lower()
        label = str(topic.get("label") or "").strip()
        return pet_name, pet_type, animal or label.lower()

    def pet_open_fallback(self, topic: dict) -> str:
        pet_name, pet_type, animal = self.pet_subject_parts(topic)
        if pet_name and pet_type and pet_name.casefold() != pet_type.casefold():
            return f"Ik weet ook nog dat jij een {pet_type} hebt die {pet_name} heet."
        if pet_type:
            return f"Ik weet ook nog dat jij een {pet_type} als huisdier hebt."
        if pet_name:
            return f"Ik weet ook nog dat jouw huisdier {pet_name} heet."
        if animal:
            return f"Ik weet ook nog dat jij {animal} leuk vindt."
        return "Ik weet ook nog dat dieren iets zijn waar jij eerder over vertelde."

    def pet_followup_fallback(self, topic: dict) -> str:
        pet_name, pet_type, animal = self.pet_subject_parts(topic)
        if pet_name and pet_type and pet_name.casefold() != pet_type.casefold():
            return f"Wat voor {pet_type} is {pet_name} eigenlijk?"
        if pet_type:
            return f"Hoe gaat het met jouw {pet_type}? Doet die nog leuke dingen?"
        if pet_name:
            return f"Hoe gaat het eigenlijk met {pet_name}?"
        if animal:
            return f"Wat vind jij zo leuk aan {animal}?"
        return "Wat vind jij zo leuk aan dieren?"

    def sport_followup_prompt(self, sport: str) -> str:
        clean = str(sport or "").strip().lower()
        team_sports = ("voetbal", "hockey", "basketbal", "handbal", "korfbal", "volleybal")
        racket_sports = ("tennis", "padel", "badminton", "tafeltennis")
        speed_sports = ("schaatsen", "hardlopen", "zwemmen", "atletiek", "rennen")
        form_sports = ("judo", "karate", "turnen", "dans", "dansen", "gymnastiek")
        if any(word in clean for word in team_sports):
            return "Ben jij dan meer van snel rennen, goed overspelen, of juist lekker fanatiek meedoen?"
        if any(word in clean for word in racket_sports):
            return "Vind je vooral het slaan, het mikken, of het spannende spel tegen iemand anders leuk?"
        if any(word in clean for word in speed_sports):
            return "Vind je vooral snelheid leuk, oefenen om beter te worden, of dat je het op je eigen manier kunt doen?"
        if any(word in clean for word in form_sports):
            return "Vind je vooral nieuwe bewegingen leren, sterk worden, of dat je goed moet opletten leuk?"
        return "Wat vind je daar meestal het fijnst aan: bezig zijn, beter worden, of samen met anderen iets doen?"

    def neutral_topic_segments(self, topic: dict, include_close: bool) -> list:
        domain = topic.get("domain")
        prompts = {
            "sport": "Vind je bewegen vooral leuk als je het samen doet, of juist liever op je eigen manier?",
            "boeken": "Vind je verhalen meestal leuker als ze grappig zijn, spannend zijn, of juist ergens over gaan dat echt kan gebeuren?",
            "muziek": "Vind je muziek meestal fijner als die rustig is, vrolijk is, of als je mee kunt zingen?",
            "huisdier": "Vind je dieren vooral leuk om naar te kijken, om te verzorgen, of omdat ze soms grappige dingen doen?",
        }
        reflections = {
            "sport": "Dat snap ik. Soms is het ook fijn als iets gewoon bij je past zonder dat je er een precies woord voor hoeft te hebben.",
            "boeken": "Dat snap ik. Verhalen kunnen op heel verschillende manieren leuk zijn.",
            "muziek": "Dat snap ik. Muziek kan ook best afhangen van je bui.",
            "huisdier": "Dat snap ik. Over dieren kun je eigenlijk altijd wel iets vertellen.",
        }
        label = {
            "sport": "sport",
            "boeken": "boeken",
            "muziek": "muziek",
            "huisdier": "dieren",
        }.get(domain, topic.get("label") or "dit")
        segments = [
            {
                "content_plan": self.d.l1(f"Dan praat ik even wat algemener over {label}."),
                "expects_response": False,
                "used_fields": {},
            },
            {
                "content_plan": self.d.l1(prompts.get(domain, f"Wat vind jij daar meestal leuk aan?")),
                "expects_response": True,
                "response_mode": "acknowledge",
                "llm_turn": True,
                "used_fields": {},
                "l3": {
                    "script_phase": "part1_topic1",
                    "topic": domain or "topic",
                    "response_function": "acknowledge",
                    "question_allowed": False,
                    "local_context": f"Leo and the child are talking generally about {label}.",
                    "fallback": "Dat snap ik wel.",
                },
            },
            {
                "content_plan": self.d.l1(reflections.get(domain, "Dat snap ik.")),
                "expects_response": False,
                "used_fields": {},
            },
        ]
        return self.with_optional_close(segments, include_close)

    def scenario_has_any_steps(self, step_ids: list[str]) -> bool:
        utterances = self.d.cri_scenario().get("utterances") or {}
        if not isinstance(utterances, dict):
            return False
        return any(step_id in utterances for step_id in step_ids)

    def generic_topic_segments(self, topic: dict, generic_prefix: str, include_close: bool) -> list:
        if topic.get("neutral_after_correction"):
            return self.neutral_topic_segments(topic, include_close)

        domain = topic.get("domain")
        label = self.topic_label(topic)
        input_fields = list((topic.get("current_values") or {}).keys())
        if domain == "sport":
            recall_fallback = f"Ik weet ook nog dat jij zelf {label} doet."
            open_fallback = f"In welke positie speel jij met {label}?"
            question_fallback = (
                "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                f"maar ik val waarschijnlijk al om voor de warming-up. Wat vind jij zo leuk aan {label}?"
            )
            followup_fallback = self.sport_followup_prompt(label)
            value_fields = ["sports_fav_play"]
        elif domain == "boeken":
            recall_fallback = f"Ik weet ook nog dat {label} bij jouw boekenwereld hoort."
            open_fallback = f"Wat vind jij leuk aan {label}?"
            question_fallback = "Dat snap ik wel. Boeken kunnen echt leuk zijn."
            followup_fallback = f"Wat vind jij het leukste aan {label}?"
            value_fields = ["books_fav_title"]
        elif domain == "muziek":
            recall_fallback = "Ik weet ook nog dat jij muziek leuk vindt."
            open_fallback = "Wat vind jij daar het leukste aan?"
            question_fallback = "Dat snap ik wel. Muziek kan echt iets bijzonders hebben."
            followup_fallback = "Luister je dan meer naar rustige muziek of juist naar vrolijke muziek?"
            value_fields = ["music_enjoys"]
        elif domain == "huisdier":
            recall_fallback = f"Ik weet ook nog dat {label} belangrijk voor jou is."
            open_fallback = f"Wat vind jij zo leuk aan {label}?"
            question_fallback = "Dat snap ik wel. Over dieren kun je echt veel vertellen."
            followup_fallback = "Vind je dieren vooral leuk om naar te kijken, om te verzorgen, of omdat ze soms grappige dingen doen?"
            value_fields = ["pet_name", "pet_type", "animal_fav"]
        else:
            recall_fallback = f"Ik weet ook nog dat {label} bij jou hoort."
            open_fallback = f"Vertel eens, wat vind jij leuk aan {label}?"
            question_fallback = "Wat vind jij daar meestal het fijnst aan?"
            followup_fallback = "Wat zou je daar nog meer over kunnen vertellen?"
            value_fields = input_fields
        pregen_fields = [
            field for field in value_fields
            if self.d.is_known((topic.get("current_values") or {}).get(field))
        ] or input_fields
        segments = []
        include_recall = generic_prefix != "p1_t2" or self.scenario_has_any_steps([f"{generic_prefix}_recall"])
        if include_recall:
            segments.append({
                "content_plan": self.d.l2_pregen(
                    f"{generic_prefix}_recall",
                    recall_fallback,
                    pregen_fields,
                    require_input_values=bool(pregen_fields),
                    topic_sensitive=True,
                ),
                "expects_response": True,
                "response_mode": "topic_interpretation",
                "used_fields": self.topic_used_fields(topic),
            })

        open_expects_response = generic_prefix != "p1_t2"
        open_segment = {
            "content_plan": self.d.l2_pregen(
                f"{generic_prefix}_open",
                open_fallback,
                pregen_fields,
                topic_sensitive=True,
            ),
            "expects_response": open_expects_response,
            "used_fields": {} if generic_prefix == "p1_t1" else self.topic_used_fields(topic),
        }
        if open_expects_response:
            open_segment.update({
                "response_mode": "acknowledge",
                "llm_turn": True,
            })
        segments.append(open_segment)

        if generic_prefix == "p1_t1":
            segments.append({
                "content_plan": self.d.l2_pregen(
                    f"{generic_prefix}_question",
                    question_fallback,
                    pregen_fields,
                    require_input_values=domain == "sport",
                    topic_sensitive=True,
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": {},
            })

        segments.append({
            "content_plan": self.d.l2_pregen(
                f"{generic_prefix}_followup",
                followup_fallback,
                pregen_fields,
                topic_sensitive=True,
            ),
            "expects_response": True,
            "response_mode": "listen_only",
            "used_fields": {},
        })

        if include_close:
            if generic_prefix == "p1_t2":
                segments.append({
                    "content_plan": self.d.l2_pregen(
                        "p1_t2_close",
                        self.topic2_closing_line(topic.get("domain")),
                        topic_sensitive=True,
                    ),
                    "expects_response": False,
                    "used_fields": {},
                })
            else:
                segments.append({
                    "content_plan": self.d.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                    "expects_response": False,
                    "used_fields": {},
                })
        return segments

    def part1_topic_segments(
        self,
        topic: dict,
        include_close: bool = True,
        generic_prefix: str | None = None,
    ) -> list:
        """Build a scenario-authored Part 1 topic as reusable segments."""
        if generic_prefix == "p1_t1" and self.scenario_has_any_steps([
            "p1_t1_recall",
            "p1_t1_open",
            "p1_t1_question",
            "p1_t1_followup",
        ]):
            return self.generic_topic_segments(topic, "p1_t1", include_close)

        domain = topic.get("domain")
        label = self.topic_label(topic)
        if topic.get("neutral_after_correction"):
            return self.neutral_topic_segments(topic, include_close)
        if domain == "sport":
            sport = topic.get("current_values", {}).get("sports_fav_play") or label
            segments = [
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "recall",
                        "sport_recall",
                        f"Ik weet ook nog dat jij zelf {sport} doet.",
                        ["sports_fav_play"],
                        require_input_values=True,
                    ),
                    "expects_response": True,
                    "response_mode": "topic_interpretation",
                    "used_fields": {"sports_fav_play": sport},
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "open",
                        "sport_open",
                        f"In welke positie speel jij met {sport}?",
                        ["sports_fav_play"],
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                    "used_fields": {},
                    "l3": {
                        "script_phase": "part1_topic1",
                        "topic": "sport",
                        "response_function": "acknowledge",
                        "question_allowed": False,
                        "relevant_um_fields": ["sports_fav_play"],
                        "local_context": f"Leo and the child are talking about {sport}.",
                        "fallback": "Dat snap ik wel. Sport kan best snel gaan.",
                    },
                },
                {
                    "content_plan": self.d.l2_pregen(
                        self.topic_step_keys(generic_prefix, "question", "sport_question"),
                        (
                            "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                            f"maar ik val waarschijnlijk al om voor de warming-up. Wat vind jij zo leuk aan {sport}?"
                        ),
                        ["sports_fav_play"],
                        require_input_values=True,
                        topic_sensitive=True,
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {},
                },
                {
                    "content_plan": self.d.sequence(
                        self.d.l1("Dat snap ik helemaal. Dat klinkt ook echt leuk."),
                        self.topic_l2_pregen(
                            generic_prefix,
                            "followup",
                            "sport_followup",
                            self.sport_followup_prompt(sport),
                            ["sports_fav_play"],
                        ),
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {},
                },
            ]
            return self.with_optional_close(segments, include_close)

        if domain == "muziek":
            segments = [
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "open",
                        "music_open",
                        "Ik weet ook nog dat jij muziek leuk vindt. Wat vind jij daar het leukste aan?",
                        ["music_enjoys"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "question",
                        "music_ack",
                        "Dat snap ik wel. Muziek kan echt iets bijzonders hebben.",
                        ["music_enjoys"],
                    ),
                    "expects_response": False,
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "followup",
                        "music_followup",
                        "Luister je dan meer naar rustige muziek of juist naar vrolijke muziek?",
                        ["music_enjoys"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
            ]
            return self.with_optional_close(segments, include_close)

        if domain == "huisdier":
            segments = [
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "open",
                        "animals_open",
                        f"Ik weet ook nog dat {label} belangrijk voor jou is.",
                        ["pet_name", "pet_type", "animal_fav", "has_pet"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "followup",
                        "animals_followup",
                        f"Wat vind jij zo leuk aan {label}?",
                        ["pet_name", "pet_type", "animal_fav"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
            ]
            return self.with_optional_close(segments, include_close)

        if domain == "boeken":
            segments = [
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "open",
                        "books_open",
                        f"Ik weet ook nog dat {label} bij jouw boekenwereld hoort.",
                        ["books_enjoys", "books_fav_title"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "question",
                        "books_ack",
                        "Dat snap ik wel. Boeken kunnen echt leuk zijn.",
                        ["books_enjoys", "books_fav_title"],
                    ),
                    "expects_response": False,
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.topic_l2_pregen(
                        generic_prefix,
                        "followup",
                        "books_followup",
                        f"Wat vind jij het leukste aan {label}?",
                        ["books_enjoys", "books_fav_title"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
            ]
            return self.with_optional_close(segments, include_close)

        topic_fields = self.topic_label_fields(topic)
        segments = [
            {
                "content_plan": self.d.l2_slot(
                    "Ik weet ook nog dat {topic} iets is waar jij eerder over vertelde.",
                    {"topic": label},
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": {field: topic.get("current_values", {}).get(field) for field in topic_fields},
            },
            {
                "content_plan": self.topic_l2_pregen(
                    generic_prefix,
                    "followup",
                    "topic1_followup",
                    f"Wat vind jij zo leuk aan {label}?",
                    list((topic.get("current_values") or {}).keys()),
                ),
                "expects_response": True,
                "response_mode": "acknowledge",
                "llm_turn": True,
                "used_fields": {field: topic.get("current_values", {}).get(field) for field in topic_fields},
            },
            {
                "content_plan": self.d.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                "expects_response": False,
                "used_fields": {},
            },
        ]
        return segments if include_close else segments[:-1]

    def topic1_phase_segments(self, topic: dict) -> list:
        """Build the multi-turn Topic 1 phase from the Part 1 script."""
        return self.part1_topic_segments(topic, include_close=True, generic_prefix="p1_t1")

    def with_optional_close(self, segments: list, include_close: bool) -> list:
        if include_close:
            return segments + [
                {
                    "content_plan": self.d.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                    "expects_response": False,
                    "used_fields": {},
                }
            ]
        return segments

    def topic_used_fields(self, topic: dict) -> dict:
        return {
            field: value
            for field, value in (topic.get("current_values") or {}).items()
            if self.d.is_known(value)
        }


    def topic_label_fields(self, topic: dict) -> list:
        """Best-effort field list for the concrete topic label Leo says aloud."""
        current = topic.get("current_values", {}) or {}
        for field in topic.get("fields", []) or []:
            if field in current and self.d.is_known(current.get(field)):
                return [field]
        return []


    def topic2_phase_segments(self, topic: dict) -> list:
        """Build the correct re-ground topic before M2."""
        if topic.get("domain") == "huisdier":
            return self.pet_topic2_segments(topic)

        if self.scenario_has_any_steps([
            "p1_t2_recall",
            "p1_t2_open",
            "p1_t2_followup",
            "p1_t2_close",
        ]):
            return self.generic_topic_segments(topic, "p1_t2", include_close=True)

        if topic.get("domain") in ("sport", "muziek", "huisdier", "boeken"):
            return self.part1_topic_segments(topic, include_close=False, generic_prefix="p1_t2") + [
                {
                    "content_plan": self.d.l2_pregen(
                        "p1_t2_close",
                        self.topic2_closing_line(topic.get("domain")),
                        topic_sensitive=True,
                    ),
                    "expects_response": False,
                    "used_fields": {},
                }
            ]

        label = topic.get("label")

        topic_fields = self.topic_label_fields(topic)
        return [
            {
                "content_plan": self.d.l2_slot(
                    "Ik weet ook nog dat {topic} bij jou hoort.",
                    {"topic": label},
                ),
                "expects_response": True,
                "response_mode": "acknowledge",
                "llm_turn": True,
                "used_fields": {field: topic.get("current_values", {}).get(field) for field in topic_fields},
            }
        ]

    def pet_topic2_segments(self, topic: dict) -> list:
        segments = []
        if self.scenario_has_any_steps(["p1_t2_recall"]):
            segments.append({
                "content_plan": self.d.l2_pregen(
                    "p1_t2_recall",
                    self.pet_open_fallback(topic),
                    ["pet_name", "pet_type", "animal_fav", "has_pet"],
                    topic_sensitive=True,
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": self.topic_used_fields(topic),
            })

        segments.extend([
            {
                "content_plan": self.topic_l2_pregen(
                    "p1_t2",
                    "open",
                    "animals_open",
                    self.pet_open_fallback(topic),
                    ["pet_name", "pet_type", "animal_fav", "has_pet"],
                ),
                "expects_response": False,
                "used_fields": self.topic_used_fields(topic),
            },
            {
                "content_plan": self.topic_l2_pregen(
                    "p1_t2",
                    "followup",
                    "animals_followup",
                    self.pet_followup_fallback(topic),
                    ["pet_name", "pet_type", "animal_fav"],
                ),
                "expects_response": True,
                "response_mode": "listen_only",
                "used_fields": {},
            },
            {
                "content_plan": self.d.l2_pregen(
                    "p1_t2_close",
                    self.topic2_closing_line("huisdier"),
                    topic_sensitive=True,
                ),
                "expects_response": False,
                "used_fields": {},
            },
        ])
        return segments

    def topic2_closing_line(self, domain: str) -> str:
        """Topic-specific L1 comfort line before Mistake 2."""
        if domain == "huisdier":
            return (
                "Bizar leuk. Ik vind dieren altijd fascinerend. "
                "Ze doen vaak alsof zij precies weten wat er aan de hand is, en ik net niet."
            )
        if domain == "boeken":
            return (
                "Bizar leuk. Verhalen kunnen echt zo'n eigen wereld maken. "
                "Soms vergeet ik bijna dat ik gewoon in een klaslokaal sta."
            )
        if domain == "muziek":
            return (
                "Bizar leuk. Muziek blijft voor mij bijzonder. "
                "Het is alsof geluid ineens een gevoel krijgt."
            )
        if domain == "sport":
            return (
                "Bizar leuk. Sport blijft voor mij als robot best indrukwekkend. "
                "Jullie bewegen zo soepel; ik moet daar nog hard op oefenen."
            )
        return "Bizar leuk. Dat vind ik echt fijn om te horen."
