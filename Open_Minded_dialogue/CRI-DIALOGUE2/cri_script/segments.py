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

    def part1_topic_segments(self, topic: dict, include_close: bool = True) -> list:
        """Build a scenario-authored Part 1 topic as reusable segments."""
        domain = topic.get("domain")
        label = topic.get("label")
        if domain == "sport":
            sport = topic.get("current_values", {}).get("sports_fav_play") or label
            segments = [
                {
                    "content_plan": self.d.l2_pregen(
                        "sport_recall",
                        f"Ik weet ook nog dat jij zelf {sport} doet.",
                        ["sports_fav_play"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {"sports_fav_play": sport},
                },
                {
                    "content_plan": self.d.l2_pregen(
                        "sport_open",
                        f"In welke positie speel jij met {sport}?",
                        ["sports_fav_play"],
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                    "used_fields": {"sports_fav_play": sport},
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
                    "content_plan": self.d.l2_slot(
                        "Ik vraag me wel eens af of ik een goede sportrobot zou kunnen zijn, "
                        "maar ik val waarschijnlijk al om voor de warming-up. Wat vind jij zo leuk aan {sport}?",
                        {"sport": sport},
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {"sports_fav_play": sport},
                },
                {
                    "content_plan": self.d.sequence(
                        self.d.l1("Dat snap ik helemaal. Dat klinkt ook echt leuk."),
                        self.d.l2_pregen(
                            "sport_followup",
                            "Wat vind je daar meestal het fijnst aan: bezig zijn, beter worden, of samen met anderen iets doen?",
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
                    "content_plan": self.d.l2_pregen(
                        "music_open",
                        "Ik weet ook nog dat jij muziek leuk vindt. Wat vind jij daar het leukste aan?",
                        ["music_enjoys"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
                        "music_ack",
                        "Dat snap ik wel. Muziek kan echt iets bijzonders hebben.",
                        ["music_enjoys"],
                    ),
                    "expects_response": False,
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
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
                    "content_plan": self.d.l2_pregen(
                        "animals_open",
                        f"Ik weet ook nog dat {label} belangrijk voor jou is.",
                        ["pet_name", "pet_type", "animal_fav", "has_pet"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
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
                    "content_plan": self.d.l2_pregen(
                        "books_open",
                        f"Ik weet ook nog dat {label} bij jouw boekenwereld hoort.",
                        ["books_enjoys", "books_fav_title"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
                        "books_ack",
                        "Dat snap ik wel. Boeken kunnen echt leuk zijn.",
                        ["books_enjoys", "books_fav_title"],
                    ),
                    "expects_response": False,
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
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
                "content_plan": self.d.l2_pregen(
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
        return self.part1_topic_segments(topic, include_close=True)

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
            label = topic.get("label")
            return [
                {
                    "content_plan": self.d.l2_pregen(
                        "animals_open",
                        f"Ik weet ook nog dat {label} belangrijk voor jou is.",
                        ["pet_name", "pet_type", "animal_fav", "has_pet"],
                    ),
                    "expects_response": False,
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l2_pregen(
                        "animals_followup",
                        f"Wat vind jij zo leuk aan {label}?",
                        ["pet_name", "pet_type", "animal_fav"],
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": self.topic_used_fields(topic),
                },
                {
                    "content_plan": self.d.l1(self.topic2_closing_line("huisdier")),
                    "expects_response": False,
                    "used_fields": {},
                },
            ]

        if topic.get("domain") in ("sport", "muziek", "huisdier", "boeken"):
            return self.part1_topic_segments(topic, include_close=False) + [
                {
                    "content_plan": self.d.l1(self.topic2_closing_line(topic.get("domain"))),
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
