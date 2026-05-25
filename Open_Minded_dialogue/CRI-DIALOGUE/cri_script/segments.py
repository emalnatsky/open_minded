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

    def topic1_phase_segments(self, topic: dict) -> list:
        """Build the multi-turn Topic 1 phase from the Part 1 script."""
        domain = topic.get("domain")
        label = topic.get("label")
        if domain == "sport":
            sport = topic.get("current_values", {}).get("sports_fav_play") or label
            return [
                {
                    "content_plan": self.d.l2_slot(
                        "Ik weet ook nog dat jij aan {sport} doet.",
                        {"sport": sport},
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {"sports_fav_play": sport},
                },
                {
                    "content_plan": self.d.l2_slot(
                        "Waarom zit je op {sport}?",
                        {"sport": sport},
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                    "used_fields": {"sports_fav_play": sport},
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
                            "sport_followup_choice",
                            "Wat vind je daar meestal het fijnst aan: bezig zijn, beter worden, of samen met anderen iets doen?",
                            ["sports_fav_play"],
                        ),
                    ),
                    "expects_response": True,
                    "response_mode": "listen_only",
                    "used_fields": {},
                },
                {
                    "content_plan": self.d.l1("Maar jij doet natuurlijk nog meer leuke dingen."),
                    "expects_response": False,
                    "used_fields": {},
                },
            ]

        topic_fields = self.topic_label_fields(topic)
        return [
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


    def topic_label_fields(self, topic: dict) -> list:
        """Best-effort field list for the concrete topic label Leo says aloud."""
        current = topic.get("current_values", {}) or {}
        for field in topic.get("fields", []) or []:
            if field in current and self.d.is_known(current.get(field)):
                return [field]
        return []


    def topic2_phase_segments(self, topic: dict) -> list:
        """Build the correct re-ground topic before M2."""
        label = topic.get("label")
        if topic.get("domain") == "huisdier":
            pet_name = topic.get("current_values", {}).get("pet_name") or label
            return [
                {
                    "content_plan": self.d.sequence(
                        self.d.l2_slot(
                            "Ik weet ook nog dat jij een huisdier hebt die {pet_name} heet.",
                            {"pet_name": pet_name},
                        ),
                        self.d.l2_pregen(
                            "pet_name_reaction",
                            f"Dat vind ik echt een mooie naam. {pet_name} klinkt alsof er stiekem belangrijke plannen worden gemaakt als niemand kijkt.",
                            ["pet_name"],
                        ),
                    ),
                    "expects_response": False,
                    "used_fields": {"pet_name": pet_name},
                },
                {
                    "content_plan": self.d.l2_pregen(
                        "pet_kind_question",
                        f"Wat voor huisdier is {pet_name} eigenlijk?",
                        ["pet_name", "pet_type"],
                    ),
                    "expects_response": True,
                    "response_mode": "acknowledge",
                    "llm_turn": True,
                    "used_fields": {"pet_name": pet_name},
                },
                {
                    "content_plan": self.d.l1(
                        "Bizar leuk. Ik vind dieren altijd fascinerend. "
                        "Ze doen vaak alsof zij precies weten wat er aan de hand is, en ik net niet."
                    ),
                    "expects_response": False,
                    "used_fields": {},
                },
            ]

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

