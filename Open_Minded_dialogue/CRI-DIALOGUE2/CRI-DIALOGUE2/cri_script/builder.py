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

    def build_script(self) -> list:
        """
        Mila Part 1 walkthrough flow.

        This follows the content-layer reference directly:
        L1 scripted text, L2-slot UM templates, L2-pregen validated stored
        utterances, and L3 runtime branches after unpredictable child input.
        """
        um = self.d.pull_um()
        self.d.alert_condition_mismatch(um)
        name = self.d.child_display_name(um)
        first_topic = self.d.select_part1_topic1(um)
        second_topic = self.d.select_part1_topic2(um, first_topic)

        m1_plan = self.d.script_plan_mistake("M1")
        m1_topic = self.d.hobby_mistake_topic(um)
        m1_field = m1_plan.get("field") or "hobby_fav"
        m1_actual = m1_topic.get("current_values", {}).get(m1_field) or m1_topic.get("label")
        m1_wrong = m1_plan.get("wrong_value") or self.d.related_wrong_hobby_value(um)
        m1_type = m1_plan.get("type") or "related-but-wrong"

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
        general_topic = self.d.general_memory_topic(um)
        story_activity = self.d.preferred_story_activity(um)
        story_activity_spoken = story_activity[:1].upper() + story_activity[1:] if story_activity else "Iets nieuws proberen"
        story_is_baking = story_activity.strip().lower() in ("bakken", "koken", "taarten bakken")
        story_problem = "een klein deegdrama" if story_is_baking else "een klein robotdrama"
        story_question = "Heb jij eigenlijk ooit een lama zien eten?" if story_is_baking else "Heb jij eigenlijk ooit een lama iets geks zien doen?"
        hobbies = self.d.format_dutch_list(self.d.all_hobbies(um), "leuke dingen")

        self.d.logger.info(
            "UM pulled for Part 1 phase flow - child:%s topic1:%s topic2:%s m1:%s m2:%s",
            name,
            first_topic["domain"],
            second_topic["domain"],
            m1_field,
            m2_field,
        )

        return [
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
                        "Als je antwoord geeft, doe dat dan luid en duidelijk."
                    ),
                    self.d.l1("Vandaag ga ik mijn geheugen best veel gebruiken. Je mag altijd vragen wat ik over jou onthoud."),
                    self.d.l1(
                        "Ik probeer alles netjes op de goede plek te bewaren, maar soms gaat dat nog een beetje robotachtig mis. "
                        "Dus als iets niet klopt, of als jij iets wilt veranderen, mag je dat gewoon zeggen."
                    ),
                    self.d.l1("Goed, dan gaan we beginnen."),
                ),
                "tutorial_condition": self.d.tutorial_condition(um),
                "expects_response": False,
                "follow_up": "",
                "llm_turn": False,
            },
            {
                "phase": 3,
                "name": "Leo mini-story",
                "layer": "L1+L3",
                "dialogue_case": self.d.CASE_PREAUTHORED_POOL,
                "segments": [
                    {
                        "content_plan": self.d.l1(
                            f"Weet je wat ik laatst weer probeerde? {story_activity_spoken}. "
                            f"Dat klinkt heel indrukwekkend, maar eerlijk gezegd was het meer {story_problem}. "
                            f"Mijn lama-vrienden vonden het wel een succes, want die eten bijna alles op. {story_question}"
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l1(
                            "Ik vind het gewoon leuk om nieuwe dingen uit te proberen, ook als het een beetje mislukt. "
                            "Doe jij dat ook wel eens?"
                        ),
                        "expects_response": True,
                        "response_mode": "listen_only",
                    },
                    {
                        "content_plan": self.d.l1(
                            "Dat snap ik wel. Nieuwe dingen proberen kan leuk zijn, maar soms ook een beetje spannend."
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
                    self.d.l2_pregen(
                        "hobbies_bridge",
                        "Dat vind ik echt een gezellige combinatie. Daar zit van alles in: bewegen, bedenken en iets maken.",
                        ["hobbies"],
                    ),
                ),
                "expects_response": True,
                "response_mode": "listen_only",
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
                    },
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1("Dat snap ik trouwens wel."),
                            self.d.l2_pregen(
                                "m1_wrong_followup",
                                f"Iets maken met {m1_wrong} klinkt best indrukwekkend. Wat vind jij daar zo leuk aan?",
                                [m1_field],
                            ),
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
                    },
                    {
                        "content_plan": self.d.sequence(
                            self.d.l1("Dat is op zich wel een sterke keuze."),
                            self.d.l2_pregen(
                                "m2_wrong_followup",
                                f"Rond, warm, handig. Wat vind jij daar eigenlijk zo lekker aan?",
                                [m2_field],
                            ),
                        ),
                        "expects_response": True,
                        "response_mode": "mistake_interpretation",
                        "skip_if_phase_confirmed_change": True,
                    },
                ],
                "used_fields": {m2_field: m2_wrong},
                "example_child": f"Nee, {m2_wrong} klopt niet.",
                "example_leo_after": "Oeps, dan had ik dat verkeerd. Wat moet ik daarover onthouden?",
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
                "response_mode": "topic_interpretation",
                "topic": general_topic,
                "memory_link": general_topic["memory_link"],
                "llm_turn": True,
                "example_child": "Nee, er klopte iets niet.",
                "example_leo_after": "Oeps. Wil je zeggen wat er niet klopte?",
            },
        ]

