import unittest

from test_cri_dialogue2 import make_app, sample_um


class RoleModelNormalizationTests(unittest.TestCase):
    def test_empty_meaning_role_model_values_use_no_role_model_branch(self):
        for empty_value in ["geen", "nee", "niemand", "weet ik niet", "n.v.t."]:
            with self.subTest(empty_value=empty_value):
                app = make_app()
                app.USE_FAKE_PERSONA_UM = False
                um = sample_um()
                um["role_model"] = empty_value
                app.pull_um = lambda um=um: um
                app.last_cri_scenario_loaded = True
                app.last_cri_scenario = {
                    "utterances": {
                        "p3_norolemodel_ack": {
                            "default": "[STUB] Dat snap ik. Soms leer je van verschillende mensen iets."
                        },
                    },
                    "mistakes": [],
                }

                script = app.build_script()
                phase33 = script[15]
                segments = phase33["segments"]

                self.assertEqual(phase33["phase_id"], "3.3")
                self.assertEqual(phase33["used_fields"], {})
                self.assertEqual(len(segments), 2)
                self.assertIn("is er niet echt een vaste persoon", app.turn_text(segments[0]))
                self.assertNotIn(empty_value, app.turn_text(segments[0]).lower())

    def test_memory_review_speaks_empty_meaning_role_model_naturally(self):
        app = make_app()
        um = sample_um()
        um["role_model"] = "niemand"
        app.last_um_preview = um

        segments, fields = app.script.memory_review_group_segments(um)
        future_segment = segments[-1]
        future_text = app.turn_text(future_segment)

        self.assertIn("role_model", fields)
        self.assertNotIn("niemand", future_text.lower())
        self.assertIn("niet echt een vaste persoon", future_text)
        self.assertIn("je later dierenarts wilt worden", future_text)
        self.assertEqual(
            future_segment["used_fields"],
            {"aspiration": "dierenarts worden", "role_model": "niemand"},
        )


if __name__ == "__main__":
    unittest.main()
