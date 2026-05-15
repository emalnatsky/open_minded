"""
Integration tests for CRI scenario endpoints.

Requires:
  - FastAPI server running: python main.py
  - GraphDB running with the open-memory-robots repository

Run: python -m pytest tests/test_cri_scenario.py -v

Tests the full flow:
  1. Create a test child with some UM data
  2. Generate a scenario with mistakes + utterances
  3. Read the scenario back
  4. Update a single utterance (swap stub for real text)
  5. Log interaction events
  6. Read events back
  7. Delete scenario
  8. Clean up
"""

import pytest
import requests

BASE = "http://localhost:8000"
TEST_CHILD = "test_cri_scenario_001"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def setup_and_teardown():
    """Create a test child with UM data before tests, clean up after."""
    # Clean up any previous test data
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}")

    # Create child
    requests.post(f"{BASE}/api/um/", json={
        "child_id": TEST_CHILD,
        "age": 10,
    })

    # Populate some UM fields so the scenario has real values to reference
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {
            "hobby_fav": "tekenen",
            "hobbies": "tekenen, hockey, bakken",
            "fav_food": "pannenkoeken",
            "aspiration": "dierenarts",
            "sports_enjoys": "ja",
            "sports_fav": "hockey",
        },
        "source": "test_setup",
    })

    # Write school_strength as a separate call (node field)
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"school_strength": "taal"},
        "source": "test_setup",
    })

    yield

    # Clean up
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}/scenario")
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}")


# ── Test: No scenario exists yet ──────────────────────────────────────────────

class TestScenarioNotExists:

    def test_get_scenario_returns_404_when_none_exists(self):
        # Delete any leftover scenario first
        requests.delete(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        assert r.status_code == 404


# ── Test: Create scenario ─────────────────────────────────────────────────────

class TestCreateScenario:

    def test_create_scenario_with_mistakes_and_utterances(self):
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/generate", json={
            "mistakes": [
                {
                    "id": "M1",
                    "target_field": "hobby_fav",
                    "wrong_value": "bakken",
                    "mistake_type": "related-but-wrong",
                    "spt_level": "orientation",
                    "step": 4,
                },
                {
                    "id": "M2",
                    "target_field": "fav_food",
                    "wrong_value": "pizza",
                    "mistake_type": "completely-wrong",
                    "spt_level": "orientation",
                    "step": 6,
                },
            ],
            "utterances": [
                {
                    "step_id": "hobby_bridge_reaction",
                    "layer": "L2-pregen",
                    "branch": "default",
                    "text": "[STUB] Tekenen, hockey en bakken - wat een leuke combinatie!",
                },
                {
                    "step_id": "m1_followup",
                    "layer": "L2-pregen",
                    "branch": "default",
                    "text": "[STUB] Wat vind jij het leukste aan bakken?",
                },
                {
                    "step_id": "m2_followup",
                    "layer": "L2-pregen",
                    "branch": "default",
                    "text": "[STUB] Pizza is best lekker. Wat vind jij daar het lekkerst aan?",
                },
            ],
        })
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["mistakes_created"] == 2
        assert data["utterances_created"] == 3


# ── Test: Read scenario ───────────────────────────────────────────────────────

class TestReadScenario:

    def test_get_scenario_returns_mistakes(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        assert r.status_code == 200
        scenario = r.json()["data"]["scenario"]

        mistakes = scenario["mistakes"]
        assert len(mistakes) == 2

        m1 = next(m for m in mistakes if m["id"] == "M1")
        assert m1["target_field"] == "hobby_fav"
        assert m1["wrong_value"] == "bakken"
        assert m1["mistake_type"] == "related-but-wrong"

        m2 = next(m for m in mistakes if m["id"] == "M2")
        assert m2["target_field"] == "fav_food"
        assert m2["wrong_value"] == "pizza"

    def test_get_scenario_returns_utterances(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        scenario = r.json()["data"]["scenario"]

        utterances = scenario["utterances"]
        assert "hobby_bridge_reaction" in utterances
        assert "m1_followup" in utterances
        assert "m2_followup" in utterances

        # Check the default branch exists
        assert "default" in utterances["hobby_bridge_reaction"]
        assert "[STUB]" in utterances["hobby_bridge_reaction"]["default"]

    def test_get_scenario_has_version_and_timestamp(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        scenario = r.json()["data"]["scenario"]
        assert scenario["version"] == "1.0"
        assert scenario["generated_at"] != "unknown"


# ── Test: Update utterance ────────────────────────────────────────────────────

class TestUpdateUtterance:

    def test_update_existing_utterance(self):
        """Overwrite a stub with real text."""
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/utterance", json={
            "step_id": "m1_followup",
            "layer": "L2-pregen",
            "branch": "default",
            "text": "Wat vind jij het allerleukste aan bakken? Ik maak zelf altijd een puinhoop.",
        })
        assert r.status_code == 200
        assert r.json()["data"]["updated"] is True

        # Verify the text was actually updated
        r2 = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        utterances = r2.json()["data"]["scenario"]["utterances"]
        assert "puinhoop" in utterances["m1_followup"]["default"]
        assert "[STUB]" not in utterances["m1_followup"]["default"]

    def test_add_new_branch_to_existing_step(self):
        """Add a corrected branch alongside the existing default."""
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/utterance", json={
            "step_id": "m1_followup",
            "layer": "L2-pregen",
            "branch": "corrected",
            "text": "Tekenen! Dat is echt creatief. Wat teken jij het liefst?",
        })
        assert r.status_code == 200

        # Both branches should now exist
        r2 = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        m1 = r2.json()["data"]["scenario"]["utterances"]["m1_followup"]
        assert "default" in m1
        assert "corrected" in m1

    def test_update_utterance_fails_without_scenario(self):
        """Can't add utterance if no scenario exists."""
        # Use a child with no scenario
        requests.post(f"{BASE}/api/um/", json={"child_id": "test_no_scenario_child"})
        r = requests.post(f"{BASE}/api/um/test_no_scenario_child/scenario/utterance", json={
            "step_id": "test", "layer": "L2-pregen", "branch": "default", "text": "test",
        })
        assert r.status_code == 404
        # Clean up
        requests.delete(f"{BASE}/api/um/test_no_scenario_child")


# ── Test: Interaction events ──────────────────────────────────────────────────

class TestInteractionEvents:

    def test_log_correction_event(self):
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/event", json={
            "event_type": "correction",
            "mistake_id": "M1",
            "field": "hobby_fav",
            "wrong_value": "bakken",
            "child_response": "Nee, tekenen!",
            "corrected": True,
            "phase": "early",
            "step": 4,
            "session_id": "cri_test_session",
        })
        assert r.status_code == 200
        assert r.json()["data"]["logged"] is True

    def test_log_no_correction_event(self):
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/event", json={
            "event_type": "no_correction",
            "mistake_id": "M2",
            "field": "fav_food",
            "wrong_value": "pizza",
            "corrected": False,
            "phase": "early",
            "step": 6,
            "session_id": "cri_test_session",
        })
        assert r.status_code == 200

    def test_log_nudge_event(self):
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/event", json={
            "event_type": "nudge_triggered",
            "corrected": False,
            "phase": "early",
            "step": 7,
            "session_id": "cri_test_session",
        })
        assert r.status_code == 200

    def test_get_events_returns_all_logged(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario/events")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["total"] >= 3  # the 3 events we just logged

        events = data["events"]
        event_types = [e["event_type"] for e in events]
        assert "correction" in event_types
        assert "no_correction" in event_types
        assert "nudge_triggered" in event_types

    def test_correction_event_has_child_response(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario/events")
        events = r.json()["data"]["events"]
        correction = next(e for e in events if e["event_type"] == "correction")
        assert correction["child_response"] == "Nee, tekenen!"
        assert correction["corrected"] is True
        assert correction["mistake_id"] == "M1"


# ── Test: Delete scenario ─────────────────────────────────────────────────────

class TestDeleteScenario:

    def test_delete_scenario(self):
        r = requests.delete(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        assert r.status_code == 200

    def test_get_after_delete_returns_404(self):
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        assert r.status_code == 404

    def test_events_gone_after_delete(self):
        """Events are part of the scenario — should be deleted too."""
        r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario/events")
        # Either 404 (no scenario) or empty events
        if r.status_code == 200:
            assert r.json()["data"]["total"] == 0

    def test_delete_nonexistent_returns_404(self):
        r = requests.delete(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        assert r.status_code == 404


# ── Test: Recreate after delete ───────────────────────────────────────────────

class TestRecreateScenario:

    def test_can_recreate_after_delete(self):
        """Verify scenarios can be regenerated cleanly after deletion."""
        r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/scenario/generate", json={
            "mistakes": [
                {
                    "id": "M1",
                    "target_field": "hobby_fav",
                    "wrong_value": "hockey",
                    "mistake_type": "related-but-wrong",
                    "spt_level": "orientation",
                    "step": 4,
                },
            ],
            "utterances": [
                {
                    "step_id": "m1_followup",
                    "layer": "L2-pregen",
                    "branch": "default",
                    "text": "New scenario text after recreation.",
                },
            ],
        })
        assert r.status_code == 200

        # Verify it's the new data, not leftover from before
        r2 = requests.get(f"{BASE}/api/um/{TEST_CHILD}/scenario")
        scenario = r2.json()["data"]["scenario"]
        assert len(scenario["mistakes"]) == 1
        assert scenario["mistakes"][0]["wrong_value"] == "hockey"
        assert "New scenario" in scenario["utterances"]["m1_followup"]["default"]