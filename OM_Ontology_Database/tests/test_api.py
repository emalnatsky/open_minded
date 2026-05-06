"""
API integration tests.

Requires:
  - The FastAPI server running: python main.py
  - GraphDB running with the open-memory-robots repository
  - SKIP_API_KEY_CHECK = True in config.py (or set X-API-Key header)

Run: python -m pytest tests/test_api.py -v

Updated for UM v5.1 field names.
"""

import pytest
import requests

BASE = "http://localhost:8000"


# ── Fixtures ──────────────────────────────────────────────────────────────────

TEST_CHILD = "test_child_pytest_001"

@pytest.fixture(autouse=True)
def cleanup():
    """Delete the test child before and after each test."""
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}")
    yield
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}")


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_graphdb_connected():
    r = requests.get(f"{BASE}/health/graphdb")
    assert r.status_code == 200
    assert r.json()["status"] == "connected"


# ── Schema ────────────────────────────────────────────────────────────────────

def test_schema_returns_known_fields():
    r = requests.get(f"{BASE}/api/schema")
    assert r.status_code == 200
    fields = r.json()["data"]["fields"]
    assert "hobbies" in fields
    assert "fav_food" in fields
    assert "aspiration" in fields


# ── Create ────────────────────────────────────────────────────────────────────

def test_create_child():
    r = requests.post(f"{BASE}/api/um/", json={
        "child_id": TEST_CHILD,
        "age": 9,
        "grade": "groep 7",
    })
    assert r.status_code == 200
    assert r.json()["data"]["child_id"] == TEST_CHILD


def test_create_duplicate_returns_409():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    r = requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    assert r.status_code == 409


def test_create_invalid_age_returns_422():
    r = requests.post(f"{BASE}/api/um/", json={
        "child_id": TEST_CHILD,
        "age": 99,    # out of range 6-13
    })
    assert r.status_code == 422


# ── Read ──────────────────────────────────────────────────────────────────────

def test_read_profile():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD, "age": 10})
    r = requests.get(f"{BASE}/api/um/{TEST_CHILD}")
    assert r.status_code == 200
    profile = r.json()["data"]["profile"]
    assert "scalars" in profile


def test_read_nonexistent_child_returns_404():
    r = requests.get(f"{BASE}/api/um/does_not_exist_xyz")
    assert r.status_code == 404


def test_read_by_category():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "pizza"},
        "source": "test",
    })
    r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/category/eten")
    assert r.status_code == 200
    assert "scalars" in r.json()["data"]


# ── Update ────────────────────────────────────────────────────────────────────

def test_update_scalar_field():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "pizza"},
        "source": "child_reported",
    })
    assert r.status_code == 200
    assert len(r.json()["data"]["written"]) == 1


def test_update_node_field():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"hobbies": "voetbal"},
        "source": "child_reported",
    })
    assert r.status_code == 200
    written = r.json()["data"]["written"]
    assert any(w["field"] == "hobbies" for w in written)


def test_update_unknown_field_is_skipped():
    """Unknown fields must not reach SPARQL — they are skipped with an error."""
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    r = requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"this_field_does_not_exist": "some_value"},
    })
    assert r.status_code == 200
    skipped = r.json()["data"]["skipped"]
    assert len(skipped) == 1
    assert "this_field_does_not_exist" in skipped[0]["field"]


def test_update_preserves_history():
    """Writing the same field twice should create a HistoryEntry."""
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "pizza"}, "source": "test",
    })
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "sushi"}, "source": "test",
    })
    r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/history/fav_food")
    history = r.json()["data"]["history"]
    assert len(history) >= 1
    assert history[0]["old_value"] == "pizza"
    assert history[0]["new_value"] == "sushi"


def test_dry_run_does_not_write():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "pizza"},
        "source": "test",
        "dry_run": True,
    })
    # Field should not be present since it was a dry run
    r = requests.get(f"{BASE}/api/um/{TEST_CHILD}/field/fav_food")
    assert r.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_field():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD})
    requests.post(f"{BASE}/api/um/{TEST_CHILD}/fields", json={
        "fields": {"fav_food": "pizza"}, "source": "test",
    })
    r = requests.delete(f"{BASE}/api/um/{TEST_CHILD}/field/fav_food")
    assert r.status_code == 200

    # Confirm it's gone
    r2 = requests.get(f"{BASE}/api/um/{TEST_CHILD}/field/fav_food")
    assert r2.status_code == 404


def test_delete_child_removes_all_data():
    requests.post(f"{BASE}/api/um/", json={"child_id": TEST_CHILD, "age": 9})
    requests.delete(f"{BASE}/api/um/{TEST_CHILD}")
    r = requests.get(f"{BASE}/api/um/{TEST_CHILD}")
    assert r.status_code == 404