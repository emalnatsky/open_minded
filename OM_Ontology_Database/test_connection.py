"""
Test script — Run this AFTER main.py is running.

Open a second terminal and run:
    python test_connection.py

This will:
1. Check if the API is running
2. Check if GraphDB is connected
3. Create a test child
4. Read the profile back
5. Update some fields
6. Read again to verify
7. Delete a field
8. Clean up

If all steps say OK, your setup is working!

Updated for UM v5.1 field names.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test(step_name, response):
    """Pretty-print test results."""
    status = "OK" if response.status_code in [200, 204] else "FAIL"
    print(f"\n{'='*50}")
    print(f"  [{status}] {step_name}")
    print(f"  Status: {response.status_code}")
    try:
        data = response.json()
        print(f"  Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception:
        print(f"  Response: {response.text[:200]}")
    print(f"{'='*50}")
    return status == "OK"


def run_tests():
    print("\n" + "#"*60)
    print("  Testing Open-Memory Robots API")
    print("#"*60)

    all_passed = True

    # 1. Health check
    r = requests.get(f"{BASE_URL}/")
    all_passed &= test("API Health Check", r)

    # 2. GraphDB connection
    r = requests.get(f"{BASE_URL}/health/graphdb")
    all_passed &= test("GraphDB Connection", r)

    # 3. Create a test child
    r = requests.post(f"{BASE_URL}/api/um/", json={
        "child_id": "test_child_001",
        "age": 9,
        "grade": "groep 7"
    })
    all_passed &= test("Create Child", r)

    # 4. Read the profile
    r = requests.get(f"{BASE_URL}/api/um/test_child_001")
    all_passed &= test("Read Profile", r)

    # 5. Update fields (simulating check-in answers)
    r = requests.post(f"{BASE_URL}/api/um/test_child_001/fields", json={
        "fields": {
            "hobbies": "voetbal en tekenen",
            "fav_food": "pizza",
            "sports_enjoys": "ja",
            "aspiration": "astronaut"
        },
        "source": "child_reported",
        "session_id": "checkin_test_001"
    })
    all_passed &= test("Update Fields", r)

    # 6. Read again — should now have the new fields
    r = requests.get(f"{BASE_URL}/api/um/test_child_001")
    all_passed &= test("Read Updated Profile", r)

    # 7. Get a specific field with provenance
    r = requests.get(f"{BASE_URL}/api/um/test_child_001/field/fav_food")
    all_passed &= test("Read Specific Field", r)

    # 8. Get the inspect view (categorized for GUI)
    r = requests.get(f"{BASE_URL}/api/um/test_child_001/inspect")
    all_passed &= test("Inspect Profile (GUI view)", r)

    # 9. Delete a field (child says "forget my favorite food")
    r = requests.delete(f"{BASE_URL}/api/um/test_child_001/field/fav_food")
    all_passed &= test("Delete Field", r)

    # 10. Verify deletion
    r = requests.get(f"{BASE_URL}/api/um/test_child_001")
    all_passed &= test("Verify Deletion", r)

    # 11. Get the schema
    r = requests.get(f"{BASE_URL}/api/schema")
    all_passed &= test("Get Schema", r)

    # 12. Clean up — delete the test child entirely
    r = requests.delete(f"{BASE_URL}/api/um/test_child_001")
    all_passed &= test("Delete Test Child (cleanup)", r)

    # Summary
    print("\n" + "#"*60)
    if all_passed:
        print("  ALL TESTS PASSED! Your setup is working.")
    else:
        print("  SOME TESTS FAILED. Check the output above.")
    print("#"*60 + "\n")


if __name__ == "__main__":
    run_tests()