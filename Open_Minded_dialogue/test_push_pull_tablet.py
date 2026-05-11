"""
TABLET TEST!!!!
Integration test that simulates a CRI conversation turn where:
  1. PULL : reads hobby_fav from Eunike's API (should be "snorkelen")
  2. SIMULATE: child corrects the robot ("Nee, ik doe aan bungeejumpen!")
  3. PUSH : writes the corrected value back to Eunike's API
  4. VERIFY:  reads hobby_fav again to confirm the update stuck
"""

import requests
import json
import time

BASE_URL  = "http://localhost:8000"
CHILD_ID  = "Julianna_dutch"
FIELD     = "hobby_fav"
OLD_VALUE = "snorkelen"
NEW_VALUE = "bungeejumpen"

def sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):  print(f"  yaaaas  {msg}")
def fail(msg): print(f"  nooooo  {msg}")
def info(msg): print(f"  →   {msg}")

# --------------------------------------------------------------------------- PULL: read current hobby_fav------------------------------------------------------------------
sep(" PULL: read current hobby_fav")

r = requests.get(f"{BASE_URL}/api/um/{CHILD_ID}/field/{FIELD}", timeout=5)

if r.status_code == 200:
    current = r.json().get("data", {}).get("value", "")
    ok(f"hobby_fav pulled successfully: '{current}'")
    if current == OLD_VALUE:
        ok(f"Value matches expected '{OLD_VALUE}' ✓")
    else:
        info(f"Value is '{current}' (expected '{OLD_VALUE}') — will still test push")
elif r.status_code == 404:
    fail(f"Field '{FIELD}' not found for child '{CHILD_ID}'")
    info("Make sure Eunike has populated the dummy child with hobby_fav=snorkelen")
    exit(1)
else:
    fail(f"Unexpected status {r.status_code}: {r.text}")
    exit(1)

# ----------------------------------------------------------------SIMULATE: robot mentions wrong hobby, child corrects--------------------------------------------------------
sep("SIMULATE: dialogue turn")

print(f"""
  LEO  [L2 template]:  "Ik weet dat jij graag aan {OLD_VALUE} doet. Klopt dat?"

  KIND [corrects]:     "Nee! Ik doe aan {NEW_VALUE}!"

  INTENT CLASSIFIER returns:
    intent: um_update
    field:  hobby_fav
    value:  {NEW_VALUE}
    confidence: 0.95

  LEO  [handle_intent → um_update]:
    "Oh, je hebt gelijk! Ik pas het aan."
    "Dus jouw hobby is {NEW_VALUE}, niet {OLD_VALUE}. Leuk!"
""")

info("Simulating handle_intent() → um_update → POST to Eunike's API ...")
time.sleep(1)

# ----------------------------------------------------------------------PUSH: write corrected value to Eunike's API-------------------------------------------------------------------------
sep("STEP 3 — PUSH: write hobby_fav = bungeejumpen")

payload = {
    "fields": {FIELD: NEW_VALUE},
    "source": "child_correction",
    "session_id": "test_push_pull_001"
}

r = requests.post(
    f"{BASE_URL}/api/um/{CHILD_ID}/fields",
    json=payload,
    timeout=5
)

if r.status_code == 200:
    data    = r.json().get("data", {})
    written = data.get("written", [])
    skipped = data.get("skipped", [])
    if any(w.get("field") == FIELD for w in written):
        ok(f"hobby_fav written successfully → '{NEW_VALUE}'")
    elif skipped:
        fail(f"Field was skipped: {skipped}")
        info("Check Eunike's validation — value might have failed schema check")
    else:
        info(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
else:
    fail(f"POST failed with status {r.status_code}: {r.text}")
    exit(1)

# ---------------------------------------------------------------VERIFY: read hobby_fav again and confirm it changed -----------------------------------------------------------------
sep("VERIFY: read hobby_fav again")

time.sleep(0.5)  

r = requests.get(f"{BASE_URL}/api/um/{CHILD_ID}/field/{FIELD}", timeout=5)

if r.status_code == 200:
    updated = r.json().get("data", {}).get("value", "")
    if updated == NEW_VALUE:
        ok(f"hobby_fav is now '{updated}' ✓")
        ok("PUSH + PULL working correctly!")
    else:
        fail(f"hobby_fav is '{updated}' — expected '{NEW_VALUE}'")
        info("The write may have been rejected by Eunike's validation")
else:
    fail(f"GET failed with status {r.status_code}: {r.text}")

# ---------------------------------------------------------------------------VERIFY via /inspect (what tablet sees)------------------------------------------------------------------
sep("VERIFY via /inspect (tablet view)")

r = requests.get(f"{BASE_URL}/api/um/{CHILD_ID}/inspect", timeout=5)

if r.status_code == 200:
    categories = r.json().get("data", {}).get("categories", {})
    hobby_cat  = categories.get("hobby", {})
    scalars    = hobby_cat.get("scalars", {})
    hobby_val  = scalars.get(FIELD, {}).get("value", "NOT FOUND")

    if hobby_val == NEW_VALUE:
        ok(f"Tablet /inspect shows hobby_fav = '{hobby_val}' ✓")
        ok("Tablet will display the updated value on next poll!")
    else:
        fail(f"Tablet /inspect shows hobby_fav = '{hobby_val}' (expected '{NEW_VALUE}')")
else:
    fail(f"/inspect failed with status {r.status_code}")

# ------------------------------------------------------------------------RESTORE original value (so test is repeatable)---------------------------------------------------------------------
sep("RESTORE: reset hobby_fav back to snorkelen")

print("  Waiting 20 seconds so you can see the tablet update...")
time.sleep(20)  # change it if you feel like its too much i just needed to clearly see the tablet change without refresh

r = requests.post(
    f"{BASE_URL}/api/um/{CHILD_ID}/fields",
    json={"fields": {FIELD: OLD_VALUE}, "source": "test_restore"},
    timeout=5
)

if r.status_code == 200:
    ok(f"hobby_fav restored to '{OLD_VALUE}' — test is repeatable")
else:
    info(f"Restore failed (non-critical): {r.status_code}")

sep("TEST COMPLETE")
print()
