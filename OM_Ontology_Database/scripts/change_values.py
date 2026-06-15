import json
import sys

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_FILE  = r"data\fix.json"   # path to your input JSON
OUTPUT_FILE = r"data\fix.json"   # where to save the result


# Set to None to update ALL children, or a list of child_ids to target specific ones
# e.g. TARGET_CHILDREN = ["2", "5"]
TARGET_CHILDREN = None

# ── WHAT TO CHANGE ─────────────────────────────────────────────────────────────

# Mistake field updates: target_field → new wrong_value
MISTAKE_UPDATES = {
    "aspiration" : "Schooldirecteur",
    # "target_field" : "new wrong_value",
}

# Utterance text updates: (step_id, branch) → new text
UTTERANCE_UPDATES = {
    ("p3_m4_followup_wrong_aspiration", "default"): "Later wil jij schooldirecteur worden, toch?",
    # ("some_other_step_id", "corrected"): "Some other new text",
}


# ───────────────────────────────────────────────────────────────────────────────


def apply_updates(data):
    for scenario in data.get("scenarios", []):
        child_id = scenario.get("child_id")

        if TARGET_CHILDREN is not None and child_id not in TARGET_CHILDREN:
            continue

        for mistake in scenario.get("mistakes", []):
            field = mistake.get("target_field")
            if field in MISTAKE_UPDATES:
                old = mistake["wrong_value"]
                mistake["wrong_value"] = MISTAKE_UPDATES[field]
                print(f"  [mistake] child_id={child_id} | {field}: '{old}' → '{MISTAKE_UPDATES[field]}'")

        for utt in scenario.get("utterances", []):
            key = (utt.get("step_id"), utt.get("branch"))
            if key in UTTERANCE_UPDATES:
                old = utt["text"]
                utt["text"] = UTTERANCE_UPDATES[key]
                print(f"  [utterance] child_id={child_id} | step_id={key[0]} branch={key[1]}")
                print(f"    before: {old}")
                print(f"    after:  {utt['text']}")

    return data


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Targeting children: {'ALL' if TARGET_CHILDREN is None else TARGET_CHILDREN}\n")
    updated = apply_updates(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
