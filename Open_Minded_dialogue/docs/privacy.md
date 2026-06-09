# Privacy and Data Handling

This document covers what data the system collects, where it is stored, and which files must never leave the researcher's laptop.

---

## 1. Sensitive Files

The following files contain identifiable participant data and are excluded from version control via `.gitignore`. They are never committed, pushed, or shared outside the research team:

| File | Contents |
|------|----------|
| `util/test_config.pl` | Participant ID, real names, condition, researcher name |
| `_local/config/.env` | OpenAI API credentials and local STT backend settings |
| `_local/session_state.json` | Current session state (refreshed every turn) |
| `_local/session_config.local.json` | Session settings (refreshed each run) |
| `_local/conversations/*.json` | Full conversation transcripts |

Before sharing any artifact externally — for example, when uploading code to a public repository or sending logs to a collaborator — confirm none of these files are included.

---

## 2. What Is Logged

Each session produces a JSON conversation log in `_local/conversations/`. The log contains:

- The child's first name (as set in `util/test_config.pl`)
- The child's participant ID
- Researcher name
- Condition
- Every utterance by Leo and the child, with timestamps
- Intent classifications, mistake states, and memory access events
- The starting and ending UM values

The log does **not** contain audio recordings. Only transcribed text is stored.

---

## 3. Before Sharing Logs

When sharing conversation logs for analysis (for example, sending to a collaborator or including in a thesis appendix), remove or anonymise:

- `child_name` field — replace with a pseudonym or short code
- `child_id` field — replace with a study-internal participant code (e.g. `P01`, `P02`)
- `researcher_name` field — typically left in if researchers are part of the team, removed for external sharing
- Any utterance from the child that identifies them by name, school, or location

---

## 4. GraphDB Data

The user-model data in GraphDB originates from a Qualtrics survey filled in before the session. The TTL dataset includes child IDs, names, ages, hobbies, and other personal preferences. GraphDB itself runs locally on the researcher's laptop — this data does not leave the laptop.

When done with a study cohort, the TTL file should be archived securely or deleted, depending on the study's data retention agreement.

---

## 5. OpenAI API Considerations

The dialogue sends child utterances to the OpenAI API for intent classification and L3 generation. Per OpenAI's policy, these requests may be retained for a short window for abuse monitoring, but are not used for model training when sent through the standard API.

To minimise data sent to the API:

- The intent classifier sends only the child's transcript and a small turn context (Leo's previous line, current topic, relevant UM fields). It does not send the full conversation history
- The L3 module sends similar context plus the response function. It also does not send the full history

No child names or IDs are sent to the API. The classifier and L3 modules receive only the linguistic content needed to do their jobs.

