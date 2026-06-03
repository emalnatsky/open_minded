# Running a Session

This document describes the terminal-by-terminal launch sequence, the operator controls available during a session, and the visual state indicators on NAO.

---

## 1. Pre-flight Checklist

Before starting the launch sequence, confirm:

- `util/test_config.pl` is updated with the current participant's ID, names, and condition
- `_local/.env` contains a valid `OPENAI_API_KEY`
- GraphDB is running and the correct TTL dataset is loaded
- NAO, laptop, and tablet are all on the same router network
- The DJI microphone receiver is plugged in and selected as the system input
- NAO is powered on and standing upright

If any of these is missing, resolve it first.

---

## 2. Terminal Launch Order

Eight terminals are needed. Start them in the listed order — each step depends on the previous one being ready.

| # | Service | Command |
|---|---------|---------|
| 1 | Redis | `redis-server` |
| 2 | GraphDB | Start GraphDB Desktop and load the TTL dataset |
| 3 | API | `cd OM_Ontology_Database && python main.py` |
| 4 | GPT component | `run-gpt` |
| 5 | Whisper component | `run-whisper` |
| 6 | Tablet server | `cd Open_Minded_dialogue/UM-TABLET && python um_tablet_server.py` |
| 7 | Dialogue script | `cd Open_Minded_dialogue/CRI-DIALOGUE2 && python CRI-BRANCH-BASIC4_0.py` |

After Terminal 6 (the tablet server) starts, open `http://<laptop-ip>:8080` on the iPad. The book cover should appear with the participant's name.

The dialogue script in Terminal 7 will prompt for session mode (new or resume) and then for the starting phase. Press Enter to accept defaults for a fresh session.

---

## 3. Operator Controls During a Session

While Leo is listening for the child's response, two keyboard controls are available in the dialogue terminal:

### R + Enter — Re-ask

Use when Leo clearly misheard, or when the child was interrupted or distracted mid-sentence. Whisper is cancelled, Leo apologises (*"Sorry, ik heb je vorige antwoord niet goed gehoord. Kun je het nog een keer zeggen?"*), and listens again for the same prompt.

### P + Enter — Pause

Use when the child needs to look at the tablet without Leo continuing. The dialogue waits indefinitely. When the child is finished, press Enter to resume. Leo will listen again for the same prompt without apologising (since the child was simply reading, not speaking).

Both controls are bounded by a maximum of five retries per turn, so a stuck session will eventually advance with an empty transcript.

---

## 4. NAO Visual State Indicators

NAO's eye LEDs indicate the dialogue state:

| Colour | Meaning |
|--------|---------|
| White | Idle, or Leo is speaking |
| Green | Leo is listening (Whisper is active) |
| Blue | Dialogue is paused (P+Enter pressed) |

## 5. Ending a Session

When the dialogue reaches its final phase, the conversation log is saved to `_local/conversations/<child_id>_<timestamp>.json`. Stop the script with Ctrl+C, then the supporting services (tablet server, Whisper, GPT, API, Redis) in reverse order.

Before starting the next participant, update `util/test_config.pl` with the new participant's information and confirm `continueSession(false).`.

---

## 6. Next Steps

If anything goes wrong during the session, see [troubleshooting.md](troubleshooting.md).

For privacy and data handling guidelines, see [privacy.md](privacy.md).
