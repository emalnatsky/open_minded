# Initial Setup

This document covers the first-time setup needed before running any session: API credentials, the per-participant session configuration file, and the experimental condition codes used throughout the codebase.

---

## 1. Credentials

The dialogue uses the OpenAI API for intent classification and for L3 runtime utterances. The API key is loaded from a local `.env` file that is never committed to version control.

Copy the example file:

```bash
mkdir -p Open_Minded_dialogue/_local/config
cp Open_Minded_dialogue/CRI-DIALOGUE2/.example_env Open_Minded_dialogue/_local/config/.env
```

Open `Open_Minded_dialogue/_local/config/.env` and insert your key:

```env
OPENAI_API_KEY=sk-your-key-here
```

The `_local/` folder is gitignored. Each researcher maintains their own copy on their laptop. For microphone sessions, also choose exactly one STT preset in that `.env` file: Mac safe, Windows NVIDIA, or Windows CPU fallback.

---

## 2. Session Configuration

Each participant's information is set in `util/test_config.pl` before the session. This file follows a Prolog-style format and contains:

- Participant ID (matches the ID used in GraphDB)
- The name NAO pronounces aloud (`first_name_cri`)
- The name shown on the tablet book cover (`first_name_tablet`)
- The researcher's name
- The experimental condition
- A flag for resuming a crashed session

Example file:

```prolog
userId('001').
localVariable(first_name_cri, "Yulianna").
localVariable(first_name_tablet, "Julianna").
localVariable(operator_name, "Researcher Name").
condition(experimental).
continueSession(false).
```

## 3. Experimental Conditions

The codebase uses two internal condition labels:

| Code | Condition | Description |
|------|-----------|-------------|
| `C` | Control | Conversational-only memory access |
| `E` | Experimental | Tablet-supported memory access (visual book) |

The `condition()` predicate in `test_config.pl` accepts several aliases for convenience:

| Written in pl file | Mapped to |
|---------------------|-----------|
| `experimental`, `E`, `C2`, `2` | `E` |
| `control`, `C`, `C1`, `1` | `C` |

Legacy values `C1` and `C2` are still accepted, but all new code uses `C` and `E`.

---

## 4. Session Resume

If the dialogue crashes mid-session, set `continueSession(true).` in `util/test_config.pl` and restart. The dialogue will prompt to resume from the most recent conversation log in `_local/conversations/`. After the participant finishes, set `continueSession(false).` again before the next session.

---

## 5. Next Steps

Once credentials and the session config are in place, continue to:

- [Network setup](network.md) — connecting NAO, laptop, and tablet to the same router
- [Microphone setup](microphone.md) — configuring the DJI wireless mic
- [Running a session](running.md) — the full terminal-by-terminal startup sequence
