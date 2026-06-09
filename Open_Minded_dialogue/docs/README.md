# CRI Dialogue System
## Overview

The system orchestrates three concurrent components:

- **GraphDB API** — stores each child's persistent user model (UM) populated from a prior Qualtrics survey
- **CRI dialogue script** (this folder) which drives the conversation, performs intent classification, manages mistakes, and logs the session
- **Tablet memory book** — a live-updating tablet interface, used only in the experimental condition

All three components run on the laptop and communicate over the local network. The NAO robot connects via the SIC framework over Wi-Fi.

---

## Repository Layout

The repository contains two folders. **Only `CRI-DIALOGUE2/` is shared between researchers.** The `_local/` folder contains participant data and credentials and must never be committed or uploaded.

```
CRI-DIALOGUE2/
├── CRI-BRANCH-BASIC4_0.py   # Main entry point
├── cri_actions/             # Intent: action mapping, nudges
├── cri_classifier/          # GPT-based intent classifier with stub fallback
├── cri_logger/              # Conversation logging and session resume
├── cri_memory/              # Phase-scoped memory access logic
├── cri_script/              # Script builder, content plans, topic segments
├── cri_um/                  # GraphDB client and UM helpers
├── fake_personas/           # JSON personas for offline development
├── tests/                   # Unit tests
└── README.md

_local/                      # GITIGNORED: do not share
├── config/.env              # OpenAI API key and local STT settings
├── session_config.local.json
├── session_state.json       # Written each turn for the tablet
└── conversations/           # JSON conversation logs
```

---

## Documentation

Detailed instructions are split across the `docs/` folder:

| Topic | File |
|-------|------|
| Initial setup (credentials, session config, conditions) | [docs/setup.md](docs/setup.md) |
| Network configuration (NAO ↔ router ↔ laptop ↔ tablet) | [docs/network.md](docs/network.md) |
| Microphone configuration (DJI wireless mic) | [docs/microphone.md](docs/microphone.md) |
| Running a session (terminal order, operator controls) | [docs/running.md](docs/running.md) |
| Troubleshooting common issues | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Privacy and data handling | [docs/privacy.md](docs/privacy.md) |

---

## Quickstart

For an experienced operator already familiar with the setup, the minimum sequence is:

1. Edit `util/test_config.pl` with the participant's ID, names, researcher, and condition
2. Start GraphDB and load the correct TTL dataset
3. Open eight terminals in the order listed in [docs/running.md](docs/running.md)
4. Open `http://<laptop-ip>:8080` on the tablet
5. Run `python CRI-BRANCH-BASIC4_0.py` and follow the on-screen prompts

For first-time setup, follow each documentation file in sequence:
**setup → network → microphone → running**.

---

## Testing

Unit tests cover the script builder, intent classifier behaviour, memory access logic, and tablet state writes:

```bash
python -m unittest discover -s tests
```

For offline testing without GraphDB, set `USE_FAKE_PERSONA_UM = True` in `config.py`. The dialogue will load a JSON persona from `fake_personas/` instead of querying the live API.
