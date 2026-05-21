# CRI-DIALOGUE2

This is the shareable modular version of the CRI dialogue script.

Only upload or share this inner `CRI-DIALOGUE2` folder. Do not upload the sibling `_local` folder, because `_local` is for private runtime files such as your real API key, launchers, conversation logs, and local session config.

## Local Setup

Create this folder next to the shareable folder:

```text
CRI-DIALOGUE2/
  CRI-DIALOGUE2/
  _local/
```

Copy `.example_env` to:

```text
CRI-DIALOGUE2/_local/.env
```

Then fill in your own OpenAI key:

```env
OPENAI_API_KEY=your_key_here
```

You can also copy the included `_local_template` folder to the sibling `_local` folder:

```text
CRI-DIALOGUE2/
  CRI-DIALOGUE2/       # this shared folder
  _local/              # private copy of _local_template
```

Never upload your real `_local` folder.

## Runtime Files

The script writes private/local files to `_local`:

```text
_local/conversations/
_local/session_config.local.json
_local/session_state.json
```

Fake personas for keyboard testing are included inside this shareable folder:

```text
CRI-DIALOGUE2/fake_personas/
```

## Running

Personal launchers can live in:

```text
CRI-DIALOGUE2/_local/launchers/
```

The script itself is:

```text
CRI-BRANCH-BASIC4_0.py
```

## Tests

Run the shareable package tests from this folder:

```powershell
python -m unittest discover -s tests
```
