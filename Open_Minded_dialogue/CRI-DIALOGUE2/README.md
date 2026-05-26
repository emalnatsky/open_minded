# CRI-DIALOGUE2

This is the shareable modular version of the CRI dialogue script.

Only upload or share this inner `CRI-DIALOGUE2` folder. Do not upload the sibling `_local` folder, because `_local` is for private runtime files such as your real API key, conversation logs, and local session config.

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

## Condition Codes

The dialogue uses these condition values consistently in config, fake personas, logs, and UM data:

```text
C = Control: conversational-only memory access
E = Experiment: transmedial metaphor-supported memory access
```

Legacy values like `C1` and `C2` are still accepted as input, but new files should use `C` and `E`.

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

The script itself is:

```text
CRI-BRANCH-BASIC4_0.py
```

## Tests

Run the shareable package tests from this folder:

```powershell
python -m unittest discover -s tests
```
