# Local Runtime Template

This folder is safe to share. It contains dummy files that show the private `_local` structure.

To use it, copy this folder next to the shareable `CRI-DIALOGUE2` folder and rename it to `_local`:

```text
CRI-DIALOGUE2/
  CRI-DIALOGUE2/      # shared code
  _local/             # private local copy of this template
```

Then rename `.example_env` to `.env` and paste your own OpenAI API key.

For a safe roster example, copy:

```text
_local_template\session_rosters\example_day.json
```

to:

```text
_local\session_rosters\example_day.json
```

Condition values:

```text
C = Control: conversational-only memory access
E = Experiment: transmedial metaphor-supported memory access
```

Do not upload your real `_local` folder. It can contain API keys, real child rosters, conversation logs, and session state.
