# Local Runtime Template

This folder is safe to share. It contains dummy files that show the private `_local` structure.

To use it, copy this folder next to the shareable `CRI-DIALOGUE2` folder and rename it to `_local`:

```text
CRI-DIALOGUE2/
  CRI-DIALOGUE2/      # shared code
  _local/             # private local copy of this template
```

Then rename `.example_env` to `.env` and paste your own OpenAI API key.

Do not upload your real `_local` folder. It can contain API keys, conversation logs, local launchers, and session state.
