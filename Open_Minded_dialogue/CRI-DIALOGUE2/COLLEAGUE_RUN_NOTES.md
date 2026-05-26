# CRI Dialogue 2 - Colleague Notes

## Can you run it if only this inner folder is pushed?

Yes, but only if you already have the rest of the Open Minded project and services available. This folder contains the Dialogue 2 code, tests, fake personas, and the shareable `_local_template`. It does not contain GraphDB, the UM API code, Redis, Whisper, or the SIC GPT service.

Best option: place this folder at:

```text
Open_Minded_dialogue\CRI-DIALOGUE2\CRI-DIALOGUE2
```

Then keep or recreate this private sibling folder locally:

```text
Open_Minded_dialogue\CRI-DIALOGUE2\_local
```

Do not commit `_local`. It can contain API keys, real rosters, logs, and local session state.

## Minimal local setup

1. Copy the included template folder:

```text
CRI-DIALOGUE2\_local_template
```

to the sibling location:

```text
Open_Minded_dialogue\CRI-DIALOGUE2\_local
```

2. Rename `_local\.example_env` to `_local\.env` and add your own OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

3. Make sure the normal stack services are running:

```text
GraphDB
UM API on http://localhost:8000
Redis
SIC GPT service
Whisper service if using microphone input
```

For keyboard testing, Whisper is not used by Dialogue 2, but it is harmless if it is running.

4. If the services are already running, you can run from this folder:

```powershell
python CRI-BRANCH-BASIC4_0.py
```

The session interface will load the example roster/config from `_local`.

5. If tablet condition run in new terminal after inputting the interface information for the CRI-BRANCH-BASIC4_0.py
```powershell
python UM-TABLET/um_tablet_server.py
```
BEWARE! TO STOP THE DETCH MESSAGES RUN THIS COMMAND BEFORE THE TABLET SERVER: 
```powershell
python -u YOUR_SCRIPT.py 2>&1 | grep -v "PATTERN_TO_IGNORE"
```
## Memory access flow

The child can ask what Leo remembers at any point in the conversation, for example: "Wat weet je over mij?" or "Wat heb je onthouden?" The intent classifier routes that to `um_inspect`.

Leo only repeats memory fields that have been used or discussed so far in the conversation. This is intentional: the child should inspect what Leo actually used, not the whole database. In condition C, Leo reads the memory back conversationally in child-friendly grouped blocks. In condition E, Leo routes the child to the tablet/geheugenboek flow and activates the same mentioned-so-far memory fields for tablet display.

At Part 3 phase 3.6, memory inspection is offered explicitly to every child. Part 3 phase 3.7 then does the memory review/co-construction step: Leo checks clusters in script order and the child can confirm, correct, delete, add, or say they are not sure.

## Nudge flow

The nudge happens after the first two deliberate mistakes only when both mistakes occurred and the child corrected none of them. In that case, Leo asks a gentle meta-question like whether everything he said was right. If the child says something was wrong, Leo asks what was wrong and lets the child correct it.

If the child says everything was fine, Leo offers memory inspection instead of pushing for a correction. The nudge is not meant to force the child; it gives one extra opportunity to notice and repair Leo's memory.

## Script implementation status

All main Dialogue 2 parts are implemented.

Part 1 includes greeting, condition-specific tutorial, Leo mini-story, correct hobby bridge, Topic 1, Mistake 1 on `hobby_fav`, Topic 2, Mistake 2 on `fav_food`, and the conditional nudge.

Part 2 includes the school joke transition, robot-school self-disclosure, correct `fav_subject` plus interest link, and Mistake 3 on `school_strength` with correction and no-correction branches.

Part 3 includes the school-to-future bridge, Leo aspiration self-disclosure, role model rapport with UM-present and UM-missing branches, Mistake 4 on `aspiration`, correction space plus personalized reflection, explicit memory inspection, memory review/co-construction, and closing.

## L3 runtime responses

L3 is implemented according to Lena's prompt specification. It is used only at specific scripted call points where Leo needs a short runtime response to unpredictable child input. The L3 call receives structured variables such as `script_phase`, `topic`, `response_function`, `question_allowed`, the child's previous utterance, Leo's previous utterance, the next scripted line, local context, and only the relevant UM fields for that turn.

The L3 system prompt defines Leo as a warm NAO school robot speaking Dutch to children aged 8-11. It includes Lena's safety rules: do not reveal UM fields that were not explicitly provided to the L3 call, do not ask verification questions unless allowed, do not overpraise, avoid sensitive topics, keep it short, and do not overlap with the next scripted line.

Outputs are validated before use. If an L3 response is too long, asks a question when questions are not allowed, or breaks safety expectations, Leo falls back to a safe prewritten line. This means L3 can make the dialogue feel natural while the scripted structure and study design stay controlled.

## Testing

Run tests from the repository root with:

```powershell
python -m unittest "Open_Minded_dialogue\CRI-DIALOGUE2\CRI-DIALOGUE2\tests\test_cri_dialogue2.py"
```

Or from inside this folder:

```powershell
python -m unittest discover -s tests
```
