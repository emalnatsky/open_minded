# Run CRI Dialogue - Mac

This guide explains how to run one CRI dialogue session in the lab on macOS.

## 1. Physical Setup

1. Turn on the router.
2. Connect the router to the laptop with an Ethernet cable.
3. Connect the laptop to the Wi-Fi.
4. Turn on NAO and wait until it is fully started.
5. Connect the tablet to the same router Wi-Fi.
6. Plug in the DJI microphone receiver.
7. In macOS sound settings, make the DJI/lab microphone the default input.

## 2. Check Local Environment

Make sure this private file exists:

```text
Open_Minded_dialogue/_local/config/.env
```

It must contain your OpenAI key and the default STT/NAO settings:

```env
OPENAI_API_KEY=your-key-here
CRI_STT_DEVICE=auto
CRI_STT_COMPUTE_TYPE=auto
CRI_STT_MODEL=auto
CRI_STT_GPU_INDEX=0
CRI_NAO_AUTO_DISCOVER=true
CRI_NAO_DISCOVERY_TIMEOUT_SECONDS=3
```

Normally leave `CRI_NAO_IP` empty or commented out. CRI will auto-discover NAO. If NAO is not found, CRI asks for the IP in the terminal.

Mac STT note: `auto` uses the Mac-safe CPU/int8 path. Do not set `CRI_STT_DEVICE=mps` or `CRI_STT_DEVICE=metal` for this RealtimeSTT/faster-whisper/CTranslate2 setup.

## 3. Start GraphDB First

1. Open GraphDB Desktop.
2. Start repository `open-memory-robots`.
3. Make sure the TTL/data for the test children is loaded.
4. GraphDB should be reachable at:

```text
http://localhost:7200
```

GraphDB must be running before the UM API is started.

## 4. Start The Stack Manually

Open separate Terminal windows or tabs. Keep every terminal open while the session runs.

Replace `/path/to/open_minded_main` with the folder where this repository is stored.

### Terminal 1: Redis

```bash
cd /path/to/open_minded_main
redis-server
```

### Terminal 2: UM API

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
cd OM_Ontology_Database
python main.py
```

Check in a browser:

```text
http://localhost:8000/health/graphdb
```

Only continue when this says the GraphDB health is OK.

### Terminal 3: SIC GPT Service

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
run-gpt
```

### Terminal 4: SIC Webserver

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
run-webserver
```

### Terminal 5: Tablet Server

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
cd Open_Minded_dialogue/UM-TABLET
python um_tablet_server.py
```

### Terminal 6: CRI Dialogue

Choose one mode before starting CRI.

Keyboard test, no microphone and no NAO:

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
export CRI_CHILD_INPUT_MODE=keyboard
export CRI_CONNECT_NAO=false
export CRI_USE_DESKTOP_MIC=true
export CRI_OUTPUT_MODE=print
cd Open_Minded_dialogue/CRI-DIALOGUE2
python CRI-BRANCH-BASIC4_0.py
```

Microphone test, laptop prints Leo instead of using NAO voice:

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
export CRI_CHILD_INPUT_MODE=microphone
export CRI_CONNECT_NAO=false
export CRI_USE_DESKTOP_MIC=true
export CRI_OUTPUT_MODE=print
cd Open_Minded_dialogue/CRI-DIALOGUE2
python CRI-BRANCH-BASIC4_0.py
```

Real NAO session:

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
export CRI_CHILD_INPUT_MODE=microphone
export CRI_CONNECT_NAO=true
export CRI_USE_DESKTOP_MIC=true
export CRI_OUTPUT_MODE=nao
cd Open_Minded_dialogue/CRI-DIALOGUE2
python CRI-BRANCH-BASIC4_0.py
```

## 5. Fill In Session Information

When CRI starts, fill in or accept the defaults:

```text
Child ID:
Child name:
Researcher:
Condition:
Start at phase [1.1/1]:
```

Use:

- `E` for experimental condition with tablet.
- `C` for control condition without tablet.
- `1` or `1.1` for a normal full session.

The child ID must exist in GraphDB.

## 6. NAO Connection

The NAO IP should normally stay empty in config.

CRI will:

1. search the laptop's active networks;
2. connect automatically if one NAO is found;
3. ask for manual IP if no NAO is found.

If asked, press NAO's chest button, listen to the IP, and type it into the terminal.

Example:

```text
Enter NAO IP manually, press Enter to retry, or type Q to quit: <NAO_IP_FROM_CHEST_BUTTON>
```

The manual IP is only used for that session. It is not saved.

## 7. Tablet

Only use the tablet for condition `E`.

The CRI terminal prints:

```text
Experimental condition URL: http://<laptop-ip>:8080
```

Open that exact URL on the tablet browser.

Do not type `localhost` on the tablet. `localhost` means the tablet itself, not the laptop.

Expected result:

- the memory book opens;
- the child name appears on the cover;
- pages update during the interaction.

## 8. During The Conversation

The terminal should show:

- phase banners;
- `[LEO]: ...`;
- child transcripts;
- warnings/errors.

NAO eyes:

- white = Leo is speaking or idle;
- green = Leo is listening.

The child should only answer when NAO's eyes are green.

## 9. Ending The Session

1. Let the final phase finish.
2. Check that the conversation log is saved in:

```text
Open_Minded_dialogue/_local/conversations
```

3. Stop CRI with `Ctrl+C` if needed.
4. Stop the other terminals.
5. For the next participant, restart CRI and fill in the new child ID/name/condition.

## 10. Common Problems

### NAO Not Found

- Press NAO's chest button and listen for the IP.
- Type that IP when CRI asks for it.
- Check laptop and NAO are on the same router network.

### Tablet Says "Verbinden"

- Check the tablet is on the same Wi-Fi as the laptop.
- Use the printed `Experimental condition URL`.
- Restart the tablet server if needed.
- Refresh the tablet browser.

### Microphone Does Not Transcribe

Check the default microphone:

```bash
cd /path/to/open_minded_main
source .venv/bin/activate
python Open_Minded_dialogue/CRI-DIALOGUE2/list_audio_devices.py
```

If the wrong mic is selected, set `CRI_STT_MIC_INDEX` in:

```text
Open_Minded_dialogue/_local/config/.env
```

### GraphDB / UM Error

- Make sure GraphDB Desktop is running.
- Make sure repository `open-memory-robots` is active.
- Check:

```text
http://localhost:8000/health/graphdb
```
