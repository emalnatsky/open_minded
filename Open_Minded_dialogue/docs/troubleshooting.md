# Troubleshooting

Common issues and how to diagnose them. Issues are grouped by the component that is most likely the source.

---

## 1. NAO Connection

### NAO is unreachable

- Press NAO's chest button to hear its current IP address out loud
- Confirm the laptop and NAO are on the same router network
- Test connectivity: `ping <robot_ip>`
- If the IP has changed, update the default in `CRI-BRANCH-BASIC4_0.py`:
  ```python
  def __init__(self, openai_env_path=None, nao_ip="10.0.0.241"):
  ```

### NAO is reserved by another client

The dialogue will log: `Device 10.0.0.241 is already reserved by another client`. This usually clears within a few seconds as SIC times out the stale reservation. If it does not, restart the dialogue script.

## 2. Tablet

### The tablet does not update

- Confirm the tablet server is running (Terminal 6)
- Confirm the iPad is on the same router network as the laptop
- The tablet polls the laptop every two seconds. If the dialogue has not yet started a session, the tablet will show a blank state until `session_state.json` is first written

### The tablet shows the wrong child's name

The tablet reads `child_name` from `_local/session_state.json`, which the dialogue writes at the start of each phase. If the wrong name appears, confirm `util/test_config.pl` has the correct `first_name_tablet` value and restart the dialogue.

---

## 3. Whisper (Speech-to-Text)

### Whisper returns no transcript

- Confirm the Whisper component is running (Terminal 5)
- Confirm the DJI microphone is selected as the system input
- Speak louder or move closer to the transmitter
- Check the transmitter battery

### Whisper transcribes in English instead of Dutch

The language is hardcoded in `whisper_stt.py` (line 214 for the OpenAI API call, line 223 for the local fallback). It should be set to `language="nl"`. If transcripts are in English, verify the file:

```bash
grep -n "language" venv_sic/lib/python3.11/site-packages/sic_framework/services/openai_whisper_stt/whisper_stt.py
```

If running Whisper on NAO (rather than on the laptop), the same change must be made in NAO's local copy of `whisper_stt.py`. See [microphone.md](microphone.md) for the production configuration that runs Whisper on the laptop.

### WhisperComponent fails to start

The dialogue logs: `Could not connect to WhisperComponent. Is SIC running on the device?`. This usually means a stale component reservation in Redis. Try:

```bash
redis-cli FLUSHALL
```

Then restart the dialogue.

Or restart run gpt and then run whisper.
Then restart the dialogue.

---

## 4. OpenAI API

### Authentication errors

- Confirm `Open_Minded_dialogue/_local/config/.env` exists and contains `OPENAI_API_KEY`
- Confirm the key has no extra whitespace or quotes around it
- Confirm the laptop has internet connectivity
- Test the key independently:
  ```bash
  curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
  ```

### Rate limit or quota errors

The intent classifier and L3 module both use the OpenAI API. If quota is exceeded, both will fall back to default behaviour (stub classifier, pre-authored fallback utterances). The dialogue continues but the conversation will feel more rigid.

---

## 5. GraphDB and User Model

### Child not found in the API

The dialogue logs: `Child '001' not found. Has Eunike loaded the data?`. Confirm:

- GraphDB is running
- The correct TTL dataset is loaded
- The child ID in `util/test_config.pl` matches an existing ID in the dataset

Test the API directly:

```bash
curl http://localhost:8000/api/um/<child_id>/inspect
```

### Wrong UM data appears

If the dialogue uses outdated or wrong values, the TTL dataset may be the wrong one. Reload the correct TTL in GraphDB.

---

## 6. Logging

### Excessive SIC log output flooding the dialogue terminal

The SIC framework broadcasts log messages from all connected components, including the tablet server's poll logs. Until the SIC framework supports per-component log levels, the workaround is to filter them on output:

```bash
python -u um_tablet_server.py 2>&1 | grep -v "Fetched"
```

This suppresses the poll lines while keeping all other log output visible.

---

## 7. Session Recovery After a Crash

If the dialogue terminates mid-session:

1. Stop the script (Ctrl+C if still running)
2. Set `continueSession(true).` in `util/test_config.pl`
3. Restart the dialogue. It will prompt to resume from the most recent log in `_local/conversations/`
4. After the session ends, set `continueSession(false).` again before the next participant

The resume logic restores which UM fields Leo has already mentioned, which mistakes have been seen, and the starting phase index. Whisper transcripts from the prior portion are not replayed.

---

## 8. When in Doubt

Most issues resolve with a clean restart. The recommended sequence:

1. Stop the dialogue (Ctrl+C in Terminal 7)
2. Stop the tablet server (Ctrl+C in Terminal 6)
3. Wait five seconds for SIC to release its component reservations
4. Restart Terminal 6, then Terminal 7

If a clean restart does not resolve the issue, restart Redis and the SIC components (Terminals 1, 4, 5) as well.
