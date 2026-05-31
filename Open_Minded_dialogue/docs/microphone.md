# Microphone Setup

The DJI receiver plugs into the laptop and acts as the system's default input device. NAO still handles speech output and LED indicators, but audio capture is routed through the laptop, which gives much better transcription quality than NAO's built-in microphone.

---

## 1. Hardware Setup

1. Plug the DJI receiver into the laptop's USB-C port
2. Power on the DJI transmitter and pair it with the receiver (refer to DJI documentation if pairing is lost)
3. Attach the transmitter to the child's clothing, typically clipped near the collar

---

## 2. System Configuration (macOS)

1. Open **System Settings → Sound → Input**
2. Select the DJI receiver as the active input device
3. Speak into the transmitter and confirm the input level meter responds
4. Adjust input volume if the level is too low or too high

---

## 3. Dialogue Configuration

In `config.py`, set:

```python
USE_DESKTOP_MIC = True
CONNECT_NAO     = True
```

This routes audio capture through the laptop (and therefore through the DJI mic) while keeping NAO connected for speech output and LED indicators.

The two flags are independent:

| `USE_DESKTOP_MIC` | `CONNECT_NAO` | Behaviour |
|-------------------|---------------|-----------|
| `True` | `True` | DJI mic + NAO speaks ← **production setup** |
| `False` | `True` | NAO mic + NAO speaks (no DJI needed) |
| `True` | `False` | Laptop mic only, no robot (desktop testing) |

---

## 4. Verifying Audio Capture

Before starting a session, do a quick test:

1. Run the dialogue with the participant ID set to a test ID
2. When Leo asks a question, speak into the DJI mic
3. Confirm the terminal shows a transcript

If no transcript appears, see the Whisper troubleshooting section in [troubleshooting.md](troubleshooting.md).

---

## 5. Common Pitfalls

- **DJI mic not selected as system input.** This is the most common cause of silent transcription (especially if the receiver gets unplugged and then plugged again)
- **Transmitter battery low.** The DJI transmitter holds a few hours of charge but loses pairing reliability when the battery is below 10%

---

## 6. Next Steps

- [Running a session](running.md) — full terminal sequence to launch the system
