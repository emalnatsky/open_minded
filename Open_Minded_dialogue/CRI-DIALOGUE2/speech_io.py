"""
Speech input/output for the CRI dialogue.
SpeechIO wraps:
  - NAO TTS (say)
  - RealtimeSTT AudioToTextRecorder (listen)
  - Keyboard input fallback (listen)
  - Transcript review loop (review_transcript / listen_with_review)
  - strip_non_bmp — sanitise text before sending to NAO TTS

Constructed once in CRI_ScriptedDialogue.setup() and stored as self.speech.
The dialogue calls self.speech.say(...) and self.speech.listen_with_review().

Dependencies injected at construction time so this module never imports
from the dialogue class or from config directly.

Eye colour contract:
  white → NAO is speaking (default, unchanged)
  green → NAO finished speaking, child's turn (set at end of say())
  white → transcript received, idle (set at end of listen())

STT backend: RealtimeSTT (pip install RealtimeSTT)
  Uses AudioToTextRecorder with:
    - model="small"       good Dutch accuracy, ~450 MB, downloaded on first run
    - language="nl"       Dutch
    - enable_realtime_transcription=False   final result only (no streaming)
    - post_speech_silence_duration=0.6      stops ~0.6s after child goes quiet
    - initial_prompt      updated each turn with Leo's last utterance so Whisper
                          knows the question context before decoding the answer

  The recorder is initialised once at construction time and reused across
  all listen() calls for the session.

If the child says nothing (empty transcript), listen_with_retry() asks Leo
to repeat the question up to max_listen_retries times before giving up.
"""

import re
import time
import unicodedata
import logging

from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

# ── Whisper context prefix ────────────────────────────────────────────────────
# Prepended to initial_prompt every turn. Primes Whisper's language model with
# the domain before decoding — biggest single accuracy gain for Dutch children.
_CONTEXT_PREFIX = (
    "Dit is een gesprek in het Nederlands tussen een kind (8–12 jaar) en "
    "een sociale robot genaamd Leo. "
    "Het kind beantwoordt vragen over hobby's, school en interesses. "
    "Veelvoorkomende woorden: tekenen, bakken, voetbal, hockey, rekenen, taal, "
    "gym, lezen, katten, honden, pizza, pannenkoeken. "
)

# ── Retry prompts (Dutch, rotated so they don't sound identical) ──────────────
_RETRY_PROMPTS = (
    "Hmm, ik heb je niet goed gehoord. Kun je het nog een keer proberen?",
    "Sorry, dat heb ik gemist. Kun je het nog eens zeggen?",
    "Ik heb je helaas niet verstaan. Kun je het nog één keer proberen?",
)

# ── Whisper silence hallucinations ────────────────────────────────────────────
# Whisper emits these token strings for silent audio instead of returning "".
_WHISPER_SILENCE_ARTEFACTS = frozenset({
    ".", "..", "...",
    "[stilte]", "[silence]",
    "[muziek]", "[music]",
    "[ondertiteling]", "[subtitles]",
    "[lacht]", "[gelach]", "[applaus]",
    "(stilte)", "(muziek)",
    "dank u", "dank u wel",
})


class SpeechIO:
    """Handles all audio in/out for one session."""

    def __init__(
        self,
        *,
        nao=None,
        use_nao_output: bool = None,
        simulation_mode: bool = False,
        use_keyboard_input_fn=None,
        review_transcripts: bool = True,
        log_event_fn=None,
        simulated_history: list = None,
        generate_simulated_response_fn=None,
        set_last_utterance_fn=None,
        # ── Accepted from main script but no longer used internally ──────────
        # (old SICWhisper stack passed these; RealtimeSTT handles routing itself)
        whisper=None,
        use_desktop_mic: bool = True,
        stt_timeout: float = None,
        stt_phrase_limit: float = None,
        # ── RealtimeSTT knobs ────────────────────────────────────────────────
        stt_model: str = "small",
        stt_language: str = "nl",
        stt_post_speech_silence: float = 0.6,
        stt_realtime_processing_pause: float = 0.2,
        stt_mic_index: int = None,
        stt_beam_size: int = 5,
        stt_device: str = "cpu",
        stt_compute_type: str = "int8",
        # ── Retry knob ───────────────────────────────────────────────────────
        max_listen_retries: int = 3,
        pronunciation_overrides_path: str = None,
    ):
        """
        Args:
            nao:                            SIC NAO device handle (or None)
            use_nao_output:                 True → use NAO TTS and LEDs
            simulation_mode:                True → use simulated child responses
            use_keyboard_input_fn:          callable() → bool
            review_transcripts:             True → researcher review loop after listening
            log_event_fn:                   callable(event_type, **kwargs)
            simulated_history:              list to append {"speaker", "text"} dicts to
            generate_simulated_response_fn: callable() → str
            set_last_utterance_fn:          callable(text) — kept for backwards compat
            whisper:                        ignored (legacy SICWhisper param)
            use_desktop_mic:                ignored (legacy routing param)
            stt_timeout:                    ignored (legacy SICWhisper param)
            stt_phrase_limit:               ignored (legacy SICWhisper param)
            stt_model:                      Whisper model size
            stt_language:                   BCP-47 language code ("nl" = Dutch)
            stt_post_speech_silence:        Seconds of silence before cut-off
            stt_realtime_processing_pause:  VAD chunk interval in seconds
            stt_mic_index:                  Mic device index (None = OS default)
            stt_beam_size:                  Whisper beam search width
            stt_device:                     RealtimeSTT device ("cpu" for study laptops)
            stt_compute_type:               Faster-Whisper compute type
            max_listen_retries:             Total listen attempts before giving up
        """
        self.nao = nao
        self.use_nao_output = (
            bool(nao) if use_nao_output is None else bool(use_nao_output)
        )
        self.simulation_mode = simulation_mode
        self._use_keyboard_input = use_keyboard_input_fn or (lambda: False)
        self.review_transcripts = review_transcripts
        self._log_event = log_event_fn or (lambda *a, **kw: None)
        self._simulated_history = simulated_history if simulated_history is not None else []
        self._generate_simulated_response = generate_simulated_response_fn or (lambda: "")
        self._set_last_utterance = set_last_utterance_fn or (lambda t: None)
        self._max_listen_retries = max_listen_retries
        self._pronunciation_overrides = {}
        if pronunciation_overrides_path:
            import json, os
            if os.path.exists(pronunciation_overrides_path):
                with open(pronunciation_overrides_path, "r", encoding="utf-8") as f:
                    self._pronunciation_overrides = json.load(f)
                logger.info("Loaded %d TTS pronunciation overrides.", len(self._pronunciation_overrides))



        # Tracks the last Leo utterance so Whisper gets question context next listen().
        self._last_leo_text: str = ""

        # ── Initialise RealtimeSTT recorder ──────────────────────────────────
        self._recorder = None
        if not simulation_mode and not (use_keyboard_input_fn and use_keyboard_input_fn()):
            recorder_kwargs = dict(
                model=stt_model,
                language=stt_language,
                post_speech_silence_duration=stt_post_speech_silence,
                realtime_processing_pause=stt_realtime_processing_pause,
                use_microphone=True,
                device=stt_device,
                compute_type=stt_compute_type,
                enable_realtime_transcription=False,  # final result only
                beam_size=stt_beam_size,
                initial_prompt=_CONTEXT_PREFIX,       # updated per turn in listen()
                # VAD tuning — stricter detection so NAO's own TTS / room noise
                # is less likely to be mistaken for the child speaking.
                silero_sensitivity=0.6,               # higher = stricter "is this speech"
                webrtc_sensitivity=3,                 # 0-3, higher = least sensitive
                min_length_of_recording=0.5,          # ignore <0.5s blips
            )
            if stt_mic_index is not None:
                recorder_kwargs["input_device_index"] = stt_mic_index

            logger.info(
                "Initialising RealtimeSTT recorder "
                "(model=%s, language=%s, silence=%.1fs) — "
                "first run downloads the model...",
                stt_model, stt_language, stt_post_speech_silence,
            )
            self._recorder = AudioToTextRecorder(**recorder_kwargs)
            logger.info("RealtimeSTT recorder ready.")

            # Mute the mic until it's actually the child's turn. The recorder
            # keeps running but processes silence, so NAO's own TTS won't be
            # transcribed and won't trigger a recording.
            self._set_mic(False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self):
        """Cleanly stop the RealtimeSTT recorder. Call at end of session."""
        if self._recorder is not None:
            try:
                self._recorder.shutdown()
                logger.info("RealtimeSTT recorder shut down.")
            except Exception as e:
                logger.debug("Error shutting down recorder: %s", e)
            self._recorder = None

    def _set_mic(self, enabled: bool):
        """
        Enable or disable the recorder's microphone input.

        While disabled the recorder still runs but only sees silence, so it
        will not transcribe NAO's own voice or trigger a recording. We mute
        during TTS and between turns, unmute only while actively listening.
        """
        if self._recorder is None:
            return
        try:
            self._recorder.set_microphone(enabled)
            logger.debug("Recorder microphone %s", "ON" if enabled else "OFF")
        except Exception as e:
            logger.debug("set_microphone(%s) failed: %s", enabled, e)

    # ── Output ────────────────────────────────────────────────────────────────

    def say(self, text: str):
        """
        Speak text via NAO TTS (or print in desktop/simulation mode).

        Eyes go GREEN immediately after speech finishes — the child's visual
        cue that it is their turn to speak.
        """
        if not text or not text.strip():
            return
        text = self.strip_non_bmp(text)
        text = self._apply_pronunciation_overrides(text)
        self._last_leo_text = text   # kept for Whisper context on next listen()
        logger.info("LEO: %s", text)
        self._set_last_utterance(text)
        if self.simulation_mode:
            self._simulated_history.append({"speaker": "LEO", "text": text})
        self._log_event("utterance", speaker="LEO", text=text)
        if self.simulation_mode or not self.use_nao_output or not self.nao:
            print(f"\n[LEO]: {text}\n")
        else:
            from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
            # Mute the mic so the recorder ignores NAO's own voice.
            self._set_mic(False)
            self.nao.tts.request(NaoqiTextToSpeechRequest(text, language="Dutch"))
            time.sleep(len(text) * 0.01)
            time.sleep(0.2)               # let the last syllable's echo clear
            self._set_eyes("green")       # ← child's turn; fires as soon as speech ends

    # ── Input ─────────────────────────────────────────────────────────────────

    def listen(self) -> str:
        """
        Single listen attempt — returns transcript or "" if nothing heard.

        Eyes are already green from say(). They reset to white once a
        transcript is received.
        """
        if self.simulation_mode:
            logger.info("Simulating child response...")
            transcript = self._generate_simulated_response()
            logger.info("Simulated child: %s", transcript or "(nothing)")
            self._simulated_history.append({"speaker": "CHILD", "text": transcript or "(nothing)"})
            self._log_event("utterance", speaker="CHILD", text=transcript or "(nothing)", simulated=True)
            return transcript

        if self._use_keyboard_input():
            transcript = input("[CHILD]: ").strip()
            logger.info("Child typed: %s", transcript or "(nothing)")
            self._log_event("utterance", speaker="CHILD", text=transcript or "(nothing)", input_mode="keyboard")
            return transcript

        logger.info("Listening via RealtimeSTT...")
        try:
            # Update Whisper context with Leo's last question before decoding.
            self._set_context_prompt()

            # Unmute the mic — now the recorder will hear the child.
            self._set_mic(True)

            # Blocks until VAD silence cut-off, then returns Whisper transcription.
            transcript = self._recorder.text()
            transcript = transcript.strip() if transcript else ""

            self._set_eyes("white")

            # Print prominently so the researcher can monitor in the terminal.
            _print_transcript(transcript)

            logger.info("Child: %s", transcript or "(nothing)")
            self._log_event(
                "utterance",
                speaker="CHILD",
                text=transcript or "(nothing)",
                input_mode="microphone",
                stt_backend="RealtimeSTT",
            )
            return transcript

        except Exception as e:
            self._set_eyes("white")
            logger.error("STT error: %s", e)
            self._log_event("stt_error", error=str(e))
            return ""
        finally:
            # Always mute again — Leo speaks next and we don't want the
            # recorder transcribing his voice.
            self._set_mic(False)

    def listen_with_retry(self, max_retries: int = None) -> str:
        """
        Listen with automatic retry when nothing is heard.

        Only retries in microphone mode; keyboard and simulation fall
        through to a single listen() call.

        On each empty result Leo says a short Dutch retry prompt (rotated
        across attempts). _last_leo_text is preserved so Whisper context
        still points to the original question, not the retry phrase.

        Returns the first non-empty transcript, or "" after all attempts.
        The caller decides whether to skip or escalate.
        """
        n = max_retries if max_retries is not None else self._max_listen_retries

        if self.simulation_mode or self._use_keyboard_input():
            return self.listen()

        for attempt in range(n):
            transcript = self.listen()
            if not self._is_empty_transcript(transcript):
                return transcript

            if attempt < n - 1:
                saved = self._last_leo_text             # preserve original context
                prompt = _RETRY_PROMPTS[min(attempt, len(_RETRY_PROMPTS) - 1)]
                self._say_system(prompt)
                self._last_leo_text = saved             # restore for Whisper
                self._log_event("listen_retry", attempt=attempt + 1, max_retries=n)

        logger.warning("listen_with_retry: no response after %d attempt(s).", n)
        self._log_event("listen_failed", attempts=n)
        return ""

    def review_transcript(self, transcript: str) -> str:
        """Let the researcher approve the transcript or re-listen."""
        if not self.review_transcripts:
            return transcript

        while True:
            print("\n" + "-" * 72)
            print(f"[HEARD]: {transcript or '(nothing)'}")
            choice = input("Press Enter to continue, or R + Enter to listen again: ").strip().lower()
            print("-" * 72)

            if choice == "":
                self._log_event("transcript_review", action="accepted", transcript=transcript or "(nothing)")
                return transcript
            if choice == "r":
                self._log_event("transcript_review", action="retry_requested", transcript=transcript or "(nothing)")
                self._set_eyes("green")
                transcript = self.listen_with_retry()
                continue

            print("Please press Enter to continue, or type R and press Enter to listen again.")

    def listen_with_review(self) -> str:
        """Listen with retry, then optionally let the researcher approve."""
        if self._use_keyboard_input():
            return self.listen()
        return self.review_transcript(self.listen_with_retry())

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_empty_transcript(transcript: str) -> bool:
        """True if transcript is empty or a known Whisper silence hallucination."""
        if not transcript:
            return True
        cleaned = transcript.strip()
        if not cleaned or len(cleaned) <= 1:
            return True
        return cleaned.lower() in _WHISPER_SILENCE_ARTEFACTS

    def _say_system(self, text: str):
        """
        Speak a retry prompt WITHOUT updating _last_leo_text.

        Keeping _last_leo_text on the original question means Whisper context
        on the next listen() still points to what Leo actually asked.
        Eyes go green after, signalling child's turn again.
        """
        text = self.strip_non_bmp(text)
        logger.info("LEO (retry): %s", text)
        self._log_event("utterance", speaker="LEO", text=text, system_message=True)
        if self.simulation_mode or not self.use_nao_output or not self.nao:
            print(f"\n[LEO]: {text}\n")
            return
        from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
        self._set_mic(False)          # don't transcribe Leo's retry prompt
        self.nao.tts.request(NaoqiTextToSpeechRequest(text, language="Dutch"))
        time.sleep(len(text) * 0.01)
        time.sleep(0.2)
        self._set_eyes("green")

    def _set_context_prompt(self):
        """Update Whisper initial_prompt with Leo's last utterance before decoding."""
        if self._recorder is None:
            return
        if self._last_leo_text:
            prompt = (
                _CONTEXT_PREFIX
                + f'Leo zei net: "{self._last_leo_text}" '
                + "Het kind antwoordt nu: "
            )
        else:
            prompt = _CONTEXT_PREFIX
        try:
            self._recorder.initial_prompt = prompt
        except AttributeError:
            pass  # older RealtimeSTT versions; not fatal

    # ── LEDs ──────────────────────────────────────────────────────────────────

    def _set_eyes(self, color: str):
        """Change NAO eye LED color. Silently ignored in desktop/simulation mode."""
        if not self.nao or not self.use_nao_output or self.simulation_mode:
            return
        try:
            from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
            colors = {
                "green": (0, 1, 0),
                "white": (1, 1, 1),
                "blue":  (0, 0, 1),
                "red":   (1, 0, 0),
                "off":   (0, 0, 0),
            }
            r, g, b = colors.get(color, (1, 1, 1))
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", r, g, b, 0))
        except Exception as e:
            logger.debug("Could not set eye LEDs: %s", e)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def strip_non_bmp(text: str) -> str:
        """Remove emoji-style characters that NAO TTS can choke on."""
        safe_chars = []
        for char in str(text or ""):
            if ord(char) > 0xFFFF:
                continue
            if unicodedata.category(char) in ("So", "Sk"):
                continue
            safe_chars.append(char)
        return "".join(safe_chars)

    def _apply_pronunciation_overrides(self, text: str) -> str:
        if not self._pronunciation_overrides:
            return text
        for word, pronunciation in self._pronunciation_overrides.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', pronunciation, text, flags=re.IGNORECASE)
        return text


# ── Module-level helpers ──────────────────────────────────────────────────────

def _print_transcript(transcript: str):
    """Print the child's transcript prominently for researcher monitoring."""
    line = "─" * 64
    text = transcript if transcript else "(nothing heard)"
    print(f"\n{line}")
    print(f"  CHILD: {text}")
    print(f"{line}\n")
