"""
Speech input/output for the CRI dialogue.
SpeechIO wraps:
  - NAO TTS (say)
  - Whisper STT (listen)
  - Keyboard input fallback (listen)
  - Transcript review loop (review_transcript / listen_with_review)
  - strip_non_bmp — sanitise text before sending to NAO TTS

Constructed once in CRI_ScriptedDialogue.setup() and stored as self.speech.
The dialogue calls self.speech.say(...) and self.speech.listen_with_review().

Dependencies injected at construction time so this module never imports
from the dialogue class or from config directly.
"""

import time
import unicodedata
import logging

logger = logging.getLogger(__name__)


class SpeechIO:
    """Handles all audio in/out for one session."""

    def __init__(
        self,
        *,
        nao=None,
        whisper=None,
        use_desktop_mic: bool = False,
        simulation_mode: bool = False,
        use_keyboard_input_fn=None,
        stt_timeout: int = 20,
        stt_phrase_limit: int = 18,
        review_transcripts: bool = True,
        log_event_fn=None,
        simulated_history: list = None,
        generate_simulated_response_fn=None,
        set_last_utterance_fn=None,
    ):
        """
        Args:
            nao:                          SIC NAO device handle (or None for desktop mode)
            whisper:                      SIC Whisper handle (or None for keyboard mode)
            use_desktop_mic:              True → skip NAO, print Leo's lines to terminal
            simulation_mode:              True → use simulated child responses
            use_keyboard_input_fn:        callable() → bool, returns True when in keyboard mode
            stt_timeout:                  seconds Whisper waits for any speech
            stt_phrase_limit:             seconds max for a single Whisper phrase
            review_transcripts:           True → show transcript review loop after listening
            log_event_fn:                 callable(event_type, **kwargs) for conversation logging
            simulated_history:            list to append {"speaker", "text"} dicts to
            generate_simulated_response_fn: callable() → str for simulation mode
            set_last_utterance_fn:        callable(text) to update self.last_leo_utterance
        """
        self.nao = nao
        self.whisper = whisper
        self.use_desktop_mic = use_desktop_mic
        self.simulation_mode = simulation_mode
        self._use_keyboard_input = use_keyboard_input_fn or (lambda: False)
        self.stt_timeout = stt_timeout
        self.stt_phrase_limit = stt_phrase_limit
        self.review_transcripts = review_transcripts
        self._log_event = log_event_fn or (lambda *a, **kw: None)
        self._simulated_history = simulated_history if simulated_history is not None else []
        self._generate_simulated_response = generate_simulated_response_fn or (lambda: "")
        self._set_last_utterance = set_last_utterance_fn or (lambda t: None)

    # ── Output ────────────────────────────────────────────────────────────────

    def say(self, text: str):
        """Speak text via NAO TTS (or print in desktop/simulation mode)."""
        if not text or not text.strip():
            return
        text = self.strip_non_bmp(text)
        logger.info("LEO: %s", text)
        self._set_last_utterance(text)
        if self.simulation_mode:
            self._simulated_history.append({"speaker": "LEO", "text": text})
        self._log_event("utterance", speaker="LEO", text=text)
        if self.simulation_mode or self.use_desktop_mic:
            print(f"\n[LEO]: {text}\n")
        else:
            from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
            # ~0.01s per character is a safe estimate for NAO's TTS speed.
            time.sleep(len(text) * 0.01)

    # ── Input ─────────────────────────────────────────────────────────────────

    def listen(self) -> str:
        """Capture child speech and return transcript string."""
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

        logger.info("Listening...")
        try:
            # Green eyes = Leo is listening
            self._set_eyes("green")

            from sic_framework.services.openai_whisper_stt.whisper_stt import GetTranscript
            result = self.whisper.request(
                GetTranscript(
                    timeout=self.stt_timeout,
                    phrase_time_limit=self.stt_phrase_limit,
                )
            )
            transcript = result.transcript.strip() if result and result.transcript else ""

            # Back to white = Leo is done listening
            self._set_eyes("white")

            logger.info("Child: %s", transcript or "(nothing)")
            self._log_event("utterance", speaker="CHILD", text=transcript or "(nothing)", input_mode="microphone")
            return transcript
        except Exception as e:
            self._set_eyes("white")  # always restore on error
            logger.error("STT error: %s", e)
            self._log_event("stt_error", error=str(e))
            return ""

    def review_transcript(self, transcript: str) -> str:
        """Let the researcher approve Whisper text or listen again before continuing."""
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
                transcript = self.listen()
                continue

            print("Please press Enter to continue, or type R and press Enter to listen again.")

    def listen_with_review(self) -> str:
        """Listen once, then optionally let the researcher approve the transcript."""
        if self._use_keyboard_input():
            return self.listen()
        return self.review_transcript(self.listen())

    # ── Utility ───────────────────────────────────────────────────────────────

    def _set_eyes(self, color: str):
        """Change NAO eye LED color. Silently ignored in desktop/simulation mode."""
        if not self.nao or self.use_desktop_mic or self.simulation_mode:
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
