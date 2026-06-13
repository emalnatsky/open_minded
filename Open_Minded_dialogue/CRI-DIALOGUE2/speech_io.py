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
  white → Leo is speaking or idle
  green → RealtimeSTT/mic is actively listening
  white → transcript received, timeout, error, or cleanup

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

NAO TTS speed:
  Every utterance is prefixed with the ACAPELA control tag \\rspd=92\\ which
  slows NAO's speech to 92% of normal — slightly slower and clearer for
  children. The prefix is applied to both the main say() path and the
  retry-prompt path (_say_system).
"""

import re
import queue
import sys
import threading
import time
import unicodedata
import logging
import os
from difflib import SequenceMatcher

from RealtimeSTT import AudioToTextRecorder

try:
    from terminal_output import LoadingStatus
except Exception:  # pragma: no cover - fallback for standalone imports
    LoadingStatus = None

try:
    import config as _cri_config
except Exception:  # pragma: no cover - standalone imports outside CRI
    _cri_config = None

logger = logging.getLogger(__name__)

_REALTIME_STT_QUEUE_WARNING_TEXT = "Audio queue size exceeds latency limit"
_DEFAULT_STT_QUALITY_FILTER_ENABLED = bool(
    getattr(_cri_config, "STT_QUALITY_FILTER_ENABLED", True)
)
_DEFAULT_STT_QUALITY_FILTER_SCOPE = str(
    getattr(_cri_config, "STT_QUALITY_FILTER_SCOPE", "all") or "all"
)


class _RealtimeSTTQueueWarningFilter(logging.Filter):
    """Suppress noisy RealtimeSTT queue spam while keeping a count for CRI."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.last_size = None
        self._lock = threading.Lock()

    def filter(self, record):
        message = record.getMessage()
        if _REALTIME_STT_QUEUE_WARNING_TEXT not in message:
            return True
        size = None
        match = re.search(r"Current size:\s*(\d+)", message)
        if match:
            try:
                size = int(match.group(1))
            except ValueError:
                size = None
        with self._lock:
            self.count += 1
            self.last_size = size
        return False

    def snapshot(self):
        with self._lock:
            return self.count, self.last_size


_REALTIME_STT_QUEUE_WARNING_FILTER = _RealtimeSTTQueueWarningFilter()


def _install_realtimestt_queue_warning_filter():
    realtimestt_logger = logging.getLogger("realtimestt")
    if _REALTIME_STT_QUEUE_WARNING_FILTER not in realtimestt_logger.filters:
        realtimestt_logger.addFilter(_REALTIME_STT_QUEUE_WARNING_FILTER)
    return _REALTIME_STT_QUEUE_WARNING_FILTER

# ── NAO TTS speed prefix ──────────────────────────────────────────────────────
# ACAPELA control tag understood by NAO's TTS engine. \rspd=92\ sets the
# speaking rate to 92% of normal. Prepended to every utterance Leo speaks.
_TTS_SPEED_PREFIX = "\\rspd=92\\"

# ── Whisper context prefix ────────────────────────────────────────────────────
# Prepended to initial_prompt every turn. Primes Whisper's language model with
# the domain before decoding — biggest single accuracy gain for Dutch children.
_CONTEXT_PREFIX = (
    "Dit is een gesprek in het Nederlands tussen een kind van 9 tot 11 jaar "
    "en een sociale robot genaamd Leo. Het kind vertelt over hobby's, "
    "lievelingseten, school en wat het later wil worden. "
    "Mogelijke woorden: "
    # food: p/b, loanwords, ones Whisper mis-decodes
    "pasta, pizza, pannenkoeken, patat, friet, biefstuk, shoarma, kapsalon, "
    "doner, lasagne, spaghetti, sushi, gyoza, dim sum, mochi, shakshuka; "
    # hobby: voicing-confusable + niche terms
    "voetbal, hockey, padel, gamen, tekenen, dansen, bakken, paardrijden, "
    "skateboarden, turnen, acrogym, basketbal, badminton, keepen, "
    "hobbyhorsen, minecraft, editen; "
    # school strength
    "rekenen, gym, spelling, begrijpend lezen, biologie, geschiedenis, "
    "aardrijkskunde, drama, tekenen, handvaardigheid, taal, natuur, techniek; "
    "verkeer, engels; "
    # aspiration: the hard ones
    "chirurg, profvoetballer, topsporter, youtuber, tiktokker, advocaat, "
    "dierenarts, orthodontist, programmeur, ingenieur, piloot, politieagent, "
    "pizzabakker, architect; "
    # high-frequency yes/no + correction phrases
    "ja, nee, dat klopt, dat klopt niet, weet ik niet, weet ik nog niet. "
)
# ── Known STT mishears, fixed after transcription ─────────────────────────────
# Deterministic repairs for domain words Whisper confuses (p/b, etc.).
# Keys are lowercase, punctuation-stripped. Add as you find them in testing.
_STT_CORRECTIONS = {
    "basta": "pasta",
    "daal": "taal",
    "doa": "taal"
}
# ── Retry prompts (Dutch, rotated so they don't sound identical) ──────────────
_RETRY_PROMPTS = (
    "Ik heb je niet goed gehoord. Kun je het nog een keer proberen?",
    "Sorry, dat heb ik gemist. Kun je het nog eens zeggen?",
    "Ik hoorde je niet goed. Wil je het nog een keer zeggen?",
    "Kun je dat nog een keer zeggen?",
    "Volgens mij miste ik je antwoord. Probeer het nog eens.",
    "Ik verstond je niet helemaal. Zeg het maar nog een keer.",
    "Oeps, dat kwam niet goed bij mij binnen. Wil je het herhalen?",
    "Ik hoorde vooral ruis. Kun je het nog eens proberen?",
    "Dat ging net te zacht voor mij. Zeg het nog maar een keer.",
    "Ik ben even de draad kwijt. Wat zei je?",
)

_SHORT_STT_SAFE_ANSWERS = frozenset({
    "ja", "nee", "jawel", "jazeker", "klopt", "ok", "oke", "pizza", "pasta",
    "lasagne", "sushi", "friet", "patat", "shoarma", "computer", "minecraft",
    "youtube", "tiktok", "hockey", "voetbal", "gamen", "game", "gym",
    "rekenen", "taal", "turnen", "tekenen", "dansen", "bakken",
})

_STT_DOMAIN_WORDS = frozenset({
    "ja", "nee", "niet", "klopt", "ik", "mijn", "jij", "je", "dat", "dit",
    "de", "het", "een", "is", "zijn", "was", "voor", "naar", "met", "op",
    "in", "en", "of", "maar", "wel", "goed", "weet", "denk", "vind",
    "wil", "hoef", "hou", "houd", "van", "leuk",
    "lekker", "lievelingseten", "lievelingsvak", "favoriete", "hobby",
    "school", "later", "worden", "sport", "dieren", "dier", "vrienden",
    "pizza", "pasta", "lasagne", "sushi", "friet", "patat", "shoarma",
    "computer", "minecraft", "youtube", "tiktok", "hockey", "voetbal",
    "gamen", "games", "gym", "rekenen", "taal", "turnen", "tekenen",
    "dansen", "bakken", "muziek", "buiten", "spelen",
})

_FOREIGN_STT_PHRASES = (
    "je suis", "je ne sais pas", "bonjour", "bonsoir", "merci", "oui",
    "avec", "pourquoi", "parce que", "porque", "quiero", "hola", "gracias",
    "vamos", "entonces", "no quiero", "no se", "i don't know",
    "i dont know", "what do you mean", "yes sure", "i want", "i like",
)
# --- Whisper silence hallucinations ────────────────────────────────────────────
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

_TTS_MOJIBAKE_REPLACEMENTS = {
    "â€”": ", ",
    "â€“": ", ",
    "â€•": ", ",
    "âˆ’": "-",
    "â€¦": "...",
    "â€˜": "'",
    "â€™": "'",
    "â€š": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€ž": '"',
    "â†’": " ",
    "â†�": " ",
    "âœ“": " ",
    "âœ”": " ",
    "âœ—": " ",
    "âœ˜": " ",
    "âš ": " ",
    "Ã¢": ", ",
    "Ã©": "é",
    "Ã¨": "è",
    "Ã«": "ë",
    "Ãª": "ê",
    "Ã¯": "ï",
    "Ã¶": "ö",
    "Ã¼": "ü",
    "Ã¡": "á",
    "Ã ": "à",
    "Ã±": "ñ",
    "Ã§": "ç",
    "Â": "",
}

_TTS_CHAR_REPLACEMENTS = {
    "\u00a0": " ",
    "\u1680": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": ", ",
    "\u2014": ", ",
    "\u2015": ", ",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u00b4": "'",
    "\u0060": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u2026": "...",
}


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
        is_memory_risk_turn_fn=None,
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
        stt_model: str = "auto",
        stt_language: str = "nl",
        stt_post_speech_silence: float = 1.6,
        stt_realtime_processing_pause: float = 0.2,
        stt_mic_index: int = None,
        stt_beam_size: int = 5,
        stt_device: str = "auto",
        stt_compute_type: str = "auto",
        stt_gpu_device_index: int = 0,
        tts_char_seconds: float = None,
        tts_tail_buffer_seconds: float = None,
        leo_echo_similarity: float = None,
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
            stt_device:                     RealtimeSTT/Faster-Whisper device
            stt_compute_type:               Faster-Whisper compute type
            stt_gpu_device_index:           CUDA GPU index for transcription
            tts_char_seconds:                Minimum physical speech seconds per character
            tts_tail_buffer_seconds:         Extra silence after NAO speech before listening
            leo_echo_similarity:             Threshold for rejecting transcripts that repeat Leo
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
        self._is_memory_risk_turn = is_memory_risk_turn_fn or (lambda: False)
        self._simulated_history = simulated_history if simulated_history is not None else []
        self._generate_simulated_response = generate_simulated_response_fn or (lambda: "")
        self._set_last_utterance = set_last_utterance_fn or (lambda t: None)
        self._max_listen_retries = max_listen_retries
        self._stt_timeout = self._positive_float(stt_timeout)
        self._stt_phrase_limit = self._positive_float(stt_phrase_limit)
        self._stt_listen_timeout = self._effective_listen_timeout(
            self._stt_timeout,
            self._stt_phrase_limit,
        )
        self._tts_char_seconds = (
            self._positive_float(tts_char_seconds)
            or self._env_positive_float("CRI_TTS_CHAR_SECONDS", 0.055)
        )
        self._tts_tail_buffer_seconds = (
            self._positive_float(tts_tail_buffer_seconds)
            or self._env_positive_float("CRI_TTS_TAIL_BUFFER_SECONDS", 0.8)
        )
        self._leo_echo_similarity = self._bounded_similarity(
            self._positive_float(leo_echo_similarity)
            or self._env_positive_float("CRI_LEO_ECHO_SIMILARITY", 0.82)
        )
        self._stale_audio_tolerance_seconds = self._env_positive_float(
            "CRI_STT_STALE_AUDIO_TOLERANCE_SECONDS",
            0.4,
        )
        self._stt_allowed_latency_limit = self._env_positive_int(
            "CRI_STT_ALLOWED_LATENCY_LIMIT",
            80,
        )
        self._stt_quality_filter_enabled = self._env_bool(
            "CRI_STT_QUALITY_FILTER",
            _DEFAULT_STT_QUALITY_FILTER_ENABLED,
        )
        self._stt_quality_filter_scope = self._env_quality_filter_scope(
            "CRI_STT_QUALITY_FILTER_SCOPE",
            _DEFAULT_STT_QUALITY_FILTER_SCOPE,
        )
        self._stt_queue_warning_filter = _install_realtimestt_queue_warning_filter()
        self._pronunciation_overrides = {}
        if pronunciation_overrides_path:
            import json
            if os.path.exists(pronunciation_overrides_path):
                with open(pronunciation_overrides_path, "r", encoding="utf-8") as f:
                    self._pronunciation_overrides = json.load(f)
                logger.info("Loaded %d TTS pronunciation overrides.", len(self._pronunciation_overrides))

        # Tracks the last Leo utterance so Whisper gets question context next listen().
        self._last_leo_text: str = ""
        self._stt_spinner_lock = threading.Lock()
        self._stt_spinner_stop_event = None
        self._stt_spinner_thread = None
        self._stt_spinner_text = "recording"
        self._stt_spinner_visible = False

        # ── Initialise RealtimeSTT recorder ──────────────────────────────────
        self._recorder = None
        if not simulation_mode and not (use_keyboard_input_fn and use_keyboard_input_fn()):
            resolved_stt_model = (stt_model or "auto").strip() or "auto"
            resolved_stt_device = (stt_device or "auto").strip().lower() or "auto"
            resolved_stt_compute_type = (stt_compute_type or "auto").strip().lower() or "auto"
            if resolved_stt_model.lower() == "auto":
                resolved_stt_model = "small"
            if resolved_stt_device == "auto":
                resolved_stt_device = "cpu"
            if resolved_stt_compute_type == "auto":
                resolved_stt_compute_type = "int8"
            recorder_kwargs = dict(
                model=resolved_stt_model,
                language=stt_language,
                post_speech_silence_duration=stt_post_speech_silence,
                realtime_processing_pause=stt_realtime_processing_pause,
                # Keep RealtimeSTT's audio reader worker alive; CRI mutes it
                # immediately after construction and only unmutes during listen().
                use_microphone=True,
                enable_realtime_transcription=False,  # final result only
                spinner=False,                         # CRI owns terminal status
                beam_size=stt_beam_size,
                device=resolved_stt_device,
                compute_type=resolved_stt_compute_type,
                gpu_device_index=stt_gpu_device_index,
                initial_prompt=_CONTEXT_PREFIX,       # updated per turn in listen()
                # VAD tuning — stricter detection so NAO's own TTS / room noise
                # is less likely to be mistaken for the child speaking.
                silero_sensitivity=0.6,               # higher = stricter "is this speech"
                webrtc_sensitivity=3,                 # 0-3, higher = least sensitive
                min_length_of_recording=0.5,          # ignore <0.5s blips
                min_gap_between_recordings=0.4,       # avoid queued carryover between turns
                pre_recording_buffer_duration=0.25,   # keep only a tiny onset buffer
                allowed_latency_limit=self._stt_allowed_latency_limit,
            )
            if stt_mic_index is not None:
                recorder_kwargs["input_device_index"] = stt_mic_index
            logger.info(
                "Initialising RealtimeSTT recorder "
                "(model=%s, language=%s, silence=%.1fs) — "
                "first run downloads the model...",
                resolved_stt_model, stt_language, stt_post_speech_silence,
            )
            logger.info(
                "RealtimeSTT target: device=%s, compute=%s, gpu=%s.",
                resolved_stt_device,
                resolved_stt_compute_type,
                stt_gpu_device_index,
            )
            self._recorder = AudioToTextRecorder(**recorder_kwargs)
            logger.info("RealtimeSTT recorder ready.")
            # Start muted and purge any startup fragments before the first
            # child turn. listen() is the only place that opens the mic.
            self._reset_stt_audio_state("after_recorder_init")

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def shutdown(self):
        """Cleanly stop the RealtimeSTT recorder. Call at end of session."""
        self._stop_stt_spinner()
        self._set_mic(False)
        self._set_eyes("white")
        if self._recorder is not None:
            self._abort_recorder_safely()
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

    def warm_up_stt(self):
        """
        Warm up Whisper / RealtimeSTT without waiting for microphone speech.

        Important: do NOT call self._recorder.text() here, because text()
        waits for real speech and can block forever during startup.
        """
        if self.simulation_mode or self._use_keyboard_input() or self._recorder is None:
            return
        logger.info("Warming up Whisper with dummy silent audio...")
        try:
            import numpy as np
            self._set_context_prompt()
            self._set_mic(False)
            # 1 second of silent float32 audio at 16 kHz.
            dummy_audio = np.zeros(16000, dtype=np.float32)
            _ = self._recorder.perform_final_transcription(
                audio_bytes=dummy_audio,
                use_prompt=True,
            )
            logger.info("Whisper warm-up complete.")
            self._log_event("stt_warmup", status="ok")
        except Exception as e:
            logger.warning("Whisper warm-up failed: %s", e)
            self._log_event("stt_warmup", status="failed", error=str(e))
        finally:
            self._set_mic(False)

    # ── Output ────────────────────────────────────────────────────────────────
    def _tts_text(self, text: str) -> str:
        """Prefix text with the NAO TTS speed control tag (\\rspd=92\\)."""
        return _TTS_SPEED_PREFIX + text

    def _prepare_leo_output_text(self, text: str) -> str:
        text = self.strip_non_bmp(text)
        text = self._apply_pronunciation_overrides(text)
        return self._sanitize_tts_text(text)

    @classmethod
    def _sanitize_tts_text(cls, text: str) -> str:
        """Normalize Leo text into punctuation/symbols NAO's Dutch TTS can handle."""
        text = str(text or "")
        for bad, replacement in _TTS_MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, replacement)
        text = unicodedata.normalize("NFKC", text)

        safe_chars = []
        for char in text:
            replacement = _TTS_CHAR_REPLACEMENTS.get(char)
            if replacement is not None:
                safe_chars.append(replacement)
                continue
            if ord(char) > 0xFFFF:
                continue
            category = unicodedata.category(char)
            if category[0] == "C":
                continue
            if category[0] == "Z":
                safe_chars.append(" ")
                continue
            if category[0] == "S":
                safe_chars.append(" ")
                continue
            if category[0] == "P" and ord(char) > 0x7F:
                safe_chars.append(" ")
                continue
            safe_chars.append(char)
        return cls._clean_tts_spacing("".join(safe_chars))

    @staticmethod
    def _clean_tts_spacing(text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,;:!?]){2,}", r"\1", text)
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"([,.;:!?])(?=[^\s\"'])", r"\1 ", text)
        text = re.sub(r"\.\s+\.\s+\.", "...", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _ascii_tts_fallback(cls, text: str) -> str:
        fallback = (
            unicodedata.normalize("NFKD", str(text or ""))
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        fallback = cls._clean_tts_spacing(fallback)
        return fallback or "Sorry, ik kan dat nu niet goed zeggen."

    def _request_nao_tts(self, text: str) -> tuple[str, bool]:
        from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest

        fallback_text = self._ascii_tts_fallback(text)
        attempts = (("primary", text), ("ascii_fallback", fallback_text))
        first_error = None
        for attempt, candidate in attempts:
            try:
                self.nao.tts.request(
                    NaoqiTextToSpeechRequest(self._tts_text(candidate), language="Dutch")
                )
                if attempt == "ascii_fallback":
                    self._log_event(
                        "tts_retry",
                        status="ascii_fallback_succeeded",
                        text=candidate,
                        primary_error=str(first_error) if first_error else "",
                    )
                return candidate, True
            except Exception as e:
                if attempt == "primary":
                    first_error = e
                    logger.warning("NAO TTS rejected sanitized text, retrying ASCII fallback: %s", e)
                    self._log_event(
                        "tts_error",
                        status="primary_failed",
                        error=str(e),
                        text=text,
                        fallback_text=fallback_text,
                    )
                    continue
                logger.error("NAO TTS failed after ASCII fallback: %s", e)
                self._log_event(
                    "tts_error",
                    status="failed",
                    error=str(e),
                    primary_error=str(first_error) if first_error else "",
                    text=text,
                    fallback_text=fallback_text,
                )
                print(f"\n[TTS ERROR]: Leo kon deze zin niet uitspreken. {e}\n")
                return text, False

    def _wait_for_tts_handoff(self, text: str, started_at: float, finished_at: float = None):
        """Keep the mic muted until NAO's physical speech and echo tail should be gone."""
        finished_at = time.monotonic() if finished_at is None else finished_at
        elapsed = max(0.0, finished_at - started_at)
        estimated_speech = len(str(text or "")) * self._tts_char_seconds
        remaining_speech = max(0.0, estimated_speech - elapsed)
        wait_seconds = remaining_speech + self._tts_tail_buffer_seconds
        self._log_event(
            "tts_handoff_wait",
            elapsed=round(elapsed, 3),
            estimated_speech=round(estimated_speech, 3),
            wait_seconds=round(wait_seconds, 3),
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _print_leo_terminal(self, text: str):
        """Show Leo's spoken text after clearing any active STT status line."""
        self._stop_stt_spinner()
        if LoadingStatus is not None:
            LoadingStatus.stop_active(keep_line=True)
        print(f"\n[LEO]: {text}\n")

    def say(self, text: str):
        """
        Speak text via NAO TTS (or print in desktop/simulation mode).

        Eyes stay WHITE while Leo is speaking. listen() turns them GREEN only
        once the microphone/STT path is actively listening.
        """
        if not text or not text.strip():
            return
        text = self._prepare_leo_output_text(text)
        if not text:
            return
        self._last_leo_text = text   # kept for Whisper context on next listen()
        logger.info("LEO: %s", text)
        self._set_last_utterance(text)
        if self.simulation_mode:
            self._simulated_history.append({"speaker": "LEO", "text": text})
        self._log_event("utterance", speaker="LEO", text=text)
        self._reset_stt_audio_state("before_leo_speech")
        self._print_leo_terminal(text)
        if self.simulation_mode or not self.use_nao_output or not self.nao:
            return
        else:
            # Mute the mic so the recorder ignores NAO's own voice.
            self._set_mic(False)
            self._set_eyes("white")
            started_at = time.monotonic()
            spoken_text, spoken = self._request_nao_tts(text)
            if spoken:
                self._wait_for_tts_handoff(spoken_text, started_at)

    # ── Input ─────────────────────────────────────────────────────────────────
    def listen(self) -> str:
        """
        Single listen attempt — returns transcript or "" if nothing heard.

        Eyes turn GREEN only while microphone/STT listening is active, then
        return to WHITE before transcript output, retry review, or cleanup.
        """
        if self.simulation_mode:
            logger.info("Simulating child response...")
            transcript = self._generate_simulated_response()
            logger.info("Simulated child: %s", transcript or "(nothing)")
            self._simulated_history.append({"speaker": "CHILD", "text": transcript or "(nothing)"})
            self._log_event("utterance", speaker="CHILD", text=transcript or "(nothing)", simulated=True)
            return transcript

        if self._use_keyboard_input():
            self._stop_stt_spinner()
            transcript = input("[CHILD]: ").strip()
            logger.info("Child typed: %s", transcript or "(nothing)")
            self._log_event("utterance", speaker="CHILD", text=transcript or "(nothing)", input_mode="keyboard")
            return transcript

        logger.info("Listening via RealtimeSTT...")
        queue_warning_snapshot = None
        try:
            # Update Whisper context with Leo's last question before decoding.
            self._set_context_prompt()
            self._reset_stt_audio_state("before_listen")
            listen_started_at = time.time()
            queue_warning_snapshot = self._stt_queue_warning_snapshot()
            # Unmute the mic — now the recorder will hear the child.
            self._set_mic(True)
            self._set_eyes("green")
            self._start_stt_spinner("recording")
            # Blocks until VAD silence cut-off, then returns Whisper transcription.
            transcript, timed_out = self._recorder_text_with_timeout()
            stale_audio = self._recording_started_before(listen_started_at)
            self._set_mic(False)
            if timed_out:
                self._set_eyes("white")
                self._stop_stt_spinner()
                self._report_stt_queue_overflow_if_needed(queue_warning_snapshot, "timeout")
                self._reset_stt_audio_state("after_timeout")
                return ""
            transcript = transcript.strip() if transcript else ""
            if transcript and stale_audio:
                logger.warning("Rejected stale STT audio from before current listen: %s", transcript)
                self._log_event("stt_rejected", reason="stale_audio", text=transcript)
                self._set_eyes("white")
                self._stop_stt_spinner()
                self._report_stt_queue_overflow_if_needed(queue_warning_snapshot, "stale_audio")
                self._reset_stt_audio_state("after_stale_audio")
                return ""
            transcript = self._clean_stt_transcript(transcript)
            stt_rejection_reason = self._stt_quality_runtime_rejection_reason(transcript)
            if stt_rejection_reason:
                logger.warning("Rejected STT transcript (%s): %s", stt_rejection_reason, transcript)
                self._log_event("stt_rejected", reason=stt_rejection_reason, text=transcript)
                self._set_eyes("white")
                self._stop_stt_spinner()
                self._report_stt_queue_overflow_if_needed(queue_warning_snapshot, stt_rejection_reason)
                self._reset_stt_audio_state(f"after_{stt_rejection_reason}")
                return ""
            self._set_eyes("white")
            self._stop_stt_spinner()
            self._report_stt_queue_overflow_if_needed(queue_warning_snapshot, "listen")
            self._reset_stt_audio_state("after_success")
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
            self._stop_stt_spinner()
            self._report_stt_queue_overflow_if_needed(queue_warning_snapshot, "error")
            self._reset_stt_audio_state("after_error")
            logger.error("STT error: %s", e)
            self._log_event("stt_error", error=str(e))
            return ""
        finally:
            # Always mute again — Leo speaks next and we don't want the
            # recorder transcribing his voice.
            self._set_mic(False)
            self._set_eyes("white")
            self._stop_stt_spinner()

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
            self._stop_stt_spinner()
            print("\n" + "-" * 72)
            print(f"[HEARD]: {transcript or '(nothing)'}")
            self._stop_stt_spinner()
            choice = input("Press Enter to continue, or R + Enter to listen again: ").strip().lower()
            self._stop_stt_spinner()
            print("-" * 72)
            if choice == "":
                self._log_event("transcript_review", action="accepted", transcript=transcript or "(nothing)")
                return transcript
            if choice == "r":
                self._log_event("transcript_review", action="retry_requested", transcript=transcript or "(nothing)")
                transcript = self.listen_with_retry()
                continue
            self._stop_stt_spinner()
            print("Please press Enter to continue, or type R and press Enter to listen again.")

    def listen_with_review(self) -> str:
        """Listen with retry, then optionally let the researcher approve."""
        if self._use_keyboard_input():
            return self.listen()
        return self.review_transcript(self.listen_with_retry())

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _positive_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @classmethod
    def _env_positive_float(cls, name: str, default: float) -> float:
        return cls._positive_float(os.environ.get(name)) or default

    @staticmethod
    def _env_positive_int(name: str, default: int) -> int:
        try:
            number = int(str(os.environ.get(name, "")).strip())
        except (TypeError, ValueError):
            return default
        return number if number > 0 else default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _normalise_quality_filter_scope(value: str) -> str:
        scope = str(value or "memory").strip().lower()
        if scope in {"0", "false", "off", "none", "disabled"}:
            return "off"
        if scope == "all":
            return "all"
        return "memory"

    @classmethod
    def _env_quality_filter_scope(cls, name: str, default: str) -> str:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            raw = default
        return cls._normalise_quality_filter_scope(raw)

    @staticmethod
    def _bounded_similarity(value: float) -> float:
        if value is None:
            return 0.82
        return max(0.0, min(float(value), 1.0))

    def _stt_queue_warning_snapshot(self):
        filter_obj = getattr(self, "_stt_queue_warning_filter", None)
        if filter_obj is None:
            return 0, None
        return filter_obj.snapshot()

    def _report_stt_queue_overflow_if_needed(self, start_snapshot, reason: str):
        filter_obj = getattr(self, "_stt_queue_warning_filter", None)
        if filter_obj is None or start_snapshot is None:
            return
        start_count, _start_size = start_snapshot
        current_count, current_size = filter_obj.snapshot()
        delta = max(0, current_count - start_count)
        if not delta:
            return
        self._log_event(
            "stt_queue_overflow",
            reason=reason,
            warnings=delta,
            last_size=current_size,
            allowed_latency_limit=self._stt_allowed_latency_limit,
        )
        logger.warning(
            "RealtimeSTT audio queue overflowed during %s (%d warning%s suppressed, "
            "last size=%s, limit=%s).",
            reason,
            delta,
            "" if delta == 1 else "s",
            current_size if current_size is not None else "unknown",
            self._stt_allowed_latency_limit,
        )

    @staticmethod
    def _effective_listen_timeout(*values):
        positive = [value for value in values if value and value > 0]
        return max(positive) if positive else None

    def _clear_terminal_status_line(self):
        """Clear one STT spinner/status line without adding dialogue text."""
        try:
            sys.stdout.write("\r" + (" " * 96) + "\r")
            sys.stdout.flush()
            self._stt_spinner_visible = False
        except Exception as e:
            logger.debug("Could not clear terminal status line: %s", e)

    def _start_stt_spinner(self, text: str = "recording"):
        """Start CRI-owned STT status while the microphone is open."""
        self._disable_realtimestt_halo()
        with self._stt_spinner_lock:
            thread = self._stt_spinner_thread
            if thread is not None and thread.is_alive():
                self._stt_spinner_text = text
                return
            stop_event = threading.Event()
            self._stt_spinner_stop_event = stop_event
            self._stt_spinner_text = text
            thread = threading.Thread(
                target=self._run_stt_spinner,
                args=(stop_event,),
                name="cri-stt-spinner",
                daemon=True,
            )
            self._stt_spinner_thread = thread
        thread.start()

    def _run_stt_spinner(self, stop_event):
        frames = ("|", "/", "-", "\\")
        frame_index = 0
        while not stop_event.is_set():
            try:
                with self._stt_spinner_lock:
                    text = self._stt_spinner_text
                sys.stdout.write(f"\r{frames[frame_index % len(frames)]} {text}")
                sys.stdout.flush()
                self._stt_spinner_visible = True
            except Exception as e:
                logger.debug("Could not write STT spinner: %s", e)
                break
            frame_index += 1
            stop_event.wait(0.12)

    def _disable_realtimestt_halo(self):
        """Keep RealtimeSTT's own Halo spinner from recreating itself."""
        recorder = self._recorder
        if recorder is None:
            return False
        disabled = False
        try:
            recorder.spinner = False
        except Exception:
            pass
        halo = getattr(recorder, "halo", None)
        if halo is not None:
            disabled = True
            try:
                halo.stop()
            except Exception as e:
                logger.debug("Could not stop RealtimeSTT spinner: %s", e)
            try:
                recorder.halo = None
            except Exception:
                pass
        return disabled

    def _stop_stt_spinner(self):
        """
        Stop CRI's STT spinner and clear any leftover RealtimeSTT Halo output.

        The dialogue owns the terminal before printing Leo, child, or
        researcher-review text.
        """
        thread = None
        with self._stt_spinner_lock:
            stop_event = self._stt_spinner_stop_event
            if stop_event is not None:
                stop_event.set()
            thread = self._stt_spinner_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        with self._stt_spinner_lock:
            if self._stt_spinner_thread is thread:
                self._stt_spinner_thread = None
                self._stt_spinner_stop_event = None
        had_halo = self._disable_realtimestt_halo()
        if thread is not None or had_halo or self._stt_spinner_visible:
            self._clear_terminal_status_line()
            time.sleep(0.02)
            self._clear_terminal_status_line()

    @staticmethod
    def _drain_queue(q) -> int:
        if q is None:
            return 0
        drained = 0
        while True:
            try:
                q.get_nowait()
                drained += 1
            except queue.Empty:
                break
            except Exception:
                break
        return drained

    @staticmethod
    def _clear_collection(obj) -> bool:
        if obj is None:
            return False
        try:
            obj.clear()
            return True
        except AttributeError:
            return False
        except Exception:
            return False

    def _reset_stt_audio_state(self, reason: str):
        """Drop all queued/remembered STT audio so the next listen starts fresh."""
        recorder = self._recorder
        if recorder is None:
            return
        self._set_mic(False)

        try:
            recorder.clear_audio_queue()
        except AttributeError:
            pass
        except Exception as e:
            logger.debug("Could not clear recorder audio queue during %s: %s", reason, e)

        raw_drained = self._drain_queue(getattr(recorder, "audio_queue", None))
        recorded_drained = self._drain_queue(getattr(recorder, "recorded_audio_queue", None))

        for attr in (
            "frames",
            "last_frames",
            "audio_buffer",
            "audio_buffer_metadata",
            "last_words_buffer",
            "text_storage",
        ):
            self._clear_collection(getattr(recorder, attr, None))

        for attr, value in (
            ("audio", None),
            ("last_transcription_bytes", None),
            ("last_transcription_bytes_b64", None),
            ("last_transcription_metadata", None),
            ("last_preroll_selection", None),
            ("_pending_preroll_selection", None),
            ("realtime_stabilized_text", ""),
            ("realtime_stabilized_safetext", ""),
            ("continuous_listening", False),
            ("start_recording_on_voice_activity", False),
            ("stop_recording_on_voice_deactivity", False),
            ("is_recording", False),
            ("is_webrtc_speech_active", False),
            ("is_silero_speech_active", False),
            ("wakeword_detected", False),
            ("wake_word_detect_time", 0),
            ("listen_start", 0),
            ("recording_start_time", 0),
            ("recording_start_monotonic", 0),
            ("recording_stop_time", 0),
            ("last_recording_start_time", 0),
            ("last_recording_stop_time", 0),
            ("backdate_stop_seconds", 0.0),
            ("backdate_resume_seconds", 0.0),
            ("speech_end_silence_start", 0),
            ("speech_end_silence_candidate_start", 0),
            ("silero_check_time", 0),
        ):
            try:
                setattr(recorder, attr, value)
            except Exception:
                pass

        for attr in ("start_recording_event", "stop_recording_event"):
            event = getattr(recorder, attr, None)
            if event is not None:
                try:
                    event.clear()
                except Exception:
                    pass

        if raw_drained or recorded_drained:
            self._log_event(
                "stt_audio_reset",
                reason=reason,
                raw_audio_chunks=raw_drained,
                recorded_chunks=recorded_drained,
            )
        logger.debug(
            "STT audio reset (%s): raw=%d recorded=%d",
            reason,
            raw_drained,
            recorded_drained,
        )

    def _recording_started_before(self, listen_started_at: float) -> bool:
        """True when RealtimeSTT appears to have returned audio from an older turn."""
        recorder = self._recorder
        if recorder is None or not listen_started_at:
            return False
        starts = []
        for attr in ("last_recording_start_time", "recording_start_time"):
            try:
                value = float(getattr(recorder, attr, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                starts.append(value)
        if not starts:
            return False
        return min(starts) < (listen_started_at - self._stale_audio_tolerance_seconds)

    def _clear_recorder_audio_queue(self):
        """Backward-compatible alias for the full RealtimeSTT state reset."""
        self._reset_stt_audio_state("clear_recorder_audio_queue")

    def _abort_recorder_safely(self):
        """Interrupt a stuck RealtimeSTT listen without blocking the dialogue."""
        recorder = self._recorder
        if recorder is None:
            return
        try:
            interrupt = getattr(recorder, "interrupt_stop_event", None)
            if interrupt is not None:
                interrupt.set()
        except Exception as e:
            logger.debug("Could not set RealtimeSTT interrupt event: %s", e)

        def abort():
            try:
                recorder.abort()
            except AttributeError:
                pass
            except Exception as e:
                logger.debug("RealtimeSTT abort failed: %s", e)

        thread = threading.Thread(target=abort, name="cri-stt-abort", daemon=True)
        thread.start()
        thread.join(timeout=1.0)
        if thread.is_alive():
            logger.warning("RealtimeSTT abort did not finish within 1 second.")

    def _recorder_text_with_timeout(self):
        """Return (transcript, timed_out) from RealtimeSTT."""
        if self._recorder is None:
            return "", False
        timeout = self._stt_listen_timeout
        if not timeout:
            return self._recorder.text(), False

        result_queue = queue.Queue(maxsize=1)

        def transcribe():
            try:
                result_queue.put(("ok", self._recorder.text()))
            except BaseException as e:
                result_queue.put(("error", e))

        thread = threading.Thread(
            target=transcribe,
            name="cri-stt-text",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            self._abort_recorder_safely()
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("RealtimeSTT text worker did not stop within 1 second.")
            self._set_mic(False)
            self._stop_stt_spinner()
            self._log_event(
                "stt_timeout",
                timeout=timeout,
                stt_timeout=self._stt_timeout,
                stt_phrase_limit=self._stt_phrase_limit,
            )
            logger.warning("RealtimeSTT listen timed out after %.2f seconds.", timeout)
            return "", True

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            return "", False
        if status == "error":
            raise payload
        return payload, False

    @staticmethod
    def _is_empty_transcript(transcript: str) -> bool:
        """True if transcript is empty or a known Whisper silence hallucination."""
        if not transcript:
            return True
        cleaned = transcript.strip()
        if not cleaned or len(cleaned) <= 1:
            return True
        return cleaned.lower() in _WHISPER_SILENCE_ARTEFACTS

    def _clean_stt_transcript(self, transcript: str) -> str:
        """Apply deterministic repairs and reject obvious STT hallucination loops."""
        if not transcript:
            return ""
        fixed = self._fix_known_mishears(transcript.strip())
        collapsed = self._collapse_repetition_loop(fixed)
        if collapsed != fixed:
            self._log_event(
                "stt_repetition_collapsed",
                original=fixed,
                collapsed=collapsed,
            )
            fixed = collapsed
        if self._looks_like_suspicious_loop(fixed):
            logger.warning("Rejected suspicious STT transcript: %s", fixed)
            self._log_event("stt_rejected", reason="suspicious_loop", text=fixed)
            return ""
        echo, similarity = self._looks_like_leo_echo(fixed)
        if echo:
            logger.warning("Rejected likely Leo echo transcript: %s", fixed)
            self._log_event(
                "stt_rejected",
                reason="leo_echo",
                text=fixed,
                leo_text=self._last_leo_text,
                similarity=round(similarity, 3),
            )
            return ""
        return self.normalize_transcript(fixed)

    def _stt_quality_rejection_reason(self, transcript: str) -> str:
        """Return a rejection reason for obvious non-Dutch or gibberish STT."""
        if not transcript:
            return ""
        text = self.normalize_transcript(transcript).lower().strip()
        words = self._normalise_stt_words(text)
        if not words:
            return "suspected_gibberish"

        compact = " ".join(words)
        if compact in _SHORT_STT_SAFE_ANSWERS:
            return ""
        if len(words) <= 2 and all(word.isalnum() for word in words):
            if not any(self._contains_phrase(compact, phrase) for phrase in _FOREIGN_STT_PHRASES):
                return ""

        if any(self._contains_phrase(compact, phrase) for phrase in _FOREIGN_STT_PHRASES):
            return "suspected_non_dutch"

        detected_language, language_probability = self._detected_stt_language()
        if (
            detected_language
            and detected_language not in {"nl", "nld", "dutch"}
            and language_probability >= 0.80
            and not self._has_dutch_or_domain_signal(words)
        ):
            return "suspected_non_dutch"

        if self._looks_like_gibberish_text(text, words):
            return "suspected_gibberish"
        return ""

    def stt_quality_rejection_reason(self, transcript: str) -> str:
        """Public wrapper used by tests/demo code without touching internals."""
        return self._stt_quality_rejection_reason(transcript)

    def stt_quality_filter_decision(
        self,
        transcript: str,
        *,
        memory_risk: bool = False,
        scope: str = None,
        enabled: bool = None,
    ) -> dict:
        """Return how the STT quality filter would handle a transcript."""
        reason = self._stt_quality_rejection_reason(transcript)
        filter_enabled = self._stt_quality_filter_enabled if enabled is None else bool(enabled)
        filter_scope = self._normalise_quality_filter_scope(
            self._stt_quality_filter_scope if scope is None else scope
        )
        reject = bool(
            filter_enabled
            and reason
            and filter_scope != "off"
            and (filter_scope == "all" or (filter_scope == "memory" and memory_risk))
        )
        if reject:
            result = f"reject:{reason}"
        elif reason and filter_enabled and filter_scope == "memory" and not memory_risk:
            result = f"pass-log:{reason}"
        else:
            result = "pass"
        return {
            "result": result,
            "reject": reject,
            "reason": reason,
            "scope": filter_scope,
            "enabled": filter_enabled,
            "memory_risk": bool(memory_risk),
        }

    def _stt_quality_runtime_rejection_reason(self, transcript: str) -> str:
        """Apply strict STT filtering only in the configured runtime scope."""
        decision = self.stt_quality_filter_decision(
            transcript,
            memory_risk=self._current_turn_is_memory_risk(),
        )
        reason = decision["reason"]
        if not reason:
            return ""
        if decision["reject"]:
            return reason
        if decision["result"].startswith("pass-log"):
            self._log_event(
                "stt_suspicious_passed",
                reason=reason,
                text=transcript,
                scope=decision["scope"],
                memory_risk=decision["memory_risk"],
            )
            logger.debug(
                "Passed suspicious STT transcript outside memory-risk scope (%s): %s",
                reason,
                transcript,
            )
        return ""

    def _current_turn_is_memory_risk(self) -> bool:
        try:
            return bool(self._is_memory_risk_turn())
        except Exception as e:
            logger.debug("Could not determine STT memory-risk turn: %s", e)
            return False

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        phrase = phrase.lower().strip()
        if not phrase:
            return False
        if " " in phrase or "'" in phrase:
            return phrase in text
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    def _detected_stt_language(self) -> tuple[str, float]:
        recorder = getattr(self, "_recorder", None)
        language = str(getattr(recorder, "detected_language", "") or "").lower()
        try:
            probability = float(getattr(recorder, "detected_language_probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            probability = 0.0
        return language, probability

    @staticmethod
    def _has_dutch_or_domain_signal(words: list[str]) -> bool:
        return any(word in _STT_DOMAIN_WORDS or word in _SHORT_STT_SAFE_ANSWERS for word in words)

    @classmethod
    def _looks_like_gibberish_text(cls, text: str, words: list[str]) -> bool:
        if len(text) >= 8:
            alnum = sum(ch.isalnum() for ch in text)
            if alnum / max(1, len(text)) < 0.45:
                return True
        if any(len(word) > 24 for word in words):
            return True
        if len(words) >= 4 and not cls._has_dutch_or_domain_signal(words):
            vowel_words = sum(1 for word in words if re.search(r"[aeiou]", word))
            if vowel_words / len(words) < 0.35:
                return True
        return False

    @staticmethod
    def _normalise_stt_words(text: str):
        return re.findall(r"[a-z0-9À-ÖØ-öø-ÿ]+(?:['-][a-z0-9À-ÖØ-öø-ÿ]+)?", text.lower())

    @classmethod
    def _collapse_repetition_loop(cls, transcript: str) -> str:
        words = cls._normalise_stt_words(transcript)
        if len(words) < 6:
            return transcript

        comma_parts = [
            " ".join(cls._normalise_stt_words(part))
            for part in re.split(r"[,;]+", transcript)
        ]
        comma_parts = [part for part in comma_parts if part]
        if len(comma_parts) >= 3 and len(set(comma_parts)) == 1:
            return comma_parts[0]

        for phrase_len in range(1, min(5, len(words) // 3 + 1)):
            phrase = words[:phrase_len]
            chunks = [
                words[i:i + phrase_len]
                for i in range(0, len(words), phrase_len)
                if len(words[i:i + phrase_len]) == phrase_len
            ]
            if len(chunks) < 3:
                continue
            matches = sum(1 for chunk in chunks if chunk == phrase)
            if matches / len(chunks) >= 0.8:
                return " ".join(phrase)

        return transcript

    @classmethod
    def _looks_like_suspicious_loop(cls, transcript: str) -> bool:
        words = cls._normalise_stt_words(transcript)
        if len(words) < 30:
            return False
        unique_ratio = len(set(words)) / max(1, len(words))
        return unique_ratio < 0.25

    def _looks_like_leo_echo(self, transcript: str) -> tuple[bool, float]:
        transcript_words = self._normalise_stt_words(transcript)
        leo_words = self._normalise_stt_words(self._last_leo_text)
        if len(transcript_words) < 4 or len(leo_words) < 4:
            return False, 0.0

        transcript_text = " ".join(transcript_words)
        leo_text = " ".join(leo_words)
        if len(transcript_text) < 18:
            return False, 0.0

        if transcript_text == leo_text or transcript_text in leo_text:
            return True, 1.0

        similarity = SequenceMatcher(None, transcript_text, leo_text).ratio()
        return similarity >= self._leo_echo_similarity, similarity

    @staticmethod
    def normalize_transcript(text: str) -> str:
        """Flatten STT output before classification while keeping readable words."""
        text = str(text or "")
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", "").replace("\u201d", "")
        text = re.sub(r"[\u2012\u2013\u2014\u2015\u2212-]", " ", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _fix_known_mishears(self, transcript: str) -> str:
        """Repair domain-specific STT confusions (e.g. basta→pasta)."""
        if not transcript:
            return transcript
        fixed = []
        for word in transcript.split():
            key = word.lower().strip(".,!?")
            if key in _STT_CORRECTIONS:
                replacement = _STT_CORRECTIONS[key]
                # carry over trailing punctuation from the original token
                trailer = word[len(word.rstrip(".,!?")):]
                fixed.append(replacement + trailer)
            else:
                fixed.append(word)
        return " ".join(fixed)

    def _say_system(self, text: str):
        """
        Speak a retry prompt WITHOUT updating _last_leo_text.

        Keeping _last_leo_text on the original question means Whisper context
        on the next listen() still points to what Leo actually asked.

        Eyes stay WHITE while Leo speaks. The next listen() call owns the
        GREEN active-listening cue.
        """
        text = self._prepare_leo_output_text(text)
        if not text:
            return
        logger.info("LEO (retry): %s", text)
        self._log_event("utterance", speaker="LEO", text=text, system_message=True)
        self._reset_stt_audio_state("before_retry_prompt")
        self._print_leo_terminal(text)
        if self.simulation_mode or not self.use_nao_output or not self.nao:
            return
        self._set_mic(False)          # don't transcribe Leo's retry prompt
        self._set_eyes("white")
        started_at = time.monotonic()
        spoken_text, spoken = self._request_nao_tts(text)
        if spoken:
            self._wait_for_tts_handoff(spoken_text, started_at)

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
