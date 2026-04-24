import time
from os import environ
from os.path import abspath, join

from dotenv import load_dotenv
from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoWakeUpRequest,
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)
from sic_framework.services.openai_whisper_stt.whisper_stt import (
    GetTranscript,
    SICWhisper,
    WhisperConf,
)
from sic_framework.services.llm import GPT, GPTConf, GPTRequest
from intent_classifier import StubIntentClassifier


class CRI_ScriptedDialogue(SICApplication):
    """
    Child-Robot Interaction scripted dialogue.

    Full loop: utterance → Whisper STT → intent classifier → branch → NAO TTS

    Layer 1 — intent branch (handles special child utterances):
        inspect         → robot reads back hardcoded UM field (Redis later)
        update_memory   → robot confirms it will remember
        update_dialogue → robot confirms it heard the correction
        delete          → robot confirms it will forget
        question        → robot gives a short canned answer
        social          → robot mirrors briefly

    Layer 2 — normal scripted flow (add / answer / none):
        llm_turn=True   → GPT generates personalised response (later need to be replaced by Nebula)
        llm_turn=False  → hardcoded follow_up line

    Graceful fallback: if LLM fails, uses follow_up as fallback.
    """

    SCRIPT = [
        {
            "question":  "Hallo, ik ben Nao! Hoe heet jij?",
            "follow_up": "Leuk je te ontmoeten!",
            "llm_turn":  False,
        },
        {
            "question":  "Ik vind het zelf leuk om spelletjes te bedenken en mysteries op te lossen. Wat vind je echt leuk om te doen?",
            "follow_up": "Wow, zo cool!",
            "llm_turn":  True,
        },
        {
            "question":  "Welke vakken vind jij leuk?",
            "follow_up": "Ik vind alle vakken leuk.",
            "llm_turn":  True,
        },
        {
            "question":  "Ik hou van mysteries oplossen, ik denk dat ik een hele goede detective zou zijn. Wat zou jij later willen worden?",
            "follow_up": "Dat is fantastisch!",
            "llm_turn":  True,
        },
        {
            "question":  "Wat is iets of iemand die jou inspireert?",
            "follow_up": "Goede keuze! Bedankt voor het gesprek. Tot ziens!",
            "llm_turn":  False,
        },
    ]

    # Whisper settings
    STT_TIMEOUT      = 20
    STT_PHRASE_LIMIT = 15

    # LLM settings
    LLM_FALLBACK = "Wauw, dat klinkt heel leuk!"
    LLM_SYSTEM   = (
        "You are a friendly child robot called Nao talking to a young child aged 8-11. "
        "Reply in short sentences (max 25 words). Be warm, enthusiastic, and simple yet "
        "still interesting and unpredictable. Do not ask a question. Speak in Dutch."
    )

    # Hardcoded UM values for inspect — will be replaced with Redis get_field() later
    HARDCODED_UM = {
        "fav_food":    "pizza",
        "hobby_fav":   "tekenen",
        "aspiration":  "dokter",
        "fav_subject": "gym",
    }

    def __init__(self, openai_env_path=None, nao_ip="10.0.0.165"):
        super(CRI_ScriptedDialogue, self).__init__()

        self.nao_ip          = nao_ip
        self.openai_env_path = openai_env_path
        self.nao             = None
        self.whisper         = None
        self.gpt             = None
        self.clf             = None

        self.set_log_level(sic_logging.INFO)
        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        self.logger.info("Setting up NAO + Whisper STT + GPT + Intent Classifier …")

        if self.openai_env_path:
            load_dotenv(self.openai_env_path)

        if "OPENAI_API_KEY" not in environ:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Set it in your shell or pass openai_env_path pointing to a .env file."
            )

        # Intent classifier — loads field names from um_field_schema.json
        self.clf = StubIntentClassifier()

        # Connect to NAO
        self.logger.info("Connecting to NAO at %s …", self.nao_ip)
        self.nao = Nao(ip=self.nao_ip)
        self.logger.info("NAO connected.")

        # Whisper STT — no callback, blocking requests only
        whisper_conf = WhisperConf(openai_key=environ["OPENAI_API_KEY"])
        self.whisper = SICWhisper(input_source=self.nao.mic, conf=whisper_conf)
        time.sleep(1)

        # GPT — used for llm_turn=True turns
        gpt_conf = GPTConf(
            openai_key=environ["OPENAI_API_KEY"],
            system_message=self.LLM_SYSTEM,
            model="gpt-4o-mini",
            max_tokens=40,
            temp=0.7,
        )
        self.gpt = GPT(conf=gpt_conf)

        self.logger.info("Setup complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def say(self, text: str):
        """Blocking TTS via NAO."""
        self.logger.info("NAO says: %s", text)
        self.nao.tts.request(NaoqiTextToSpeechRequest(text))

    def listen(self) -> str:
        """Blocking STT via Whisper. Returns transcript or empty string."""
        self.logger.info("Listening …")
        try:
            result = self.whisper.request(
                GetTranscript(
                    timeout=self.STT_TIMEOUT,
                    phrase_time_limit=self.STT_PHRASE_LIMIT,
                )
            )
            transcript = result.transcript.strip() if result and result.transcript else ""
            self.logger.info("Child said: %s", transcript or "(nothing recognised)")
            return transcript
        except Exception as e:
            self.logger.error("STT error: %s", e)
            return ""

    def llm_response(self, child_input: str) -> str:
        """
        Send child_input to GPT, return a short Dutch robot utterance.
        Falls back to LLM_FALLBACK on any error or empty response.
        """
        if not child_input:
            self.logger.warning("Empty transcript — using fallback.")
            return self.LLM_FALLBACK

        prompt = (
            f"Het kind zei: \"{child_input}\". "
            f"Reageer warm en enthousiast in één korte zin."
        )
        self.logger.info("Sending to GPT: %s", prompt)

        try:
            reply = self.gpt.request(GPTRequest(prompt=prompt, stream=False))
            response = reply.response.strip() if reply and reply.response else ""

            if not response:
                self.logger.warning("GPT returned empty — using fallback.")
                return self.LLM_FALLBACK

            first_sentence = response.split(".")[0].strip() + "."
            self.logger.info("GPT response: %s", first_sentence)
            return first_sentence

        except Exception as e:
            self.logger.error("LLM error: %s — using fallback.", e)
            return self.LLM_FALLBACK

    def handle_intent(self, result, turn: dict, transcript: str) -> bool:
        """
        Layer 1 — intent branch for special child utterances.

        Returns True  → special intent was handled, skip layer 2.
        Returns False → normal flow, layer 2 runs (LLM or follow_up).

        NOTE: inspect uses HARDCODED_UM for now.
        Replace with get_field(child_id, field) when Redis is connected.
        """
        intent = result.intent
        field  = result.field

        if intent == "inspect":
            # Hardcoded lookup — swap with Redis get_field() later
            value = self.HARDCODED_UM.get(field, "dat weet ik nog niet")
            self.say(f"Ik weet dat jouw {field or 'antwoord'} {value} is.")
            return True

        elif intent in ("update_memory", "update_dialogue"):
            self.say("Oké, ik onthoud dat!")
            return True

        elif intent == "delete":
            self.say("Oké, ik vergeet dat!")
            return True

        elif intent == "question":
            self.say("Dat is een goede vraag! Ik vertel het je later.")
            return True

        elif intent == "social":
            self.say("Haha ja! Oké, verder!")
            return True

        # add / answer / none → fall through to layer 2
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self.logger.info("Starting CRI Scripted Dialogue …")

        try:
            self.nao.autonomous.request(NaoWakeUpRequest())

            for i, turn in enumerate(self.SCRIPT):
                if self.shutdown_event.is_set():
                    break

                # Ask the question
                self.say(turn["question"])
                time.sleep(0.5)

                # Listen — only once per turn
                transcript = self.listen()
                time.sleep(0.8)

                # Classify what the child said
                result = self.clf.classify(transcript)
                self.logger.info("Intent: %s", result.to_dict())

                # Layer 1 — special intent handling
                handled = self.handle_intent(result, turn, transcript)

                # Layer 2 — normal scripted flow (only runs if layer 1 didn't handle it)
                if not handled:
                    if turn["llm_turn"]:
                        follow_up = self.llm_response(transcript)
                    else:
                        follow_up = turn["follow_up"]
                    self.say(follow_up)

                if i < len(self.SCRIPT) - 1:
                    time.sleep(1.5)

            self.logger.info("Dialogue completed successfully.")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        except Exception as e:
            self.logger.error("Error during dialogue: %s", e)
        finally:
            try:
                self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass
            self.logger.info("Shutting down.")
            self.shutdown()


if __name__ == "__main__":
    dialogue_app = CRI_ScriptedDialogue(
        openai_env_path=abspath(join("conf", ".env")),
        nao_ip="10.0.0.165",  # ← replace with your NAO's IP
    )
    dialogue_app.run()
