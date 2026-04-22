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


class CRI_ScriptedDialogue(SICApplication):
    """
    Minimal Child-Robot Interaction scripted dialogue.
    4-5 turn scripted exchange with a simple LLM call injected, so GPT generates
    a short personalised response instead of a canned line (Nebula? rn since i already used gpt
    im going with it but will change later?)

    Graceful fallback: if the LLM call fails or times out, a default
    response is used so the dialogue continues uninterrupted.
    """

    # Dialogue script.
    # Turns with llm_turn=True skip the hardcoded follow_up and instead
    # call GPT with the child's transcript as input.
    #fallback is the follow_up

    SCRIPT = [
        {
            "question":  "Hallo, ik ben Nao! Hoe heet jij?",
            "follow_up": "Leuk je te ontmoeten!",
            "llm_turn":  False,
        },
        {
            "question":  "Ik vind het zelf leuk om spelletjes te bedenken en mysteries op te lossen. Wat vind je echt leuk om te doen? ",
            "follow_up": "Wow, zo cool!",
            "llm_turn":  True,
        },
        {
            "question":  "Welke vakken vind jij leuk?",
            "follow_up": "Ik vind alle vakken leuk.",
            "llm_turn":  True,   
        },
        {
            "question":  "Ik hou van mysteries oplossen, ik denk dat ik een hele goede detective zou zijn… Wat zou jij later willen worden?",
            "follow_up": "Dat is fantastisch!",
            "llm_turn":  True,
        },
        {
            "question":  "Wat is iets of iemand die jou inspireert?",
            "follow_up": "Goede keuze! Bedankt voor het gesprek. Tot ziens!",
            "llm_turn":  False,
        },
    ]

    # Whisper settings for seconds
    STT_TIMEOUT      = 20
    STT_PHRASE_LIMIT = 15

    # LLM settings
    LLM_FALLBACK = "That sounds really fun!"
    LLM_SYSTEM   = (
        "You are a friendly child robot called Nao talking to a young child aged 8-11. "
        "Reply in short sentences (max 25 words). Be warm, enthusiastic, and simple yet still interesting and upredictable. "
        "Do not ask a question. And speak in Dutch language"
    )

    def __init__(self, openai_env_path=None, nao_ip="10.0.0.165"):
        super(CRI_ScriptedDialogue, self).__init__()

        self.nao_ip          = nao_ip
        self.openai_env_path = openai_env_path
        self.nao             = None
        self.whisper         = None
        self.gpt             = None

        self.set_log_level(sic_logging.INFO)
        self.setup()

    # Setup

    def setup(self):
        self.logger.info("Setting up NAO + Whisper STT + GPT …")

        if self.openai_env_path:
            load_dotenv(self.openai_env_path)

        if "OPENAI_API_KEY" not in environ:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Set it in your shell or pass openai_env_path pointing to a .env file."
            )

        # Connect to NAO
        self.logger.info("Connecting to NAO at %s …", self.nao_ip)
        self.nao = Nao(ip=self.nao_ip)
        self.logger.info("NAO connected.")

        # Whisper STT : no callback, blocking requests only
        whisper_conf = WhisperConf(openai_key=environ["OPENAI_API_KEY"])
        self.whisper = SICWhisper(input_source=self.nao.mic, conf=whisper_conf)
        time.sleep(1)

        # GPT : used for the single LLM turn
        gpt_conf = GPTConf(
            openai_key=environ["OPENAI_API_KEY"],
            system_message=self.LLM_SYSTEM,
            model="gpt-4o-mini",
            max_tokens=40,      # one short sentence 
            temp=0.7,
        )
        self.gpt = GPT(conf=gpt_conf)

        self.logger.info("Setup complete.")

#helpers

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
        Send child_input to GPT and return a short robot utterance.
        Falls back to LLM_FALLBACK on any error or empty response.
        """
        if not child_input:
            self.logger.warning("Empty transcript — using fallback response.")
            return self.LLM_FALLBACK

        prompt = (
            f"The child just said they like: \"{child_input}\". "
            f"React warmly in one short sentence."
        )
        self.logger.info("Sending to GPT: %s", prompt)

        try:
            reply = self.gpt.request(
                GPTRequest(prompt=prompt, stream=False)
            )
            response = reply.response.strip() if reply and reply.response else ""

            if not response:
                self.logger.warning("GPT returned empty response — using fallback.")
                return self.LLM_FALLBACK

            # Trim to one sentence and cap length for TTS naturalness
            first_sentence = response.split(".")[0].strip() + "."
            self.logger.info("GPT response: %s", first_sentence)
            return first_sentence

        except Exception as e:
            self.logger.error("LLM error: %s — using fallback.", e)
            return self.LLM_FALLBACK

    #m`in

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

                # Capture child response
                transcript = self.listen()
                time.sleep(0.8)

                # Choose follow-up: LLM or scripted
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
        nao_ip="10.0.0.165",  
    )
    dialogue_app.run()
