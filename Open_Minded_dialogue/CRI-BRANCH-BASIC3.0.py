import sys
import os
import time
import requests
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

# ---------------------------------------------------------------------------Path setup------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_INTENT = os.path.join(_HERE, "CRI-INTENT")
sys.path.append(_INTENT)

from stub_intent_classifier import StubIntentClassifier
from gpt_intent_classifier import GPTIntentClassifier, REPEAT_SENTINEL


class CRI_ScriptedDialogue(SICApplication):
    """
    Minimal L1 + L2 + L3 implementation for NAO test.

    L1: static scripted utterances (greeting, closing, correction ack)
    L2: UM slot-filling: pulls dummy (julianna_dutch) child fields from Eunike's API
    L3: LLM generates personalised follow-up to child's response (thru the flag)

    UM connection:
        GET http://localhost:8000/api/um/{child_id}/field/{field_name}
        No API key needed for reads.
        If a field is not set, robot says "dat weet ik nog niet".
    """

    # UM connection
    UM_API_BASE = "http://localhost:8000"
    CHILD_ID    = "Julianna_dutch"

    # Whisper
    STT_TIMEOUT      = 20
    STT_PHRASE_LIMIT = 18

    # LLM
    LLM_FALLBACK = "Wauw, dat klinkt heel leuk!"
    LLM_SYSTEM   = (
        "Jij bent een vriendelijke robot genaamd Leo en je praat tegen een Nederlands kind van 8 tot 11 jaar oud."
        "Geef antwoord in één korte zin (maximaal 25 woorden)."
        "Wees warm, enthousiast en geschikt voor de leeftijden tussen 8 en 11."
        "Vraag geen vragen. Praat in het Nederlands."
    )

    # Desktop mic flag
    USE_DESKTOP_MIC = False

    def __init__(self, openai_env_path=None, nao_ip="10.0.0.165"):
        super(CRI_ScriptedDialogue, self).__init__()
        self.nao_ip          = nao_ip
        self.openai_env_path = openai_env_path
        self.nao             = None
        self.whisper         = None
        self.gpt             = None
        self.clf             = None
        self.desktop         = None
        self.set_log_level(sic_logging.INFO)
        self.setup()

    # ------------------------------------------------------------------Setup---------------------------------------------------------------

    def setup(self):
        self.logger.info("Setting up CRI pipeline …")

        if self.openai_env_path:
            load_dotenv(self.openai_env_path)

        if "OPENAI_API_KEY" not in environ:
            raise RuntimeError("OPENAI_API_KEY not found.")

        # Intent classifier: GPT with stub fallback
        try:
            self.clf = GPTIntentClassifier(
                openai_key=environ["OPENAI_API_KEY"],
                schema_path=os.path.join(_INTENT, "um_field_schema.json"),
                contract_path=os.path.join(_INTENT, "intent_classification_contract.json"),
            )
            self.logger.info("GPTIntentClassifier ready.")
        except Exception as e:
            self.logger.warning("GPTIntentClassifier failed (%s) — using stub.", e)
            self.clf = StubIntentClassifier(
                schema_path=os.path.join(_INTENT, "um_field_schema.json")
            )

        self.logger.info("UM: LIVE — %s, child=%s", self.UM_API_BASE, self.CHILD_ID)

        # NAO
        if not self.USE_DESKTOP_MIC:
            self.logger.info("Connecting to NAO at %s …", self.nao_ip)
            self.nao = Nao(ip=self.nao_ip)
            self.logger.info("NAO connected.")

        # Whisper
        if self.USE_DESKTOP_MIC:
            from sic_framework.devices.desktop import Desktop
            self.desktop = Desktop()
            self.whisper = SICWhisper(
                input_source=self.desktop.mic,
                conf=WhisperConf(openai_key=environ["OPENAI_API_KEY"])
            )
        else:
            self.whisper = SICWhisper(
                input_source=self.nao.mic,
                conf=WhisperConf(openai_key=environ["OPENAI_API_KEY"])
            )
        time.sleep(1.0)

        # GPT for L3 responses
        self.gpt = GPT(conf=GPTConf(
            openai_key=environ["OPENAI_API_KEY"],
            system_message=self.LLM_SYSTEM,
            model="gpt-4o-mini",
            max_tokens=40,
            temp=0.7,
        ))
        self.logger.info("Setup complete.")

    # ------------------------------------------------------------------UM pulling----------------------------------------------------------------

    def get_field(self, field: str) -> str:
        """
        Pull a single UM field from Eunike's API.
        GET /api/um/{child_id}/field/{field_name} — no API key needed.

        Returns the value as a Dutch string.
        Returns 'dat weet ik nog niet' if field not set or API unreachable.
        """
        if not field:
            return "dat weet ik nog niet"

        url = f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/field/{field}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                value = resp.json().get("data", {}).get("value")
                if value:
                    self.logger.info("UM[%s] = %s", field, value)
                    return str(value)
                return "dat weet ik nog niet"
            elif resp.status_code == 404:
                self.logger.info("UM field '%s' not set for child '%s'.", field, self.CHILD_ID)
                return "dat weet ik nog niet"
            else:
                self.logger.warning("UM API returned %d for field '%s'.", resp.status_code, field)
                return "dat weet ik nog niet"
        except requests.exceptions.ConnectionError:
            self.logger.error("UM API not reachable at %s — is Eunike's main.py running?", self.UM_API_BASE)
            return "dat weet ik nog niet"
        except Exception as e:
            self.logger.error("UM error for field '%s': %s", field, e)
            return "dat weet ik nog niet"

    def build_script(self) -> list:
        """
        6-turn script at runtime + pulling UM fields via API for L2 slot-filling.

        L1:  static (no UM needed)
        L2:  UM field injected into question string at runtime
        L3:  llm_turn=True: LLM generates the follow-up response
        """
        name        = self.get_field("name")
        hobby       = self.get_field("hobby_fav")
        food        = self.get_field("fav_food")
        fav_subject = self.get_field("fav_subject")
        aspiration  = self.get_field("aspiration")
        animal      = self.get_field("animal_fav")
        pet         = self.get_field("pet_name")

        self.logger.info(
            "UM pulled — name:%s hobby:%s food:%s subject:%s aspiration:%s animal:%s pet:%s",
            name, hobby, food, fav_subject, aspiration, animal, pet
        )

        return [
            {   # Turn 1: L1 static greeting
                "layer":     "L1",
                "question":  (
                    f"Hallo! Ik ben Leo. Wat fijn dat je er bent, {name}! "
                    f"We hebben elkaar al een beetje leren kennen. "
                    f"Nu kunnen we echt even lekker samen kletsen."
                ),
                "follow_up": "Fijn!",
                "llm_turn":  False,
            },
            {   # Turn 2: L2+L3 hobby reference
                "layer":     "L2+L3",
                "question":  (
                    f"Ik weet al iets over jou! "
                    f"Ik heb onthouden dat jij graag {hobby} doet. "
                    f"Klopt dat?"
                ),
                "follow_up": "Interessant!",
                "llm_turn":  True,
            },
            {   # Turn 3: L2+L3 food reference
                "layer":     "L2+L3",
                "question":  (
                    f"En ik weet ook dat jouw lievelingseten {food} is. "
                    f"Ik vind zelf pannenkoeken het allerlekkerst! "
                    f"Wat eet jij het liefst?"
                ),
                "follow_up": "Mmm, dat klinkt heerlijk!",
                "llm_turn":  True,
            },
            {   # Turn 4: L2+L3 correct UM refs (animal + pet)
                "layer":     "L2+L3",
                "question":  (
                    f"Oh, en ik weet dat jij {animal} heel leuk vindt. "
                    f"En je hebt er zelf ook een — {pet}! "
                    f"Wat doet {pet} het liefst?"
                ),
                "follow_up": "Dat klinkt leuk!",
                "llm_turn":  True,
            },
            {   # Turn 5: L2+L3 aspiration reference
                "layer":     "L2+L3",
                "question":  (
                    f"Ik hoor dat jij later {aspiration} wilt worden. "
                    f"Klopt dat?"
                ),
                "follow_up": "Dat is een mooie droom!",
                "llm_turn":  True,
            },
            {   # Turn 6: L1 static closing
                "layer":     "L1",
                "question":  (
                    "Bedankt voor het gesprek! "
                    "Ik heb heel veel geleerd over jou vandaag. "
                    "Tot de volgende keer!"
                ),
                "follow_up": "",
                "llm_turn":  False,
            },
        ]

    # ------------------------------------------------------------------Helpers----------------------------------------------------------------

    def say(self, text: str):
        """Speak text via NAO TTS and wait for it to finish before returning."""
        if not text or not text.strip():
            return
        self.logger.info("LEO: %s", text)
        if self.USE_DESKTOP_MIC:
            print(f"\n[LEO]: {text}\n")
        else:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
            # Wait proportional to text length so Whisper doesn't start
            # listening while NAO is still speaking.
            # ~0.01s per character is a safe estimate for NAO's TTS speed.
            speaking_time = len(text) * 0.01
            time.sleep(speaking_time)

    def listen(self) -> str:
        self.logger.info("Listening …")
        try:
            result = self.whisper.request(
                GetTranscript(
                    timeout=self.STT_TIMEOUT,
                    phrase_time_limit=self.STT_PHRASE_LIMIT,
                )
            )
            transcript = result.transcript.strip() if result and result.transcript else ""
            self.logger.info("Child: %s", transcript or "(nothing)")
            return transcript
        except Exception as e:
            self.logger.error("STT error: %s", e)
            return ""

    def classify_with_repeat(self, transcript: str):
        """GPT classify → low confidence → ask to repeat once → stub fallback."""
        result = self.clf.classify(transcript)
        if result.intent == REPEAT_SENTINEL:
            self.logger.info("Low confidence — asking to repeat.")
            self.say("Kun je dat nog een keer zeggen?")
            time.sleep(0.8)
            transcript = self.listen()
            result     = self.clf.classify_retry(transcript)
        self.logger.info("Intent: %s", result.to_dict())
        return result

    def llm_response(self, child_input: str) -> str:
        """L3 — GPT generates personalised Dutch follow-up."""
        if not child_input:
            return self.LLM_FALLBACK
        prompt = (
            f"Het kind zei: \"{child_input}\". "
            f"Reageer warm en enthousiast in één korte zin in het Nederlands."
        )
        try:
            reply = self.gpt.request(GPTRequest(prompt=prompt, stream=False))
            response = reply.response.strip() if reply and reply.response else ""
            return (response.split(".")[0].strip() + ".") if response else self.LLM_FALLBACK
        except Exception as e:
            self.logger.error("LLM error: %s", e)
            return self.LLM_FALLBACK

    def handle_intent(self, result, transcript: str) -> bool:
        """
        Route classified intent to correct response.
        Returns True if handled (skip L3), False to fall through to L3.
        """
        intent = result.intent
        field  = result.field

        if intent == "um_inspect":
            value = self.get_field(field)
            self.say(f"Ik weet dat jouw {field or 'antwoord'} {value} is.")
            return True

        elif intent == "um_update":
            self.say("Oh, je hebt gelijk! Ik pas het aan.")
            if field and result.value:
                old = self.get_field(field)
                self.say(f"Dus jouw {field} is {result.value}, niet {old}. Leuk!")
                # TODO: write correction to Eunike's API (Sherissa's task):
                # requests.post(f"{self.UM_API_BASE}/api/um/{self.CHILD_ID}/fields",
                #     json={"fields": {field: result.value}, "source": "child_correction"})
            return True

        elif intent == "dialogue_update":
            self.say("Oké, ik hoorde dat. Klopt dat?")
            return True

        elif intent == "um_delete":
            self.say("Oké, ik vergeet dat!")
            return True

        elif intent == "dialogue_question":
            self.say("Dat is een goede vraag! Ik vertel het je later.")
            return True

        elif intent == "dialogue_social":
            self.say("Haha ja! Oké, verder!")
            return True

        # um_add / dialogue_answer / dialogue_none → L3
        return False

    # ------------------------------------------------------------------Main loop------------------------------------------------------------------

    def run(self):
        self.logger.info("Starting CRI — minimal L1+L2+L3 NAO test …")

        script = self.build_script()
        self.logger.info("Script ready — %d turns.", len(script))

        try:
            if not self.USE_DESKTOP_MIC:
                self.nao.autonomous.request(NaoWakeUpRequest())

            for i, turn in enumerate(script):
                if self.shutdown_event.is_set():
                    break

                self.logger.info("=== Turn %d/%d [%s] ===", i + 1, len(script), turn["layer"])

                # Say the question — say() now waits for NAO to finish speaking
                self.say(turn["question"])

                # Extra buffer before Whisper starts listening
                time.sleep(0.5)

                # Last turn — no response needed
                if i == len(script) - 1:
                    break

                transcript = self.listen()
                time.sleep(0.8)

                result  = self.classify_with_repeat(transcript)
                handled = self.handle_intent(result, transcript)

                if not handled:
                    if turn["llm_turn"] and transcript:
                        self.say(self.llm_response(transcript))   # L3
                    else:
                        self.say(turn["follow_up"])                # L1 fallback

                if i < len(script) - 2:
                    time.sleep(1.0)

            self.logger.info("Dialogue completed.")

        except KeyboardInterrupt:
            self.logger.info("Interrupted.")
        except Exception as e:
            self.logger.error("Error: %s", e)
        finally:
            try:
                if not self.USE_DESKTOP_MIC:
                    self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass
            self.logger.info("Shutting down.")
            self.shutdown()


if __name__ == "__main__":
    dialogue_app = CRI_ScriptedDialogue(
        openai_env_path=abspath(join(_HERE, "conf", ".env")),
        nao_ip="10.0.0.165",  # ← replace with your NAO's IP
    )
    dialogue_app.run()
