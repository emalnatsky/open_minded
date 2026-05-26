"""Structured L3 runtime utterances for short Leo responses.

L3 is the only place where GPT may improvise inside the scripted dialogue.
The script still controls the phase order and next line; L3 only produces one
short context-aware Dutch utterance between scripted turns.
"""

import json
import logging
import re

from sic_framework.services.llm import GPTRequest

logger = logging.getLogger(__name__)


L3_SYSTEM_PROMPT = """
You are Leo, a small NAO school robot.
You speak to Dutch children aged 8-11 in a primary school setting.
You are warm, playful, kind, slightly self-ironic, and naturally imperfect.
You want to become a very good and helpful school robot.
You sound lively and personal, never formal, clinical, sarcastic, or overly dramatic.

You are in the middle of a scripted interaction. Produce exactly one short
Dutch utterance that fits the current turn.

Leo backstory, light reference only:
- You are not good at sport and you fall over a lot.
- You like learning new things and admire teachers.
- You love books and stories.
- You once became Robot Bookworm and spent time in Cloudbib-9 with flying
  books, a lama friend, and a chaotic lama postbode.
Only use a tiny reference when it fits naturally. Do not force the backstory.

Response functions:
- acknowledge: validate what the child just said. Do not open a new subtopic.
- wrap_up: close the current subtopic warmly. Do not introduce new information.
- bridge: acknowledge and transition toward the next script phase without
  overlapping with the next scripted line.

Safety rules:
- NEVER reveal or reference UM fields not listed in relevant_um_fields.
- NEVER ask the child to verify or confirm a stored value. No "klopt dat?",
  no "is dat zo?", unless explicitly instructed.
- NEVER praise too enthusiastically. No "Wat goed dat je dat zegt!" or "Knap!".
- NEVER introduce sensitive or heavy topics such as family problems, illness,
  bullying, violence, or fear.
- NEVER contradict something Leo has already stated in this conversation.
- NEVER produce more than 25 words. If in doubt, be shorter.
- Only ask a question if question_allowed is true.
- Do not overlap with or pre-empt next_script_line.
- Output plain Dutch text only. No labels, no quotes, no markdown.

Good examples:
acknowledge, sport:
Child: Ik sta meestal voorin.
Leo: Voorin, lekker! Daar moet je snel zijn.

wrap_up, school:
Child: Ehm, luisteren?
Leo: Haha, goed geraden! Luisteren ging best oke. De rest was soms wat lastiger.

bridge, future:
Child: Soms wel. Ik wil iets met dieren doen.
Leo: Iets met dieren, dat klinkt mooi. Ik denk zelf ook best vaak na over later.
""".strip()


class L3Runtime:
    """Build, call, validate, and log Lena's structured L3 runtime prompt."""

    DEFAULT_FALLBACKS = {
        "acknowledge": "Dat snap ik wel.",
        "wrap_up": "Dat is leuk om te horen.",
        "bridge": "Dat past mooi bij wat we net bespraken.",
    }

    FORBIDDEN_PHRASES = (
        "klopt dat",
        "is dat zo",
        "wat goed dat je dat zegt",
        "knap",
    )

    SENSITIVE_WORDS = (
        "pesten",
        "gepest",
        "ziek",
        "ziekte",
        "familieproblemen",
        "geweld",
        "bang",
    )

    SHORT_OR_CONTROL_VALUES = {
        "ja",
        "nee",
        "true",
        "false",
        "new",
        "returning",
        "c",
        "e",
        "een beetje",
        "misschien",
        "dat weet ik nog niet",
    }

    def __init__(self, dialogue):
        self.d = dialogue

    def is_enabled(self, turn: dict) -> bool:
        return isinstance((turn or {}).get("l3"), dict)

    def fallback_for(self, l3: dict) -> str:
        function = str(l3.get("response_function") or "acknowledge").strip()
        return str(l3.get("fallback") or self.DEFAULT_FALLBACKS.get(function, self.d.LLM_FALLBACK))

    def relevant_um_fields(self, turn: dict) -> dict:
        l3 = turn.get("l3") or {}
        configured = l3.get("relevant_um_fields") or {}
        if isinstance(configured, dict):
            fields = configured
        else:
            fields = {}
            for field in configured:
                value = (turn.get("used_fields") or {}).get(field)
                if not self.d.is_known(value):
                    value = (self.d.last_um_preview or {}).get(field)
                if self.d.is_known(value):
                    fields[field] = value

        return {
            str(field): str(value)
            for field, value in fields.items()
            if self.d.is_known(value)
        }

    def user_prompt(self, child_input: str, turn: dict, relevant_fields: dict) -> str:
        l3 = turn.get("l3") or {}
        script_phase = l3.get("script_phase") or f"phase_{turn.get('phase', '')}"
        topic = l3.get("topic") or (turn.get("topic") or {}).get("domain") or ""
        response_function = l3.get("response_function") or "acknowledge"
        question_allowed = bool(l3.get("question_allowed", False))
        local_context = l3.get("local_context") or turn.get("local_context") or ""
        next_script_line = turn.get("next_script_line") or l3.get("next_script_line") or ""

        return (
            "Generate one short Leo utterance for the current script turn.\n\n"
            "Current Turn\n"
            f"- Script phase: {script_phase}\n"
            f"- Topic: {topic}\n"
            f"- Response function: {response_function}\n"
            f"- Question allowed: {str(question_allowed).lower()}\n\n"
            "Dialogue Context\n"
            f"- Leo's previous utterance: {getattr(self.d, 'last_leo_utterance', '')}\n"
            f"- Child's utterance: {child_input}\n"
            f"- Local context: {local_context}\n\n"
            "Constraints\n"
            f"- Next scripted Leo line (do NOT overlap): {next_script_line}\n"
            "- Relevant UM fields (only these may be referenced): "
            f"{json.dumps(relevant_fields, ensure_ascii=False)}\n\n"
            "Output: one Leo utterance in Dutch. No labels, no quotes, no explanation. Maximum 20 words."
        )

    def sanitize_output(self, raw: str) -> str:
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:text)?|```$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^(leo|antwoord|output)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.strip("\"'` ")
        text = " ".join(text.split())
        if hasattr(self.d, "speech") and hasattr(self.d.speech, "strip_non_bmp"):
            text = self.d.speech.strip_non_bmp(text)
        return text

    def word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))

    def normalized(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").casefold()).strip()

    def value_tokens(self, value: str) -> list:
        raw_values = []
        if hasattr(self.d, "split_memory_values"):
            raw_values.extend(self.d.split_memory_values(value))
        raw_values.append(str(value or ""))

        tokens = []
        for raw in raw_values:
            clean = self.normalized(raw).strip(" .,!?:;()[]")
            if len(clean) < 4 or clean in self.SHORT_OR_CONTROL_VALUES:
                continue
            tokens.append(clean)
            for match in re.findall(r"\(([^)]+)\)", clean):
                if len(match) >= 4 and match not in self.SHORT_OR_CONTROL_VALUES:
                    tokens.append(match)
        return list(dict.fromkeys(tokens))

    def forbidden_um_value(self, text: str, relevant_fields: dict) -> str:
        output = self.normalized(text)
        allowed_values = set()
        for value in relevant_fields.values():
            allowed_values.update(self.value_tokens(value))

        for field, value in (self.d.last_um_preview or {}).items():
            if field in relevant_fields:
                continue
            for token in self.value_tokens(value):
                if token in allowed_values:
                    continue
                if re.search(rf"\b{re.escape(token)}\b", output):
                    return token
        return ""

    def overlaps_next_line(self, text: str, next_script_line: str) -> bool:
        words = re.findall(r"\b\w+\b", self.normalized(text), flags=re.UNICODE)
        next_text = self.normalized(next_script_line)
        if len(words) < 4 or not next_text:
            return False
        for index in range(0, len(words) - 3):
            phrase = " ".join(words[index:index + 4])
            if phrase in next_text:
                return True
        return False

    def validation_error(self, text: str, turn: dict, relevant_fields: dict) -> str:
        if not text:
            return "empty"
        if self.word_count(text) > 25:
            return "too_long"

        l3 = turn.get("l3") or {}
        if not bool(l3.get("question_allowed", False)) and "?" in text:
            return "question_not_allowed"

        lowered = self.normalized(text)
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in lowered:
                return f"forbidden_phrase:{phrase}"
        for word in self.SENSITIVE_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                return f"sensitive_word:{word}"

        forbidden_value = self.forbidden_um_value(text, relevant_fields)
        if forbidden_value:
            return f"forbidden_um_value:{forbidden_value}"

        if self.overlaps_next_line(text, turn.get("next_script_line") or ""):
            return "overlaps_next_script_line"

        return ""

    def call_model(self, system_prompt: str, user_prompt: str) -> str:
        if self.d.gpt is not None:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            reply = self.d.gpt.request(GPTRequest(prompt=prompt, stream=False))
            return reply.response.strip() if reply and reply.response else ""

        reply = self.d.openai_client.chat.completions.create(
            model=self.d.TOPIC_CHANGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            temperature=0.4,
        )
        return reply.choices[0].message.content.strip()

    def generate(self, child_input: str, turn: dict) -> str:
        l3 = turn.get("l3") or {}
        fallback = self.fallback_for(l3)
        relevant_fields = self.relevant_um_fields(turn)
        user_prompt = self.user_prompt(child_input, turn, relevant_fields)
        raw_output = ""
        final_output = fallback
        used_fallback = True
        validation = "fallback_before_model"

        try:
            if not child_input:
                validation = "empty_child_input"
            else:
                raw_output = self.call_model(L3_SYSTEM_PROMPT, user_prompt)
                candidate = self.sanitize_output(raw_output)
                validation = self.validation_error(candidate, turn, relevant_fields)
                if not validation:
                    final_output = candidate
                    used_fallback = False
        except Exception as exc:
            validation = f"exception:{exc}"
            self.d.logger.error("L3 runtime error: %s", exc)

        self.d.log_conversation_event(
            "l3_call",
            script_phase=l3.get("script_phase"),
            topic=l3.get("topic"),
            response_function=l3.get("response_function"),
            question_allowed=bool(l3.get("question_allowed", False)),
            child_input=child_input,
            last_leo_utterance=getattr(self.d, "last_leo_utterance", ""),
            next_script_line=turn.get("next_script_line") or l3.get("next_script_line"),
            local_context=l3.get("local_context") or turn.get("local_context"),
            relevant_um_fields=relevant_fields,
            raw_output=raw_output,
            final_output=final_output,
            used_fallback=used_fallback,
            validation=validation,
        )
        return final_output
