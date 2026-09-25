"""
Question answering over captured text, using a local Qwen 2.5 7B Instruct model
served by Ollama (https://ollama.com).

Setup:
    ollama pull qwen2.5:7b-instruct
    (keep the Ollama app open, or run `ollama serve`)

Optional .env settings:
    OLLAMA_HOST = http://localhost:11434
    LLM_MODEL   = qwen2.5:7b-instruct
"""

import json
import logging
import os
import re
import calendar
import threading
from datetime import date
from typing import Iterator, List

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

# The model replies with exactly this when the user wants everything read out;
# the app then speaks Apple Vision's text itself instead of the model's version.
READ_ALL = "READ_ALL"

SYSTEM_PROMPT = """You help a blind or low-vision person understand text that their camera captured.
The captured text below was extracted by OCR. It may contain recognition mistakes, broken lines,
or text from more than one object.

Rules:
- Your reply is spoken aloud by a text-to-speech voice. Use plain sentences only: no markdown,
  no bullet symbols, no tables, no emojis.
- If the user wants the whole text read out (for example "read everything", "read it all",
  "what does it say"), reply with exactly: READ_ALL
- If the user asks you to read only certain parts (for example the ingredients, the dosage,
  the expiry date, the address), read those parts exactly as written, with no extra commentary.
  Fix an obvious OCR mistake only when you are certain.
- If the user asks a question, answer it briefly, in one to three sentences, using only the
  captured text. If the text does not contain the answer, say so plainly and mention what the
  text is about.
- Never invent details that are not in the text. Be precise with medicine, dosage, dates,
  prices and safety information.
- Today's date is {today}.
- For expiry questions, trust the "Date checks" below: they were computed by the app and are
  correct. Do not redo the date arithmetic yourself. If there is no expiry date, say the text
  does not show one; never guess it from a manufacturing date.

Date checks:
{date_checks}

Captured text:
<<<
{document}
>>>"""

# Split finished sentences out of a growing stream of text. A sentence only
# counts as finished once whitespace and a capital letter follow, so "0.5%"
# and "Rs. 25.00" aren't cut in the middle.
_SENTENCE_END = re.compile(r"(.+?[.!?])\s+(?=[A-Z\"'(])|([^\n]+?)\n+", re.S)

# Requests that clearly mean "read the whole thing": handled without the model
_READ_ALL_REQUEST = re.compile(
    r"^(?:please |can you |could you )*"
    r"(?:read(?: it| this| that| out| aloud| everything| all| the whole thing| the (?:whole|entire|full) (?:text|thing|page|label))*"
    r"|(?:read|say|tell me)(?: me)? (?:everything|all(?: of it)?|the (?:whole|entire|full) (?:text|thing|page|label))"
    r"|what does (?:it|this|that) say|what(?:'s| is) written(?: here| on it)?)"
    r"(?: (?:please|for me|out loud|aloud|again))*[?.!]*$"
)


def wants_read_all(request: str) -> bool:
    """True for plain "read it all" requests, so they skip the model entirely."""
    return bool(_READ_ALL_REQUEST.match(request.strip().lower()))


# ---------------- expiry dates (computed in code; small models get these wrong) ----------------
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_EXPIRY_LABEL = r"(?:exp(?:iry|iration)?(?:\s*date)?|use\s*(?:by|before)|best\s*before)\.?\s*[:.\-]?\s*"
_EXPIRY_DATE = re.compile(
    _EXPIRY_LABEL + r"(?:"
    r"(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{2,4})"          # 15/04/2028
    r"|(?P<m2>\d{1,2})[/\-.](?P<y2>\d{2,4})"                            # 04/2028, 04-28
    r"|(?P<mon>[a-z]{3})[a-z]*\.?[\s/\-.]*(?P<y3>\d{2,4})"               # APR 2028, Apr-28
    r")",
    re.I,
)


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def expiry_checks(text: str, today: date = None) -> str:
    """One line per expiry date found, saying whether it has passed."""
    today = today or date.today()
    lines = []
    for m in _EXPIRY_DATE.finditer(text):
        try:
            if m.group("d"):
                exp = date(_year(m.group("y")), int(m.group("m")), int(m.group("d")))
            else:
                if m.group("m2"):
                    month, year = int(m.group("m2")), _year(m.group("y2"))
                else:
                    month, year = _MONTHS.get(m.group("mon").lower()[:3]), _year(m.group("y3"))
                    if not month:
                        continue
                # Month-only dates are good until the end of that month
                exp = date(year, month, calendar.monthrange(year, month)[1])
        except ValueError:
            continue
        status = "HAS EXPIRED" if exp < today else "has NOT expired"
        lines.append(f'"{m.group(0).strip()}" means valid until {exp.strftime("%d %B %Y")}: it {status}.')
    return "\n".join(lines) or "No expiry date found in the text."


class DocAssistant:
    """Keeps the captured text plus the conversation about it."""

    def __init__(self):
        self.document = ""
        self.history: List[dict] = []
        self._generation = 0          # bumped by cancel(); stale streams stop early
        self._lock = threading.Lock()

    # ---------------- document ----------------
    def set_document(self, text: str) -> None:
        """New capture: replace the text and forget the previous conversation."""
        with self._lock:
            self.document = text
            self.history = []
            self._generation += 1

    def has_document(self) -> bool:
        return bool(self.document.strip())

    def cancel(self) -> None:
        """Stop any answer currently being generated."""
        with self._lock:
            self._generation += 1

    # ---------------- model ----------------
    def warm_up(self) -> bool:
        """Load the model into memory so the first question isn't slow."""
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate",
                              json={"model": LLM_MODEL, "keep_alive": "30m"}, timeout=120)
            r.raise_for_status()
            logger.info(f"✅ {LLM_MODEL} loaded in Ollama")
            return True
        except Exception as e:
            logger.error(f"❌ Could not load {LLM_MODEL} from Ollama at {OLLAMA_HOST}: {e}")
            return False

    def ask(self, question: str) -> Iterator[str]:
        """
        Stream the answer sentence by sentence.
        Yields READ_ALL alone when the user wants the whole text read.
        Stops early if cancel() or set_document() is called.
        """
        with self._lock:
            generation = self._generation
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(document=self.document, today=date.today().strftime("%d %B %Y"),
                                                                   date_checks=expiry_checks(self.document))}]
            messages += self.history
            messages.append({"role": "user", "content": question})

        answer = ""
        buffer = ""
        decided = False  # whether we've ruled READ_ALL in or out
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": "30m",
                    "options": {"temperature": 0.2, "num_ctx": 8192},
                },
                stream=True,
                timeout=(5, 120),
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if generation != self._generation:
                        logger.info("⏹️ Answer cancelled")
                        return
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    piece = chunk.get("message", {}).get("content", "")
                    answer += piece
                    buffer += piece

                    if not decided:
                        head = answer.strip()
                        if head.startswith(READ_ALL):
                            self._remember(question, READ_ALL, generation)
                            yield READ_ALL
                            return
                        if len(head) >= len(READ_ALL) or not READ_ALL.startswith(head):
                            decided = True
                        else:
                            continue

                    while True:
                        m = _SENTENCE_END.match(buffer)
                        if not m:
                            break
                        sentence = (m.group(1) or m.group(2) or "").strip()
                        buffer = buffer[m.end():]
                        if sentence:
                            yield sentence

                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            yield "I can't reach the language model. Please make sure Ollama is running."
            return
        except Exception as e:
            logger.error(f"❌ Model error: {e}")
            yield "Sorry, something went wrong while thinking about that."
            return

        if generation != self._generation:
            return
        if answer.strip() == READ_ALL or (not decided and answer.strip().startswith(READ_ALL)):
            self._remember(question, READ_ALL, generation)
            yield READ_ALL
            return
        if buffer.strip():
            yield buffer.strip()
        self._remember(question, answer.strip(), generation)

    def _remember(self, question: str, answer: str, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self.history += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}]
                self.history = self.history[-12:]  # last 6 exchanges
