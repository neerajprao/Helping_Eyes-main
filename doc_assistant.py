"""
Question answering over captured text, using a local Qwen 2.5 7B Instruct model
served by Ollama (https://ollama.com), with an optional web lookup (DuckDuckGo)
when the captured text doesn't contain the answer.

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
from typing import Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

# The model replies with exactly this when the user wants everything read out;
# the app then speaks Apple Vision's text itself instead of the model's version.
READ_ALL = "READ_ALL"

# The model starts its reply with this tag when the captured text doesn't hold
# the answer. The tag is never spoken; the app uses it to offer a web lookup.
NOT_IN_TEXT = "[NOT_IN_TEXT]"

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
  captured text.
- If the captured text does not contain the answer, start your reply with the tag
  [NOT_IN_TEXT] and then say in one sentence that the text doesn't say. Do not answer from
  your own knowledge, even if you know the answer. Explaining what an ingredient or term
  means also counts as not in the text unless the text itself explains it.
  Examples:
    Question: Is it waterproof?  (the text doesn't mention water resistance)
    Reply: [NOT_IN_TEXT] The text doesn't say whether it is waterproof.
    Question: What is zinc oxide?  (the text only lists zinc oxide as an ingredient)
    Reply: [NOT_IN_TEXT] The text lists zinc oxide as an ingredient but doesn't explain what it is.
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

QUERY_PROMPT = """Write one short web search query (at most 8 words) that would answer the
user's question. Use the product or document name from the captured text if it helps.
Reply with the query only, no quotes.

Captured text (first part):
{document}

Question: {question}"""

WEB_PROMPT = """You help a blind or low-vision person. The text their camera captured did not
answer their question, so the app searched the web. Answer using only the web results below.

Rules:
- Your reply is spoken aloud. Plain sentences only, no markdown, no lists, no URLs.
- Start with "According to the web," so the user knows this is not from their item.
- Answer in two to four sentences. If the results don't answer the question, say so.
- For medicine, dosage or health questions, end with: "Please confirm with a pharmacist or doctor."
- Never present web information as if it were printed on the user's item.

Question: {question}

Web results:
{results}"""

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

# Facts that only the item itself can tell you: never offered as a web lookup
_ITEM_ONLY = re.compile(r"expir|exp\b|best before|use by|batch|lot\b|mfg|manufactur(ed|ing) date|price|mrp|cost", re.I)


def wants_read_all(request: str) -> bool:
    """True for plain "read it all" requests, so they skip the model entirely."""
    return bool(_READ_ALL_REQUEST.match(request.strip().lower()))


# Backup for when the model forgets the [NOT_IN_TEXT] tag
_SAYS_NOT_IN_TEXT = re.compile(
    r"\b(text|label|it|this|they|directions|instructions|package|packaging)\s+"
    r"(does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(say|mention|show|state|specify|include|contain|provide|explain|list|indicate)"
    r"|\bnot\s+(mentioned|stated|specified|provided|shown|listed|explained|included)\b"
    r"|\bno\s+(information|mention|details?)\s+(about|on|of|regarding)\b",
    re.I,
)


def web_lookup_allowed(question: str) -> bool:
    """False for questions about this particular item (expiry, batch, price)."""
    return not _ITEM_ONLY.search(question)


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


# ---------------- web search ----------------
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the Mac."""
    from ddgs import DDGS  # imported here so the app still starts if ddgs is missing
    return DDGS(timeout=10).text(query, max_results=max_results) or []


class DocAssistant:
    """Keeps the captured text plus the conversation about it."""

    def __init__(self):
        self.document = ""
        self.history: List[dict] = []
        self.last_not_in_text = False  # last answer said the text doesn't have it
        self._generation = 0           # bumped by cancel(); stale streams stop early
        self._lock = threading.Lock()

    # ---------------- document ----------------
    def set_document(self, text: str) -> None:
        """New capture: replace the text and forget the previous conversation."""
        with self._lock:
            self.document = text
            self.history = []
            self.last_not_in_text = False
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
        Answer from the captured text, streamed sentence by sentence.
        Yields READ_ALL alone when the user wants the whole text read.
        Sets last_not_in_text when the text doesn't contain the answer.
        """
        with self._lock:
            generation = self._generation
            self.last_not_in_text = False
            system = SYSTEM_PROMPT.format(document=self.document,
                                          today=date.today().strftime("%d %B %Y"),
                                          date_checks=expiry_checks(self.document))
            messages = [{"role": "system", "content": system}] + self.history
            messages.append({"role": "user", "content": question})

        answer = []
        for sentence in self._stream(messages, generation, markers=(READ_ALL, NOT_IN_TEXT)):
            if sentence == READ_ALL:
                self._remember(question, READ_ALL, generation)
                yield READ_ALL
                return
            if sentence == NOT_IN_TEXT:
                self.last_not_in_text = True
                continue
            answer.append(sentence)
            yield sentence
        if answer and _SAYS_NOT_IN_TEXT.search(answer[0]):
            self.last_not_in_text = True
        self._remember(question, " ".join(answer), generation)

    def ask_web(self, question: str) -> Iterator[str]:
        """Search the web for the question and answer from the results."""
        with self._lock:
            generation = self._generation
            document = self.document

        query = self._search_query(question, document) or question
        logger.info(f"🌐 Searching the web for: {query}")
        try:
            results = web_search(query)
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            yield "Sorry, I couldn't search online right now."
            return
        if generation != self._generation:
            return
        if not results:
            yield "I searched online but found nothing useful."
            return

        sources = "\n\n".join(f"{r.get('title', '')}\n{r.get('body', '')}" for r in results)
        messages = [{"role": "user", "content": WEB_PROMPT.format(question=question, results=sources)}]
        answer = []
        for sentence in self._stream(messages, generation):
            answer.append(sentence)
            yield sentence
        self._remember(question, " ".join(answer), generation)

    # ---------------- internals ----------------
    def _search_query(self, question: str, document: str) -> Optional[str]:
        """Ask the model for a short search query that includes the product name."""
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/chat", json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": QUERY_PROMPT.format(
                    document=document[:1500], question=question)}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 30},
            }, timeout=30)
            r.raise_for_status()
            query = r.json()["message"]["content"].strip().strip('"').splitlines()[0]
            return query[:120] or None
        except Exception as e:
            logger.warning(f"⚠️ Couldn't build a search query: {e}")
            return None

    def _stream(self, messages: List[dict], generation: int, markers=()) -> Iterator[str]:
        """
        Stream a chat reply sentence by sentence. If the reply starts with one
        of `markers`, that marker is yielded on its own first (READ_ALL ends
        the reply; other markers are stripped and the rest is streamed).
        """
        answer = ""
        buffer = ""
        decided = not markers  # whether the reply's opening marker (if any) is known
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
                        head = answer.lstrip()
                        found = next((m for m in markers if head.startswith(m)), None)
                        if found == READ_ALL:
                            yield READ_ALL
                            return
                        if found:
                            yield found
                            buffer = head[len(found):].lstrip()
                            decided = True
                        elif any(m.startswith(head) for m in markers) and not chunk.get("done"):
                            continue  # could still turn into a marker
                        else:
                            decided = True

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

        if generation == self._generation and buffer.strip():
            yield buffer.strip()

    def _remember(self, question: str, answer: str, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self.history += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}]
                self.history = self.history[-12:]  # last 6 exchanges
