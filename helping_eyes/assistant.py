"""
Question answering over captured text, using any chat language model that speaks the
OpenAI-style chat API (Gemini, Groq, OpenRouter, OpenAI, a local server, ...), with an
optional web lookup (DuckDuckGo) when the captured text doesn't contain the answer.

Settings (environment or .env); nothing here is specific to one provider:
    LLM_BASE_URL  chat API address (default: Google Gemini's OpenAI-compatible endpoint)
    LLM_API_KEY   the provider's API key
    LLM_MODEL     model name (default: gemini-3.1-flash-lite)

The page's model menu picks between two providers per request:
    gemini        the settings above
    qwen3b, qwen7b  models on this computer, served by Ollama (https://ollama.com)
    QWEN_BASE_URL  Ollama's OpenAI-compatible address (default: http://localhost:11434/v1)
    QWEN_3B_MODEL  the 3B model's name in `ollama list` (default: qwen2.5:3b-instruct)
    QWEN_7B_MODEL  the 7B model's name in `ollama list` (default: qwen2.5:7b-instruct)
"""

import json
import logging
import os
import re
import calendar
import threading
from datetime import date
from typing import Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1").rstrip("/")
QWEN_MODELS = {"qwen3b": os.getenv("QWEN_3B_MODEL", "qwen2.5:3b-instruct"),
               "qwen7b": os.getenv("QWEN_7B_MODEL", "qwen2.5:7b-instruct")}
PROVIDERS = ("gemini", *QWEN_MODELS)

# The model replies with exactly this when the user wants everything read out;
# the app then speaks the OCR text itself instead of the model's version.
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
- Questions about the page number, book title or chapter are answered from the "Page facts"
  below: they were read from the printed page numbers, header and headings and are reliable.

Date checks:
{date_checks}

Page facts:
{facts}

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
- Say only what the question asks for: the specific fact, value or part they asked about.
  No background, no history, no related facts, no extra tips, and no summary of the results.
  If they ask for one thing (a price, a date, a dosage, an ingredient), give just that.
- Start with "According to the web," so the user knows this is not from their item.
- Keep it to one or two short sentences. If the results don't answer the question, say so in one sentence.
- For medicine, dosage or health questions only, end with: "Please confirm with a pharmacist or doctor."
- Never present web information as if it were printed on the user's item.

Question: {question}

Web results:
{results}"""

MSG_UNREACHABLE = "I can't reach the language model. Please check the internet connection."
MSG_NO_KEY = "No API key is set for the language model. Please set LLM_API_KEY."
MSG_BAD_KEY = "The language model rejected the request. Please check the API key."
MSG_NO_MODEL = "The language model name was not found. Please check LLM_MODEL and LLM_BASE_URL."
MSG_BUSY = "The language model is busy right now, or today's free limit is used up. Please try again in a minute."
MSG_QWEN_DOWN = "I can't reach the Qwen model on this computer. Please start Ollama, or choose Gemini."
MSG_FAILED = "Sorry, something went wrong while thinking about that."

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


def expiry_dates(text: str) -> List[Tuple[str, date]]:
    """(printed text, last valid day) for every expiry date found in the text."""
    found = []
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
        found.append((m.group(0).strip(), exp))
    return found


def expiry_checks(text: str, today: date = None) -> str:
    """One line per expiry date found, saying whether it has passed."""
    today = today or date.today()
    lines = [f'"{printed}" means valid until {exp.strftime("%d %B %Y")}: it {"HAS EXPIRED" if exp < today else "has NOT expired"}.'
             for printed, exp in expiry_dates(text)]
    return "\n".join(lines) or "No expiry date found in the text."


# Questions with an exact answer are answered in code, not by the model
_EXPIRY_QUESTION = re.compile(r"\b(expir\w*|exp(iry)? date|best before|use by|out of date|still (good|valid|usable|okay|ok))\b", re.I)
_PAGE_QUESTION = re.compile(r"^\s*(what|which)\s+page(\s+(is\s+(this|it)|am\s+i\s+on|are\s+we\s+on))?\s*\??\s*$", re.I)


def expiry_answer(text: str, today: date = None) -> Optional[str]:
    """Direct spoken answer to an expiry question, or None when no expiry date is printed."""
    today = today or date.today()
    dates = expiry_dates(text)
    if not dates:
        return None
    parts = []
    for _, exp in dates:
        when = exp.strftime("%d %B %Y").lstrip("0")
        parts.append(f"Yes, it has expired. It was valid until {when}." if exp < today
                     else f"No, it has not expired. It is valid until {when}.")
    return " ".join(dict.fromkeys(parts))


# ---------------- web search ----------------
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the Mac."""
    from ddgs import DDGS  # imported here so the app still starts if ddgs is missing
    return DDGS(timeout=10).text(query, max_results=max_results) or []


def _why(response) -> str:
    """The provider's own explanation of an error (it says which model or key is wrong), shortened for the log."""
    try:
        return " ".join(response.text.split())[:300] or "(no details)"
    except Exception:
        return "(no details)"


def _endpoint(provider: str) -> Tuple[str, dict, str, str]:
    """(chat API address, headers, model name, API key) of a provider; read at call time."""
    if provider in QWEN_MODELS:
        return QWEN_BASE_URL, {"Content-Type": "application/json"}, QWEN_MODELS[provider], ""
    return (LLM_BASE_URL,
            {"Content-Type": "application/json", **({"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {})},
            LLM_MODEL, LLM_API_KEY)


class DocAssistant:
    """Keeps the captured text plus the conversation about it."""

    def __init__(self):
        self.document = ""
        self.facts = ""
        self.page_label = ""
        self.history: List[dict] = []
        self.provider = "gemini"       # which language model answers: one of PROVIDERS
        self.last_not_in_text = False  # last answer said the text doesn't have it
        self._generation = 0           # bumped by cancel(); stale streams stop early
        self._lock = threading.Lock()

    # ---------------- document ----------------
    def set_document(self, text: str, facts: str = "", page_label: str = "") -> None:
        """New capture: replace the text and forget the previous conversation.
        facts: page facts (page numbers, running title, headings) for book pages;
        page_label: e.g. "Pages 82 and 83", used to answer "what page is this?"."""
        with self._lock:
            self.document = text
            self.facts = facts
            self.page_label = page_label
            self.history = []
            self.last_not_in_text = False
            self._generation += 1

    def update_document(self, text: str, facts: str = "", page_label: str = "") -> None:
        """Same item, slightly different view: replace the text but keep the conversation."""
        with self._lock:
            self.document = text
            self.facts = facts
            self.page_label = page_label

    def set_provider(self, provider: str) -> None:
        """Choose the language model for the next answers (unknown names are ignored)."""
        if provider in PROVIDERS:
            self.provider = provider

    def has_document(self) -> bool:
        return bool(self.document.strip())

    def cancel(self) -> None:
        """Stop any answer currently being generated."""
        with self._lock:
            self._generation += 1

    # ---------------- model ----------------
    def ask(self, question: str) -> Iterator[str]:
        """
        Answer from the captured text, streamed sentence by sentence.
        Yields READ_ALL alone when the user wants the whole text read.
        Sets last_not_in_text when the text doesn't contain the answer.
        """
        with self._lock:
            generation = self._generation
            self.last_not_in_text = False

        # Exact answers from code (reliable with any model size)
        direct = None
        if _EXPIRY_QUESTION.search(question):
            direct = expiry_answer(self.document)
        elif _PAGE_QUESTION.match(question) and self.page_label:
            direct = f"This is {self.page_label[0].lower() + self.page_label[1:]}."
        if direct:
            yield direct
            self._remember(question, direct, generation)
            return

        with self._lock:
            system = SYSTEM_PROMPT.format(document=self.document,
                                          today=date.today().strftime("%d %B %Y"),
                                          date_checks=expiry_checks(self.document),
                                          facts=self.facts or "None (not a book page).")
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
        logger.info(f"Searching the web for: {query}")
        try:
            results = web_search(query)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
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
            base, headers, model, _ = _endpoint(self.provider)
            r = requests.post(f"{base}/chat/completions", headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": QUERY_PROMPT.format(
                    document=document[:1500], question=question)}],
                "temperature": 0,
                "max_tokens": 30,
            }, timeout=30)
            r.raise_for_status()
            query = r.json()["choices"][0]["message"]["content"].strip().strip('"').splitlines()[0]
            return query[:120] or None
        except Exception as e:
            logger.warning(f"Couldn't build a search query: {e}")
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
        base, headers, model, key = _endpoint(self.provider)
        try:
            with requests.post(
                f"{base}/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "stream": True, "temperature": 0.2},
                stream=True,
                timeout=(5, 120),
            ) as r:
                if not r.ok:
                    r.content                                  # read the error body now: it is gone once the connection closes
                r.raise_for_status()
                for raw in r.iter_lines():
                    if generation != self._generation:
                        logger.info("Answer cancelled")
                        return
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue                               # blank keep-alive lines, comments
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    choices = json.loads(payload).get("choices") or []
                    choice = choices[0] if choices else {}
                    piece = (choice.get("delta") or {}).get("content") or ""
                    finished = bool(choice.get("finish_reason"))
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
                        elif any(m.startswith(head) for m in markers) and not finished:
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

                    if finished:
                        break
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            logger.error(f"Language model returned HTTP {status} for model {model}: {_why(e.response)}")
            if self.provider in QWEN_MODELS:
                yield MSG_NO_MODEL.replace("LLM_MODEL and LLM_BASE_URL", "the Qwen model name (and that Ollama has it)") if status == 404 else MSG_FAILED
            elif not key and status in (400, 401, 403):
                yield MSG_NO_KEY                               # providers answer a missing key with 400, 401 or 403
            else:
                yield (MSG_BAD_KEY if status in (401, 403) else MSG_NO_MODEL if status == 404
                       else MSG_BUSY if status == 429 else MSG_FAILED)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            yield MSG_QWEN_DOWN if self.provider in QWEN_MODELS else MSG_UNREACHABLE
            return
        except Exception as e:
            logger.error(f"Model error: {e}")
            yield MSG_FAILED
            return

        if generation == self._generation and buffer.strip():
            yield buffer.strip()

    def _remember(self, question: str, answer: str, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self.history += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}]
                self.history = self.history[-12:]  # last 6 exchanges
