"""
Spoken and typed commands handled by the application itself, not the model, plus the
spoken voice (text to speech).

classify() turns one request into a Command. The server routes on it (server.py),
so the browser needs no command logic of its own.

synthesize() turns one sentence into MP3 audio with a free Microsoft Edge neural voice (edge-tts: no
account, key or card). It is the last section of this file. The page uses the browser's own voice if
this fails. Settings (environment or .env):
    TTS_VOICE    voice name (default en-US-JennyNeural, female; also en-US-AriaNeural, en-US-AvaNeural, en-GB-SoniaNeural)
    TTS_RATE     speaking speed, e.g. +10% or -10% (default +0%)
    TTS_ENABLED  set to 0 to always use the browser voice
"""

import asyncio
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from assistant import wants_read_all

_STOP = re.compile(r"^(stop|stop (it|talking|speaking|reading)|be quiet|quiet|shut up|cancel)[.!]*$")
_REPEAT = re.compile(r"^(repeat|repeat that|say (that|it) again|again|pardon|what)[.?!]*$")
_NEW_CAPTURE = re.compile(r"^(next|next page|new page|scan( again)?|capture( again)?|read something else|new (text|item|document))[.!]*$")
_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|please do|do it|go ahead|yes please|look it up|search)[.!]*$")
_NO = re.compile(r"^(no|nope|nah|no thanks|no thank you|don't|never mind|leave it)[.!]*$")
_BOOK_ON = re.compile(r"^(start |turn on |enter |switch to )?(book|reading) mode( on)?[.!]*$|^read (a|my|this) book[.!]*$")
_BOOK_OFF = re.compile(r"^(stop|exit|leave|end|turn off|quit) (book|reading) mode[.!]*$|^(book|reading) mode off[.!]*$|^(normal|regular) mode[.!]*$")
# "look it up", "search online", "google that" -> web search for the last question
_SEARCH_LAST = re.compile(r"^(please )?(look (it|that) up|search|google)( (it|that))?( (online|on the internet|on the web|the web|the internet))?( please)?[.!]*$")
# "search the web for X", "look up X", "google X" -> web search for X
_SEARCH_FOR = re.compile(r"^(?:please )?(?:search (?:the web |online |the internet )?for|look up|google) (.+?)[.?!]*$")


# Words that make a request possibly mean "search the web" (then the language model decides), and wordings that
# plainly agree to or refuse the offer "Should I look it up online?"
_WEB_WORDS = re.compile(r"\b(web|online|internet|google|search|look(ing)? (it|that|this)? ?up|browse|find out|check)\b")
_AGREES = re.compile(r"^(yes|yeah|yep|yup|yea|sure|ok|okay|alright|please|go ahead|do it|go for it|definitely|absolutely|of course|why not|sounds good|fine)\b"
                     r"|\b(look (it|that|this) up|search|google|check online|go online|on the (web|internet))\b")
_REFUSES = re.compile(r"^(no|nope|nah|don't|do not|never mind|nevermind|leave it|skip|forget it|not now|not really)\b")


@dataclass
class Command:
    """name is one of: stop, repeat, book_off, book_on, new_capture, search_last,
    search_for (query holds the phrase), read_all, question, yes, no, empty."""
    name: str
    query: str = ""


def classify(text: str, offer_pending: bool = False, judge=None) -> Command:
    """
    Decide what one request means. offer_pending is True right after the app asked
    "Should I look it up online?": only then do yes / no answer that offer.

    judge(text, offer_pending) -> "yes" | "no" | "search" | "search: <topic>" | "other" is the language model's
    reading of what was said; it is asked only when the wording is not already clear, so any phrasing that
    means "look it up online" works, not just the exact ones listed above.
    """
    request = text.strip()
    lower = request.lower()
    if not request:
        return Command("empty")
    if offer_pending:
        if _YES.match(lower):
            return Command("yes")
        if _NO.match(lower):
            return Command("no")
        if _REFUSES.match(lower):
            return Command("no")
        if _AGREES.search(lower):
            return Command("yes")
        verdict = _judged(judge, request, True)
        if verdict == "yes":
            return Command("yes")
        if verdict == "no":
            return Command("no")
        # anything else is a new request; the offer is dropped by the caller
    if _STOP.match(lower):
        return Command("stop")
    if _REPEAT.match(lower):
        return Command("repeat")
    if _BOOK_OFF.match(lower):
        return Command("book_off")
    if _BOOK_ON.match(lower):
        return Command("book_on")
    if _NEW_CAPTURE.match(lower):
        return Command("new_capture")
    m = _SEARCH_FOR.match(lower)
    if m:
        return Command("search_for", m.group(1))
    if _SEARCH_LAST.match(lower):
        return Command("search_last")
    if wants_read_all(lower):
        return Command("read_all")
    if _WEB_WORDS.search(lower):
        verdict = _judged(judge, request, False)
        if verdict.startswith("search"):
            topic = verdict.partition(":")[2].strip()
            return Command("search_for", topic) if topic else Command("search_last")
    return Command("question", request)


def _judged(judge, text: str, offer_pending: bool) -> str:
    """The language model's reading of the request ("other" when there is no judge or it fails)."""
    if judge is None:
        return "other"
    try:
        return (judge(text, offer_pending) or "other").strip().lower()
    except Exception:
        return "other"


# ---------------------------------------------------------------- spoken voice (text to speech)
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_ENABLED = os.getenv("TTS_ENABLED", "1") != "0"
MAX_CHARS = 700                 # one sentence or a short chunk; keeps each call quick
_CACHE_SIZE = 60                # repeated sentences ("Got it...") are not synthesized twice

_cache: "OrderedDict[str, bytes]" = OrderedDict()
_lock = threading.Lock()


class TtsError(Exception):
    pass


# A number above 9999 is nearly always a serial, batch or phone number: say its digits one by one,
# never "twelve thousand three hundred...". Comma-grouped forms (12,345,678) count too.
_LONG_NUMBER = re.compile(r"(?<![\d,])(?:\d{1,3}(?:,\d{3})+|\d{5,})(?!\d)")


def spoken_form(text: str) -> str:
    """The text as it is sent to the voice: numbers above 9999 spelled out digit by digit."""
    def digits(m):
        number = m.group(0).replace(",", "")
        return " ".join(number) if int(number) > 9999 else m.group(0)
    return _LONG_NUMBER.sub(digits, text)


async def _speak(text: str) -> bytes:
    import edge_tts
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


def synthesize(text: str) -> bytes:
    """MP3 audio of one sentence. Raises TtsError when the voice service cannot be reached."""
    text = text.strip()[:MAX_CHARS]
    if not text:
        raise TtsError("empty text")
    with _lock:
        if text in _cache:
            _cache.move_to_end(text)
            return _cache[text]
    try:
        audio = asyncio.run(asyncio.wait_for(_speak(spoken_form(text)), timeout=30))
    except Exception as e:                              # no network, service changed, timeout, ...
        raise TtsError(f"{type(e).__name__}: {e}")
    if not audio:
        raise TtsError("the voice service returned no audio")
    with _lock:
        _cache[text] = audio
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return audio
