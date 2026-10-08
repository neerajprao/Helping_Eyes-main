"""
Spoken and typed commands handled by the application itself, not the model, plus the
spoken voice (text to speech).

classify() turns one request into a Command. The server routes on it (server.py),
so the browser needs no command logic of its own.

synthesize() turns one sentence into WAV audio with a Gemini voice (the last section of this file).
The page falls back to the browser voice on any error. Settings (environment or .env):
    TTS_MODEL    Gemini speech model (default gemini-3.8-flash-lite-tts)
    TTS_VOICE    voice name, e.g. Kore, Puck, Charon, Aoede, Zephyr (default Kore)
    TTS_ENABLED  set to 0 to keep the browser voice
The key is LLM_API_KEY, the same one as for the language model.
"""

import base64
import io
import logging
import os
import re
import threading
import wave
from collections import OrderedDict
from dataclasses import dataclass

import requests

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


@dataclass
class Command:
    """name is one of: stop, repeat, book_off, book_on, new_capture, search_last,
    search_for (query holds the phrase), read_all, question, yes, no, empty."""
    name: str
    query: str = ""


def classify(text: str, offer_pending: bool = False) -> Command:
    """
    Decide what one request means. offer_pending is True right after the app asked
    "Should I look it up online?": only then do yes / no answer that offer.
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
    return Command("question", request)


# ---------------------------------------------------------------- spoken voice (text to speech)
logger = logging.getLogger(__name__)

TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-lite-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
TTS_ENABLED = os.getenv("TTS_ENABLED", "1") != "0" and bool(os.getenv("LLM_API_KEY", ""))
MAX_CHARS = 700                 # one sentence or a short chunk; keeps each call quick
_CACHE_SIZE = 60                # repeated sentences ("Got it...") are not synthesized twice

_cache: "OrderedDict[str, bytes]" = OrderedDict()
_lock = threading.Lock()


class TtsError(Exception):
    pass


def _wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def synthesize(text: str) -> bytes:
    text = text.strip()[:MAX_CHARS]
    if not text:
        raise TtsError("empty text")
    key = f"{TTS_MODEL}|{TTS_VOICE}|{text}"
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent",
            headers={"x-goog-api-key": os.getenv("LLM_API_KEY", "")},
            json={"contents": [{"parts": [{"text": text}]}],
                  "generationConfig": {"responseModalities": ["AUDIO"],
                                       "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": TTS_VOICE}}}}},
            timeout=40)
    except requests.exceptions.RequestException as e:
        raise TtsError(f"unreachable: {e}")
    if not r.ok:
        raise TtsError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        audio = base64.b64decode(part["data"])
        mime = part.get("mimeType", "").lower()
    except (KeyError, IndexError, ValueError) as e:
        raise TtsError(f"no audio in the reply: {e}")
    if "wav" not in mime:                               # raw 16-bit mono PCM, e.g. "audio/L16;codec=pcm;rate=24000"
        rate = 24000
        for piece in mime.replace(";", " ").split():
            if piece.startswith("rate="):
                rate = int(piece[5:])
        audio = _wav(audio, rate)
    with _lock:
        _cache[key] = audio
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return audio
