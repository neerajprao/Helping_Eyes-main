"""
Spoken and typed commands handled by the application itself, not the model.

classify() turns one request into a Command. The server routes on it (server.py),
so the browser needs no command logic of its own.
"""

import re
from dataclasses import dataclass

from doc_assistant import wants_read_all

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
