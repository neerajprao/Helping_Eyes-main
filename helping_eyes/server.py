"""
Helping Eyes: the application (one codebase for a laptop and for the cloud).

A FastAPI server plus a browser page. The browser handles the camera, speech
recognition and speech output; the server does everything else:

    WS   /ws/live         preview frames in, guidance out (live.py): "Move closer",
                          hold still, auto-capture; page turns in book mode
    POST /api/capture     full-size photo -> text kept for questions
    POST /api/book/page   full-size photo of a book view -> what to read, from where
    POST /api/ask         a spoken or typed request -> commands, answers, web lookup,
                          streamed as NDJSON (the routing lives in commands.py)
    GET  /api/health      health check
    GET  /                the page (web/index.html)

Run it:
    uvicorn server:app --port 7860          (then open http://localhost:7860)

Settings (environment or .env), the same for every deployment:
    OCR_ENGINE   apple (macOS default) or rapidocr (everywhere else)
    LLM_MODEL    qwen2.5:7b-instruct by default; the cloud image sets qwen2.5:3b-instruct
    OLLAMA_HOST  Ollama address (default http://localhost:11434)
"""

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from commands import classify
from doc_assistant import LLM_MODEL, OLLAMA_HOST, READ_ALL, DocAssistant, web_lookup_allowed
from vision import BookReader, BookWatcher, LiveGuide, read_page
from text_vision import OCR_ENGINE, find_text_region, read_text_enhanced

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("helping_eyes.web")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SESSION_TTL = 30 * 60          # seconds a session (captured item, conversation) is kept
MAX_SESSIONS = 200
MAX_SIDE = 1600                # photos are downscaled to this for OCR speed
RATE_LIMIT = 30                # requests per minute per client
MAX_SOCKETS = 2                # live streams per client
_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

app = FastAPI(title="Helping Eyes", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# One OCR job at a time: the OCR engines are not meant to be called from several threads at once
_ocr_lock = threading.Lock()


# ---------------- sessions and rate limiting ----------------
class Session:
    """Everything the server remembers about one browser: the captured item and the conversation."""

    def __init__(self):
        self.doc = DocAssistant()
        self.last_question = ""
        self.last_answer = ""
        self.pending_search = ""       # question waiting for a yes / no to "Should I look it up online?"
        self.book_mode = False
        self.guide = LiveGuide()       # normal mode: live guidance
        self.watcher = BookWatcher()   # book mode: page turns
        self.reader = BookReader()     # book mode: what to read
        self.touched = time.time()


_sessions: Dict[str, Session] = {}
_sessions_lock = threading.Lock()
_hits: Dict[str, deque] = defaultdict(deque)
_sockets: Dict[str, int] = defaultdict(int)


def _get_session(sid: str) -> Session:
    if not _SID.match(sid or ""):
        raise HTTPException(400, "Missing or invalid session id.")
    now = time.time()
    with _sessions_lock:
        for key in [k for k, s in _sessions.items() if now - s.touched > SESSION_TTL]:
            del _sessions[key]
        session = _sessions.get(sid)
        if session is None:
            if len(_sessions) >= MAX_SESSIONS:
                raise HTTPException(503, "The server is busy. Please try again in a few minutes.")
            session = _sessions[sid] = Session()
        session.touched = now
        return session


def _client(request) -> str:
    headers, host = request.headers, request.client.host if request.client else "?"
    return headers.get("x-forwarded-for", host).split(",")[0].strip()


def _rate_limit(request: Request) -> None:
    client = _client(request)
    now = time.time()
    hits = _hits[client]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a moment.")
    hits.append(now)


# ---------------- helpers ----------------
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _sentences(text: str) -> Iterator[str]:
    """Split text into speakable pieces: paragraphs, then sentences."""
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in _SENTENCES.split(paragraph.strip()):
            if sentence.strip():
                yield sentence.strip()


def _event(kind: str, **data) -> str:
    return json.dumps({"type": kind, **data}) + "\n"


def _stream(events: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _decode(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "The file is not an image.")
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


# ---------------- API ----------------
@app.on_event("startup")
def _warm_up() -> None:
    # Load the language model and the OCR engine in the background so the first request is fast
    def work():
        DocAssistant().warm_up()
        # Real text, not a blank image: the OCR models only load when they have something to read,
        # and the first capture would otherwise take many seconds
        sample = np.full((240, 900, 3), 255, np.uint8)
        cv2.putText(sample, "Helping Eyes warm up 123", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
        with _ocr_lock:
            find_text_region(sample)
            read_text_enhanced(sample)
            read_page(sample, split=False, enhance=True)
        logger.info("Warm-up finished")
    threading.Thread(target=work, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "favicon.svg"), media_type="image/svg+xml")


@app.get("/api/health")
def health() -> dict:
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).json()
        models = [m["name"] for m in tags.get("models", [])]
        model_ready = any(m.split(":")[0] == LLM_MODEL.split(":")[0] and m.endswith(LLM_MODEL.split(":")[-1]) for m in models)
    except Exception:
        model_ready = False
    return {"status": "ok", "model": LLM_MODEL, "model_ready": model_ready, "ocr": OCR_ENGINE}


@app.post("/api/capture")
async def capture(request: Request, image: UploadFile = File(...), sid: str = Form(""),
                  book: bool = Form(False)) -> dict:
    """Read all text in a photo and keep it for questions."""
    _rate_limit(request)
    session = _get_session(sid)
    frame = _decode(await image.read())

    def work():
        with _ocr_lock:
            if book:    # a two-page spread: split it, find page numbers, order the columns
                text, layout = read_page(frame, split=True, enhance=True)
                return text, layout, "book"
            # a single item: the untouched frame, plus corrected versions when it looks like it needs them
            text, method = read_text_enhanced(frame)
            return text, None, method

    started = time.time()
    text, layout, method = await run_in_threadpool(work)
    seconds = round(time.time() - started, 2)
    logger.info(f"capture: {len(text)} characters in {seconds} s (book={book}, version={method})")
    if not text.strip():
        return {"ok": False, "text": "", "seconds": seconds,
                "message": "I can't see any text. Move closer, hold the item steady and try again."}

    session.pending_search = ""
    session.last_answer = ""
    session.doc.set_document(text, layout.summary() if layout else "", layout.label() if layout else "")
    return {"ok": True, "text": text, "seconds": seconds, "method": method,
            "label": layout.label() if layout else "",
            "running_title": (layout.running_title() or "") if layout else "",
            "message": "Got it. What would you like to know? You can also say, read everything."}


@app.post("/api/book/page")
async def book_page(request: Request, image: UploadFile = File(...), sid: str = Form(""),
                    event: str = Form("changed"), read_pos: int = Form(0),
                    reading: bool = Form(False), thinking: bool = Form(False)) -> dict:
    """
    Book mode: the page view became steady (a page turn, or the view moved).
    Replies with what to speak and from where, plus the layout for the on-screen overlay.
    """
    _rate_limit(request)
    session = _get_session(sid)
    frame = _decode(await image.read())

    def work():
        with _ocr_lock:
            return session.reader.process(frame, event, read_pos, reading, thinking, session.doc)

    started = time.time()
    result = await run_in_threadpool(work)
    result["seconds"] = round(time.time() - started, 2)
    if result["action"] in ("new", "resume"):
        session.pending_search = ""
        session.last_answer = result["text"][result["from_pos"]:]
    return result


class Question(BaseModel):
    sid: str
    question: str = ""


@app.post("/api/ask")
def ask(q: Question, request: Request) -> StreamingResponse:
    """
    One spoken or typed request. Replies as NDJSON, one event per line:
        sentence      {text, source?}   speak this (source "document" or "web")
        command       {name}            the page must do something: stop, new_capture, book_on, book_off
        offer_search  {text}            "Should I look it up online?"
        done
    """
    _rate_limit(request)
    session = _get_session(q.sid)
    request_text = q.question.strip()

    def sentences(text: str, source: Optional[str] = None) -> Iterator[str]:
        for s in _sentences(text):
            yield _event("sentence", text=s, **({"source": source} if source else {}))

    def web(question: str) -> Iterator[str]:
        yield _event("sentence", text="Let me look that up.")
        answer = []
        for s in session.doc.ask_web(question):
            answer.append(s)
            yield _event("sentence", text=s, source="web")
        if answer:
            session.last_answer = " ".join(answer)

    def events() -> Iterator[str]:
        started = time.time()
        pending, session.pending_search = session.pending_search, ""
        cmd = classify(request_text, offer_pending=bool(pending))

        if cmd.name == "empty":
            pass
        elif cmd.name == "yes":
            session.doc.cancel()
            yield from web(pending)
        elif cmd.name == "no":
            yield _event("sentence", text="Okay.")
        elif cmd.name == "stop":
            session.doc.cancel()
            yield _event("command", name="stop")
        elif cmd.name == "repeat":
            yield _event("command", name="stop")
            yield from sentences(session.last_answer or "Nothing to repeat yet.")
        elif cmd.name in ("book_on", "book_off"):
            session.doc.cancel()
            session.book_mode = cmd.name == "book_on"
            session.reader.reset()
            session.watcher.reset()
            yield _event("command", name=cmd.name)
            yield _event("sentence", text="Book mode. Hold the book open in front of the camera. I'll read each page, "
                                          "and the next one when you turn it." if session.book_mode else "Normal mode.")
        elif cmd.name == "new_capture":
            session.doc.cancel()
            session.guide.rearm()
            yield _event("command", name="new_capture")
            yield _event("sentence", text="Turn the page and hold it still. I'll read it." if session.book_mode
                         else "Okay. Show me the next thing.")
        elif cmd.name in ("search_for", "search_last"):
            question = cmd.query or session.last_question
            session.doc.cancel()
            if question:
                yield from web(question)
            else:
                yield _event("sentence", text="What should I look up?")
        elif not session.doc.has_document():
            yield _event("sentence", text="I haven't captured any text yet. Hold something up to the camera.")
        else:
            session.doc.cancel()
            if cmd.name == "read_all":
                session.last_answer = session.doc.document
                yield from sentences(session.doc.document, "document")
            else:
                session.last_question = cmd.query
                first, answer = None, []
                for s in session.doc.ask(cmd.query):
                    if first is None:
                        first = round(time.time() - started, 2)
                    if s == READ_ALL:
                        session.last_answer = session.doc.document
                        yield from sentences(session.doc.document, "document")
                        answer = []
                        break
                    answer.append(s)
                    yield _event("sentence", text=s)
                if answer:
                    session.last_answer = " ".join(answer)
                # The text didn't have it: offer to look it up (never for expiry, batch or price)
                if session.doc.last_not_in_text and web_lookup_allowed(cmd.query):
                    session.pending_search = cmd.query
                    yield _event("offer_search", text="Should I look it up online?")
                logger.info(f"ask: first sentence after {first} s, total {time.time() - started:.1f} s")
        yield _event("done", seconds=round(time.time() - started, 2))

    return _stream(events())


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    """
    Live camera stream. The page sends one JPEG preview frame at a time (and waits for the reply):
        text message  {"type": "mode", "mode": "normal" | "book"}
                      {"type": "rearm"}                     ready for a new item
                      {"type": "capture_result", "ok": bool}
        binary        a JPEG frame -> a "guide" reply (normal mode) or a "book" reply (page turns)
    """
    client = _client(ws)
    if _sockets[client] >= MAX_SOCKETS:
        await ws.close(code=1013)
        return
    sid = ws.query_params.get("sid", "")
    try:
        session = _get_session(sid)
    except HTTPException:
        await ws.close(code=1008)
        return
    await ws.accept()
    _sockets[client] += 1
    book = session.book_mode
    session.guide.reset()
    session.watcher.reset()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            session.touched = time.time()
            if message.get("bytes"):
                frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if book:
                    reply = session.watcher.update(frame)
                else:
                    def work():
                        with _ocr_lock:
                            return session.guide.update(frame)
                    reply = await run_in_threadpool(work)
                await ws.send_json(reply)
            elif message.get("text"):
                try:
                    control = json.loads(message["text"])
                except ValueError:
                    continue
                kind = control.get("type")
                if kind == "mode":
                    book = control.get("mode") == "book"
                    session.book_mode = book
                    session.watcher.reset()
                    session.reader.reset()
                    session.guide.reset()
                elif kind == "rearm":
                    session.guide.rearm()
                elif kind == "capture_result":
                    session.guide.capture_result(bool(control.get("ok")))
    except WebSocketDisconnect:
        pass
    finally:
        _sockets[client] = max(0, _sockets[client] - 1)
