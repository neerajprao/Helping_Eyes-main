import cv2
import time
import threading
import queue
import os
import re
import sys
import numpy as np
import speech_recognition as sr
import logging
from typing import Tuple, Optional
from collections import deque
from dotenv import load_dotenv

load_dotenv()  # before importing modules that read settings from .env

from speech import Speaker
from text_vision import find_text_region, read_text
from doc_assistant import DocAssistant, READ_ALL, LLM_MODEL, wants_read_all, web_lookup_allowed
from book_mode import PageTurnDetector, read_page

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# The web-search library logs every request it makes; keep the console readable
for noisy in ("ddgs", "primp", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ================= CONFIG =================
# 0 = first camera macOS lists (a plugged-in USB camera usually comes first)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ================= CAPTURE CONTROL =================
MIN_TEXT_CHARS = 8         # ignore specks: need this many characters to count as text
MIN_TEXT_HEIGHT = 14       # px on the 720p preview; smaller text -> "Move closer"
DETECT_INTERVAL = 0.1      # look for text 10x a second (plenty for guidance, saves CPU)
HOLD_TIME = 0.5            # seconds the text must stay steady before capturing
REARM_AFTER = 1.5          # seconds with no text in view before a new capture is allowed
ECHO_GUARD = 0.7           # seconds after the app stops talking before the mic listens

# ================= SPEECH =================
speaker = Speaker()
doc = DocAssistant()

def speak(text: str) -> None:
    """Queue text for speech synthesis"""
    if not text or not text.strip():
        return
    clean = re.sub(r"[*#_`~]", "", text).strip()
    if not clean:
        return
    print(f"Assistant: {clean}")
    speaker.say(clean)

def stop_speech() -> None:
    """Stop talking and stop any answer still being generated"""
    print("Stopping speech")
    doc.cancel()
    speaker.stop()

# ================= SHARED STATE =================
_running = True
_thinking = False          # the model is working on an answer
_rearm_requested = False   # user asked for a new capture
_last_answer = ""          # for "repeat"
_last_question = ""        # for "look it up"
_pending_search = ""       # question waiting for a yes/no to "Should I look it up online?"
_book_mode = False         # read a book page by page (see book_mode.py)
_requests: "queue.Queue[str]" = queue.Queue()

# Short spoken commands handled by the app itself, not the model
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

def request_new_capture() -> None:
    global _rearm_requested, _pending_search
    stop_speech()
    _pending_search = ""
    _rearm_requested = True
    if _book_mode:
        speak("Turn the page and hold it still. I'll read it.")
    else:
        speak("Okay. Show me the next thing.")

def web_answer(question: str) -> None:
    """Search the web for the question and speak the answer"""
    global _thinking, _last_answer
    speak("Let me look that up.")
    _thinking = True
    answer = []
    try:
        for sentence in doc.ask_web(question):
            answer.append(sentence)
            speak(sentence)
    finally:
        _thinking = False
    if answer:
        _last_answer = " ".join(answer)

def set_book_mode(on: bool) -> None:
    """Switch between normal capture and book reading mode"""
    global _book_mode, _pending_search
    stop_speech()
    _pending_search = ""
    _book_mode = on
    if on:
        speak("Book mode. Hold the book open in front of the camera. I'll read each page, and the next one when you turn it.")
    else:
        speak("Normal mode.")

def _speech_chunks(text: str, paragraphs, max_len: int = 600):
    """
    (start, chunk) pieces of the page for tracked speech: one per paragraph,
    long paragraphs split at sentence ends. Each chunk is a separate `say`,
    so the ~1 s start-up pause falls between paragraphs.
    """
    for a, b in paragraphs:
        start = a
        while start < b:
            end = b
            if end - start > max_len:
                cut = max(text.rfind(". ", start, start + max_len),
                          text.rfind("? ", start, start + max_len),
                          text.rfind("! ", start, start + max_len))
                end = cut + 1 if cut > start else start + max_len
            yield start, text[start:end]
            start = end

def load_page(text: str, page_no: int, layout) -> None:
    """A new book page was read: make it the current text and speak it,
    tracking the spoken word so the screen can point at it"""
    global _pending_search, _last_answer
    stop_speech()
    _pending_search = ""
    doc.set_document(text)  # questions now refer to this page
    logger.info(f"Page {page_no}, {len(text)} characters:\n{text}")
    _last_answer = text
    speak(f"Page {page_no}.")
    print(f"Assistant: {text}")
    for start, chunk in _speech_chunks(text, layout.paragraphs):
        speaker.say(chunk, track_from=start)

def speak_document() -> None:
    """Read everything Apple Vision captured, exactly as captured"""
    global _last_answer
    _last_answer = doc.document
    speak(doc.document)

# ================= REQUEST HANDLING =================
def handle_request(text: str) -> None:
    """Route one spoken or typed request: app command, web lookup, read-all, or the model"""
    global _thinking, _last_answer, _last_question, _pending_search

    request = text.strip()
    lower = request.lower()
    if not request:
        return
    print(f"You: {request}")

    # Answer to "Should I look it up online?"
    pending, _pending_search = _pending_search, ""
    if pending:
        if _YES.match(lower):
            stop_speech()
            web_answer(pending)
            return
        if _NO.match(lower):
            speak("Okay.")
            return
        # Anything else is a new request; the offer is dropped

    if _STOP.match(lower):
        stop_speech()
        return
    if _REPEAT.match(lower):
        stop_speech()
        speak(_last_answer or "Nothing to repeat yet.")
        return
    if _BOOK_OFF.match(lower):
        set_book_mode(False)
        return
    if _BOOK_ON.match(lower):
        set_book_mode(True)
        return
    if _NEW_CAPTURE.match(lower):
        request_new_capture()
        return

    m = _SEARCH_FOR.match(lower)
    if m or _SEARCH_LAST.match(lower):
        question = m.group(1) if m else _last_question
        stop_speech()
        if question:
            web_answer(question)
        else:
            speak("What should I look up?")
        return

    if not doc.has_document():
        speak("I haven't captured any text yet. Hold something up to the camera.")
        return

    stop_speech()

    if wants_read_all(lower):
        speak_document()
        return

    _last_question = request
    _thinking = True
    answer = []
    try:
        for sentence in doc.ask(request):
            if sentence == READ_ALL:
                speak_document()
                return
            answer.append(sentence)
            speak(sentence)  # speak each sentence as soon as it's ready
    finally:
        _thinking = False
    if answer:
        _last_answer = " ".join(answer)

    # The text didn't have it: offer to look it up (never for expiry, batch or
    # price, which only the item itself can tell you)
    if doc.last_not_in_text and web_lookup_allowed(request):
        _pending_search = request
        speak("Should I look it up online?")

def request_worker() -> None:
    """Handles requests one at a time, off the camera loop"""
    while _running:
        try:
            text = _requests.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            handle_request(text)
        except Exception as e:
            logger.error(f"Request error: {e}")

# ================= VOICE INPUT =================
def voice_listener() -> None:
    """Listens for spoken questions and commands (not while the app is talking)"""
    try:
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 5  # don't wait forever on Google's server
        recognizer.pause_threshold = 1.0  # a 1 s pause ends the question
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Voice control active. Ask a question after the text is captured.")

    except Exception as e:
        logger.error(f"Voice listener initialization failed: {e}")
        return

    while _running:
        try:
            # Wait until the app has been quiet for a moment (room echo)
            if speaker.spoke_since(time.time() - ECHO_GUARD) or _thinking:
                time.sleep(0.1)
                continue

            started = time.time()
            with mic as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=12)

            # The app started talking while we were recording: that's our own
            # voice in the recording, not the user. Throw it away.
            if speaker.spoke_since(started):
                continue

            try:
                command = recognizer.recognize_google(audio)
                print(f"Heard: {command}")
                _requests.put(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.warning(f"Google API error: {e}")

        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            logger.error(f"Voice error: {e}")
            time.sleep(1)

# ================= TYPED INPUT =================
def typed_listener() -> None:
    """Type a question in the terminal and press Enter (handy for testing)"""
    if not sys.stdin or not sys.stdin.isatty():
        return
    print("You can also type a question here and press Enter.")
    for line in sys.stdin:
        if not _running:
            break
        if line.strip():
            _requests.put(line.strip())

# ================= BACKGROUND FRAME READER =================
class BackgroundFrameReader(threading.Thread):
    """
    Continuously reads frames in background to prevent buffer overflow
    Stores latest frame in a ring buffer
    """

    def __init__(self, camera_index: int, buffer_size: int = 5):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.frame_id = 0
        self.running = True
        self.connected = False
        self.consecutive_failures = 0
        self.max_failures = 10

    def connect(self) -> bool:
        """Open the webcam"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Cannot open webcam {self.camera_index}. On macOS, allow Camera access for your terminal / VS Code.")
                self.close()
                return False

            # Landscape 720p matches the display and keeps detection fast
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                logger.info(f"Camera opened at {w}x{h}")
                self.connected = True
                self.consecutive_failures = 0
                with self.buffer_lock:
                    self.buffer.append(frame)
                    self.frame_id += 1
                return True
            logger.warning("Camera opened but no valid frame returned")
            self.close()
            return False
        except Exception as e:
            logger.error(f"Camera error: {e}")
            self.close()
            return False

    def run(self):
        """Continuously read frames in background"""
        while self.running:
            try:
                if self.cap is None or not self.connected:
                    if not self.connect():
                        time.sleep(2)
                        continue

                ret, frame = self.cap.read()

                if ret and frame is not None and frame.size > 0:
                    with self.buffer_lock:
                        self.buffer.append(frame)
                        self.frame_id += 1
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    logger.warning(f"Camera read failure {self.consecutive_failures}/{self.max_failures}")

                    if self.consecutive_failures >= self.max_failures:
                        logger.info("Reopening camera...")
                        self.close()
                        time.sleep(2)
                        if not self.connect():
                            time.sleep(2)

            except Exception as e:
                logger.error(f"Camera error: {e}")
                self.consecutive_failures += 1
                time.sleep(1)

    def get_frame(self) -> Tuple[int, Optional[np.ndarray]]:
        """Get latest frame and its number (0, None when nothing yet)"""
        with self.buffer_lock:
            if len(self.buffer) > 0:
                return self.frame_id, self.buffer[-1]
        return 0, None

    def close(self):
        """Close camera safely"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.connected = False
        except Exception as e:
            logger.error(f"Error closing camera: {e}")

    def stop(self):
        """Stop the background reader"""
        self.running = False
        time.sleep(0.5)
        self.close()

# ================= CAPTURE =================
def capture_document(frame: np.ndarray) -> bool:
    """Read all text in the full-resolution frame and keep it for questions"""
    try:
        text = read_text(frame)
    except Exception as e:
        logger.error(f"Apple Vision error: {e}")
        speak("Could not read the text. Try again.")
        return False

    if not text.strip():
        return False

    global _pending_search
    stop_speech()
    _pending_search = ""
    doc.set_document(text)
    logger.info(f"Captured {len(text)} characters:\n{text}")
    speak("Got it. What would you like to know? You can also say, read everything.")
    return True

# ================= MAIN =================
def main():
    """Main loop: find text live, guide the user, capture it when held still"""
    global _running, _rearm_requested

    # Open the webcam on the main thread first: macOS can only show the
    # camera-permission prompt from the main thread.
    frame_reader = BackgroundFrameReader(CAMERA_INDEX)
    if not frame_reader.connect():
        logger.error("Failed to open webcam")
        return
    frame_reader.start()

    if not doc.warm_up():
        speak(f"Warning. I can't reach the language model. Please start Ollama and pull {LLM_MODEL}.")

    speak("System online. Show me something to read.")

    for target in (request_worker, voice_listener, typed_listener):
        threading.Thread(target=target, daemon=True).start()

    stable_start = 0
    is_stable = False
    armed = True                 # a new capture is allowed
    last_text_seen = time.time()

    # ================= GUIDE BOX CONFIG =================
    DISPLAY_W, DISPLAY_H = 1280, 720
    GUIDE_MARGIN_X = 30
    GUIDE_MARGIN_Y = 20
    GUIDE_X1, GUIDE_Y1 = GUIDE_MARGIN_X, GUIDE_MARGIN_Y
    GUIDE_X2, GUIDE_Y2 = DISPLAY_W - GUIDE_MARGIN_X, DISPLAY_H - GUIDE_MARGIN_Y

    GUIDE_COLOR_IDLE   = (180, 180, 180)
    GUIDE_COLOR_DETECT = (0, 200, 255)
    GUIDE_COLOR_HOLD   = (0, 165, 255)
    GUIDE_COLOR_READY  = (0, 255, 0)
    LINE_COLOR         = (255, 100, 0)

    last_spoken_hint = ""
    last_hint_time = 0
    last_light_warning = 0
    HINT_COOLDOWN = 2.0

    # ================= HELPER FUNCTIONS =================
    def draw_guide_box(img, color, label=None):
        corner_len = 30
        thickness = 2
        corners = [(GUIDE_X1, GUIDE_Y1, 1, 1), (GUIDE_X2, GUIDE_Y1, -1, 1),
                   (GUIDE_X1, GUIDE_Y2, 1, -1), (GUIDE_X2, GUIDE_Y2, -1, -1)]
        for (cx, cy, dx, dy) in corners:
            cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, thickness + 1)
            cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, thickness + 1)
        cv2.rectangle(img, (GUIDE_X1, GUIDE_Y1), (GUIDE_X2, GUIDE_Y2), color, 1)
        if label:
            cv2.putText(img, label, (GUIDE_X1 + 10, GUIDE_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def text_inside_guide(x1, y1, x2, y2):
        overlap_x1, overlap_y1 = max(x1, GUIDE_X1), max(y1, GUIDE_Y1)
        overlap_x2, overlap_y2 = min(x2, GUIDE_X2), min(y2, GUIDE_Y2)
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        text_area = (x2 - x1) * (y2 - y1)
        return overlap_area / text_area if text_area > 0 else 0.0

    def get_guidance(box, lines):
        """Spoken hint to bring the text fully into view ("" when it's fine)"""
        x1, y1, x2, y2 = box
        heights = sorted(l.box[3] - l.box[1] for l in lines)
        if heights and heights[len(heights) // 2] < MIN_TEXT_HEIGHT:
            return "Move closer"
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        gx, gy = (GUIDE_X1 + GUIDE_X2) // 2, (GUIDE_Y1 + GUIDE_Y2) // 2
        # Only nudge when the text is also near an edge (i.e. likely cut off)
        near_x_edge = x1 <= GUIDE_X1 + 5 or x2 >= GUIDE_X2 - 5
        near_y_edge = y1 <= GUIDE_Y1 + 5 or y2 >= GUIDE_Y2 - 5
        if near_x_edge and abs(cx - gx) > 80:
            return "Move Left" if cx > gx else "Move Right"
        if near_y_edge and abs(cy - gy) > 80:
            return "Move Up" if cy > gy else "Move Down"
        return ""

    def handle_key():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('s'):
            stop_speech()
        elif key == ord('r'):
            request_new_capture()
        elif key == ord('b'):
            set_book_mode(not _book_mode)
        elif key == ord('a'):
            if doc.has_document():
                stop_speech()
                speak_document()
            else:
                speak("Nothing captured yet.")
        return True

    # ================= MAIN LOOP =================
    logger.info("Starting main loop")
    window = "Smart Reader"
    no_frame_counter = 0
    last_frame_id = 0
    last_detect_time = 0
    box, lines = None, []

    # ================= BOOK MODE STATE =================
    page_detector = PageTurnDetector()
    book_active = False          # book mode was running on the previous frame
    page_layout = None           # columns / paragraphs of the page being read
    layout_scale = (1.0, 1.0)    # full frame -> display
    page_no = 0
    turn_prompted = True         # "Turn the page" already said for this page

    def draw_layout(img):
        """Draw the book-mode analysis: gutter, columns, numbered paragraphs"""
        if page_layout is None or page_detector.state == "TURNING":
            return
        sx, sy = layout_scale
        if page_layout.gutter_x:
            gx = int(page_layout.gutter_x * sx)
            cv2.line(img, (gx, 0), (gx, DISPLAY_H), (0, 255, 255), 2)
        for x1, _, x2, _ in page_layout.columns:
            cv2.rectangle(img, (int(x1 * sx), GUIDE_Y1), (int(x2 * sx), GUIDE_Y2), (255, 120, 0), 1)
        for i, (x1, y1, x2, y2) in enumerate(page_layout.blocks, 1):
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(img, p1, p2, (0, 200, 0), 2)
            badge = (max(14, p1[0] - 20), p1[1] + 12)   # in the margin, left of the paragraph
            cv2.circle(img, badge, 14, (0, 0, 220), -1)
            cv2.putText(img, str(i), (badge[0] - 7 if i < 10 else badge[0] - 12, badge[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def draw_reading_highlight(img):
        """Highlight the line being spoken and box the exact word"""
        pos = speaker.position
        if pos is None or page_layout is None or page_detector.state == "TURNING":
            return
        span = page_layout.line_at(pos)
        if span is None:
            return
        sx, sy = layout_scale
        x1, y1, x2, y2 = span.box
        overlay = img.copy()
        cv2.rectangle(overlay, (int(x1 * sx) - 4, int(y1 * sy) - 3), (int(x2 * sx) + 4, int(y2 * sy) + 3),
                      (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        word = page_layout.word_at(pos)
        if word:
            wx1, wy1, wx2, wy2 = (int(word[0] * sx), int(word[1] * sy), int(word[2] * sx), int(word[3] * sy))
            cv2.rectangle(img, (wx1 - 3, wy1 - 3), (wx2 + 3, wy2 + 3), (0, 120, 255), 2)

    def draw_book_status(img, note=""):
        cv2.rectangle(img, (0, DISPLAY_H - 40), (DISPLAY_W, DISPLAY_H), (40, 40, 40), -1)
        status = (f"BOOK MODE   Page {page_no}   {note or page_detector.state}   "
                  f"motion {page_detector.motion:.1f} (still < {page_detector.motion_off:.1f})   B = exit")
        cv2.putText(img, status, (15, DISPLAY_H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    try:
        while True:
            try:
                frame_id, frame = frame_reader.get_frame()

                if frame is None:
                    no_frame_counter += 1
                    if no_frame_counter > 30:
                        logger.warning("No frames for 1 second, waiting for camera...")
                        time.sleep(1)
                        no_frame_counter = 0
                    else:
                        time.sleep(0.033)
                    continue
                no_frame_counter = 0

                # Same frame as last time: nothing new to look at, just keep
                # the window responsive (avoids spinning a CPU core at 100%)
                if frame_id == last_frame_id:
                    if not handle_key():
                        break
                    time.sleep(0.005)
                    continue
                last_frame_id = frame_id

                if _rearm_requested:
                    _rearm_requested = False
                    armed = True
                    is_stable = False

                # Preview is shown as the camera sees it (not mirrored)
                small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                display = small.copy()

                # ===== BOOK MODE: read page by page, triggered by page turns =====
                # Runs even while speaking, so turning the page mid-read moves on.
                if _book_mode:
                    if not book_active:
                        book_active = True
                        page_detector.reset()
                        page_layout, page_no, turn_prompted = None, 0, True

                    if page_detector.update(small):
                        draw_book_status(display, "READING PAGE...")
                        cv2.imshow(window, display)
                        cv2.waitKey(1)
                        try:
                            text, page_layout = read_page(frame)
                        except Exception as e:
                            logger.error(f"Page reading error: {e}")
                            text, page_layout = "", None
                        layout_scale = (DISPLAY_W / frame.shape[1], DISPLAY_H / frame.shape[0])
                        if text.strip():
                            page_no += 1
                            load_page(text, page_no, page_layout)
                            turn_prompted = False
                        else:
                            speak("I can't see any text on this page.")
                    elif page_no and not turn_prompted and not (speaker.is_speaking or _thinking):
                        speak("Turn the page.")
                        turn_prompted = True

                    draw_layout(display)
                    draw_reading_highlight(display)
                    draw_book_status(display)
                    cv2.imshow(window, display)
                    if not handle_key():
                        break
                    continue
                elif book_active:
                    book_active = False

                busy = speaker.is_speaking or _thinking
                if busy:
                    label = "THINKING..." if _thinking and not speaker.is_speaking else "SPEAKING...  S = stop"
                    draw_guide_box(display, GUIDE_COLOR_IDLE, label)
                    cv2.imshow(window, display)
                    if not handle_key():
                        break
                    continue

                # ===== LOW LIGHT DETECTION =====
                if np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) < 50:
                    cv2.putText(display, "LOW LIGHT", (GUIDE_X1 + 10, GUIDE_Y2 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if armed and time.time() - last_light_warning > 5:
                        speak("Low light detected")
                        last_light_warning = time.time()

                # ===== FIND TEXT (Apple Vision, fast mode) =====
                # Between checks, the last result is redrawn on the new frame.
                if time.time() - last_detect_time >= DETECT_INTERVAL:
                    last_detect_time = time.time()
                    try:
                        box, lines = find_text_region(small, MIN_TEXT_CHARS)
                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                        box, lines = None, []
                    if box is not None:
                        last_text_seen = time.time()

                # Text gone for a moment -> ready for the next item
                if not armed and time.time() - last_text_seen > REARM_AFTER:
                    armed = True
                    logger.info("Ready for a new capture")

                if box is not None:
                    for line in lines:
                        lx1, ly1, lx2, ly2 = line.box
                        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), LINE_COLOR, 1)
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), LINE_COLOR, 2)

                if not armed:
                    # Already captured this item: stay quiet and wait for questions
                    draw_guide_box(display, GUIDE_COLOR_READY, "CAPTURED - ASK ME   R = new capture   A = read all")
                elif box is not None:
                    hint = get_guidance(box, lines)
                    overlap = text_inside_guide(*box)

                    if overlap >= 0.65 and not hint:
                        if not is_stable:
                            is_stable = True
                            stable_start = time.time()
                        held = time.time() - stable_start
                        if held < HOLD_TIME:
                            draw_guide_box(display, GUIDE_COLOR_HOLD, f"HOLD {HOLD_TIME - held:.1f}s")
                        else:
                            draw_guide_box(display, GUIDE_COLOR_READY, "CAPTURING...")
                            cv2.imshow(window, display)
                            cv2.waitKey(1)
                            if capture_document(frame):
                                armed = False
                            is_stable = False
                    else:
                        is_stable = False
                        draw_guide_box(display, GUIDE_COLOR_DETECT, "ADJUST POSITION")
                        if hint:
                            now = time.time()
                            if hint != last_spoken_hint or now - last_hint_time > HINT_COOLDOWN:
                                speak(hint)
                                last_spoken_hint, last_hint_time = hint, now
                else:
                    is_stable = False
                    draw_guide_box(display, GUIDE_COLOR_IDLE, "SHOW TEXT HERE")

                cv2.imshow(window, display)
                if not handle_key():
                    break

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.5)
                continue

    finally:
        logger.info("Shutting down...")
        _running = False
        doc.cancel()
        speaker.shutdown()
        try:
            frame_reader.stop()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # lets macOS actually close the window
        except Exception:
            pass
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
