import cv2
import time
import threading
import os
import re
import numpy as np
import speech_recognition as sr
import logging
from typing import Tuple, Optional
from collections import deque
from dotenv import load_dotenv
from speech import Speaker
from text_vision import find_text_region, read_text, scale_box, mirror_box

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
load_dotenv()
# 0 = first camera macOS lists (a plugged-in USB camera usually comes first)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ================= READ CONTROL =================
READ_COOLDOWN = 3          # seconds between two reads
MIN_TEXT_CHARS = 8         # ignore specks: need this many characters to count as text
MIN_TEXT_HEIGHT = 14       # px on the 720p preview; smaller text -> "Move closer"

# ================= SPEECH =================
speaker = Speaker()

def speak(text: str) -> None:
    """Queue text for speech synthesis"""
    if not text or not text.strip():
        return
    clean = re.sub(r"[*#_`~]", "", text).strip()
    if not clean:
        return
    print(f"🗣️ {clean[:120]}")
    logger.info(f"Speaking: {clean[:80]}")
    speaker.say(clean)

def stop_speech() -> None:
    """Stop all speech and clear queue"""
    print("🛑 Stopping speech")
    speaker.stop()

def restart_capture():
    """Stop speech and restart capture flow"""
    global _restart_capture
    print("🔄 Restarting capture")
    stop_speech()
    _restart_capture = True

# ================= VOICE CONTROL =================
_voice_thread_running = True
_force_next_capture = False
_restart_capture = False

def voice_listener(get_last_text_callback) -> None:
    """Voice command listener with error recovery"""
    global _force_next_capture, _voice_thread_running

    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        print("🎤 Voice control active...")
        logger.info("Voice listener started")

    except Exception as e:
        logger.error(f"❌ Voice listener initialization failed: {e}")
        return

    while _voice_thread_running:
        try:
            if speaker.is_speaking:
                time.sleep(0.1)
                continue

            with mic as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"🎤 Heard: {command}")
                logger.info(f"Command: {command}")

                if "stop" in command:
                    stop_speech()
                elif "repeat" in command:
                    stop_speech()
                    last_text = get_last_text_callback()
                    if last_text and last_text.strip():
                        speak(last_text)
                    else:
                        speak("Nothing to repeat")
                elif "next" in command:
                    stop_speech()
                    _force_next_capture = True
                    speak("Next content")
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.warning(f"⚠️ Google API error: {e}")

        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            logger.error(f"❌ Voice error: {e}")
            time.sleep(1)

# ================= TEXT UTILS =================
def normalize_text(t: str) -> str:
    """Normalize text for comparison"""
    t = re.sub(r"[^a-z0-9\s]", "", t.lower())
    return re.sub(r"\s+", " ", t).strip()

def similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    if not a or not b:
        return 0.0
    sa, sb = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

# ================= READING WITH APPLE VISION =================
IGNORE_THRESHOLD = 0.96
PARTIAL_THRESHOLD = 0.80
LAST_READ = 0
_same_announced = False

def analyze_image(frame: np.ndarray, last_text: str) -> str:
    """Read all text in the full-resolution (unmirrored) frame and speak what's new"""
    global LAST_READ, _force_next_capture, _same_announced

    if time.time() - LAST_READ < READ_COOLDOWN:
        return last_text
    LAST_READ = time.time()

    try:
        text = read_text(frame)
    except Exception as e:
        logger.error(f"❌ Apple Vision error: {e}")
        speak("Could not read text. Try again.")
        return last_text

    if not text.strip():
        speak("No text found")
        return last_text

    logger.info(f"📖 Read {len(text)} characters")

    if _force_next_capture:
        _force_next_capture = False
        _same_announced = False
        speak(text)
        return text

    sim = similarity(text, last_text)
    logger.info(f"📊 Similarity: {sim:.2f}")

    if sim >= IGNORE_THRESHOLD:
        # Say it once per page, not every time the user keeps holding it
        if not _same_announced:
            speak("Same content")
            _same_announced = True
        return last_text

    _same_announced = False

    if PARTIAL_THRESHOLD <= sim < IGNORE_THRESHOLD:
        old_words = set(normalize_text(last_text).split())
        delta_words = [w for w in normalize_text(text).split() if w not in old_words]
        if len(delta_words) >= 5:
            delta = " ".join(delta_words)
            speak("New text")
            speak(delta)
            return last_text + " " + delta
        return last_text

    speak(text)
    return text

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
        self.running = True
        self.connected = False
        self.consecutive_failures = 0
        self.max_failures = 10

    def connect(self) -> bool:
        """Open the webcam"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"❌ Cannot open webcam {self.camera_index}. On macOS, allow Camera access for your terminal / VS Code.")
                self.close()
                return False

            # Landscape 720p matches the display and keeps detection fast
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                logger.info(f"✅ Camera opened at {w}x{h}")
                self.connected = True
                self.consecutive_failures = 0
                with self.buffer_lock:
                    self.buffer.append(frame)
                return True
            logger.warning("⚠️ Camera opened but no valid frame returned")
            self.close()
            return False
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
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
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    logger.warning(f"⚠️ Camera read failure {self.consecutive_failures}/{self.max_failures}")

                    if self.consecutive_failures >= self.max_failures:
                        logger.info("🔄 Reopening camera...")
                        self.close()
                        time.sleep(2)
                        if not self.connect():
                            time.sleep(2)

            except Exception as e:
                logger.error(f"❌ Camera error: {e}")
                self.consecutive_failures += 1
                time.sleep(1)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame from buffer"""
        with self.buffer_lock:
            if len(self.buffer) > 0:
                return True, self.buffer[-1]
        return False, None

    def close(self):
        """Close camera safely"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.connected = False
        except Exception as e:
            logger.error(f"⚠️ Error closing camera: {e}")

    def stop(self):
        """Stop the background reader"""
        self.running = False
        time.sleep(0.5)
        self.close()

# ================= MAIN =================
def main():
    """Main loop: find text live, guide the user, read it when held still"""
    global _force_next_capture, _voice_thread_running, _restart_capture

    # Open the webcam on the main thread first: macOS can only show the
    # camera-permission prompt from the main thread.
    frame_reader = BackgroundFrameReader(CAMERA_INDEX)
    if not frame_reader.connect():
        logger.error("❌ Failed to open webcam")
        return
    frame_reader.start()

    speak("System online. Show me something to read.")

    last_text = ""

    # Start voice listener
    voice_thread = threading.Thread(
        target=voice_listener,
        args=(lambda: last_text,),
        daemon=True
    )
    voice_thread.start()

    stable_start = 0
    is_stable = False
    HOLD_TIME = 0.5

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
        """Hints use mirrored coordinates, so left/right match the user's view"""
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
            restart_capture()
        return True

    # ================= MAIN LOOP =================
    logger.info("🎬 Starting main loop")
    window = "Smart Reader"
    no_frame_counter = 0

    try:
        while True:
            try:
                ret, frame = frame_reader.get_frame()

                if not ret or frame is None:
                    no_frame_counter += 1
                    if no_frame_counter > 30:
                        logger.warning("⚠️ No frames for 1 second, waiting for camera...")
                        time.sleep(1)
                        no_frame_counter = 0
                    else:
                        time.sleep(0.033)
                    continue
                no_frame_counter = 0

                if _restart_capture:
                    is_stable = False
                    stable_start = 0
                    _force_next_capture = True
                    _restart_capture = False
                    speak("Ready for next capture")

                # Detection runs on the unmirrored image (mirrored text can't be read);
                # the preview is mirrored like a selfie view.
                small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                display = cv2.flip(small, 1)

                if speaker.is_speaking:
                    draw_guide_box(display, GUIDE_COLOR_IDLE, "READING...")
                    cv2.imshow(window, display)
                    if not handle_key():
                        break
                    continue

                # ===== LOW LIGHT DETECTION =====
                if np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) < 50:
                    cv2.putText(display, "LOW LIGHT", (GUIDE_X1 + 10, GUIDE_Y2 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_light_warning > 5:
                        speak("Low light detected")
                        last_light_warning = time.time()

                # ===== FIND TEXT (Apple Vision, fast mode) =====
                try:
                    box, lines = find_text_region(small, MIN_TEXT_CHARS)
                except Exception as e:
                    logger.error(f"❌ Detection error: {e}")
                    box, lines = None, []

                if box is not None:
                    for line in lines:
                        lx1, ly1, lx2, ly2 = mirror_box(line.box, DISPLAY_W)
                        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), LINE_COLOR, 1)
                    box = mirror_box(box, DISPLAY_W)
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), LINE_COLOR, 2)

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
                            last_text = analyze_image(frame, last_text)
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
                logger.info("⏹️ Keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                time.sleep(0.5)
                continue

    finally:
        logger.info("🔴 Shutting down...")
        _voice_thread_running = False
        speaker.shutdown()
        try:
            frame_reader.stop()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # lets macOS actually close the window
        except Exception:
            pass
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
