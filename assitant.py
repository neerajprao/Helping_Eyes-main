import cv2
import numpy as np
import google.generativeai as genai
import time
import os
import sys
import torch
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
import re
from speech import Speaker

# ================= CONFIG =================
load_dotenv()
API_KEY = os.getenv("API_KEY")
PHONE_IP = os.getenv("PHONE_IP")
# Android "IP Webcam" app default; override with CAMERA_URL in .env
CAMERA_URL = os.getenv("CAMERA_URL") or f"http://{PHONE_IP}:8080/video"

if not API_KEY or not PHONE_IP:
    print("API_KEY or PHONE_IP missing in .env")
    sys.exit(1)

# ================= GEMINI =================
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# ================= YOLO =================
# Apple Silicon GPU (MPS) when available, else CPU. Override with YOLO_DEVICE.
YOLO_DEVICE = os.getenv("YOLO_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
finder_ai = YOLO("yolov8n.pt")
TARGET_CLASSES = [73]  # book

# ================= SPEECH =================
speaker = Speaker()

def speak(text):
    clean = text.replace("*", "").replace("#", "").replace("_", "")
    print(f"🗣️ {clean[:120]}")
    speaker.say(clean)

# ================= TEXT UTILITIES =================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def text_similarity(t1, t2):
    if not t1 or not t2:
        return 0.0
    w1 = set(normalize_text(t1).split())
    w2 = set(normalize_text(t2).split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def extract_new_text(new_text, old_text):
    new_words = normalize_text(new_text).split()
    old_words = set(normalize_text(old_text).split())
    unique_words = [w for w in new_words if w not in old_words]
    if len(unique_words) < 5:
        return ""
    return " ".join(unique_words)

# ================= ANALYSIS =================
IGNORE_THRESHOLD = 0.96
PARTIAL_THRESHOLD = 0.80

def analyze_image(frame, last_text):
    # HARD SPEECH GATE (absolute protection)
    speaker.wait_until_idle()

    speak("Reading")

    h, w = frame.shape[:2]
    if w > 1024:
        frame = cv2.resize(frame, (1024, int(h * (1024 / w))))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    try:
        prompt = (
            "Extract all visible text exactly as seen. "
            "Do not summarize or infer."
        )
        response = model.generate_content([prompt, pil_img])

        if not response.text or len(response.text.strip()) < 10:
            # DO NOT SPEAK HERE (silent failure)
            return last_text

        new_text = response.text.strip()
        sim = text_similarity(new_text, last_text)

        print(f"\nSimilarity score: {sim:.2f}")

        # -------- CASE 1: Almost identical --------
        if sim >= IGNORE_THRESHOLD:
            speak("Same content. No new text.")
            return last_text

        # -------- CASE 2: Partial overlap --------
        elif PARTIAL_THRESHOLD <= sim < IGNORE_THRESHOLD:
            delta = extract_new_text(new_text, last_text)
            if delta:
                speak("New text detected.")
                speak(delta)
                return last_text + " " + delta
            else:
                speak("No significant new text.")
                return last_text

        # -------- CASE 3: New content --------
        else:
            speak(new_text)
            return new_text

    except Exception as e:
        print("Gemini error:", e)
        # DO NOT SPEAK HERE
        return last_text

# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Camera not accessible at {CAMERA_URL}")
        return

    speak("System online. Show me a book.")

    last_guidance_time = 0
    stable_start = 0
    is_stable = False
    last_text = ""

    HOLD_TIME = 0.7
    TOLERANCE = 100
    MICRO_TOL = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)

        # Vision loop pauses while speaking
        if speaker.is_speaking:
            cv2.putText(display, "Reading...",
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)
            cv2.imshow("Smart Reader", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        results = finder_ai(display, conf=0.3, verbose=False, device=YOLO_DEVICE)
        found = False

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in TARGET_CLASSES:
                    found = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 3)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    fx = display.shape[1] // 2
                    fy = display.shape[0] // 2

                    offx = cx - fx
                    offy = cy - fy

                    area = (x2 - x1) * (y2 - y1)
                    coverage = area / (display.shape[0] * display.shape[1])

                    centered = True

                    if abs(offx) > TOLERANCE + MICRO_TOL:
                        centered = False
                        if time.time() - last_guidance_time > 2:
                            speak("Move Left" if offx > 0 else "Move Right")
                            last_guidance_time = time.time()

                    elif abs(offy) > TOLERANCE + MICRO_TOL:
                        centered = False
                        if time.time() - last_guidance_time > 2:
                            speak("Move Up" if offy > 0 else "Move Down")
                            last_guidance_time = time.time()

                    elif coverage < 0.15:
                        centered = False
                        if time.time() - last_guidance_time > 2:
                            speak("Bring Closer")
                            last_guidance_time = time.time()

                    if centered:
                        if not is_stable:
                            is_stable = True
                            stable_start = time.time()

                        held = time.time() - stable_start

                        if held < HOLD_TIME:
                            cv2.putText(display,
                                        f"HOLD STILL {HOLD_TIME-held:.1f}s",
                                        (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 255, 255), 3)
                        else:
                            cv2.putText(display, "CAPTURING",
                                        (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 255, 0), 3)

                            last_text = analyze_image(frame, last_text)
                            is_stable = False
                    else:
                        is_stable = False

                    break

        if not found:
            is_stable = False
            cv2.putText(display, "Searching...",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.imshow("Smart Reader", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    speaker.shutdown()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # lets macOS actually close the window

if __name__ == "__main__":
    main()
