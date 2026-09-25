"""
Live YOLOv8n object detection on a USB webcam — all 80 COCO objects.

Run from the project root:
    source .venv/bin/activate
    python dummy/live_detect.py                     # USB camera (default)
    python dummy/live_detect.py --camera builtin    # laptop's FaceTime camera
    python dummy/live_detect.py --camera AN-VC550   # pick a camera by name
    python dummy/live_detect.py --camera 0          # pick by OpenCV index
    python dummy/live_detect.py --conf 0.4          # stricter confidence

Keys (click the video window first):
    q / Esc  quit
    + / -    raise / lower the confidence threshold
    m        toggle mirror (selfie) view
    s        save a screenshot into dummy/
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "yolov8n.pt"  # downloaded here automatically on first run
WINDOW = "YOLOv8n Live - 80 objects"
PANEL_W = 260


BUILT_IN_NAMES = ("facetime", "built-in", "integrated")


def list_cameras() -> list:
    """
    Camera names in OpenCV's index order (macOS only; [] elsewhere).
    OpenCV's AVFoundation backend numbers cameras by sorting their unique IDs,
    so sorting system_profiler's list the same way gives index -> name.
    """
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=10).stdout
        cams = json.loads(out).get("SPCameraDataType", [])
    except Exception:
        return []
    cams.sort(key=lambda c: c.get("spcamera_unique-id", ""))
    return [c.get("_name", "?") for c in cams]


def resolve_camera(choice: str) -> int:
    """
    "usb"      -> first camera that isn't the laptop's built-in one (the default)
    "0", "1"   -> that OpenCV index
    other text -> first camera whose name contains it, e.g. "AN-VC550"
    """
    if choice.isdigit():
        return int(choice)

    names = list_cameras()
    for i, name in enumerate(names):
        print(f"  camera {i}: {name}")

    if choice.lower() == "usb":
        for i, name in enumerate(names):
            if not any(b in name.lower() for b in BUILT_IN_NAMES):
                return i
        print("No USB camera found, falling back to camera 0")
        return 0

    for i, name in enumerate(names):
        if choice.lower() in name.lower():
            return i
    print(f"No camera named '{choice}', falling back to camera 0")
    return 0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def draw_panel(frame, counts: Counter, fps: float, conf: float, device: str):
    """Right-hand panel: FPS, settings and a live list of detected objects."""
    h = frame.shape[0]
    panel = frame[:, -PANEL_W:]
    panel[:] = (panel * 0.35).astype(panel.dtype)  # darken for readability

    x = frame.shape[1] - PANEL_W + 12
    lines = [
        (f"FPS: {fps:5.1f}", (0, 255, 0)),
        (f"Device: {device}", (200, 200, 200)),
        (f"Confidence >= {conf:.2f}", (200, 200, 200)),
        (f"Objects: {sum(counts.values())}", (0, 255, 255)),
        ("", None),
    ]
    for name, n in counts.most_common():
        lines.append((f"{n} x {name}", (255, 255, 255)))
    if not counts:
        lines.append(("(nothing detected)", (150, 150, 150)))

    y = 30
    for text, color in lines:
        if y > h - 50:
            cv2.putText(frame, "...", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            break
        if text:
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 26

    cv2.putText(frame, "q quit  +/- conf  m mirror  s save", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Live YOLOv8n detection of all 80 COCO objects")
    parser.add_argument("--camera", default="usb",
                        help='"usb" (default), "builtin", a camera name, or an index')
    parser.add_argument("--conf", type=float, default=0.35, help="minimum confidence (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size in pixels")
    args = parser.parse_args()

    device = pick_device()
    model = YOLO(str(MODEL_PATH))
    print(f"Loaded {MODEL_PATH.name}: {len(model.names)} classes, running on {device}")

    camera = resolve_camera("facetime" if args.camera.lower() == "builtin" else args.camera)
    print(f"Using camera {camera}")

    # Open the camera on the main thread (required for the macOS permission prompt)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open webcam {camera}. On macOS, allow Camera access for your "
              "terminal / VS Code in System Settings > Privacy & Security > Camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    conf = args.conf
    mirror = True
    fps = 0.0
    last = time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            # No `classes=` filter -> all 80 COCO objects are detected
            result = model.predict(frame, conf=conf, imgsz=args.imgsz,
                                   device=device, verbose=False)[0]

            annotated = result.plot(line_width=2)  # boxes + "label confidence"
            counts = Counter(model.names[int(c)] for c in result.boxes.cls)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
            last = now

            draw_panel(annotated, counts, fps, conf, device)
            cv2.imshow(WINDOW, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            elif key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("s"):
                path = HERE / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), annotated)
                print(f"Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # lets macOS actually close the window


if __name__ == "__main__":
    main()
