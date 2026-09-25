"""
Text finding and reading with Apple's Vision framework (macOS only).

Vision both locates text (a box per line) and reads it, fully on-device.
    fast=True  -> ~15 ms per 720p frame; good for live "is there text, where?"
    fast=False -> ~65 ms; accurate reading once the user holds still

Optional .env setting:
    TEXT_LANGUAGES = comma-separated languages, e.g. "en-US,hi-IN".
                     Unset = Vision detects the language automatically.
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

if sys.platform != "darwin":
    raise RuntimeError("text_vision.py uses Apple Vision and only runs on macOS.")

import objc
import Vision
from Foundation import NSData

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels, top-left origin

_LANGUAGES = [l.strip() for l in os.getenv("TEXT_LANGUAGES", "").split(",") if l.strip()]


@dataclass
class TextLine:
    text: str
    confidence: float
    box: Box


def recognize_text(image: np.ndarray, fast: bool = False) -> List[TextLine]:
    """Find and read every line of text in a BGR image, in reading order."""
    h, w = image.shape[:2]
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return []

    # Vision objects are autoreleased; drain them every call so a live loop
    # doesn't slowly grow memory.
    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelFast if fast
            else Vision.VNRequestTextRecognitionLevelAccurate
        )
        request.setUsesLanguageCorrection_(not fast)
        if _LANGUAGES:
            request.setRecognitionLanguages_(_LANGUAGES)
        elif hasattr(request, "setAutomaticallyDetectsLanguage_"):
            request.setAutomaticallyDetectsLanguage_(True)

        success, _ = handler.performRequests_error_([request], None)
        if not success:
            return []

        lines = []
        for obs in request.results() or []:
            candidates = obs.topCandidates_(1)
            if not candidates:
                continue
            best = candidates[0]
            bb = obs.boundingBox()  # normalised 0..1, origin bottom-left
            x1 = int(bb.origin.x * w)
            x2 = int((bb.origin.x + bb.size.width) * w)
            y1 = int((1.0 - bb.origin.y - bb.size.height) * h)
            y2 = int((1.0 - bb.origin.y) * h)
            lines.append(TextLine(str(best.string()), float(best.confidence()),
                                  (max(0, x1), max(0, y1), min(w, x2), min(h, y2))))
        return lines


def find_text_region(image: np.ndarray, min_chars: int = 8) -> Tuple[Optional[Box], List[TextLine]]:
    """
    Fast check for readable text. Returns (box around all text, lines),
    or (None, lines) when there are fewer than `min_chars` letters/digits.
    """
    lines = [l for l in recognize_text(image, fast=True)
             if len(re.sub(r"[^0-9A-Za-z\u0080-￿]", "", l.text)) >= 2]
    chars = sum(len(re.sub(r"\s", "", l.text)) for l in lines)
    if chars < min_chars:
        return None, lines
    x1 = min(l.box[0] for l in lines)
    y1 = min(l.box[1] for l in lines)
    x2 = max(l.box[2] for l in lines)
    y2 = max(l.box[3] for l in lines)
    return (x1, y1, x2, y2), lines


def read_text(image: np.ndarray) -> str:
    """Accurate read of all text, one line per detected line."""
    return "\n".join(l.text for l in recognize_text(image, fast=False) if l.text.strip())


def scale_box(box: Box, sx: float, sy: float) -> Box:
    return (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))


def mirror_box(box: Box, width: int) -> Box:
    """Flip a box horizontally, to draw it on a mirrored (selfie) preview."""
    return (width - box[2], box[1], width - box[0], box[3])
