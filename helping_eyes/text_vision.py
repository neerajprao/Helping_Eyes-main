"""
Text finding and reading.

Two OCR engines with the same interface:
    apple     Apple's Vision framework (macOS only, on-device, default on a Mac)
                fast=True  -> ~15 ms per 720p frame; live "is there text, where?"
                fast=False -> ~65 ms; accurate reading once the user holds still
    rapidocr  RapidOCR (open-source, ONNX, CPU; used by the cloud web version
              and on Linux). Has no separate fast mode; ~0.35 s per image.

Optional .env settings:
    OCR_ENGINE     = "apple" or "rapidocr" (default: apple on macOS, else rapidocr)
    TEXT_LANGUAGES = comma-separated languages for Apple Vision, e.g. "en-US,hi-IN".
                     Unset = Vision detects the language automatically.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels, top-left origin

OCR_ENGINE = os.getenv("OCR_ENGINE") or ("apple" if sys.platform == "darwin" else "rapidocr")
_LANGUAGES = [l.strip() for l in os.getenv("TEXT_LANGUAGES", "").split(",") if l.strip()]


@dataclass
class TextLine:
    text: str
    confidence: float
    box: Box
    # (start, end, box) of each word in `text`; filled when word_boxes=True
    words: List[Tuple[int, int, Box]] = field(default_factory=list)


# =====================================================================
# Apple Vision (macOS)
# =====================================================================
if OCR_ENGINE == "apple":
    if sys.platform != "darwin":
        raise RuntimeError("OCR_ENGINE=apple needs macOS; use OCR_ENGINE=rapidocr on other systems.")
    import objc
    import Quartz
    import Vision
    from Foundation import NSData

    def _to_cgimage(image: np.ndarray):
        """Hand raw pixels to Vision as a CGImage: lossless and faster than
        encoding to JPEG (JPEG artefacts caused misreads on small print)."""
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        h, w = rgba.shape[:2]
        data = NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        return Quartz.CGImageCreate(
            w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
            Quartz.kCGRenderingIntentDefault,
        )

    def _to_pixels(bb, w: int, h: int) -> Box:
        """Vision's normalised, bottom-left-origin rectangle -> pixel box, top-left origin."""
        x1 = int(bb.origin.x * w)
        x2 = int((bb.origin.x + bb.size.width) * w)
        y1 = int((1.0 - bb.origin.y - bb.size.height) * h)
        y2 = int((1.0 - bb.origin.y) * h)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    def _recognize_apple(image: np.ndarray, fast: bool, word_boxes: bool) -> List[TextLine]:
        h, w = image.shape[:2]

        # Vision objects are autoreleased; drain them every call so a live loop
        # doesn't slowly grow memory.
        with objc.autorelease_pool():
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_to_cgimage(image), None)

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
                text = str(best.string())
                line = TextLine(text, float(best.confidence()), _to_pixels(obs.boundingBox(), w, h))
                if word_boxes and not fast:
                    for m in re.finditer(r"\S+", text):
                        rect, _ = best.boundingBoxForRange_error_((m.start(), m.end() - m.start()), None)
                        if rect is not None:
                            line.words.append((m.start(), m.end(), _to_pixels(rect.boundingBox(), w, h)))
                lines.append(line)
            return lines


# =====================================================================
# RapidOCR (any platform, CPU)
# =====================================================================
_rapid = None


def _quad_to_box(quad, w: int, h: int) -> Box:
    pts = np.asarray(quad, dtype=float)
    return (max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min())),
            min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max())))


def _recognize_rapid(image: np.ndarray, word_boxes: bool) -> List[TextLine]:
    global _rapid
    if _rapid is None:
        from rapidocr import RapidOCR   # loaded on first use (model load takes a moment)
        _rapid = RapidOCR()
    h, w = image.shape[:2]
    result = _rapid(image, return_word_box=word_boxes)
    if result.boxes is None:
        return []
    word_results = getattr(result, "word_results", None) or [()] * len(result.txts)

    lines = []
    for quad, text, score, words in zip(result.boxes, result.txts, result.scores, word_results):
        text = str(text)
        line = TextLine(text, float(score), _quad_to_box(quad, w, h))
        if word_boxes:
            # Locate each recognised word in the line text to get its offsets
            pos = 0
            for word, _, wquad in words or ():
                word = str(word)
                i = text.find(word, pos) if word.strip() else -1
                if i < 0:
                    continue
                line.words.append((i, i + len(word), _quad_to_box(wquad, w, h)))
                pos = i + len(word)
        lines.append(line)
    lines.sort(key=lambda l: (l.box[1], l.box[0]))
    return lines


# =====================================================================
# Public interface
# =====================================================================
def recognize_text(image: np.ndarray, fast: bool = False, word_boxes: bool = False) -> List[TextLine]:
    """
    Find and read every line of text in a BGR image.
    word_boxes=True also records where each word is (accurate mode only).
    """
    if OCR_ENGINE == "apple":
        return _recognize_apple(image, fast, word_boxes)
    return _recognize_rapid(image, word_boxes and not fast)


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


def read_text_enhanced(image: np.ndarray, report=None) -> Tuple[str, str]:
    """
    Accurate read that copes with skew, shadows and glare: vision.py (part A) offers
    extra versions of the image (flattened page, shadow-corrected, binarised)
    only when the frame looks like it needs them, and the version Vision is
    most confident about wins (sum of confidence x characters, so a version
    that recovers more text also scores higher). The untouched frame keeps
    the win unless another version is clearly better.
    Returns (text, name of the winning version).
    """
    from vision import capture_candidates
    best_name, best_lines, best_score = "original", [], -1.0
    for name, candidate in capture_candidates(image, report):
        lines = [l for l in recognize_text(candidate, fast=False) if l.text.strip()]
        score = sum(l.confidence * len(l.text.strip()) for l in lines)
        if name == "original":
            best_lines, best_score = lines, score * 1.05      # small bias towards the untouched frame
        elif score > best_score:
            best_name, best_lines, best_score = name, lines, score
    return "\n".join(l.text for l in best_lines), best_name
