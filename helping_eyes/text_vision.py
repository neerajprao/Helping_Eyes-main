"""
Text finding and reading, with RapidOCR (open-source, ONNX, runs on the CPU of any
computer: no platform-specific OCR, no GPU, no API key).

    find_text_region()    quick check of a live preview frame: is there text, and where?
    recognize_text()      every line of text in an image, optionally with each word's box
    read_text_enhanced()  an accurate read that copes with skew, shadows and glare
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels, top-left origin


@dataclass
class TextLine:
    text: str
    confidence: float
    box: Box
    # (start, end, box) of each word in `text`; filled when word_boxes=True
    words: List[Tuple[int, int, Box]] = field(default_factory=list)


_rapid = None


def _quad_to_box(quad, w: int, h: int) -> Box:
    pts = np.asarray(quad, dtype=float)
    return (max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min())),
            min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max())))


def recognize_text(image: np.ndarray, word_boxes: bool = False) -> List[TextLine]:
    """Find and read every line of text in a BGR image, top to bottom.
    word_boxes=True also records where each word is."""
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


def find_text_region(image: np.ndarray, min_chars: int = 8) -> Tuple[Optional[Box], List[TextLine]]:
    """
    Quick check for readable text. Returns (box around all text, lines),
    or (None, lines) when there are fewer than `min_chars` letters/digits.
    """
    lines = [l for l in recognize_text(image)
             if len(re.sub(r"[^0-9A-Za-z\u0080-￿]", "", l.text)) >= 2]
    chars = sum(len(re.sub(r"\s", "", l.text)) for l in lines)
    if chars < min_chars:
        return None, lines
    x1 = min(l.box[0] for l in lines)
    y1 = min(l.box[1] for l in lines)
    x2 = max(l.box[2] for l in lines)
    y2 = max(l.box[3] for l in lines)
    return (x1, y1, x2, y2), lines


def read_text_enhanced(image: np.ndarray, report=None) -> Tuple[str, str]:
    """
    Accurate read that copes with skew, shadows and glare: vision.py (part A) offers
    extra versions of the image (flattened page, shadow-corrected, binarised)
    only when the frame looks like it needs them, and the version the OCR is
    most confident about wins (sum of confidence x characters, so a version
    that recovers more text also scores higher). The untouched frame keeps
    the win unless another version is clearly better.
    Returns (text, name of the winning version).
    """
    from vision import capture_candidates
    best_name, best_lines, best_score = "original", [], -1.0
    for name, candidate in capture_candidates(image, report):
        lines = [l for l in recognize_text(candidate) if l.text.strip()]
        score = sum(l.confidence * len(l.text.strip()) for l in lines)
        if name == "original":
            best_lines, best_score = lines, score * 1.05      # small bias towards the untouched frame
        elif score > best_score:
            best_name, best_lines, best_score = name, lines, score
    return "\n".join(l.text for l in best_lines), best_name
