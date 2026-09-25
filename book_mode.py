"""
Book reading mode: the computer vision that decides WHEN to read a page,
WHERE the pages are, and in WHAT ORDER to read the text.

Apple Vision (text_vision.py) only recognises the letters of each line.
Everything else here is classic OpenCV / NumPy:

    PageTurnDetector   frame differencing + a small state machine
                       -> fires once per new page, after the page settles
    split_spread()     column brightness profile -> finds the book's gutter
                       (the dark shadow of the spine) and splits a two-page spread
    find_blocks()      adaptive threshold + projection profiles
                       -> text columns, then paragraphs, in reading order
    order_lines()      puts Apple Vision's lines into that reading order
    read_page()        all of the above for one camera frame
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from text_vision import TextLine, recognize_text

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


# =====================================================================
# 1. Page-turn detection (frame differencing)
# =====================================================================
class PageTurnDetector:
    """
    Watches the camera for a page turn.

        TURNING   something is moving (a hand, a page in the air)
        SETTLING  motion stopped; waiting for the page to stay still
        STEADY    page has been still for `settle_time`

    update() returns True once per NEW page: when the view becomes STEADY
    and the page looks different from the last page that was read, so a hand
    passing over the same page doesn't trigger a re-read.

    Thresholds adapt to the camera: sensor noise and small hand tremor give
    every camera a different "still" level, so the detector keeps a running
    estimate of that noise floor and measures motion relative to it.
    """

    def __init__(self, on_margin: float = 3.0, off_margin: float = 1.5,
                 settle_time: float = 0.8, change_min: float = 0.06):
        self.on_margin = on_margin      # this far above the noise floor = movement
        self.off_margin = off_margin    # within this of the noise floor = still
        self.settle_time = settle_time  # seconds of stillness before reading
        self.change_min = change_min    # fraction of the page layout that must change
        self.reset()

    @property
    def motion_on(self) -> float:
        return self.noise + self.on_margin

    @property
    def motion_off(self) -> float:
        return self.noise + self.off_margin

    def reset(self) -> None:
        """Start over: the first steady page will be read."""
        self.state = "TURNING"
        self.motion = 0.0
        self.noise = 2.0                # running estimate of the camera's still-scene motion
        self._prev: Optional[np.ndarray] = None
        self._recent: deque = deque(maxlen=5)   # last few raw motion values
        self._still_since: Optional[float] = None
        self._last_signature: Optional[np.ndarray] = None

    @staticmethod
    def _small_gray(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0)

    @staticmethod
    def signature(frame: np.ndarray) -> np.ndarray:
        """
        A small binary 'fingerprint' of the page layout: text lines appear as
        dark bands. Adaptive threshold makes it robust to lighting changes.
        """
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 10)
        # Smear horizontally so a line of text becomes one band; tiny shifts of
        # the book then change the signature much less than a new page does
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        return cv2.resize(b, (80, 45), interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        g = self._small_gray(frame)
        if self._prev is None:
            self._prev = g
            return False
        # Median of the last 5 frame differences: one noisy frame can't
        # restart the "hold still" timer, but real movement shows immediately
        self._recent.append(float(np.mean(cv2.absdiff(g, self._prev))))
        self._prev = g
        self.motion = float(np.median(self._recent))

        # Learn the noise floor from quiet frames (slow moving average)
        if self.motion < self.motion_off:
            self.noise = 0.95 * self.noise + 0.05 * self.motion

        if self.motion > self.motion_on:
            self.state = "TURNING"
            self._still_since = None
            return False

        if self.state == "STEADY":
            return False

        if self.motion > self.motion_off:
            self._still_since = None     # between the thresholds: not still yet
            return False

        if self._still_since is None:
            self._still_since = now
            self.state = "SETTLING"
        if now - self._still_since < self.settle_time:
            return False

        self.state = "STEADY"
        sig = self.signature(frame)
        if self._last_signature is not None:
            changed = float(np.mean(cv2.absdiff(sig, self._last_signature))) / 255.0
            if changed < self.change_min:
                return False             # same page as before
        self._last_signature = sig
        return True


# =====================================================================
# 2. Two-page spread splitting (gutter detection)
# =====================================================================
def split_spread(image: np.ndarray, min_depth: float = 0.12) -> Optional[int]:
    """
    Find the gutter of an open book: the darkest vertical valley in the
    middle of the image. Returns its x position, or None for a single page.

    Column-wise mean brightness is smoothed; the valley must be at least
    `min_depth` (12%) darker than the pages on both sides. The white gap
    between two text columns is BRIGHTER than the text, so it isn't mistaken
    for a gutter.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    profile = gray[int(h * 0.1):int(h * 0.9)].mean(axis=0)
    k = max(3, w // 100)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")

    lo, hi = int(w * 0.35), int(w * 0.65)
    x = lo + int(np.argmin(profile[lo:hi]))
    valley = profile[x]
    left = np.median(profile[int(w * 0.15):lo])
    right = np.median(profile[hi:int(w * 0.85)])
    if valley < (1.0 - min_depth) * min(left, right):
        return x
    return None


# =====================================================================
# 3. Layout analysis (columns and paragraphs)
# =====================================================================
@dataclass
class Block:
    box: Box
    column: int
    paragraph: int


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Start/end (exclusive) of each run of True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def ink_mask(page: np.ndarray) -> np.ndarray:
    """
    Binary image with text pixels = 255: adaptive threshold, speckles removed,
    and long straight lines (page edges, rules, table borders) removed so they
    don't fill in the gaps between columns and paragraphs.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block = max(15, (min(h, w) // 40) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # A morphological opening with a long thin kernel keeps only straight lines
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 4), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 4))))
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def find_columns(ink: np.ndarray) -> List[Tuple[int, int]]:
    """
    Vertical projection profile: count ink pixels in each x column. A gap
    between text columns is a wide run of (nearly) empty x positions; word
    gaps don't line up from line to line, so they never look empty.
    """
    h, w = ink.shape
    profile = (ink > 0).sum(axis=0).astype(np.float32)
    k = max(3, w // 200)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    has_ink = profile > max(2.0, 0.01 * h)

    min_gap = max(8, int(w * 0.025))
    spans = _runs(has_ink)
    merged: List[List[int]] = []
    for x1, x2 in spans:
        if merged and x1 - merged[-1][1] < min_gap:
            merged[-1][1] = x2       # small gap (between words): same column
        else:
            merged.append([x1, x2])
    return [(a, b) for a, b in merged if b - a >= w * 0.08]


def find_paragraphs(ink: np.ndarray, x1: int, x2: int) -> List[Box]:
    """
    Horizontal projection profile inside one column: runs of rows with ink are
    text lines; a gap clearly taller than the usual line gap starts a new paragraph.
    """
    col = ink[:, x1:x2]
    rows = (col > 0).sum(axis=1)
    lines = [(a, b) for a, b in _runs(rows > max(2, 0.01 * (x2 - x1))) if b - a >= 3]
    if not lines:
        return []

    gaps = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    heights = [b - a for a, b in lines]
    typical_gap = float(np.median(gaps)) if gaps else 0.0
    break_gap = max(typical_gap * 1.8, typical_gap + 0.8 * float(np.median(heights)))

    paragraphs, start = [], lines[0][0]
    for i, gap in enumerate(gaps):
        if gap > break_gap:
            paragraphs.append((start, lines[i][1]))
            start = lines[i + 1][0]
    paragraphs.append((start, lines[-1][1]))

    boxes = []
    for y1, y2 in paragraphs:
        xs = np.flatnonzero((col[y1:y2] > 0).any(axis=0))
        if len(xs):
            boxes.append((x1 + int(xs[0]), int(y1), x1 + int(xs[-1]) + 1, int(y2)))
    return boxes


def find_blocks(page: np.ndarray, ink: Optional[np.ndarray] = None) -> List[Block]:
    """Columns left to right, paragraphs top to bottom: the reading order."""
    ink = ink_mask(page) if ink is None else ink
    blocks = []
    for c, (x1, x2) in enumerate(find_columns(ink)):
        for p, box in enumerate(find_paragraphs(ink, x1, x2)):
            blocks.append(Block(box, c, p))
    return blocks


# =====================================================================
# 4. Reading order for Apple Vision's lines
# =====================================================================
def _join_lines(lines: List[TextLine]) -> str:
    """Join lines into flowing text, re-joining words hyphenated across lines."""
    text = ""
    for line in lines:
        t = line.text.strip()
        if not t:
            continue
        if text.endswith("-") and t[:1].islower():
            text = text[:-1] + t
        else:
            text = f"{text} {t}" if text else t
    return text


def order_lines(lines: List[TextLine], blocks: List[Block]) -> List[str]:
    """
    Assign each recognised line to the block containing its centre (or the
    nearest block), then read blocks in order and lines top to bottom.
    Returns one string per paragraph; stray lines (headers, page numbers)
    come last.
    """
    if not blocks:
        return [_join_lines(sorted(lines, key=lambda l: (l.box[1], l.box[0])))] if lines else []

    groups: List[List[TextLine]] = [[] for _ in blocks]
    strays: List[TextLine] = []
    for line in lines:
        cx = (line.box[0] + line.box[2]) / 2
        cy = (line.box[1] + line.box[3]) / 2
        best, best_d = None, float("inf")
        for i, b in enumerate(blocks):
            x1, y1, x2, y2 = b.box
            dx = max(x1 - cx, 0, cx - x2)
            dy = max(y1 - cy, 0, cy - y2)
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best, best_d = i, d
        line_h = max(1, line.box[3] - line.box[1])
        if best is not None and best_d <= 1.5 * line_h:
            groups[best].append(line)
        else:
            strays.append(line)

    paragraphs = [_join_lines(sorted(g, key=lambda l: (l.box[1], l.box[0]))) for g in groups if g]
    if strays:
        paragraphs.append(_join_lines(sorted(strays, key=lambda l: (l.box[1], l.box[0]))))
    return [p for p in paragraphs if p]


# =====================================================================
# 5. Whole frame
# =====================================================================
@dataclass
class PageLayout:
    """Everything found in one frame, in full-frame pixel coordinates (for drawing)."""
    gutter_x: Optional[int] = None
    columns: List[Box] = field(default_factory=list)
    blocks: List[Box] = field(default_factory=list)   # in reading order


def read_page(frame: np.ndarray) -> Tuple[str, PageLayout]:
    """Split the spread, lay out each page, read it with Apple Vision, order the text."""
    h, w = frame.shape[:2]
    layout = PageLayout(gutter_x=split_spread(frame))
    pages = [(0, layout.gutter_x), (layout.gutter_x, w)] if layout.gutter_x else [(0, w)]

    paragraphs: List[str] = []
    for x_start, x_end in pages:
        page = frame[:, x_start:x_end]
        ink = ink_mask(page)
        blocks = find_blocks(page, ink)
        for x1, x2 in find_columns(ink):
            layout.columns.append((x_start + x1, 0, x_start + x2, h))
        for b in blocks:
            x1, y1, x2, y2 = b.box
            layout.blocks.append((x_start + x1, y1, x_start + x2, y2))
        paragraphs += order_lines(recognize_text(page, fast=False), blocks)

    return "\n\n".join(paragraphs), layout
