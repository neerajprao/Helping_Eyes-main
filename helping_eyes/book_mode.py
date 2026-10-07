"""
Book reading mode: the computer vision that decides WHEN to read a page,
WHERE the pages are, and in WHAT ORDER to read the text.

Apple Vision (text_vision.py) only recognises the letters of each line.
Everything else here is classic OpenCV / NumPy:

    PageTurnDetector   frame differencing + a small state machine
                       -> fires once per new page, after the page settles
    split_spread()     finds the book, then its spine: the text-free (smoothest)
                       vertical strip, or the spine's shadow -> splits a two-page spread
    split_page_parts() header / footer rows -> printed page number, running title
    xy_cut()           recursive XY-cut on the recognised lines
                       -> columns, headings and paragraphs in reading order
    read_page()        all of the above for one camera frame
    continue_from()    compares a new view with the page being read: same words
                       -> carry on from the word reached instead of starting over
"""

import re
import time
from collections import deque
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from page_enhance import assess_quality, prepare_page, unproject_box, unwarp_box, warp_page
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

    update() returns an event:
        ""         nothing to do
        "changed"  the view became STEADY and the page layout differs from the
                   last page read (a new page, or the first page)
        "moved"    the view became STEADY after movement but the layout looks
                   the same: a hand passed over, the book shifted slightly, or a
                   page of very similar layout was turned (dense book pages
                   look alike), so the caller should compare the words

    Thresholds adapt to the camera: sensor noise and small hand tremor give
    every camera a different "still" level, so the detector keeps a running
    estimate of that noise floor and measures motion relative to it.
    """

    def __init__(self, on_margin: float = 3.0, off_margin: float = 1.5,
                 settle_time: float = 0.8, change_min: float = 0.06, recheck_every: float = 1.0):
        self.on_margin = on_margin      # this far above the noise floor = movement
        self.off_margin = off_margin    # within this of the noise floor = still
        self.settle_time = settle_time  # seconds of stillness before reading
        self.change_min = change_min    # fraction of the page layout that must change
        self.recheck_every = recheck_every  # seconds between layout checks while steady
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
        self._last_check = 0.0

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

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        g = self._small_gray(frame)
        if self._prev is None:
            self._prev = g
            return ""
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
            return ""

        if self.state == "STEADY":
            # A slow change (camera drifting, page sliding) never looks like
            # motion, so re-check the page layout now and then
            if now - self._last_check >= self.recheck_every:
                self._last_check = now
                return "changed" if self._is_new_page(frame) else ""
            return ""

        if self.motion > self.motion_off:
            self._still_since = None     # between the thresholds: not still yet
            return ""

        if self._still_since is None:
            self._still_since = now
            self.state = "SETTLING"
        if now - self._still_since < self.settle_time:
            return ""

        self.state = "STEADY"
        self._last_check = now
        return "changed" if self._is_new_page(frame) else "moved"

    def _is_new_page(self, frame: np.ndarray) -> bool:
        """True (and remembered) when the page layout differs from the last page read."""
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
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Start/end (exclusive) of each run of True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def find_book(image: np.ndarray) -> Box:
    """
    The open book = the bright (paper) regions: Otsu threshold on a blurred
    grayscale image, a closing to fill in the text, then the bounding box of
    all large paper regions together (a spine shadow can cut the paper into
    two halves). Falls back to the whole frame when no paper is found.
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 0.05 * w * h]
    if big and sum(cv2.contourArea(c) for c in big) > 0.2 * w * h:
        x, y, bw, bh = cv2.boundingRect(np.vstack(big))
        return (x, y, x + bw, y + bh)
    return (0, 0, w, h)


def texture_profile(gray: np.ndarray, window: int = 15) -> np.ndarray:
    """
    How "busy" each x column is: the local standard deviation of brightness,
    averaged down the column. Text is busy even when blurred; blank margins
    and the spine are smooth.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    mean = cv2.blur(g, (window, window))
    mean_sq = cv2.blur(g * g, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    h = g.shape[0]
    profile = std[int(h * 0.1):int(h * 0.9)].mean(axis=0)
    k = max(3, g.shape[1] // 60)
    return np.convolve(profile, np.ones(k) / k, mode="same")


def split_spread(image: np.ndarray, min_depth: float = 0.12) -> Optional[int]:
    """
    Find the gutter (spine) of an open book and return its x position, or
    None for a single page.

    1. Find the book (the paper). A two-page spread is clearly wider than
       tall; a single page is not.
    2. The spine's shadow: a half-open book casts a dark valley in the
       column brightness profile. When it's clear, it's the most precise clue.
    3. The text-free strip: text never crosses the spine, so the two inner
       margins form a smooth vertical strip in the middle of the book. This
       works for a flat, fully open book whose spine casts no shadow. Of the
       smooth strips, the one nearest the middle of the book wins.
    """
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = find_book(image)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 1.1 * bh:
        return None                      # taller than wide: a single page
    gray = cv2.cvtColor(image[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
    lo, hi = int(bw * 0.3), int(bw * 0.7)

    # --- clue 1: the spine's shadow ---
    profile = gray.astype(np.float32)[int(bh * 0.1):int(bh * 0.9)].mean(axis=0)
    k = max(3, bw // 100)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    x = lo + int(np.argmin(profile[lo:hi]))
    left = np.median(profile[int(bw * 0.15):lo])
    right = np.median(profile[hi:int(bw * 0.85)])
    if profile[x] < (1.0 - min_depth) * min(left, right):
        return bx1 + x

    # --- clue 2: the smooth, text-free strip nearest the middle ---
    tex = texture_profile(gray)
    text_level = float(np.median(tex[int(bw * 0.1):int(bw * 0.9)]))
    quiet = tex[lo:hi] < 0.6 * text_level
    runs = [(a, b) for a, b in _runs(quiet) if b - a >= max(4, bw * 0.015)]
    if runs:
        middle = bw / 2 - lo
        a, b = min(runs, key=lambda r: 0 if r[0] <= middle <= r[1] else min(abs(r[0] - middle), abs(r[1] - middle)))
        return bx1 + lo + (a + b) // 2

    # --- a very wide book with no clear clue: split down the middle ---
    if bw > 1.3 * bh and bw > 0.5 * w:
        return bx1 + bw // 2
    return None


# =====================================================================
# 3. Page structure: header, footer, page number, running title
# =====================================================================
@dataclass
class PageInfo:
    """What a printed page says about itself."""
    number: Optional[str] = None          # printed page number, e.g. "47" or "xii"
    running_title: Optional[str] = None   # book or chapter title in the header/footer
    headings: List[str] = field(default_factory=list)  # larger-type headings in the text


_ROMAN = re.compile(r"(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.I)
_NUMBER_ONLY = re.compile(r"^[\s\-–—.()|]*(?:page\s+)?(\d{1,4})[\s\-–—.()|]*$", re.I)
_NUMBER_THEN_TITLE = re.compile(r"^(\d{1,4})\s+(\D.*)$")
_TITLE_THEN_NUMBER = re.compile(r"^(.*\D)\s+(\d{1,4})$")
_HEADING_WORDS = re.compile(r"^(chapter|part|book|section|prologue|epilogue|preface|introduction|contents|appendix)\b", re.I)


def parse_page_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a header/footer line into (page number, remaining title text).
    Handles "47", "- 47 -", "Page 47", "xii", "47  THE SILENT RIVER" and
    "CHAPTER THREE  48".
    """
    t = text.strip()
    m = _NUMBER_ONLY.match(t)
    if m:
        return m.group(1), None
    if _ROMAN.match(t.strip(" -–—.")) and len(t.strip(" -–—.")) <= 7:
        return t.strip(" -–—.").lower(), None
    m = _NUMBER_THEN_TITLE.match(t)
    if m:
        return m.group(1), m.group(2).strip() or None
    m = _TITLE_THEN_NUMBER.match(t)
    if m:
        return m.group(2), m.group(1).strip() or None
    return None, t or None


def _rows(lines: List[TextLine]) -> List[List[TextLine]]:
    """Group lines that sit side by side (overlapping in y) into rows, top to bottom."""
    rows: List[List[TextLine]] = []
    for line in sorted(lines, key=lambda l: l.box[1]):
        cy = (line.box[1] + line.box[3]) / 2
        if rows:
            r = rows[-1]
            top, bottom = min(l.box[1] for l in r), max(l.box[3] for l in r)
            if top <= cy <= bottom:
                r.append(line)
                continue
        rows.append([line])
    for r in rows:
        r.sort(key=lambda l: l.box[0])
    return rows


def _median(values, default: float = 0.0) -> float:
    return float(np.median(values)) if len(values) else default


def split_page_parts(lines: List[TextLine]) -> Tuple[List[TextLine], List[TextLine], PageInfo]:
    """
    Separate the page's header and footer from its body.

    A header (footer) is a row at the very top (bottom) of the page that is set
    apart from the body by a gap clearly larger than the usual line spacing and
    is either short or contains a page number. Its text gives the printed page
    number and the running title. Returns (body lines, header/footer lines, info).
    """
    info = PageInfo()
    rows = _rows(lines)
    if len(rows) < 3:
        return lines, [], info

    heights = [l.box[3] - l.box[1] for l in lines]
    line_h = _median(heights, 1.0)
    gaps = [max(0, min(l.box[1] for l in b) - max(l.box[3] for l in a)) for a, b in zip(rows, rows[1:])]
    usual_gap = _median(gaps, line_h * 0.5)
    widths = [max(l.box[2] for l in r) - min(l.box[0] for l in r) for r in rows]
    usual_width = _median(widths, 1.0)

    def is_margin_row(row, gap):
        text = " ".join(l.text for l in row)
        number, _ = parse_page_number(text)
        width = max(l.box[2] for l in row) - min(l.box[0] for l in row)
        set_apart = gap > max(1.8 * usual_gap, 0.9 * line_h)
        return set_apart and (number is not None or width < 0.6 * usual_width)

    margin_rows = []
    if is_margin_row(rows[0], gaps[0]):
        margin_rows.append(rows[0])
    if is_margin_row(rows[-1], gaps[-1]):
        margin_rows.append(rows[-1])

    titles = []
    for row in margin_rows:
        for line in row:
            number, title = parse_page_number(line.text)
            if number and info.number is None:
                info.number = number
            if title and not number:
                titles.append(title)
            elif title:
                titles.append(title)
    info.running_title = " ".join(titles).strip() or None

    margin = {id(l) for r in margin_rows for l in r}
    body = [l for l in lines if id(l) not in margin]
    info.headings = [l.text.strip() for l in body
                     if (l.box[3] - l.box[1]) >= 1.35 * line_h or _HEADING_WORDS.match(l.text.strip())]
    return body, [l for l in lines if id(l) in margin], info


# =====================================================================
# 4. Reading order: recursive XY-cut on the recognised lines
# =====================================================================
def xy_cut(lines: List[TextLine], columns: Optional[List[Box]] = None) -> List[List[TextLine]]:
    """
    Recursive XY-cut, a classic document layout algorithm, run on the boxes of
    the lines Apple Vision recognised (reliable whenever the text is readable):

      1. Vertical cut: project the line boxes onto the x axis. An empty band
         between them that is wider than a column gap splits the region into
         a left and a right part (columns), read left first.
      2. Otherwise, horizontal cut: project onto the y axis. A gap clearly
         larger than the usual line spacing splits the region into an upper
         and a lower part (heading / paragraphs), read top first.
      3. Repeat on each part. A part that can't be cut is read line by line
         from the top; an indented line starts a new paragraph.

    Returns the paragraphs in reading order. Column boxes found by vertical
    cuts are appended to `columns` if given.
    """
    if not lines:
        return []
    line_h = _median([l.box[3] - l.box[1] for l in lines], 1.0)
    rows = _rows(lines)
    usual_gap = _median([max(0, min(l.box[1] for l in b) - max(l.box[3] for l in a))
                         for a, b in zip(rows, rows[1:])], line_h * 0.5)

    def gaps_along(spans):
        """Empty intervals between merged [a, b) spans."""
        spans = sorted(spans)
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]

    def cut(group: List[TextLine]) -> List[List[TextLine]]:
        if len(group) <= 1:
            return [group] if group else []
        # 1. columns
        x_gaps = [g for g in gaps_along([(l.box[0], l.box[2]) for l in group]) if g[1] - g[0] >= 1.2 * line_h]
        if x_gaps:
            a, b = max(x_gaps, key=lambda g: g[1] - g[0])
            split = (a + b) / 2
            left = [l for l in group if (l.box[0] + l.box[2]) / 2 < split]
            right = [l for l in group if (l.box[0] + l.box[2]) / 2 >= split]
            if left and right:
                if columns is not None:
                    for side in (left, right):
                        columns.append((min(l.box[0] for l in side), min(l.box[1] for l in side),
                                        max(l.box[2] for l in side), max(l.box[3] for l in side)))
                return cut(left) + cut(right)
        # 2. headings / paragraphs
        y_gaps = [g for g in gaps_along([(l.box[1], l.box[3]) for l in group])
                  if g[1] - g[0] >= max(1.8 * usual_gap, usual_gap + 0.6 * line_h)]
        if y_gaps:
            a, b = max(y_gaps, key=lambda g: g[1] - g[0])
            split = (a + b) / 2
            top = [l for l in group if (l.box[1] + l.box[3]) / 2 < split]
            bottom = [l for l in group if (l.box[1] + l.box[3]) / 2 >= split]
            if top and bottom:
                return cut(top) + cut(bottom)
        # 3. no gap left: split paragraphs at indented first lines (novel style)
        ordered = [l for r in _rows(group) for l in r]
        left_edge = _median([l.box[0] for l in ordered])
        paragraphs = [[ordered[0]]]
        for line in ordered[1:]:
            if line.box[0] - left_edge > max(0.6 * line_h, 6):
                paragraphs.append([line])
            else:
                paragraphs[-1].append(line)
        return paragraphs

    return cut(lines)


# =====================================================================
# 5. Whole frame
# =====================================================================
@dataclass
class LineSpan:
    """Where one printed line ended up in the page text, and where it is on screen."""
    start: int      # character offsets into the page text
    end: int
    box: Box        # full-frame pixel coordinates
    words: List[Tuple[int, int, Box]] = field(default_factory=list)  # same, per word


@dataclass
class PageLayout:
    """Everything found in one frame, in full-frame pixel coordinates (for drawing)."""
    gutter_x: Optional[int] = None
    columns: List[Box] = field(default_factory=list)
    blocks: List[Box] = field(default_factory=list)   # paragraphs, in reading order
    margins: List[Box] = field(default_factory=list)  # headers / footers (not read aloud)
    lines: List[LineSpan] = field(default_factory=list)
    paragraphs: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) in the page text
    pages: List[PageInfo] = field(default_factory=list)              # left to right

    def line_at(self, pos: int) -> Optional[LineSpan]:
        """The printed line that contains character `pos` of the page text."""
        for span in self.lines:
            if span.start <= pos < span.end:
                return span
        return None

    def word_at(self, pos: int) -> Optional[Box]:
        """Screen box of the word at character `pos` (or the next word on that line)."""
        span = self.line_at(pos)
        if span is None:
            return None
        for start, end, box in span.words:
            if pos < end:
                return box
        return None

    def page_numbers(self) -> List[str]:
        """Printed page numbers, left to right. A missing number on one side of
        a spread is inferred from the facing page (left = right - 1)."""
        nums = [p.number for p in self.pages]
        if len(nums) == 2:
            left, right = nums
            if left is None and right and right.isdigit() and int(right) > 1:
                nums[0] = str(int(right) - 1)
            elif right is None and left and left.isdigit():
                nums[1] = str(int(left) + 1)
        return [n for n in nums if n]

    def label(self) -> str:
        """'Page 47', 'Pages 46 and 47', or '' when no number is printed."""
        nums = self.page_numbers()
        if len(nums) == 2:
            return f"Pages {nums[0]} and {nums[1]}"
        return f"Page {nums[0]}" if nums else ""

    def running_title(self) -> Optional[str]:
        titles = [p.running_title for p in self.pages if p.running_title]
        return titles[0] if titles else None

    def summary(self) -> str:
        """Page facts for the language model, so it can answer 'what page is this?'."""
        parts = []
        if self.label():
            parts.append(self.label() + ".")
        titles = [p.running_title for p in self.pages if p.running_title]
        if titles:
            parts.append("Running header: " + "; ".join(dict.fromkeys(titles)) + ".")
        headings = [h for p in self.pages for h in p.headings]
        if headings:
            parts.append("Headings: " + "; ".join(headings) + ".")
        return " ".join(parts)


def _shift(line: TextLine, dx: int) -> TextLine:
    """The same line with its boxes moved dx pixels to the right."""
    x1, y1, x2, y2 = line.box
    words = [(a, b, (bx1 + dx, by1, bx2 + dx, by2)) for a, b, (bx1, by1, bx2, by2) in line.words]
    return TextLine(line.text, line.confidence, (x1 + dx, y1, x2 + dx, y2), words)


def _unwarp(line: TextLine, model) -> TextLine:
    """Boxes found on a dewarped page image, mapped back onto the camera frame."""
    if model is None:
        return line
    return TextLine(line.text, line.confidence, unwarp_box(line.box, model),
                    [(a, b, unwarp_box(box, model)) for a, b, box in line.words])


def _read_flat_page(frame: np.ndarray, split: bool, enhance: bool) -> Tuple[str, PageLayout]:
    """read_page() on one image: split the spread, OCR each page, order the text (see read_page)."""
    h, w = frame.shape[:2]
    layout = PageLayout(gutter_x=split_spread(frame) if split else None)
    pages = [(0, layout.gutter_x), (layout.gutter_x, w)] if layout.gutter_x else [(0, w)]

    report = assess_quality(frame) if enhance else None
    text = ""
    for x_start, x_end in pages:
        image, model = prepare_page(frame[:, x_start:x_end], report) if enhance else (frame[:, x_start:x_end], None)
        lines = [_shift(_unwarp(l, model), x_start) for l in recognize_text(image, fast=False, word_boxes=True)
                 if l.text.strip()]
        body, margin_lines, info = split_page_parts(lines)
        layout.pages.append(info)
        layout.margins += [l.box for l in margin_lines]

        for paragraph in xy_cut(body, layout.columns):
            layout.blocks.append((min(l.box[0] for l in paragraph), min(l.box[1] for l in paragraph),
                                  max(l.box[2] for l in paragraph), max(l.box[3] for l in paragraph)))
            para_start = None
            for line in paragraph:
                t = line.text.strip()
                if para_start is None:
                    if text:
                        text += "\n\n"
                    para_start = len(text)
                elif text.endswith("-") and t[:1].islower():
                    text = text[:-1]          # "exam-" + "ple" -> "example"
                else:
                    text += " "
                start = len(text)
                text += t
                lead = len(line.text) - len(line.text.lstrip())   # strip() shifted the offsets
                words = [(start + a - lead, start + b - lead, box) for a, b, box in line.words]
                layout.lines.append(LineSpan(start, len(text), line.box, words))
            layout.paragraphs.append((para_start, len(text)))

    return text, layout


def _unproject_layout(layout: PageLayout, matrix: np.ndarray, flat_height: int) -> None:
    """Map a layout found on the flattened page back onto the camera frame (in place)."""
    back = lambda box: unproject_box(box, matrix)
    if layout.gutter_x:
        point = cv2.perspectiveTransform(np.array([[[layout.gutter_x, flat_height / 2]]], np.float32), np.linalg.inv(matrix))
        layout.gutter_x = int(point[0, 0, 0])
    layout.columns = [back(b) for b in layout.columns]
    layout.blocks = [back(b) for b in layout.blocks]
    layout.margins = [back(b) for b in layout.margins]
    layout.lines = [LineSpan(l.start, l.end, back(l.box), [(a, b, back(box)) for a, b, box in l.words])
                    for l in layout.lines]


def read_page(frame: np.ndarray, split: bool = True, enhance: bool = False) -> Tuple[str, PageLayout]:
    """
    Split the spread, read each page with Apple Vision, separate headers and
    footers (page number, running title), and order the body text with XY-cut.
    Paragraphs are separated by a blank line; lines within a paragraph are
    joined with spaces (re-joining words hyphenated across lines). The layout
    remembers which characters came from which printed line and word, so the
    app can show exactly where it is reading. split=False treats the image as
    one page (e.g. a single label photographed in the web version).

    enhance=True first prepares the image (page_enhance.py): a page photographed at
    an angle is flattened with a homography, curved lines (a page bending into the
    spine) are straightened, and dark / shadowed / glary frames are tone-corrected.
    Boxes are mapped back, so the layout is still in camera coordinates.
    """
    if enhance:
        flat, matrix = warp_page(frame)
        if matrix is not None:
            text, layout = _read_flat_page(flat, split, enhance)
            if text.strip():
                _unproject_layout(layout, matrix, flat.shape[0])
                return text, layout
            # nothing read from the flattened page: fall back to the frame as the camera saw it
    return _read_flat_page(frame, split, enhance)


# =====================================================================
# 6. Same words after the camera or book moved: carry on reading
# =====================================================================
def _words(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, normalised word) for each word, ignoring punctuation-only tokens."""
    words = []
    for m in re.finditer(r"\S+", text):
        norm = re.sub(r"[^\w]", "", m.group().lower())
        if norm:
            words.append((m.start(), m.end(), norm))
    return words


def continue_from(old_text: str, old_pos: int, new_text: str,
                  min_run: int = 3, min_overlap: float = 0.4) -> Optional[int]:
    """
    Compare the page being read with a new view of it.

    old_pos is how far reading had got in old_text (a character offset; use
    len(old_text) when the page was read to the end).

    Returns the character offset in new_text to carry on reading from:
    the word reading had reached if it is still in view, otherwise the first
    word after the part already read (len(new_text) when nothing new is in
    view). Returns None when the views don't share enough words, i.e. it's a
    different page.

    The words are aligned with difflib's SequenceMatcher. Only matching runs of
    at least `min_run` words count, so common words ("the", "and") on a
    different page don't look like an overlap.
    """
    old_words, new_words = _words(old_text), _words(new_text)
    if not old_words or not new_words:
        return None
    matcher = SequenceMatcher(None, [w[2] for w in old_words], [w[2] for w in new_words], autojunk=False)
    runs = [b for b in matcher.get_matching_blocks() if b.size >= min_run]
    matched = sum(b.size for b in runs)
    if matched < min_overlap * min(len(old_words), len(new_words)):
        return None                              # different page

    # First old word not finished yet (the one being spoken, or the next one)
    reached = next((i for i, w in enumerate(old_words) if w[1] > old_pos), len(old_words))

    for b in runs:                               # that word is still in view
        if b.a <= reached < b.a + b.size:
            return new_words[b.b + reached - b.a][0]

    earlier = [b for b in runs if b.a + b.size <= reached]
    if earlier:                                  # continue after what was already read
        j = earlier[-1].b + earlier[-1].size
        return new_words[j][0] if j < len(new_words) else len(new_text)

    return new_words[runs[0].b][0]               # reading hadn't reached the visible part yet
