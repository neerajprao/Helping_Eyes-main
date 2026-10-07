"""
Live camera logic, kept on the server so the browser only draws and speaks.

    LiveGuide   normal mode: is there text, is it in view, hold still, capture
                (spoken guidance: "Move closer", "Move Left", blur / glare coaching)
    BookWatcher book mode: page-turn detection on the live frames
    BookReader  book mode: read a page, decide whether it is a new page or the same
                page seen from a moved view, and say what to speak and from where

The browser sends small preview frames (the server never needs the full-size
picture until it is told to capture) and gets back one JSON-able dict per frame.
All boxes in replies are normalised to 0-1 so the page can draw them at any size.
"""

import logging
import re
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from book_mode import PageLayout, PageTurnDetector, continue_from, read_page
from page_enhance import ENABLED as ENHANCE, QualityReport, assess_quality
from text_vision import find_text_region

logger = logging.getLogger(__name__)

# ---- capture control (same values as the desktop app used) ----
MIN_TEXT_CHARS = 8         # need this many characters to count as text
MIN_TEXT_HEIGHT = 14       # px on a 720-line preview; smaller text -> "Move closer"
HOLD_TIME = 0.5            # seconds the text must stay steady before capturing
REARM_AFTER = 1.5          # seconds with no text in view before a new capture is allowed
QUALITY_PATIENCE = 3.0     # seconds to coach about blur / glare before capturing anyway
QUALITY_COOLDOWN = 5.0     # seconds between spoken image-quality hints
HINT_COOLDOWN = 2.0        # seconds before the same position hint is spoken again
GUIDE_MARGIN_X = 30 / 1280  # the guide box, as fractions of the frame
GUIDE_MARGIN_Y = 20 / 720
GUIDE = (GUIDE_MARGIN_X, GUIDE_MARGIN_Y, 1 - GUIDE_MARGIN_X, 1 - GUIDE_MARGIN_Y)

Box = Tuple[int, int, int, int]


def _norm(box: Box, w: int, h: int) -> List[float]:
    return [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]


# =====================================================================
# Normal mode
# =====================================================================
class LiveGuide:
    """
    Feed it preview frames with update(). It says what to draw and speak, and
    when to take the full-size photo ("capture": true). The page then calls
    capture_result(ok) so a successful capture is not repeated until the item
    leaves the view (or rearm() is called: "next", "r").
    """

    def __init__(self, detect: Callable = find_text_region, assess: Callable = assess_quality,
                 clock: Callable[[], float] = time.time):
        self._detect, self._assess, self._clock = detect, assess, clock
        self.reset()

    def reset(self) -> None:
        self.armed = True
        self.is_stable = False
        self.stable_start = 0.0
        self.last_text_seen = self._clock()
        self.last_hint = ""
        self.last_hint_time = 0.0
        self.last_quality_hint_time = 0.0
        self.quality = QualityReport()

    def rearm(self) -> None:
        self.armed = True
        self.is_stable = False

    def capture_result(self, ok: bool) -> None:
        if ok:
            self.armed = False
        self.is_stable = False

    # ---- helpers (ported from the desktop app) ----
    @staticmethod
    def _inside_guide(box: Box, w: int, h: int) -> float:
        gx1, gy1, gx2, gy2 = GUIDE[0] * w, GUIDE[1] * h, GUIDE[2] * w, GUIDE[3] * h
        x1, y1, x2, y2 = box
        ox1, oy1, ox2, oy2 = max(x1, gx1), max(y1, gy1), min(x2, gx2), min(y2, gy2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        area = (x2 - x1) * (y2 - y1)
        return (ox2 - ox1) * (oy2 - oy1) / area if area > 0 else 0.0

    @staticmethod
    def _guidance(box: Box, lines, w: int, h: int) -> str:
        """Spoken hint to bring the text fully into view ("" when it's fine)."""
        x1, y1, x2, y2 = box
        gx1, gy1, gx2, gy2 = GUIDE[0] * w, GUIDE[1] * h, GUIDE[2] * w, GUIDE[3] * h
        heights = sorted(l.box[3] - l.box[1] for l in lines)
        if heights and heights[len(heights) // 2] < MIN_TEXT_HEIGHT * h / 720:
            return "Move closer"
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        slack = 5 * h / 720
        near_x_edge = x1 <= gx1 + slack or x2 >= gx2 - slack
        near_y_edge = y1 <= gy1 + slack or y2 >= gy2 - slack
        far = 80 * h / 720
        if near_x_edge and abs(cx - gcx) > far:
            return "Move Left" if cx > gcx else "Move Right"
        if near_y_edge and abs(cy - gcy) > far:
            return "Move Up" if cy > gcy else "Move Down"
        return ""

    def _coach_quality(self, now: float) -> str:
        hint = self.quality.hint()
        if hint and now - self.last_quality_hint_time > QUALITY_COOLDOWN:
            self.last_quality_hint_time = now
            return hint
        return ""

    # ---- one frame ----
    def update(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        now = self._clock()
        try:
            box, lines = self._detect(frame, MIN_TEXT_CHARS)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            box, lines = None, []
        self.quality = self._assess(frame, box) if ENHANCE else QualityReport()
        if box is not None:
            self.last_text_seen = now

        # Text gone for a moment -> ready for the next item
        if not self.armed and now - self.last_text_seen > REARM_AFTER:
            self.armed = True
            logger.info("Ready for a new capture")

        reply = {"type": "guide", "state": "NO_TEXT", "label": "SHOW TEXT HERE", "speak": "", "capture": False,
                 "box": _norm(box, w, h) if box else None,
                 "lines": [_norm(l.box, w, h) for l in lines],
                 "quality": self.quality.summary(), "problems": list(self.quality.problems)}

        if not self.armed:
            # Already captured this item: stay quiet and wait for questions
            reply.update(state="CAPTURED", label="CAPTURED - ASK ME")
        elif box is not None:
            hint = self._guidance(box, lines, w, h)
            if self._inside_guide(box, w, h) >= 0.65 and not hint:
                if not self.is_stable:
                    self.is_stable = True
                    self.stable_start = now
                held = now - self.stable_start
                blocking = {"blurry", "glare"} & set(self.quality.problems)
                if held < HOLD_TIME:
                    reply.update(state="HOLD", label=f"HOLD {HOLD_TIME - held:.1f}s")
                elif blocking and held < HOLD_TIME + QUALITY_PATIENCE:
                    # Coach for a moment; if it doesn't improve, capture anyway
                    # (the enhancement layer then tries corrected versions of the frame)
                    reply.update(state="COACH", label=self.quality.hint().upper(),
                                 speak=self._coach_quality(now))
                else:
                    reply.update(state="CAPTURING", label="CAPTURING...", capture=True)
                    self.is_stable = False
            else:
                self.is_stable = False
                reply.update(state="ADJUST", label="ADJUST POSITION")
                if hint and (hint != self.last_hint or now - self.last_hint_time > HINT_COOLDOWN):
                    reply["speak"] = hint
                    self.last_hint, self.last_hint_time = hint, now
        else:
            self.is_stable = False
            if "dark" in self.quality.problems:
                reply["speak"] = self._coach_quality(now)
        return reply


# =====================================================================
# Book mode
# =====================================================================
class BookWatcher:
    """Page-turn detection on the live frames (book_mode.PageTurnDetector)."""

    def __init__(self):
        self.detector = PageTurnDetector()

    def reset(self) -> None:
        self.detector.reset()

    def update(self, frame: np.ndarray) -> dict:
        event = self.detector.update(frame)
        d = self.detector
        return {"type": "book", "event": event, "state": d.state,
                "motion": round(d.motion, 1), "still_below": round(d.motion_off, 1)}


def speech_chunks(text: str, paragraphs, from_pos: int = 0, max_len: int = 600):
    """
    (start, chunk) pieces of the page for tracked speech, beginning at
    character `from_pos`: one per paragraph, long paragraphs split at sentence
    ends, so each chunk is spoken as its own utterance.
    """
    for a, b in paragraphs:
        if b <= from_pos:
            continue
        start = max(a, from_pos)
        while start < b:
            end = b
            if end - start > max_len:
                cut = max(text.rfind(". ", start, start + max_len),
                          text.rfind("? ", start, start + max_len),
                          text.rfind("! ", start, start + max_len))
                if cut > start:
                    end = cut + 1                      # end of a sentence
                else:
                    space = text.rfind(" ", start, start + max_len)
                    end = space if space > start else start + max_len   # between words
            yield start, text[start:end]
            start = end


def layout_json(layout: PageLayout, w: int, h: int) -> dict:
    """The page layout as JSON, with boxes normalised to 0-1 of the frame."""
    return {
        "gutter": round(layout.gutter_x / w, 4) if layout.gutter_x else None,
        "columns": [_norm(b, w, h) for b in layout.columns],
        "blocks": [_norm(b, w, h) for b in layout.blocks],
        "margins": [_norm(b, w, h) for b in layout.margins],
        "lines": [{"start": s.start, "end": s.end, "box": _norm(s.box, w, h),
                   "words": [[a, b, _norm(box, w, h)] for a, b, box in s.words]} for s in layout.lines],
    }


class BookReader:
    """
    What to do with a steady book view. process() reads the page and returns an
    action for the browser:

        new        a different page: speak intro, then the chunks from the start
        resume     the same page seen from a moved view: speak the chunks from `from_pos`
        keep       same page, barely moved while reading: carry on, change nothing
        nothing_new  same page, nothing left to read
        no_text    no text in view
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.page_text = ""
        self.layout: Optional[PageLayout] = None
        self.last_running_title = ""

    def process(self, frame: np.ndarray, event: str, read_pos: int, reading: bool, thinking: bool, doc) -> dict:
        h, w = frame.shape[:2]
        try:
            text, layout = read_page(frame, enhance=True)
        except Exception as e:
            logger.error(f"Page reading error: {e}")
            text, layout = "", None

        # Same words as before (camera or book moved)? Carry on instead of starting over
        resume = continue_from(self.page_text, read_pos, text) if (self.page_text and text.strip()) else None
        # Different printed page numbers mean a different page, whatever the words say
        if resume is not None and layout and self.layout:
            old_nums, new_nums = set(self.layout.page_numbers()), set(layout.page_numbers())
            if old_nums and new_nums and not (old_nums & new_nums):
                resume = None

        if resume is not None and event == "moved" and reading:
            return {"action": "keep"}                       # a hand passed over: keep reading

        if resume is not None:
            self.page_text, self.layout = text, layout
            doc.update_document(text, layout.summary(), layout.label())
            if re.search(r"\w", text[resume:]) and not thinking:
                logger.info(f"Same words in view: continuing from \"{text[resume:resume + 40]}...\"")
                return {"action": "resume", "text": text, "from_pos": resume, "label": layout.label(),
                        "layout": layout_json(layout, w, h),
                        "chunks": [{"start": s, "text": t} for s, t in speech_chunks(text, layout.paragraphs, resume)]}
            logger.info("Same words in view: nothing new to read")
            return {"action": "nothing_new", "text": text, "label": layout.label(), "layout": layout_json(layout, w, h)}

        if text.strip():
            self.page_text, self.layout = text, layout
            doc.set_document(text, layout.summary(), layout.label())
            logger.info(f"{layout.label() or 'Page (no number found)'}, {len(text)} characters. {layout.summary()}")
            intro = []
            if layout.label():
                intro.append(layout.label() + ".")
            title = layout.running_title()
            if title and title.lower() != self.last_running_title.lower():
                intro.append(title + ".")
                self.last_running_title = title
            return {"action": "new", "text": text, "from_pos": 0, "label": layout.label(),
                    "intro": " ".join(intro), "layout": layout_json(layout, w, h),
                    "chunks": [{"start": s, "text": t} for s, t in speech_chunks(text, layout.paragraphs, 0)]}

        return {"action": "no_text"}
