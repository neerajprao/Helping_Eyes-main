"""
Tests for vision.py part D (live guidance): the guidance state machine and the book-page decisions, with
the OCR replaced by fakes (no camera, OCR or model needed):

    python tests/test_vision_live.py        (or: python -m pytest tests/test_vision_live.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import numpy as np

import vision
from vision import PageInfo, PageLayout, QualityReport, TextLine
from assistant import DocAssistant

W, H = 1280, 720
FRAME = np.zeros((H, W, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def guide_with(box_lines, problems=(), clock=None):
    """A LiveGuide whose detector always 'sees' box_lines = (box, [line boxes]) or nothing."""
    clock = clock or Clock()
    state = {"seen": box_lines, "problems": list(problems)}

    def detect(frame, min_chars):
        if state["seen"] is None:
            return None, []
        box, line_boxes = state["seen"]
        return box, [TextLine("text", 0.9, b) for b in line_boxes]

    def assess(frame, box=None):
        return QualityReport(problems=state["problems"])

    return vision.LiveGuide(detect=detect, assess=assess, clock=clock), clock, state


CENTRED = ((300, 150, 980, 560), [(300, 150 + i * 40, 980, 175 + i * 40) for i in range(8)])


def test_no_text():
    guide, _, _ = guide_with(None)
    reply = guide.update(FRAME)
    assert reply["state"] == "NO_TEXT" and not reply["capture"] and reply["box"] is None


def test_hold_still_then_capture_once():
    guide, clock, _ = guide_with(CENTRED)
    first = guide.update(FRAME)
    assert first["state"] == "HOLD" and not first["capture"] and first["label"].startswith("HOLD")
    clock.advance(0.3)
    assert guide.update(FRAME)["state"] == "HOLD"
    clock.advance(0.3)
    reply = guide.update(FRAME)
    assert reply["state"] == "CAPTURING" and reply["capture"]
    # the hold restarts, so a second capture is not requested straight away
    assert not guide.update(FRAME)["capture"]
    guide.capture_result(True)
    assert guide.update(FRAME)["state"] == "CAPTURED"


def test_new_item_needs_the_old_one_to_leave_the_view():
    guide, clock, state = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    assert guide.update(FRAME)["capture"]
    guide.capture_result(True)
    clock.advance(1)
    assert guide.update(FRAME)["state"] == "CAPTURED"
    state["seen"] = None
    clock.advance(2)                                    # nothing in view for longer than REARM_AFTER
    assert guide.update(FRAME)["state"] == "NO_TEXT"
    state["seen"] = CENTRED
    assert guide.update(FRAME)["state"] == "HOLD"


def test_failed_capture_is_retried():
    guide, clock, _ = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    assert guide.update(FRAME)["capture"]
    guide.capture_result(False)
    assert guide.update(FRAME)["state"] == "HOLD"       # still armed


def test_rearm_on_request():
    guide, clock, _ = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    guide.update(FRAME)
    guide.capture_result(True)
    assert guide.update(FRAME)["state"] == "CAPTURED"
    guide.rearm()
    assert guide.update(FRAME)["state"] == "HOLD"


def test_small_text_says_move_closer_without_repeating():
    tiny = ((300, 150, 600, 200), [(300, 150 + i * 10, 600, 158 + i * 10) for i in range(4)])
    guide, clock, _ = guide_with(tiny)
    assert guide.update(FRAME)["speak"] == "Move closer"
    clock.advance(0.5)
    assert guide.update(FRAME)["speak"] == ""           # same hint, inside the cooldown
    clock.advance(2)
    assert guide.update(FRAME)["speak"] == "Move closer"


def test_text_cut_off_at_an_edge_says_which_way_to_move():
    at_right_edge = ((900, 150, 1270, 560), [(900, 150 + i * 40, 1270, 175 + i * 40) for i in range(8)])
    guide, _, _ = guide_with(at_right_edge)
    assert guide.update(FRAME)["speak"] == "Move Left"
    at_left_edge = ((10, 150, 400, 560), [(10, 150 + i * 40, 400, 175 + i * 40) for i in range(8)])
    guide, _, _ = guide_with(at_left_edge)
    assert guide.update(FRAME)["speak"] == "Move Right"


def test_blur_is_coached_then_captured_anyway():
    guide, clock, _ = guide_with(CENTRED, problems=["blurry"])
    guide.update(FRAME)
    clock.advance(0.6)
    reply = guide.update(FRAME)
    assert reply["state"] == "COACH" and reply["speak"] == "Hold still, the image is blurry" and not reply["capture"]
    clock.advance(1)
    assert guide.update(FRAME)["speak"] == ""           # not repeated inside the cooldown
    clock.advance(vision.QUALITY_PATIENCE)
    assert guide.update(FRAME)["capture"]


def test_dark_room_is_announced_when_no_text_is_in_view():
    guide, _, _ = guide_with(None, problems=["dark"])
    assert guide.update(FRAME)["speak"] == "Low light"


# ---------------------------------------------------------------- book pages
PAGE_A = "The river was quiet that morning. Nobody came down to the water."
PAGE_B = "Chapter two began with rain. The streets filled slowly and the lamps came on."


def fake_page(text, number=None):
    layout = PageLayout(paragraphs=[(0, len(text))])
    if number:
        layout.pages.append(PageInfo(number=number))
    return text, layout


def reader_showing(pages):
    """A BookReader that 'reads' each queued page in turn."""
    queue = list(pages)
    original = vision.read_page
    vision.read_page = lambda frame, **kwargs: queue.pop(0)
    return vision.BookReader(), original


def test_book_new_page_then_same_page_then_next_page():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A), fake_page(PAGE_B)])
    doc = DocAssistant()
    try:
        first = reader.process(FRAME, "changed", 0, False, False, doc)
        assert first["action"] == "new" and first["from_pos"] == 0 and doc.document == PAGE_A
        assert first["chunks"][0]["start"] == 0 and first["chunks"][0]["text"] == PAGE_A
        # the same words again after a hand passed over, while reading: carry on silently
        again = reader.process(FRAME, "moved", 20, True, False, doc)
        assert again["action"] == "keep"
        # a different page
        nxt = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert nxt["action"] == "new" and doc.document == PAGE_B
    finally:
        vision.read_page = original


def test_book_view_moved_continues_from_the_word_reached():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A)])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        pos = PAGE_A.index("Nobody")
        resume = reader.process(FRAME, "changed", pos, False, False, doc)
        assert resume["action"] == "resume" and resume["from_pos"] == pos
        assert resume["chunks"][0]["text"].startswith("Nobody")
    finally:
        vision.read_page = original


def test_book_page_fully_read_has_nothing_new():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A)])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        done = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert done["action"] == "nothing_new"
    finally:
        vision.read_page = original


def test_book_different_printed_numbers_mean_a_new_page_even_with_the_same_words():
    reader, original = reader_showing([fake_page(PAGE_A, "12"), fake_page(PAGE_A, "13")])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        second = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert second["action"] == "new" and second["label"] == "Page 13"
    finally:
        vision.read_page = original


def test_book_page_number_and_title_are_introduced():
    reader, original = reader_showing([fake_page(PAGE_A, "47")])
    doc = DocAssistant()
    try:
        first = reader.process(FRAME, "changed", 0, False, False, doc)
        assert first["intro"] == "Page 47."
        assert doc.page_label == "Page 47"
    finally:
        vision.read_page = original


def test_book_no_text():
    reader, original = reader_showing([("", PageLayout())])
    try:
        assert reader.process(FRAME, "changed", 0, False, False, DocAssistant())["action"] == "no_text"
    finally:
        vision.read_page = original


def test_speech_chunks_split_long_paragraphs_at_sentence_ends():
    text = ("Sentence one is here. " * 40).strip()
    chunks = list(vision.speech_chunks(text, [(0, len(text))], 0, max_len=100))
    assert len(chunks) > 5
    assert "".join(t for _, t in chunks).replace(" ", "") == text.replace(" ", "")
    assert all(t.rstrip().endswith(".") for _, t in chunks)
    # starting part-way: nothing before the start is spoken
    assert list(vision.speech_chunks(text, [(0, len(text))], 30, max_len=1000))[0][0] == 30


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for test_name, fn in tests:
        try:
            fn()
            print(f"PASS  {test_name}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"FAIL  {test_name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
