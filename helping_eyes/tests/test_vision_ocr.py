"""
Tests for vision.py part A (reading text) with the real Apple Vision on rendered text
(no camera, key or language model needed):

    python tests/test_vision_ocr.py        (or: python -m pytest tests/test_vision_ocr.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import cv2
import numpy as np

import vision

LINES = ["PARACETAMOL 500 mg TABLETS", "Take two tablets every six hours", "Do not exceed eight tablets daily"]


def render(lines=LINES, size=(720, 1280)):
    """A white picture with black text, one line per entry, top to bottom."""
    image = np.full((*size, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (100, 170 + i * 120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4, cv2.LINE_AA)
    return image


def test_recognize_text_reads_every_line_top_to_bottom():
    lines = vision.recognize_text(render())
    text = " ".join(l.text for l in lines).upper()
    for word in ("PARACETAMOL", "TABLETS", "EVERY", "EXCEED", "DAILY"):
        assert word in text, (word, text)
    tops = [l.box[1] for l in lines]
    assert tops == sorted(tops) and len(lines) >= 3
    assert text.index("PARACETAMOL") < text.index("EVERY") < text.index("EXCEED")


def test_boxes_are_inside_the_picture_and_around_the_text():
    image = render()
    h, w = image.shape[:2]
    for line in vision.recognize_text(image):
        x1, y1, x2, y2 = line.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert 0.5 <= line.confidence <= 1.0
        assert y2 - y1 > 20 and x2 - x1 > 100                 # a real line of big print, not a speck


def test_word_boxes_only_when_asked_for_and_they_match_the_text():
    plain = vision.recognize_text(render())
    assert all(l.words == [] for l in plain)
    lines = vision.recognize_text(render(), word_boxes=True)
    words = [(l, w) for l in lines for w in l.words]
    assert words, "no word boxes were returned"
    for line, (start, end, box) in words:
        assert 0 <= start < end <= len(line.text) and line.text[start:end].strip()
        assert line.box[0] - 6 <= box[0] and box[2] <= line.box[2] + 6      # a word lies within its line
    # words of a line are in reading order
    for line in lines:
        starts = [w[0] for w in line.words]
        assert starts == sorted(starts)


def test_find_text_region_on_a_blank_picture():
    box, lines = vision.find_text_region(np.full((720, 1280, 3), 255, np.uint8))
    assert box is None and lines == []


def test_find_text_region_boxes_all_the_text():
    box, lines = vision.find_text_region(render())
    assert box is not None and len(lines) >= 3
    for line in lines:
        assert box[0] <= line.box[0] and box[1] <= line.box[1] and line.box[2] <= box[2] and line.box[3] <= box[3]


def test_find_text_region_needs_enough_characters():
    image = render(["OK"])
    assert vision.find_text_region(image, min_chars=2)[0] is not None
    assert vision.find_text_region(image, min_chars=50)[0] is None


def test_read_text_enhanced_keeps_a_clean_frame_as_it_is():
    text, method = vision.read_text_enhanced(render())
    assert method == "original"
    assert "PARACETAMOL" in text.upper() and "DAILY" in text.upper()
    assert text.index("PARACETAMOL") < text.index("Take") < text.index("Do not")        # lines in order


def test_read_text_enhanced_on_a_blank_frame_returns_nothing():
    text, method = vision.read_text_enhanced(np.full((480, 640, 3), 255, np.uint8))
    assert text == "" and method in ("original", "flattened", "tone", "flattened+tone", "binarized")


def slanted_page(angle):
    """Two paragraphs of three lines each, on a page turned by `angle` degrees."""
    texts = ["The quick brown fox jumps over the lazy dog", "Pack my box with five dozen liquor jugs",
             "Sphinx of black quartz judge my vow", "How vexingly quick daft zebras jump",
             "The five boxing wizards jump quickly", "Jackdaws love my big sphinx of quartz"]
    image = np.full((900, 1300, 3), 255, np.uint8)
    for i, t in enumerate(texts):
        cv2.putText(image, t, (260, 250 + i * 60 + (30 if i >= 3 else 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return cv2.warpAffine(image, cv2.getRotationMatrix2D((650, 450), angle, 1.0), (1300, 900), borderValue=(255, 255, 255))


def test_the_slant_of_a_page_is_measured():
    for angle in (-12, -5, 0, 8, 15):
        found = vision.estimate_skew(slanted_page(angle))
        assert abs(found + angle) <= 1.0, (angle, found)           # rotating by -angle makes the lines level


def test_a_slanted_page_keeps_its_paragraphs_and_outlines_follow_the_text():
    for angle in (-10, 12):
        text, layout = vision.read_page(slanted_page(angle), split=False, enhance=True)
        assert len(layout.blocks) == 2 and len(layout.lines) == 6, (angle, len(layout.blocks), len(layout.lines))
        assert text.index("quick brown") < text.index("Pack my") < text.index("Sphinx") < text.index("How vexingly")
        assert len(layout.block_quads) == 2
        for line in layout.lines:
            tl, tr, br, bl = line.quad
            height = float(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
            slope = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
            assert 20 <= height <= 60, (angle, height)             # one line tall, not the height of the slanted bounds
            assert abs(slope + angle) <= 2.5, (angle, slope)           # the outline leans the way the text does


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
