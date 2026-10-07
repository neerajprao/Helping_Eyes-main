"""
Tests for commands.py (no camera, OCR or model needed):

    python tests/test_commands.py        (or: python -m pytest tests/test_commands.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

from commands import classify


def name(text, pending=False):
    return classify(text, offer_pending=pending).name


def test_simple_commands():
    assert name("stop") == "stop"
    assert name("Be quiet.") == "stop"
    assert name("repeat that") == "repeat"
    assert name("next page") == "new_capture"
    assert name("scan again") == "new_capture"
    assert name("Book mode") == "book_on"
    assert name("read my book") == "book_on"
    assert name("stop book mode") == "book_off"
    assert name("normal mode") == "book_off"
    assert name("") == "empty"


def test_read_all_is_not_a_model_question():
    for text in ("read everything", "Read it all", "what does it say", "please read it out loud"):
        assert name(text) == "read_all", text


def test_questions_go_to_the_model():
    cmd = classify("When does it expire?")
    assert cmd.name == "question" and cmd.query == "When does it expire?"
    assert name("read only the dosage") == "question"


def test_yes_and_no_only_answer_an_offer():
    assert name("yes", pending=True) == "yes"
    assert name("no thanks", pending=True) == "no"
    assert name("yes") != "yes"            # with no offer open, "yes" is not special
    assert name("no") == "question"
    # anything else drops the offer and is handled as a new request
    assert name("stop", pending=True) == "stop"
    assert name("what is the dose", pending=True) == "question"


def test_web_search_phrases():
    assert name("look it up") == "search_last"
    assert name("search online") == "search_last"
    cmd = classify("Look up tartrazine allergy")
    assert (cmd.name, cmd.query) == ("search_for", "tartrazine allergy")
    cmd = classify("search the web for ibuprofen dose")
    assert (cmd.name, cmd.query) == ("search_for", "ibuprofen dose")


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
