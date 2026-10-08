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


def test_numbers_above_9999_are_spoken_digit_by_digit():
    from commands import spoken_form
    assert spoken_form("Serial number 123456 ok") == "Serial number 1 2 3 4 5 6 ok"
    assert spoken_form("Batch AB12345X") == "Batch AB1 2 3 4 5X"
    assert spoken_form("Total 12,345,678 units") == "Total 1 2 3 4 5 6 7 8 units"
    assert spoken_form("Take 9999 or 500 or 1,000 mg") == "Take 9999 or 500 or 1,000 mg"    # up to 9999 stays a normal number
    assert spoken_form("Phone 9876543210.") == "Phone 9 8 7 6 5 4 3 2 1 0."


def test_synthesize_reports_a_failing_voice_service_and_caches_good_audio():
    import commands
    calls = []

    async def fake_speak(text):
        calls.append(text)
        if "fail" in text:
            raise ConnectionError("no network")
        return b"mp3-bytes"

    original = commands._speak
    commands._speak = fake_speak
    try:
        assert commands.synthesize("Serial 123456.") == b"mp3-bytes"
        assert calls == ["Serial 1 2 3 4 5 6."]                  # digits spelled out before the voice
        assert commands.synthesize("Serial 123456.") == b"mp3-bytes" and len(calls) == 1      # cached
        try:
            commands.synthesize("please fail")
            raise AssertionError("expected TtsError")
        except commands.TtsError:
            pass
        try:
            commands.synthesize("   ")
            raise AssertionError("expected TtsError")
        except commands.TtsError:
            pass
    finally:
        commands._speak = original


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
