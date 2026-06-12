"""Deepgram message parsing — segment stitching across long utterances.

Deepgram sends is_final=true segments mid-utterance whose text is never
re-sent; only speech_final marks the endpoint, carrying just the tail. The
parser must concatenate, or earlier segments silently disappear.
"""

import json

from earpiece.config import LLMSlot, Settings
from earpiece.stt.deepgram import DeepgramSTT


def make_engine() -> DeepgramSTT:
    slot = LLMSlot(base_url="x", api_key="x", model="x")
    settings = Settings(mission="m", responder=slot, watcher=slot, deepgram_api_key="dg")
    return DeepgramSTT(settings, "THEM")


def result(text: str, *, is_final: bool = False, speech_final: bool = False) -> str:
    return json.dumps(
        {
            "type": "Results",
            "is_final": is_final,
            "speech_final": speech_final,
            "duration": 1.0,
            "channel": {"alternatives": [{"transcript": text}]},
        }
    )


def test_long_utterance_segments_are_stitched():
    eng = make_engine()
    # live interim
    ev = eng._parse(result("okay so you want"))
    assert ev is not None and not ev.is_final and ev.text == "okay so you want"
    # segment finalized mid-utterance (this text is never re-sent by deepgram)
    ev = eng._parse(result("okay so you want me to name four countries", is_final=True))
    assert ev is not None and not ev.is_final
    # interim of the next segment includes the buffered text
    ev = eng._parse(result("in the west", is_final=False))
    assert ev.text == "okay so you want me to name four countries in the west"
    # endpoint: only the tail arrives, full utterance must come out
    ev = eng._parse(result("in the west of Europe", is_final=True, speech_final=True))
    assert ev.is_final
    assert ev.text == "okay so you want me to name four countries in the west of Europe"


def test_buffer_resets_between_utterances():
    eng = make_engine()
    eng._parse(result("first part", is_final=True))
    eng._parse(result("second part", is_final=True, speech_final=True))
    ev = eng._parse(result("new utterance", is_final=True, speech_final=True))
    assert ev.text == "new utterance"


def test_empty_speech_final_flushes_buffered_segments():
    eng = make_engine()
    eng._parse(result("trailing words", is_final=True))
    ev = eng._parse(result("", is_final=True, speech_final=True))
    assert ev is not None and ev.is_final and ev.text == "trailing words"


def test_empty_messages_are_ignored():
    eng = make_engine()
    assert eng._parse(result("")) is None
    assert eng._parse(result("", is_final=True)) is None
    assert eng._parse(result("", is_final=True, speech_final=True)) is None
    assert eng._parse(json.dumps({"type": "Metadata"})) is None
    assert eng._parse(b"\x00not json") is None


def test_utterance_end_flushes_when_speech_final_never_fires():
    eng = make_engine()
    eng._parse(result("the beginning of a long question", is_final=True))
    eng._parse(result("that deepgram never endpoints", is_final=True))
    ev = eng._parse(json.dumps({"type": "UtteranceEnd", "last_word_end": 7.1}))
    assert ev is not None and ev.is_final
    assert ev.text == "the beginning of a long question that deepgram never endpoints"
    # idempotent: a second UtteranceEnd (deepgram may send one per is_final) is a no-op
    assert eng._parse(json.dumps({"type": "UtteranceEnd", "last_word_end": 7.1})) is None


def test_utterance_end_falls_back_to_interim_only_words():
    eng = make_engine()
    eng._parse(result("words that never got finalized"))
    ev = eng._parse(json.dumps({"type": "UtteranceEnd", "last_word_end": 3.0}))
    assert ev is not None and ev.is_final
    assert ev.text == "words that never got finalized"


def test_retracted_interim_is_cleared_not_stuck():
    eng = make_engine()
    eng._parse(result("garbled noise words"))
    # deepgram finalizes that audio as silence — an empty final clears the grey line
    ev = eng._parse(result("", is_final=True))
    assert ev is not None and ev.is_final and ev.text == ""
