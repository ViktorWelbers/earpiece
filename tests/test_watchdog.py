"""Interim watchdog + transcript liveness — nothing may stay grey forever."""

import time

from fakes import final

from earpiece.brain.transcript import TranscriptStore
from earpiece.config import LLMSlot, Settings
from earpiece.events import TranscriptEvent
from earpiece.orchestrator import Orchestrator


def interim(speaker: str, text: str) -> TranscriptEvent:
    return TranscriptEvent(source=speaker, text=text, is_final=False, started_at=0, ended_at=0)


def make_orch() -> Orchestrator:
    slot = LLMSlot(base_url="http://localhost:1/v1", api_key="k", model="m")
    settings = Settings(
        mission="m",
        responder=slot,
        watcher=slot,
        stt_engine="whisper",
        stt_base_url="http://localhost:1/v1",
        stt_model="w",
    )
    return Orchestrator(settings)


async def test_stale_interim_is_promoted_to_final():
    orch = make_orch()
    orch.transcript.add(interim("THEM", "words stuck in grey"))
    orch.transcript.interim_updated["THEM"] = time.monotonic() - 10
    events = orch._stale_interim_events()
    assert len(events) == 1
    assert events[0].is_final and events[0].source == "THEM"
    assert events[0].text == "words stuck in grey"
    # not promoted twice while the synthetic final is in flight
    assert orch._stale_interim_events() == []


async def test_fresh_interim_is_left_alone():
    orch = make_orch()
    orch.transcript.add(interim("ME", "still being spoken"))
    assert orch._stale_interim_events() == []


async def test_promoted_final_lands_in_transcript_and_clears_interim():
    orch = make_orch()
    orch.transcript.add(interim("THEM", "words stuck in grey"))
    orch.transcript.interim_updated["THEM"] = time.monotonic() - 10
    [event] = orch._stale_interim_events()
    utt = orch.transcript.add(event)
    assert utt is not None and utt.text == "words stuck in grey"
    assert "THEM" not in orch.transcript.interim


# ---- TranscriptStore liveness primitives ----------------------------------


def test_empty_final_retracts_interim_without_utterance():
    store = TranscriptStore(mission="m")
    store.add(interim("ME", "garbled"))
    assert store.interim["ME"] == "garbled"
    utt = store.add(TranscriptEvent("ME", "", is_final=True, started_at=0, ended_at=1))
    assert utt is None
    assert "ME" not in store.interim
    assert store.utterances == []


def test_duplicate_final_within_window_is_suppressed():
    store = TranscriptStore(mission="m")
    assert store.add(final("THEM", "same words")) is not None
    assert store.add(final("THEM", "same words")) is None  # watchdog/engine race
    assert store.add(final("ME", "same words")) is not None  # other speaker unaffected
    assert len(store.utterances) == 2


def test_interim_updates_are_timestamped():
    store = TranscriptStore(mission="m")
    store.add(interim("ME", "one"))
    first = store.interim_updated["ME"]
    store.add(interim("ME", "one"))  # unchanged text must not bump the clock
    assert store.interim_updated["ME"] == first
    store.add(interim("ME", "one two"))
    assert store.interim_updated["ME"] >= first
