from fakes import FakeWatcherHandle, final

from earpiece.brain.transcript import TranscriptStore
from earpiece.brain.watcher import Watcher
from earpiece.events import Action, Decision


def store_with(*texts: str) -> TranscriptStore:
    store = TranscriptStore(mission="m")
    for text in texts:
        store.add(final("THEM", text))
    return store


async def test_decision_passthrough():
    decision = Decision(action=Action.RESPOND, reason="question asked", urgency="normal")
    watcher = Watcher(FakeWatcherHandle([decision]), "m")
    result = await watcher.decide(store_with("What is X?"), partial_answer=None)
    assert result.action is Action.RESPOND


async def test_llm_error_fails_safe_to_silent():
    watcher = Watcher(FakeWatcherHandle([RuntimeError("boom")]), "m")
    result = await watcher.decide(store_with("hi"), partial_answer=None)
    assert result.action is Action.STAY_SILENT


async def test_interrupt_without_inflight_downgrades_to_respond():
    decision = Decision(action=Action.INTERRUPT_AND_RESPOND, reason="r", urgency="high")
    watcher = Watcher(FakeWatcherHandle([decision]), "m")
    result = await watcher.decide(store_with("hi"), partial_answer=None)
    assert result.action is Action.RESPOND


async def test_partial_answer_is_shown_to_watcher():
    decision = Decision(action=Action.INTERRUPT_AND_RESPOND, reason="topic moved", urgency="high")
    handle = FakeWatcherHandle([decision])
    watcher = Watcher(handle, "m")
    result = await watcher.decide(store_with("actually, pricing"), partial_answer="The rollout is")
    assert result.action is Action.INTERRUPT_AND_RESPOND
    prompt = handle.calls[0][-1]["content"]
    assert "The rollout is" in prompt
