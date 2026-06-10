import asyncio

from fakes import FakeLLMHandle, FakeTTSSink, final

from earpiece.brain.responder import Responder
from earpiece.brain.transcript import TranscriptStore


def make_responder(answers: list[str], sink: FakeTTSSink | None = None, delta_delay: float = 0.01):
    store = TranscriptStore(mission="m")
    store.add(final("THEM", "What does the integration timeline look like?"))
    llm = FakeLLMHandle(answers, delta_delay=delta_delay)
    deltas: list[str] = []
    ends: list[tuple[str, bool]] = []
    responder = Responder(
        llm,
        store,
        on_delta=lambda aid, d: deltas.append(d),
        on_sentence=sink.on_sentence if sink else None,
        on_end=lambda aid, interrupted: ends.append((aid, interrupted)),
    )
    return responder, store, deltas, ends


async def test_natural_completion_commits_answer():
    responder, store, deltas, ends = make_responder(
        ["Typical rollout is four to six weeks for a team your size."], delta_delay=0
    )
    answer = responder.start()
    await answer.task
    assert "".join(deltas).startswith("Typical rollout")
    assert ends == [(answer.id, False)]
    last = store.as_messages()[-1]
    assert last["role"] == "assistant"
    assert "[interrupted" not in last["content"]
    assert responder.partial_answer is None


async def test_interrupt_cancels_flushes_and_marks_history():
    sink = FakeTTSSink()
    responder, store, deltas, ends = make_responder(
        ["Typical rollout is four to six weeks. " * 20], sink=sink
    )
    responder.start()
    # let a few deltas stream
    while not deltas:
        await asyncio.sleep(0.005)
    assert responder.partial_answer is not None

    await responder.interrupt()
    sink.cancel()  # orchestrator flushes TTS alongside

    assert ends[-1][1] is True  # interrupted=True surfaced
    last = store.as_messages()[-1]
    assert last["role"] == "assistant"
    assert last["content"].endswith("[interrupted — conversation moved on]")
    assert responder.partial_answer is None
    assert sink.cancelled and sink.sentences == []


async def test_interrupt_then_new_answer_addresses_new_topic():
    responder, store, deltas, _ = make_responder(
        ["Typical rollout is four to six weeks. " * 20, "On pricing: lead with platform fees."]
    )
    responder.start()
    while not deltas:
        await asyncio.sleep(0.005)
    await responder.interrupt()

    store.add(final("THEM", "Actually the bigger issue is pricing."))
    deltas.clear()
    answer2 = responder.start()
    await answer2.task
    assert "".join(deltas).startswith("On pricing")
    # exactly one in-flight answer existed at a time
    assert responder.current is answer2


async def test_sentence_chunking_for_tts():
    sink = FakeTTSSink()
    responder, _, _, _ = make_responder(
        ["First sentence is long enough to flush properly. Second one also has enough length."],
        sink=sink,
        delta_delay=0,
    )
    answer = responder.start()
    await answer.task
    assert len(sink.sentences) == 2
    assert sink.sentences[0].startswith("First sentence")


async def test_llm_error_is_surfaced_not_swallowed():
    store = TranscriptStore(mission="m")
    store.add(final("THEM", "Question?"))
    llm = FakeLLMHandle([RuntimeError("connection refused")])
    errors: list[Exception] = []
    ends: list[tuple[str, bool]] = []
    responder = Responder(
        llm,
        store,
        on_delta=lambda aid, d: None,
        on_end=lambda aid, interrupted: ends.append((aid, interrupted)),
        on_error=errors.append,
    )
    answer = responder.start()
    await answer.task  # must not raise — error is handled inside the task
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert ends and ends[0][1] is True
    assert responder.partial_answer is None  # not stuck "in flight"


async def test_nothing_to_add_is_not_spoken_or_committed():
    sink = FakeTTSSink()
    responder, store, _, _ = make_responder(["(nothing to add)"], sink=sink, delta_delay=0)
    answer = responder.start()
    await answer.task
    assert sink.sentences == []
    assert store.as_messages()[-1]["role"] == "user"  # no assistant entry committed
