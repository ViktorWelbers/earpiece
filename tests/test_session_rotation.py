"""Session rotation: every N turns the harness session is compressed into a
brief and reopened, so it never lives long enough to start continuing the
transcript instead of answering it."""

from fakes import FakeACPAgent, FakeTTSSink, final

from earpiece.brain.prompts import COMPRESS_PROMPT
from earpiece.brain.responder import Responder
from earpiece.brain.transcript import TranscriptStore
from earpiece.config import Settings

SUMMARY = "Mock interview with Namespace. Covered Postgres indexes and OCI layers."


def make_responder(agent: FakeACPAgent, sink: FakeTTSSink | None = None, **overrides):
    store = TranscriptStore(mission="m")
    deltas: list[str] = []
    responder = Responder(
        Settings(mission="m", agent_cmd="fake --acp", **overrides),
        store,
        on_delta=lambda aid, d: deltas.append(d),
        on_sentence=sink.on_sentence if sink else None,
        on_end=lambda aid, interrupted: None,
        agent=agent,
    )
    return responder, store, deltas


async def run_turns(responder: Responder, store: TranscriptStore, n: int) -> None:
    for i in range(n):
        store.add(final("THEM", f"question {i}", t=float(i)))
        await responder.start().task


async def test_rotates_after_the_configured_number_of_turns():
    agent = FakeACPAgent([["cue"] for _ in range(40)], delta_delay=0)
    responder, store, _ = make_responder(agent, agent_session_turns=15)
    await responder.start_agent()

    await run_turns(responder, store, 15)
    assert agent.new_sessions == 2  # startup + the rotation on turn 15

    await run_turns(responder, store, 14)
    assert agent.new_sessions == 2  # not yet
    await run_turns(responder, store, 1)
    assert agent.new_sessions == 3


async def test_compress_prompt_goes_to_the_old_session_and_stays_off_screen():
    sink = FakeTTSSink()
    agent = FakeACPAgent([["cue"], [SUMMARY], ["cue"]], delta_delay=0)
    responder, store, deltas = make_responder(agent, sink, agent_session_turns=1)
    await responder.start_agent()

    await run_turns(responder, store, 1)

    assert agent.prompts[1] == COMPRESS_PROMPT
    # the brief is context for the next session, not something the operator sees
    assert SUMMARY not in "".join(deltas)
    assert not any(SUMMARY.split()[0] in s for s in sink.sentences)
    # ...and it is not committed to the transcript as an answer either
    assert all(SUMMARY not in m["content"] for m in store.as_messages())


async def test_next_turn_carries_the_system_prompt_and_the_brief():
    agent = FakeACPAgent([["cue"], [SUMMARY], ["cue"]], delta_delay=0)
    responder, store, _ = make_responder(agent, agent_session_turns=1)
    await responder.start_agent()

    await run_turns(responder, store, 2)

    after = agent.prompts[2]
    assert "discreet real-time copilot" in after  # mission + system prompt re-sent
    assert f"[prior conversation]\n{SUMMARY}" in after
    assert "[transcript]\n[" in after  # the actual new lines still follow


async def test_a_failed_compression_keeps_the_session_alive():
    agent = FakeACPAgent([["cue"], [RuntimeError("harness blew up")], ["cue"]], delta_delay=0)
    responder, store, deltas = make_responder(agent, agent_session_turns=1)
    await responder.start_agent()

    await run_turns(responder, store, 1)
    assert agent.new_sessions == 1  # rotation aborted, old session retained
    assert responder.session_id == "sess-1"

    await run_turns(responder, store, 1)  # and the next turn still answers
    assert "cue" in "".join(deltas)


async def test_zero_disables_rotation():
    agent = FakeACPAgent([["cue"] for _ in range(30)], delta_delay=0)
    responder, store, _ = make_responder(agent, agent_session_turns=0)
    await responder.start_agent()

    await run_turns(responder, store, 30)
    assert agent.new_sessions == 1
    assert COMPRESS_PROMPT not in agent.prompts
