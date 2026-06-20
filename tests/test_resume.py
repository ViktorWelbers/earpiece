"""Resuming a session: the responder either reattaches to the harness session
(ACP session/load) or replays the saved transcript into a fresh one, and the
orchestrator restores both panes and auto-saves."""

from fakes import FakeACPAgent, final

from earpiece.brain.responder import Responder
from earpiece.brain.session_store import SessionRecord
from earpiece.brain.transcript import TranscriptStore
from earpiece.config import Settings
from earpiece.orchestrator import Orchestrator


def make_settings(**overrides) -> Settings:
    return Settings(mission="m", agent_cmd="fake --acp", **overrides)


def make_responder(agent: FakeACPAgent, store: TranscriptStore | None = None) -> Responder:
    return Responder(
        make_settings(),
        store or TranscriptStore(mission="m"),
        on_delta=lambda aid, d: None,
        on_end=lambda aid, interrupted: None,
        on_action=lambda *a: None,
        agent=agent,
    )


def resume_record() -> SessionRecord:
    rec = SessionRecord.new(make_settings())  # agent_cmd/cwd match make_settings
    rec.harness_session_id = "sess-42"
    return rec


def store_with_history() -> TranscriptStore:
    store = TranscriptStore(mission="m")
    store.add(final("THEM", "what is the timeline?"))
    store.drain_pending_block()
    store.add_answer("about four weeks", interrupted=False)
    return store


# -- responder reattach vs replay -------------------------------------------


async def test_reattaches_when_harness_supports_load():
    agent = FakeACPAgent(capabilities={"loadSession": True})
    responder = make_responder(agent)
    await responder.start_agent(resume_record())
    assert agent.loaded == ["sess-42"]  # session/load, not session/new
    assert agent.new_sessions == 0
    assert responder.session_id == "sess-42"
    assert responder._instructed is True  # mission already lives in that session


async def test_replays_transcript_when_load_unsupported():
    agent = FakeACPAgent(capabilities={})  # harness can't load
    responder = make_responder(agent, store_with_history())
    await responder.start_agent(resume_record())
    assert agent.loaded == []
    assert agent.new_sessions == 1  # fresh session instead
    assert responder.session_id == "sess-1"
    assert responder._instructed is False
    assert "about four weeks" in responder._resume_prefix  # history queued for turn 1


async def test_falls_back_to_replay_when_load_errors():
    agent = FakeACPAgent(capabilities={"loadSession": True}, load_error=True)
    responder = make_responder(agent, store_with_history())
    await responder.start_agent(resume_record())
    assert agent.new_sessions == 1  # session/load failed → new_session
    assert responder.session_id == "sess-1"
    assert "about four weeks" in responder._resume_prefix


async def test_does_not_reattach_across_a_different_harness():
    agent = FakeACPAgent(capabilities={"loadSession": True})
    responder = make_responder(agent, store_with_history())
    rec = resume_record()
    rec.agent_cmd = "some-other-harness acp"  # session/load only valid on same harness
    await responder.start_agent(rec)
    assert agent.loaded == []
    assert agent.new_sessions == 1


async def test_replay_prefix_is_injected_into_the_first_prompt():
    agent = FakeACPAgent([["ok"]], capabilities={})
    responder = make_responder(agent, store_with_history())
    await responder.start_agent(resume_record())
    answer = responder.start()
    await answer.task
    assert "[prior conversation]" in agent.prompts[0]
    assert "about four weeks" in agent.prompts[0]


async def test_normal_start_opens_a_fresh_session_without_prefix():
    agent = FakeACPAgent(capabilities={"loadSession": True})
    responder = make_responder(agent)
    await responder.start_agent()  # no resume
    assert agent.new_sessions == 1
    assert agent.loaded == []
    assert responder._resume_prefix == ""


# -- orchestrator restore + auto-save ---------------------------------------


def use_tmp_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("EARPIECE_CONFIG", str(tmp_path / "config.toml"))


def test_orchestrator_restores_both_panes_and_reuses_the_id(monkeypatch, tmp_path):
    use_tmp_sessions(monkeypatch, tmp_path)
    rec = SessionRecord.new(make_settings())
    rec.transcript = {
        "utterances": [{"speaker": "THEM", "text": "hello", "wall_time": "00:00:00"}],
        "history": [],
    }
    rec.answers = [{"kind": "answer", "wall_time": "00:00:00", "text": "hi", "interrupted": False}]

    orch = Orchestrator(make_settings(), resume=rec)
    assert [u.text for u in orch.transcript.utterances] == ["hello"]
    assert orch.console.answers[0].text == "hi"
    assert orch._record.id == rec.id  # overwrite the same file, don't fork a new one


def test_persist_writes_a_reloadable_record(monkeypatch, tmp_path):
    use_tmp_sessions(monkeypatch, tmp_path)
    from earpiece.brain import session_store

    orch = Orchestrator(make_settings())
    orch.transcript.add(final("THEM", "what is the cost?"))
    orch.console.on_answer_start()
    orch.console.on_delta("a1", "about four weeks")
    orch.console.on_end("a1", False)
    orch._persist()

    [reloaded] = session_store.list_sessions()
    assert reloaded.id == orch._record.id
    assert reloaded.transcript["utterances"][0]["text"] == "what is the cost?"
    assert reloaded.answers[0]["text"] == "about four weeks"
