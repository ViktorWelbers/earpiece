"""Session persistence: round-tripping a SessionRecord to disk and the
snapshot/restore of the transcript and the answers timeline that feed it."""

import json

from earpiece.brain import session_store
from earpiece.brain.session_store import SessionRecord
from earpiece.brain.transcript import TranscriptStore
from earpiece.config import Settings
from earpiece.events import TranscriptEvent
from earpiece.output.console import ConsoleView


def make_settings(**overrides) -> Settings:
    return Settings(mission="be my assistant", agent_cmd="fake --acp", **overrides)


def use_tmp_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("EARPIECE_CONFIG", str(tmp_path / "config.toml"))


# -- session_store round-trip ------------------------------------------------


def test_save_then_load_round_trips(monkeypatch, tmp_path):
    use_tmp_sessions(monkeypatch, tmp_path)
    rec = SessionRecord.new(make_settings())
    rec.harness_session_id = "sess-9"
    rec.transcript = {"mission": "m", "utterances": [], "history": []}
    rec.answers = [{"kind": "answer", "wall_time": "00:00:00", "text": "hi", "interrupted": False}]
    session_store.save(rec)

    loaded = session_store.load(rec.id)
    assert loaded.id == rec.id
    assert loaded.harness_session_id == "sess-9"
    assert loaded.answers == rec.answers


def test_save_is_atomic_no_tempfile_left(monkeypatch, tmp_path):
    use_tmp_sessions(monkeypatch, tmp_path)
    rec = SessionRecord.new(make_settings())
    session_store.save(rec)
    files = sorted(p.name for p in session_store.sessions_dir().iterdir())
    assert files == [f"{rec.id}.json"]  # no .json.tmp left behind


def test_list_sessions_newest_first_and_skips_corrupt(monkeypatch, tmp_path):
    use_tmp_sessions(monkeypatch, tmp_path)
    older = SessionRecord.new(make_settings())
    older.updated_at = 100.0
    newer = SessionRecord.new(make_settings())
    newer.updated_at = 200.0
    session_store.save(older)
    session_store.save(newer)
    session_store.save(older)  # bumps older.updated_at to now → would be newest
    newer.updated_at = 9_999_999_999.0  # force newer ahead again
    session_store.save(newer)
    (session_store.sessions_dir() / "broken.json").write_text("{ not json")

    listed = session_store.list_sessions()
    assert [r.id for r in listed] == [newer.id, older.id]  # corrupt skipped, newest first


def test_turns_counts_only_answers():
    rec = SessionRecord.new(make_settings())
    rec.answers = [
        {"kind": "answer", "text": "a"},
        {"kind": "action", "tool": "bash"},
        {"kind": "chat", "text": "hey"},
        {"kind": "answer", "text": "b"},
    ]
    assert rec.turns == 2


# -- transcript snapshot/restore --------------------------------------------


def add_final(store: TranscriptStore, speaker: str, text: str) -> None:
    store.add(
        TranscriptEvent(source=speaker, text=text, is_final=True, started_at=0.0, ended_at=0.0)
    )


def test_transcript_snapshot_restore_round_trips_and_commits_pending():
    store = TranscriptStore(mission="m")
    add_final(store, "THEM", "what is the timeline?")
    store.drain_pending_block()
    store.add_answer("about four weeks", interrupted=False)
    add_final(store, "THEM", "and the cost?")  # still pending

    snap = store.snapshot()
    assert store.has_pending is False  # snapshot committed the pending line

    fresh = TranscriptStore(mission="m")
    fresh.restore(snap)
    assert [u.text for u in fresh.utterances] == ["what is the timeline?", "and the cost?"]
    assert [e["role"] for e in snap["history"]] == ["user", "assistant", "user"]
    # restored history feeds the model on reattach failure (system prompt is index 0)
    restored_msgs = fresh.as_messages()[1:]
    assert [m["role"] for m in restored_msgs] == ["user", "assistant", "user"]
    assert "what is the timeline?" in restored_msgs[0]["content"]
    assert restored_msgs[1]["content"] == "about four weeks"
    assert "and the cost?" in restored_msgs[2]["content"]


def test_history_text_labels_both_roles():
    store = TranscriptStore(mission="m")
    add_final(store, "THEM", "hello")
    store.drain_pending_block()
    store.add_answer("hi there", interrupted=False)
    text = store.history_text()
    assert "[transcript]" in text and "[assistant]" in text
    assert "hi there" in text


# -- console answers snapshot/restore ---------------------------------------


def test_console_snapshot_restore_round_trips_mixed_timeline():
    view = ConsoleView(TranscriptStore(mission="m"))
    view.on_chat("summarize the meeting")
    view.on_answer_start()
    view.on_delta("a1", "Three action items.")
    view.on_end("a1", False)
    view.on_action("t1", "create_ticket", {"summary": "bug"}, "done")

    snap = view.snapshot_answers()
    assert [e["kind"] for e in snap] == ["chat", "answer", "action"]

    restored = ConsoleView(TranscriptStore(mission="m"))
    restored.restore_answers(json.loads(json.dumps(snap)))  # survives a JSON trip
    assert [type(e).__name__ for e in restored.answers] == [
        "ChatEntry",
        "AnswerEntry",
        "ActionEntry",
    ]
    assert restored.answers[1].text == "Three action items."
    assert restored.answers[2].tool == "create_ticket"
