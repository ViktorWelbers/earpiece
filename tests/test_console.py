from rich.text import Text

from earpiece.brain.transcript import TranscriptStore
from earpiece.output.console import ActionEntry, ConsoleView


def make_view() -> ConsoleView:
    return ConsoleView(TranscriptStore(mission="m"))


def stream(view: ConsoleView, answer_id: str, text: str, *, interrupted: bool = False) -> None:
    view.on_answer_start()
    view.on_delta(answer_id, text)
    view.on_end(answer_id, interrupted)


def test_answers_accumulate_as_timeline():
    view = make_view()
    stream(view, "a1", "Berlin.")
    stream(view, "a2", "Linux.")
    assert [e.text for e in view.answers] == ["Berlin.", "Linux."]
    assert view.answer_text == ""


def test_nothing_to_add_keeps_previous_answers():
    view = make_view()
    stream(view, "a1", "Berlin.")
    stream(view, "a2", "(nothing to add)")
    assert [e.text for e in view.answers] == ["Berlin."]


def test_interrupted_answer_is_kept_and_marked():
    view = make_view()
    stream(view, "a1", "Typical rollout is four to", interrupted=True)
    assert view.answers[0].interrupted is True
    assert view.answers[0].text == "Typical rollout is four to"


def test_action_lifecycle_updates_one_entry():
    view = make_view()
    view.on_action("t1", "create_ticket", {"summary": "bug"}, "pending")
    assert view.status.pending_action == "create_ticket"  # y/n banner up
    view.on_action("t1", "create_ticket", {}, "running")
    view.on_action("t1", "create_ticket", {}, "done")
    [entry] = view.answers
    assert entry.status == "done"
    assert "bug" in entry.args_summary
    assert view.status.pending_action == ""  # banner cleared


def test_actions_and_answers_interleave_in_timeline():
    view = make_view()
    view.on_action("s1", "web_search", {"q": "x"}, "running")
    view.on_action("s1", "web_search", {}, "done")
    stream(view, "a1", "Found it: four weeks.")
    view.on_action("s2", "web_search", {"q": "y"}, "running")  # new call -> new entry
    assert [type(e).__name__ for e in view.answers] == [
        "ActionEntry",
        "AnswerEntry",
        "ActionEntry",
    ]
    panel = view._answer_panel()  # rendering must not blow up on mixed entries
    assert panel is not None


def test_repeated_same_tool_calls_stay_separate_in_order():
    """Regression: a second call to the same tool must append a new entry at the
    end, not fold onto the first (even while the first is still pending)."""
    view = make_view()
    view.on_action("b1", "bash", {"cmd": "ls"}, "pending")  # never completes
    view.on_action("b2", "bash", {"cmd": "pwd"}, "pending")  # distinct call
    entries = [e for e in view.answers if isinstance(e, ActionEntry)]
    assert [e.call_id for e in entries] == ["b1", "b2"]  # two entries, in order
    assert all(e.status == "pending" for e in entries)
    assert view.answers[-1].call_id == "b2"  # newest sits last, correct position


def test_turn_end_clears_stale_pending_banner():
    view = make_view()
    view.on_action("t1", "create_ticket", {}, "pending")
    view.on_end("a1", True)  # interrupted turn — confirmation is moot
    assert view.status.pending_action == ""


def test_window_shows_bottom_and_scrolls_up():
    lines = [Text(str(i)) for i in range(10)]
    assert [t.plain for t in ConsoleView._window(lines, 3, 0)] == ["7", "8", "9"]  # live bottom
    assert [t.plain for t in ConsoleView._window(lines, 3, 2)] == ["5", "6", "7"]  # scrolled up 2
    assert [t.plain for t in ConsoleView._window(lines, 3, 99)] == ["0", "1", "2"]  # clamp at top
    assert ConsoleView._window(lines, 20, 5) == lines  # fits entirely → no windowing


def test_scroll_state_clamps_and_returns_to_live():
    view = make_view()
    view.scroll_down(5)  # already at bottom
    assert view.scroll == 0
    view.scroll_up(3)
    assert view.scroll == 3
    view.scroll_to_bottom()
    assert view.scroll == 0


def test_chat_message_appears_in_timeline_before_the_reply():
    view = make_view()
    view.on_chat("summarize the meeting")
    stream(view, "a1", "Three open action items.")
    assert [type(e).__name__ for e in view.answers] == ["ChatEntry", "AnswerEntry"]
    assert view.answers[0].text == "summarize the meeting"


def test_always_on_input_renders_the_buffer():
    view = make_view()
    view.set_input("draft a reply")
    assert view.input_buffer == "draft a reply"
    layout = view._render()  # always-on input bar must render without error
    assert layout is not None
