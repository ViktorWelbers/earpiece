from earpiece.brain.transcript import TranscriptStore
from earpiece.output.console import ConsoleView


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
