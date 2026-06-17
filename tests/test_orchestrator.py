"""Orchestrator decision logic: when a new utterance interrupts an in-flight
answer (voice mode) vs. coalesces and waits for it to finish (text-only mode)."""

from types import SimpleNamespace

from fakes import final

from earpiece.config import Settings
from earpiece.events import Action
from earpiece.orchestrator import Orchestrator


def make_orch(**overrides) -> Orchestrator:
    # __init__ touches no audio devices and ACPAgent doesn't spawn until start()
    return Orchestrator(Settings(mission="m", agent_cmd="fake --acp", **overrides))


def set_in_flight(orch: Orchestrator, in_flight: bool) -> None:
    orch.responder = SimpleNamespace(partial_answer="…" if in_flight else None)


def test_text_mode_does_not_interrupt_in_flight_answer():
    orch = make_orch()  # no tts_engine → text-only
    set_in_flight(orch, True)
    decision = orch._decide(final("THEM", "and pricing?"), forced=False)
    assert decision.action is Action.RESPOND  # let the current answer finish


def test_voice_mode_interrupts_in_flight_answer():
    orch = make_orch()
    orch.tts = object()  # pretend --voice was given
    set_in_flight(orch, True)
    decision = orch._decide(final("THEM", "and pricing?"), forced=False)
    assert decision.action is Action.INTERRUPT_AND_RESPOND


def test_idle_answer_responds_when_nothing_in_flight():
    orch = make_orch()
    set_in_flight(orch, False)
    decision = orch._decide(final("THEM", "and pricing?"), forced=False)
    assert decision.action is Action.RESPOND


def test_answer_end_signals_idle_only_on_natural_finish():
    orch = make_orch()
    orch._on_answer_end("a1", interrupted=True)
    assert not orch._answer_idle.is_set()  # interrupt already starts the next turn
    orch._on_answer_end("a2", interrupted=False)
    assert orch._answer_idle.is_set()  # natural finish → drain any backlog


def test_has_pending_tracks_undrained_utterances():
    orch = make_orch()
    assert orch.transcript.has_pending is False
    orch.transcript.add(final("THEM", "what is the timeline?"))
    assert orch.transcript.has_pending is True
    orch.transcript.drain_pending_block()
    assert orch.transcript.has_pending is False
