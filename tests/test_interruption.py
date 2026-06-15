"""Responder behavior over a scripted ACP agent — streaming, interruption,
tool actions, and the y/n confirmation gate. No subprocess, no network."""

import asyncio

from fakes import FakeACPAgent, FakeTTSSink, final

from earpiece.brain.responder import Responder
from earpiece.brain.transcript import TranscriptStore
from earpiece.config import LLMSlot, Settings


def make_settings(**overrides) -> Settings:
    slot = LLMSlot(base_url="http://localhost:1/v1", api_key="k", model="m")
    return Settings(
        mission="m", responder=slot, watcher=slot, agent_cmd="fake --acp", **overrides
    )


def make_responder(
    turns: list[list],
    sink: FakeTTSSink | None = None,
    delta_delay: float = 0.01,
    **settings_overrides,
):
    store = TranscriptStore(mission="m")
    store.add(final("THEM", "What does the integration timeline look like?"))
    agent = FakeACPAgent(turns, delta_delay=delta_delay)
    deltas: list[str] = []
    ends: list[tuple[str, bool]] = []
    actions: list[tuple[str, str]] = []
    responder = Responder(
        make_settings(**settings_overrides),
        store,
        on_delta=lambda aid, d: deltas.append(d),
        on_sentence=sink.on_sentence if sink else None,
        on_end=lambda aid, interrupted: ends.append((aid, interrupted)),
        on_action=lambda tool, args, status: actions.append((tool, status)),
        agent=agent,
    )
    responder.actions = actions  # type: ignore[attr-defined] — test convenience
    return responder, store, deltas, ends


async def test_natural_completion_commits_answer():
    responder, store, deltas, ends = make_responder(
        [["Typical rollout is four to six weeks for a team your size."]], delta_delay=0
    )
    answer = responder.start()
    await answer.task
    assert "".join(deltas).startswith("Typical rollout")
    assert ends == [(answer.id, False)]
    last = store.as_messages()[-1]
    assert last["role"] == "assistant"
    assert "[interrupted" not in last["content"]
    assert responder.partial_answer is None


async def test_chat_message_is_forwarded_verbatim_not_the_transcript():
    responder, store, deltas, _ = make_responder([["On it."]], delta_delay=0)
    answer = responder.start(prompt_text="summarize the meeting so far")
    await answer.task
    prompt = responder.agent.prompts[0]
    assert "summarize the meeting so far" in prompt  # operator's words forwarded
    assert "[operator]" in prompt  # labelled as a direct message on the first turn
    assert "integration timeline" not in prompt  # transcript was NOT drained
    assert "".join(deltas).startswith("On it.")


async def test_first_turn_carries_mission_then_only_new_lines():
    responder, store, _, _ = make_responder([["Answer one."], ["Answer two."]], delta_delay=0)
    answer = responder.start()
    await answer.task
    store.add(final("THEM", "And what about pricing?"))
    answer2 = responder.start()
    await answer2.task
    first, second = responder.agent.prompts
    assert "MISSION" in first and "integration timeline" in first
    assert "MISSION" not in second  # instructions only once; session keeps context
    assert "pricing" in second and "integration timeline" not in second


async def test_interrupt_cancels_flushes_and_notifies_harness():
    sink = FakeTTSSink()
    responder, store, deltas, ends = make_responder(
        [["Typical rollout is four to six weeks. " * 20]], sink=sink
    )
    responder.start()
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
    assert responder.agent.cancelled == ["sess-1"]  # session/cancel sent
    assert sink.cancelled and sink.sentences == []


async def test_sentence_chunking_for_tts():
    sink = FakeTTSSink()
    responder, _, _, _ = make_responder(
        [["First sentence is long enough to flush properly. Second one also has enough length."]],
        sink=sink,
        delta_delay=0,
    )
    answer = responder.start()
    await answer.task
    assert len(sink.sentences) == 2
    assert sink.sentences[0].startswith("First sentence")


async def test_harness_error_is_surfaced_not_swallowed():
    responder, _, _, ends = make_responder([[RuntimeError("harness crashed")]], delta_delay=0)
    errors: list[Exception] = []
    responder.on_error = errors.append
    answer = responder.start()
    await answer.task  # must not raise — error is handled inside the task
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert ends and ends[0][1] is True
    assert responder.partial_answer is None  # not stuck "in flight"


async def test_nothing_to_add_is_not_spoken_or_committed():
    sink = FakeTTSSink()
    responder, store, _, _ = make_responder([["(nothing to add)"]], sink=sink, delta_delay=0)
    answer = responder.start()
    await answer.task
    assert sink.sentences == []
    assert store.as_messages()[-1]["role"] == "user"  # no assistant entry committed


# ---- tool calls in the timeline --------------------------------------------


async def test_tool_call_updates_surface_as_actions():
    responder, _, _, _ = make_responder(
        [
            [
                {"sessionUpdate": "tool_call", "toolCallId": "t1",
                 "title": "web_search", "status": "pending",
                 "rawInput": {"query": "rollout times"}},
                {"sessionUpdate": "tool_call_update", "toolCallId": "t1",
                 "status": "completed"},
                "Search says four weeks.",
            ]
        ],
        delta_delay=0,
    )
    answer = responder.start()
    await answer.task
    assert responder.actions == [("web_search", "running"), ("web_search", "done")]


# ---- confirmation gate ------------------------------------------------------

OPTIONS = [
    {"optionId": "ok", "name": "Allow", "kind": "allow_once"},
    {"optionId": "no", "name": "Reject", "kind": "reject_once"},
]


def permission_step(outcomes: list, tool_call: dict):
    """Like the real ACP client, the permission handler runs in its own task —
    interrupting the answer turn must not kill the gate mid-decision."""

    async def handler(agent: FakeACPAgent) -> None:
        outcomes.append(
            await agent.request_permission(
                {"sessionId": "sess-1", "toolCall": tool_call, "options": OPTIONS}
            )
        )

    async def step(agent: FakeACPAgent) -> None:
        agent.permission_task = asyncio.create_task(handler(agent))
        # shield: cancelling the answer turn must not cancel the gate, exactly
        # like the real client where the reader loop owns the handler task
        await asyncio.shield(agent.permission_task)

    return step


async def test_readonly_kind_auto_approves():
    outcomes: list = []
    responder, _, _, _ = make_responder(
        [
            [
                permission_step(
                    outcomes,
                    {"toolCallId": "t1", "title": "read_page", "kind": "read"},
                ),
                "Done.",
            ]
        ],
        delta_delay=0,
    )
    await responder.start().task
    assert outcomes == [{"outcome": {"outcome": "selected", "optionId": "ok"}}]
    assert responder.pending_action is None
    assert ("read_page", "pending") not in responder.actions


async def test_auto_tools_glob_approves_without_asking():
    outcomes: list = []
    responder, _, _, _ = make_responder(
        [
            [
                permission_step(
                    outcomes,
                    {"toolCallId": "t1", "title": "jira_lookup", "kind": "execute"},
                ),
            ]
        ],
        delta_delay=0,
        agent_auto_tools="jira_*",
    )
    await responder.start().task
    assert outcomes == [{"outcome": {"outcome": "selected", "optionId": "ok"}}]


async def test_write_kind_waits_for_approval():
    outcomes: list = []
    responder, _, _, _ = make_responder(
        [
            [
                permission_step(
                    outcomes,
                    {"toolCallId": "t1", "title": "create_ticket", "kind": "execute",
                     "rawInput": {"summary": "bug"}},
                ),
                "Created the ticket.",
            ]
        ],
        delta_delay=0,
    )
    task = responder.start().task
    while responder.pending_action is None:
        await asyncio.sleep(0)
    assert responder.pending_action.tool == "create_ticket"
    assert ("create_ticket", "pending") in responder.actions

    assert responder.resolve_action(True) is True
    await task
    assert outcomes == [{"outcome": {"outcome": "selected", "optionId": "ok"}}]
    assert ("create_ticket", "running") in responder.actions
    assert responder.pending_action is None


async def test_denied_action_returns_reject_option():
    outcomes: list = []
    responder, _, _, _ = make_responder(
        [
            [
                permission_step(
                    outcomes,
                    {"toolCallId": "t1", "title": "create_ticket", "kind": "execute"},
                ),
            ]
        ],
        delta_delay=0,
    )
    task = responder.start().task
    while responder.pending_action is None:
        await asyncio.sleep(0)
    responder.resolve_action(False)
    await task
    assert outcomes == [{"outcome": {"outcome": "selected", "optionId": "no"}}]
    assert ("create_ticket", "denied") in responder.actions


async def test_interrupt_cancels_pending_permission():
    outcomes: list = []
    responder, _, _, _ = make_responder(
        [
            [
                permission_step(
                    outcomes,
                    {"toolCallId": "t1", "title": "create_ticket", "kind": "execute"},
                ),
            ]
        ],
        delta_delay=0,
    )
    responder.start()
    while responder.pending_action is None:
        await asyncio.sleep(0)
    await responder.interrupt()
    # the gate task survives the turn cancellation and answers "cancelled"
    await asyncio.wait_for(responder.agent.permission_task, 1.0)
    assert outcomes == [{"outcome": {"outcome": "cancelled"}}]
    assert responder.pending_action is None
