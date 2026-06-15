"""Scripted fakes — no audio hardware, no network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from earpiece.events import TranscriptEvent
from earpiece.llm import Message


class FakeLLMHandle:
    """Stands in for llm.LLMHandle. Streams scripted answers; records calls."""

    def __init__(self, answers: list[str] | None = None, *, delta_delay: float = 0.0) -> None:
        self.answers = list(answers or [])
        self.delta_delay = delta_delay
        self.calls: list[list[Message]] = []
        self.last_usage: dict = {}

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        self.calls.append(messages)
        item = self.answers.pop(0) if self.answers else "(nothing to add)"
        if isinstance(item, Exception):
            raise item
        for word in item.split(" "):
            if self.delta_delay:
                await asyncio.sleep(self.delta_delay)
            yield word + " "

    async def structured(self, messages: list[Message], schema):
        self.calls.append(messages)
        raise NotImplementedError("use FakeWatcherHandle for structured calls")


class FakeWatcherHandle:
    """Returns pre-scripted Decision objects (or raises)."""

    def __init__(self, decisions: list) -> None:
        self.decisions = list(decisions)
        self.calls: list[list[Message]] = []
        self.last_usage: dict = {}

    async def structured(self, messages: list[Message], schema):
        self.calls.append(messages)
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        self.calls.append(messages)
        yield "summary"


def final(source: str, text: str, t: float = 0.0) -> TranscriptEvent:
    return TranscriptEvent(source=source, text=text, is_final=True, started_at=t, ended_at=t + 1)


def interim(source: str, text: str, t: float = 0.0) -> TranscriptEvent:
    return TranscriptEvent(source=source, text=text, is_final=False, started_at=t, ended_at=t)


class FakeACPAgent:
    """Duck-types brain.acp.ACPAgent — scripted turns, no subprocess.

    Each turn is a list of steps executed in order:
      str        -> streamed word-by-word as agent_message_chunk updates
      dict       -> emitted verbatim as a session/update payload
      Exception  -> raised (harness failure)
      callable   -> awaited with the agent (e.g. a permission round-trip)
    """

    def __init__(self, turns: list[list] | None = None, *, delta_delay: float = 0.0) -> None:
        self.turns = list(turns or [])
        self.delta_delay = delta_delay
        self.prompts: list[str] = []
        self.cancelled: list[str] = []
        self.started = False
        self.stopped = False
        self.on_update = None
        self.request_permission = None

    async def start(self) -> dict:
        self.started = True
        return {"protocolVersion": 1, "agentInfo": {"name": "fake"}}

    async def stop(self) -> None:
        self.stopped = True

    async def new_session(self, cwd: str, mcp_servers: list) -> str:
        return "sess-1"

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)

    def emit(self, update: dict) -> None:
        self.on_update({"sessionId": "sess-1", "update": update})

    async def prompt(self, session_id: str, text: str) -> str:
        self.prompts.append(text)
        steps = self.turns.pop(0) if self.turns else ["(nothing to add)"]
        for step in steps:
            if isinstance(step, Exception):
                raise step
            if isinstance(step, str):
                for word in step.split(" "):
                    if self.delta_delay:
                        await asyncio.sleep(self.delta_delay)
                    self.emit(
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": word + " "},
                        }
                    )
            elif isinstance(step, dict):
                self.emit(step)
            else:
                await step(self)
        return "end_turn"


class FakeTTSSink:
    """Collects sentences the responder emits for TTS."""

    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.cancelled = False

    def on_sentence(self, sentence: str) -> None:
        self.sentences.append(sentence)

    def cancel(self) -> None:
        self.cancelled = True
        self.sentences.clear()
