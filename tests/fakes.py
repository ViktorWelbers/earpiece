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
