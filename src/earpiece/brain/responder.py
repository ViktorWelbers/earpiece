"""Streaming answers with interruption bookkeeping.

The Responder owns answer state; the Orchestrator owns the task wrapping it.
On natural end the full answer is committed to history; on cancellation the
partial text is committed with the interrupted marker.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
from collections.abc import Callable

from ..llm import LLMHandle
from .transcript import TranscriptStore

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]?\s")
_MIN_TTS_CHUNK = 40  # chars; avoid stuttering TTS on tiny fragments

NOTHING = "(nothing to add)"


class Answer:
    """One in-flight (or finished) streamed answer."""

    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.id = f"a{next(self._ids)}"
        self.text = ""
        self.task: asyncio.Task | None = None
        self.interrupted = False


class Responder:
    def __init__(
        self,
        llm: LLMHandle,
        transcript: TranscriptStore,
        on_delta: Callable[[str, str], None],  # (answer_id, delta) -> None
        on_sentence: Callable[[str], None] | None = None,  # sentence chunks for TTS
        on_end: Callable[[str, bool], None] | None = None,  # (answer_id, interrupted)
        on_error: Callable[[Exception], None] | None = None,  # surfaced in the UI
    ) -> None:
        self.llm = llm
        self.transcript = transcript
        self.on_delta = on_delta
        self.on_sentence = on_sentence
        self.on_end = on_end
        self.on_error = on_error
        self.current: Answer | None = None

    @property
    def partial_answer(self) -> str | None:
        answer = self.current
        if answer is not None and answer.task is not None and not answer.task.done():
            return answer.text or "(answer just started)"
        return None

    def start(self) -> Answer:
        """Begin a new streamed answer over the current transcript."""
        answer = Answer()
        answer.task = asyncio.create_task(self._run(answer), name=f"responder-{answer.id}")
        self.current = answer
        return answer

    async def interrupt(self) -> None:
        """Cancel the in-flight answer and commit its partial text to history."""
        answer = self.current
        if answer is None or answer.task is None or answer.task.done():
            return
        answer.interrupted = True
        answer.task.cancel()
        try:
            await answer.task
        except asyncio.CancelledError:
            pass

    async def _run(self, answer: Answer) -> None:
        tts_buffer = ""
        try:
            async for delta in self.llm.stream_chat(self.transcript.as_messages()):
                answer.text += delta
                self.on_delta(answer.id, delta)
                if self.on_sentence is not None:
                    tts_buffer += delta
                    tts_buffer = self._flush_sentences(tts_buffer)
            if self.on_sentence is not None and tts_buffer.strip():
                self._emit_sentence(tts_buffer)
            if answer.text.strip() and answer.text.strip() != NOTHING:
                self.transcript.add_answer(answer.text, interrupted=False)
            if self.on_end is not None:
                self.on_end(answer.id, False)
        except asyncio.CancelledError:
            if answer.text.strip():
                self.transcript.add_answer(answer.text, interrupted=True)
            if self.on_end is not None:
                self.on_end(answer.id, True)
            raise
        except Exception as exc:  # noqa: BLE001 — a failed call must not be invisible
            log.exception("responder stream failed")
            if answer.text.strip():
                self.transcript.add_answer(answer.text, interrupted=True)
            if self.on_error is not None:
                self.on_error(exc)
            if self.on_end is not None:
                self.on_end(answer.id, True)

    def _flush_sentences(self, buffer: str) -> str:
        """Emit complete sentences >= _MIN_TTS_CHUNK chars; return the remainder."""
        while True:
            match = _SENTENCE_END.search(buffer, _MIN_TTS_CHUNK)
            if match is None:
                return buffer
            sentence, buffer = buffer[: match.end()], buffer[match.end() :]
            self._emit_sentence(sentence)

    def _emit_sentence(self, sentence: str) -> None:
        sentence = sentence.strip()
        if sentence and sentence != NOTHING and self.on_sentence is not None:
            self.on_sentence(sentence)
