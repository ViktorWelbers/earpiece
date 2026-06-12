"""Rolling speaker-tagged transcript + append-only chat-history builder.

Append-only invariant: `as_messages()` never rewrites earlier messages — new
utterances since the last call are folded into ONE new user message, so any
provider-side prefix cache (OpenAI automatic, Anthropic, vLLM) stays valid.
The only intentional rewrite is the sliding-window summarization, which resets
the cache prefix at most once per long session.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..events import Speaker, TranscriptEvent
from ..llm import LLMHandle, Message
from .prompts import responder_system

log = logging.getLogger(__name__)


@dataclass
class Utterance:
    speaker: Speaker
    text: str
    wall_time: str  # "hh:mm:ss" — display + message content only

    def line(self) -> str:
        return f"[{self.wall_time}] {self.speaker}: {self.text}"


@dataclass
class _HistoryEntry:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class TranscriptStore:
    mission: str
    max_context_tokens: int = 60_000
    utterances: list[Utterance] = field(default_factory=list)
    # committed chat history (frozen prefix); pending utterances not yet committed
    _history: list[_HistoryEntry] = field(default_factory=list)
    _pending: list[Utterance] = field(default_factory=list)
    # live interim lines, keyed by speaker (UI only)
    interim: dict[Speaker, str] = field(default_factory=dict)
    # monotonic time each interim last changed — read by the orchestrator watchdog
    interim_updated: dict[Speaker, float] = field(default_factory=dict)
    # last finalized text per speaker (text, monotonic) — duplicate suppression
    _last_final: dict[Speaker, tuple[str, float]] = field(default_factory=dict)

    _DUP_WINDOW_SECS = 10.0

    def add(self, event: TranscriptEvent) -> Utterance | None:
        """Merge an STT event. Returns the new Utterance for finals, else None."""
        now = time.monotonic()
        if not event.is_final:
            if self.interim.get(event.source) != event.text:
                self.interim[event.source] = event.text
                self.interim_updated[event.source] = now
            return None
        self.interim.pop(event.source, None)
        self.interim_updated.pop(event.source, None)
        text = event.text.strip()
        if not text:
            return None  # retraction: STT finalized pending words as silence
        last = self._last_final.get(event.source)
        if last is not None and last[0] == text and now - last[1] < self._DUP_WINDOW_SECS:
            return None  # same text finalized twice (watchdog/engine race)
        self._last_final[event.source] = (text, now)
        utt = Utterance(
            speaker=event.source,
            text=text,
            wall_time=time.strftime("%H:%M:%S"),
        )
        self.utterances.append(utt)
        self._pending.append(utt)
        return utt

    # ---- chat history -------------------------------------------------

    def as_messages(self) -> list[Message]:
        """Commit pending utterances as one user message; return full history."""
        if self._pending:
            block = "\n".join(u.line() for u in self._pending)
            self._history.append(_HistoryEntry("user", block))
            self._pending.clear()
        messages: list[Message] = [
            {"role": "system", "content": responder_system(self.mission)}
        ]
        messages.extend({"role": e.role, "content": e.content} for e in self._history)
        return messages

    def add_answer(self, text: str, *, interrupted: bool) -> None:
        if interrupted:
            text = f"{text} [interrupted — conversation moved on]"
        self._history.append(_HistoryEntry("assistant", text))

    def recent_lines(self, n: int = 15) -> str:
        return "\n".join(u.line() for u in self.utterances[-n:])

    # ---- sliding window ------------------------------------------------

    def estimated_tokens(self) -> int:
        return sum(len(e.content) for e in self._history) // 4

    def needs_compaction(self) -> bool:
        return self.estimated_tokens() > self.max_context_tokens

    async def compact(self, summarizer: LLMHandle) -> None:
        """Summarize the oldest half of history into one context block."""
        if len(self._history) < 4:
            return
        half = len(self._history) // 2
        old, keep = self._history[:half], self._history[half:]
        digest = "\n\n".join(f"{e.role.upper()}:\n{e.content}" for e in old)
        summary = ""
        async for delta in summarizer.stream_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation transcript segment into a dense brief "
                        "for an assistant that will keep following the live conversation. "
                        "Keep names, numbers, commitments, open questions."
                    ),
                },
                {"role": "user", "content": digest},
            ]
        ):
            summary += delta
        self._history = [
            _HistoryEntry("user", f"[earlier conversation summary]\n{summary}"),
            *keep,
        ]
        log.info("compacted history: %d -> %d entries", half + len(keep), len(self._history))
