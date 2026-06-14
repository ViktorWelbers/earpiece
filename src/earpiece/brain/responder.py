"""Agentic responder: each answer turn is forwarded to an ACP agent harness.

earpiece stays an audio frontend — capture, STT, watcher, TTS, UI. The harness
(pi, claude-agent-acp, gemini --experimental-acp, ...) owns the model endpoint,
the tools (its own + the MCP servers from our config), and the conversation
context. Streamed text lands in the same on_delta/TTS path as before; tool
calls surface as actions, and permission requests pass through the operator's
y/n gate — read-only tool kinds run without asking.

The Responder owns answer state; the Orchestrator owns the task wrapping it.
On natural end the full answer is committed to history; on cancellation the
partial text is committed with the interrupted marker.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch

from ..config import Settings
from .acp import ACPAgent, acp_mcp_servers
from .prompts import responder_system
from .transcript import TranscriptStore

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]?\s")
_MIN_TTS_CHUNK = 40  # chars; avoid stuttering TTS on tiny fragments
_CONFIRM_TIMEOUT_SECS = 30.0  # unconfirmed tool call -> denied

NOTHING = "(nothing to add)"

# tool kinds that never mutate anything — run without asking
_READONLY_KINDS = ("read", "search", "fetch", "think")
# ACP tool-call statuses -> timeline statuses ("pending" is reserved for the y/n gate)
_STATUS_MAP = {"pending": "running", "in_progress": "running",
               "completed": "done", "failed": "failed"}


class Answer:
    """One in-flight (or finished) streamed answer."""

    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.id = f"a{next(self._ids)}"
        self.text = ""
        self.task: asyncio.Task | None = None
        self.interrupted = False


@dataclass
class PendingAction:
    """A tool call waiting for the operator's y/n."""

    tool: str
    args: dict
    decision: asyncio.Future  # resolves to "approve" | "deny" | "cancel"


class Responder:
    def __init__(
        self,
        settings: Settings,
        transcript: TranscriptStore,
        on_delta: Callable[[str, str], None],  # (answer_id, delta) -> None
        on_sentence: Callable[[str], None] | None = None,  # sentence chunks for TTS
        on_end: Callable[[str, bool], None] | None = None,  # (answer_id, interrupted)
        on_error: Callable[[Exception], None] | None = None,  # surfaced in the UI
        on_action: Callable[[str, dict, str], None] | None = None,  # (tool, args, status)
        agent: ACPAgent | None = None,  # injectable for tests
    ) -> None:
        self.settings = settings
        self.transcript = transcript
        self.on_delta = on_delta
        self.on_sentence = on_sentence
        self.on_end = on_end
        self.on_error = on_error
        self.on_action = on_action
        self.agent = agent or ACPAgent(settings.agent_cmd or "", cwd=settings.agent_cwd)
        self.agent.on_update = self._on_update
        self.agent.request_permission = self._on_permission
        self.current: Answer | None = None
        self.pending_action: PendingAction | None = None
        self.last_usage: dict = {}  # OpenAI usage shape, for the status bar
        self._session_id: str | None = None
        self._instructed = False  # mission goes out with the first turn
        self._tool_titles: dict[str, str] = {}  # toolCallId -> human title
        self._tts_buffer = ""

    # ------------------------------------------------------------ lifecycle

    async def start_agent(self) -> None:
        """Spawn the harness, handshake, open the session. Fails fast on a
        broken AGENT_CMD instead of erroring on the first answer."""
        init = await self.agent.start()
        agent_info = init.get("agentInfo") or {}
        log.info("acp agent ready: %s %s", agent_info.get("name", "?"),
                 agent_info.get("version", ""))
        cwd = self.settings.agent_cwd or os.getcwd()
        self._session_id = await self.agent.new_session(
            cwd, acp_mcp_servers(self.settings.mcp_servers)
        )

    async def close(self) -> None:
        await self.agent.stop()

    # ----------------------------------------------------------- public api

    @property
    def partial_answer(self) -> str | None:
        answer = self.current
        if answer is not None and answer.task is not None and not answer.task.done():
            return answer.text or "(answer just started)"
        return None

    def start(self) -> Answer:
        """Begin a new agent turn over the transcript lines since the last one."""
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
        # a tool call awaiting y/n belongs to the cancelled turn — release it
        if self.pending_action is not None and not self.pending_action.decision.done():
            self.pending_action.decision.set_result("cancel")
        answer.task.cancel()
        try:
            await answer.task
        except asyncio.CancelledError:
            pass

    def resolve_action(self, approved: bool) -> bool:
        """Operator hotkey verdict on the pending tool call (y/n)."""
        pending = self.pending_action
        if pending is None or pending.decision.done():
            return False
        pending.decision.set_result("approve" if approved else "deny")
        return True

    # ------------------------------------------------------------- turn loop

    async def _run(self, answer: Answer) -> None:
        self._tts_buffer = ""
        try:
            if self._session_id is None:  # normally done at startup (fail fast)
                await self.start_agent()
            block = self.transcript.drain_pending_block()
            if not block:
                block = "(no new transcript lines — the operator asked you to respond now)"
            if not self._instructed:
                instructions = responder_system(self.settings.mission, with_tools=True)
                block = f"{instructions}\n\n[transcript]\n{block}"
                self._instructed = True
            stop_reason = await self.agent.prompt(self._session_id or "", block)
            log.debug("turn %s finished: %s", answer.id, stop_reason)
            if self.on_sentence is not None and self._tts_buffer.strip():
                self._emit_sentence(self._tts_buffer)
            if answer.text.strip() and answer.text.strip() != NOTHING:
                self.transcript.add_answer(answer.text, interrupted=False)
            if self.on_end is not None:
                self.on_end(answer.id, False)
        except asyncio.CancelledError:
            if self._session_id is not None:
                with contextlib.suppress(Exception):
                    await self.agent.cancel(self._session_id)
            if answer.text.strip():
                self.transcript.add_answer(answer.text, interrupted=True)
            if self.on_end is not None:
                self.on_end(answer.id, True)
            raise
        except Exception as exc:  # noqa: BLE001 — a failed turn must not be invisible
            log.exception("responder turn failed")
            if answer.text.strip():
                self.transcript.add_answer(answer.text, interrupted=True)
            if self.on_error is not None:
                self.on_error(exc)
            if self.on_end is not None:
                self.on_end(answer.id, True)

    # --------------------------------------------------- updates from harness

    def _on_update(self, params: dict) -> None:
        update = params.get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            if content.get("type") == "text" and content.get("text"):
                self._append_delta(content["text"])
        elif kind in ("tool_call", "tool_call_update"):
            tool_id = update.get("toolCallId", "")
            title = update.get("title") or self._tool_titles.get(tool_id, tool_id or "tool")
            self._tool_titles[tool_id] = title
            status = _STATUS_MAP.get(update.get("status") or "pending")
            if status is not None:
                self._notify_action(title, update.get("rawInput") or {}, status)
        elif kind == "usage_update":
            self.last_usage = {
                "prompt_tokens": update.get("used") or 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            }

    def _append_delta(self, text: str) -> None:
        answer = self.current
        if answer is None or answer.task is None or answer.task.done():
            return  # stray chunk outside a turn (e.g. right after cancellation)
        answer.text += text
        self.on_delta(answer.id, text)
        if self.on_sentence is not None:
            self._tts_buffer += text
            self._tts_buffer = self._flush_sentences(self._tts_buffer)

    # ------------------------------------------------------ confirmation gate

    async def _on_permission(self, params: dict) -> dict:
        """ACP session/request_permission — reads auto, writes wait for y/n."""
        options = params.get("options") or []
        tool_call = params.get("toolCall") or {}
        tool_id = tool_call.get("toolCallId", "")
        title = tool_call.get("title") or self._tool_titles.get(tool_id, tool_id or "tool")
        args = tool_call.get("rawInput") or {}
        if self._auto_approved(title, tool_call.get("kind")):
            return self._verdict(title, args, options, approved=True)
        self.pending_action = PendingAction(
            title, args, asyncio.get_running_loop().create_future()
        )
        self._notify_action(title, args, "pending")
        try:
            verdict = await asyncio.wait_for(
                self.pending_action.decision, _CONFIRM_TIMEOUT_SECS
            )
        except TimeoutError:
            verdict = "deny"
        finally:
            self.pending_action = None
        if verdict == "cancel":  # the turn was interrupted; tear down quietly
            return {"outcome": {"outcome": "cancelled"}}
        return self._verdict(title, args, options, approved=verdict == "approve")

    def _verdict(self, title: str, args: dict, options: list, *, approved: bool) -> dict:
        self._notify_action(title, args, "running" if approved else "denied")
        option_id = _pick_option(options, "allow" if approved else "reject")
        if option_id is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {"outcome": {"outcome": "selected", "optionId": option_id}}

    def _auto_approved(self, title: str, kind: str | None) -> bool:
        if kind in _READONLY_KINDS:
            return True
        globs = [g.strip() for g in self.settings.agent_auto_tools.split(",") if g.strip()]
        return any(fnmatch(title, glob) for glob in globs)

    def _notify_action(self, tool: str, args: dict, status: str) -> None:
        log.info("tool %s %s args=%s", tool, status, args)
        if self.on_action is not None:
            self.on_action(tool, args, status)

    # ------------------------------------------------------------------- tts

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


def _pick_option(options: list, prefix: str) -> str | None:
    """Prefer the `<prefix>_once` permission option, else any `<prefix>_*`."""
    for option in options:
        if option.get("kind") == f"{prefix}_once":
            return option.get("optionId")
    for option in options:
        if str(option.get("kind", "")).startswith(prefix):
            return option.get("optionId")
    return None


def summarize_args(args: dict) -> str:
    """Compact one-line rendering of tool args for the answers timeline."""
    try:
        raw = json.dumps(args, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(args)
    raw = raw.strip("{}")
    return raw if len(raw) <= 60 else raw[:57] + "…"
