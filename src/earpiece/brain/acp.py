"""Minimal ACP (Agent Client Protocol) client over a subprocess' stdio.

JSON-RPC 2.0, one LF-delimited JSON object per line — only the slice of the
protocol earpiece needs: `initialize`, `session/new`, `session/prompt`,
`session/cancel`, incoming `session/update` notifications and
`session/request_permission` requests. The harness (opencode, claude-code-acp,
pi, gemini --experimental-acp, ...) owns the model endpoint, the tools, and the
conversation context.

Spec: https://agentclientprotocol.com
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


class ACPError(RuntimeError):
    """Protocol/transport failure talking to the agent harness."""


class ACPAgent:
    """One spawned agent harness speaking ACP on its stdio."""

    def __init__(self, command: str, cwd: str | None = None) -> None:
        self.command = command
        self.cwd = cwd
        # set by the owner before start(): update notifications and permission
        # requests from the harness
        self.on_update: Callable[[dict], None] | None = None
        self.request_permission: Callable[[dict], Awaitable[dict]] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        # tests inject these to skip the subprocess
        self._stdin: asyncio.StreamWriter | None = None
        self._stdout: asyncio.StreamReader | None = None

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> dict:
        """Spawn the harness and run the initialize handshake."""
        if self._stdout is None:
            argv = shlex.split(self.command)
            if not argv:
                raise ACPError("AGENT_CMD is empty")
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd,
                )
            except OSError as exc:
                raise ACPError(f"cannot start agent harness {argv[0]!r}: {exc}") from exc
            self._stdin = self._proc.stdin
            self._stdout = self._proc.stdout
            self._stderr_task = asyncio.create_task(
                self._pump_stderr(self._proc.stderr), name="acp-stderr"
            )
        self._reader_task = asyncio.create_task(self._read_loop(), name="acp-reader")
        return await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {},  # no fs/terminal — we are an audio frontend
                "clientInfo": {"name": "earpiece", "version": "0"},
            },
        )

    async def stop(self) -> None:
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(self._proc.wait(), 5.0)

    # ------------------------------------------------------------- requests

    async def new_session(self, cwd: str, mcp_servers: list[dict]) -> str:
        result = await self._request(
            "session/new", {"cwd": cwd, "mcpServers": mcp_servers}
        )
        return result["sessionId"]

    async def prompt(self, session_id: str, text: str) -> str:
        """One turn. Returns the stopReason (end_turn | cancelled | ...).

        Streaming output arrives via on_update while this awaits."""
        result = await self._request(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )
        return result.get("stopReason", "end_turn")

    async def cancel(self, session_id: str) -> None:
        await self._send({"jsonrpc": "2.0", "method": "session/cancel",
                          "params": {"sessionId": session_id}})

    # -------------------------------------------------------------- plumbing

    async def _request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        msg_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        await self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            return await future
        finally:
            self._pending.pop(msg_id, None)

    async def _send(self, msg: dict) -> None:
        if self._stdin is None:
            raise ACPError("agent harness is not running")
        self._stdin.write(json.dumps(msg).encode() + b"\n")
        await self._stdin.drain()

    async def _read_loop(self) -> None:
        assert self._stdout is not None
        while True:
            line = await self._stdout.readline()
            if not line:
                self._fail_pending(ACPError("agent harness closed its stdout (crashed?)"))
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("acp: non-JSON line from harness: %r", line[:200])
                continue
            try:
                self._dispatch(msg)
            except Exception:  # noqa: BLE001 — one bad message must not kill the loop
                log.exception("acp: failed handling %r", msg)

    def _dispatch(self, msg: dict) -> None:
        if "method" not in msg:  # response to one of our requests
            future = self._pending.get(msg.get("id"))
            if future is None or future.done():
                return
            if "error" in msg:
                err = msg["error"]
                future.set_exception(ACPError(f"{err.get('message')} ({err.get('code')})"))
            else:
                future.set_result(msg.get("result") or {})
            return
        if "id" in msg:  # request from the agent — needs a response
            asyncio.create_task(
                self._answer_request(msg), name=f"acp-req-{msg['method']}"
            )
            return
        # notification
        if msg["method"] == "session/update" and self.on_update is not None:
            self.on_update(msg.get("params") or {})

    async def _answer_request(self, msg: dict) -> None:
        method, msg_id = msg["method"], msg["id"]
        if method == "session/request_permission" and self.request_permission is not None:
            try:
                result = await self.request_permission(msg.get("params") or {})
            except Exception:  # noqa: BLE001 — fail safe: never grant on error
                log.exception("permission handler failed")
                result = {"outcome": {"outcome": "cancelled"}}
            await self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})
            return
        # fs/terminal/etc.: we advertised no such capabilities
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"earpiece does not support {method}"},
            }
        )

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    async def _pump_stderr(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            log.debug("harness: %s", line.decode(errors="replace").rstrip())


def acp_mcp_servers(config: dict[str, dict]) -> list[dict]:
    """Map [mcp_servers.*] config tables to ACP session/new mcpServers entries."""
    servers: list[dict] = []
    for name, cfg in config.items():
        if "url" in cfg:
            servers.append(
                {
                    "type": "sse" if cfg.get("transport") == "sse" else "http",
                    "name": name,
                    "url": cfg["url"],
                    "headers": [
                        {"name": str(k), "value": str(v)}
                        for k, v in cfg.get("headers", {}).items()
                    ],
                }
            )
            continue
        servers.append(
            {
                "name": name,
                "command": cfg["command"],
                "args": [str(a) for a in cfg.get("args", [])],
                "env": [
                    {"name": str(k), "value": str(v)}
                    for k, v in cfg.get("env", {}).items()
                ],
            }
        )
    return servers
