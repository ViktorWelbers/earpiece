"""Wire-level tests for the minimal ACP client — injected streams, no subprocess."""

import asyncio
import json

import pytest

from earpiece.brain.acp import ACPAgent, ACPError


class FakeWriter:
    """Stands in for the subprocess' stdin StreamWriter."""

    def __init__(self) -> None:
        self.lines: list[dict] = []

    def write(self, data: bytes) -> None:
        for line in data.decode().strip().splitlines():
            self.lines.append(json.loads(line))

    async def drain(self) -> None:
        pass


def make_agent() -> tuple[ACPAgent, FakeWriter, asyncio.StreamReader]:
    agent = ACPAgent("unused")
    reader = asyncio.StreamReader()
    writer = FakeWriter()
    agent._stdout = reader
    agent._stdin = writer  # type: ignore[assignment] — duck-typed
    return agent, writer, reader


def feed(reader: asyncio.StreamReader, msg: dict) -> None:
    reader.feed_data(json.dumps(msg).encode() + b"\n")


async def reply_to(writer: FakeWriter, method: str, result: dict, reader) -> None:
    """Wait for our request to be written, then feed the harness' response."""
    for _ in range(100):
        for line in writer.lines:
            if line.get("method") == method:
                feed(reader, {"jsonrpc": "2.0", "id": line["id"], "result": result})
                return
        await asyncio.sleep(0)
    raise AssertionError(f"client never sent {method}")


async def started_agent() -> tuple[ACPAgent, FakeWriter, asyncio.StreamReader]:
    agent, writer, reader = make_agent()
    start = asyncio.create_task(agent.start())
    await reply_to(writer, "initialize", {"protocolVersion": 1}, reader)
    await start
    return agent, writer, reader


async def test_initialize_handshake():
    agent, writer, reader = await started_agent()
    init = writer.lines[0]
    assert init["method"] == "initialize"
    assert init["params"]["protocolVersion"] == 1
    assert init["params"]["clientInfo"]["name"] == "earpiece"
    await agent.stop()


async def test_prompt_streams_updates_then_resolves():
    agent, writer, reader = await started_agent()
    updates: list[dict] = []
    agent.on_update = lambda params: updates.append(params["update"])

    prompt = asyncio.create_task(agent.prompt("s1", "hello"))
    await asyncio.sleep(0)
    feed(reader, {"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": "s1",
                             "update": {"sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "hi "}}}})
    await reply_to(writer, "session/prompt", {"stopReason": "end_turn"}, reader)
    assert await prompt == "end_turn"
    assert updates[0]["content"]["text"] == "hi "
    sent = next(line for line in writer.lines if line.get("method") == "session/prompt")
    assert sent["params"]["prompt"] == [{"type": "text", "text": "hello"}]
    await agent.stop()


async def test_permission_request_is_answered():
    agent, writer, reader = await started_agent()

    async def grant(params: dict) -> dict:
        assert params["toolCall"]["toolCallId"] == "t1"
        return {"outcome": {"outcome": "selected", "optionId": "ok"}}

    agent.request_permission = grant
    feed(reader, {"jsonrpc": "2.0", "id": 7, "method": "session/request_permission",
                  "params": {"sessionId": "s1", "toolCall": {"toolCallId": "t1"},
                             "options": []}})
    for _ in range(100):
        if any(line.get("id") == 7 and "result" in line for line in writer.lines):
            break
        await asyncio.sleep(0)
    response = next(line for line in writer.lines if line.get("id") == 7)
    assert response["result"]["outcome"]["optionId"] == "ok"
    await agent.stop()


async def test_unsupported_request_gets_method_not_found():
    agent, writer, reader = await started_agent()
    feed(reader, {"jsonrpc": "2.0", "id": 8, "method": "fs/read_text_file",
                  "params": {"path": "/etc/passwd"}})
    for _ in range(100):
        if any(line.get("id") == 8 for line in writer.lines):
            break
        await asyncio.sleep(0)
    response = next(line for line in writer.lines if line.get("id") == 8)
    assert response["error"]["code"] == -32601
    await agent.stop()


async def test_error_response_raises_acp_error():
    agent, writer, reader = await started_agent()
    prompt = asyncio.create_task(agent.prompt("s1", "hello"))
    await asyncio.sleep(0)
    request = next(line for line in writer.lines if line.get("method") == "session/prompt")
    feed(reader, {"jsonrpc": "2.0", "id": request["id"],
                  "error": {"code": -32603, "message": "model exploded"}})
    with pytest.raises(ACPError, match="model exploded"):
        await prompt
    await agent.stop()


async def test_harness_crash_fails_pending_requests():
    agent, writer, reader = await started_agent()
    prompt = asyncio.create_task(agent.prompt("s1", "hello"))
    await asyncio.sleep(0)
    reader.feed_eof()  # harness died
    with pytest.raises(ACPError, match="closed its stdout"):
        await prompt
    await agent.stop()


async def test_cancel_is_a_notification():
    agent, writer, reader = await started_agent()
    await agent.cancel("s1")
    note = next(line for line in writer.lines if line.get("method") == "session/cancel")
    assert "id" not in note
    assert note["params"] == {"sessionId": "s1"}
    await agent.stop()
