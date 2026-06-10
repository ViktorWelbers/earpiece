"""Provider-agnostic LLM access via OpenAI-compatible chat completions.

Works against OpenAI, Anthropic's OpenAI-compat endpoint, OpenRouter, Groq,
Ollama, vLLM, ... — anything that speaks `POST {base_url}/chat/completions`.

Cache-friendliness rule (provider independent): the conversation prefix must be
byte-stable — frozen system prompt, append-only history, no timestamps/UUIDs in
the system prompt. Callers (TranscriptStore) are responsible for upholding it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import LLMSlot

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

Message = dict  # {"role": ..., "content": ...} — OpenAI chat shape


class LLMHandle:
    """One bound (endpoint, model) slot."""

    def __init__(self, slot: LLMSlot) -> None:
        self.slot = slot
        http_client = None
        if not slot.verify_tls:
            # Self-hosted endpoints often use self-signed certs;
            # LLM_VERIFY_TLS=false opts out of verification.
            http_client = httpx.AsyncClient(verify=False)
        self._client = AsyncOpenAI(
            base_url=slot.base_url, api_key=slot.api_key, http_client=http_client
        )
        # filled from the last response that reported usage (for the status bar)
        self.last_usage: dict = {}

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield content deltas. Cancellation closes the HTTP stream cleanly."""
        stream = await self._client.chat.completions.create(
            model=self.slot.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            async for chunk in stream:
                if chunk.usage is not None:
                    self.last_usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await stream.close()

    async def structured(self, messages: list[Message], schema: type[M]) -> M:
        """One non-streaming call returning a validated `schema` instance.

        Uses response_format=json_schema when the provider supports it; otherwise
        prompt-enforced JSON with one retry on validation failure.
        """
        if self.slot.supports_json_schema:
            resp = await self._client.chat.completions.create(
                model=self.slot.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True,
                    },
                },
            )
            if resp.usage is not None:
                self.last_usage = resp.usage.model_dump()
            return schema.model_validate_json(resp.choices[0].message.content or "{}")

        # Fallback: prompt-enforced JSON + one retry.
        contract = (
            "Reply ONLY with a single JSON object matching this schema, no prose, "
            f"no code fences:\n{json.dumps(schema.model_json_schema())}"
        )
        attempt_messages = [*messages, {"role": "system", "content": contract}]
        last_error: Exception | None = None
        for _ in range(2):
            resp = await self._client.chat.completions.create(
                model=self.slot.model, messages=attempt_messages
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": raw},
                    {"role": "system", "content": f"Invalid: {exc}. Reply with valid JSON only."},
                ]
        raise last_error  # type: ignore[misc]


class LLMClient:
    def __init__(self, responder_slot: LLMSlot, watcher_slot: LLMSlot) -> None:
        self.responder = LLMHandle(responder_slot)
        self.watcher = LLMHandle(watcher_slot)
