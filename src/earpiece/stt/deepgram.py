"""Deepgram streaming STT over a raw websocket (no SDK dependency).

Interim results map to is_final=False (UI only); `speech_final` utterances map
to is_final=True (drive the brain). Reconnects with backoff on connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets

from ..config import SAMPLE_RATE, Settings
from ..events import AudioChunk, Speaker, TranscriptEvent
from .base import register

log = logging.getLogger(__name__)

_WS_BASE = "wss://api.deepgram.com/v1/listen"


@register("deepgram")
class DeepgramSTT:
    def __init__(self, settings: Settings, source: Speaker) -> None:
        assert settings.deepgram_api_key
        self.api_key = settings.deepgram_api_key
        self.source = source
        self.endpointing_ms = settings.endpointing_ms
        self.connected = False  # surfaced in the status bar

    def _url(self) -> str:
        params = {
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "interim_results": "true",
            "endpointing": self.endpointing_ms,
            "smart_format": "true",
            "model": "nova-3",
        }
        return f"{_WS_BASE}?{urlencode(params)}"

    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptEvent]:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self._url(),
                    additional_headers={"Authorization": f"Token {self.api_key}"},
                ) as ws:
                    self.connected = True
                    backoff = 1.0
                    log.info("deepgram[%s] connected", self.source)
                    sender = asyncio.create_task(self._pump_audio(ws, audio))
                    try:
                        async for raw in ws:
                            event = self._parse(raw)
                            if event is not None:
                                yield event
                    finally:
                        sender.cancel()
                        try:
                            await sender
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect on any transport error
                self.connected = False
                log.warning(
                    "deepgram[%s] disconnected (%s); retry in %.0fs", self.source, exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _pump_audio(self, ws, audio: AsyncIterator[AudioChunk]) -> None:
        async for chunk in audio:
            await ws.send(chunk.pcm)

    def _parse(self, raw: str | bytes) -> TranscriptEvent | None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if msg.get("type") != "Results":
            return None
        alt = (msg.get("channel") or {}).get("alternatives") or [{}]
        text = (alt[0].get("transcript") or "").strip()
        if not text:
            return None
        now = time.monotonic()
        duration = float(msg.get("duration") or 0.0)
        return TranscriptEvent(
            source=self.source,
            text=text,
            is_final=bool(msg.get("speech_final")),
            started_at=now - duration,
            ended_at=now,
        )
