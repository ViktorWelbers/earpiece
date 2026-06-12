"""Deepgram streaming STT over a raw websocket (no SDK dependency).

Deepgram finalizes long utterances in segments: mid-utterance messages with
is_final=true carry text that is never re-sent, and the speech_final endpoint
message carries only the words since the last finalized segment. So segments
are buffered per utterance and concatenated at speech_final — that full text
maps to is_final=True (drives the brain); everything earlier maps to
is_final=False (UI only). Reconnects with backoff on connection loss.
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
        self._segments: list[str] = []  # finalized segments of the current utterance
        self._last_interim = ""  # words seen only in interim results so far

    def _url(self) -> str:
        params = {
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "interim_results": "true",
            "endpointing": self.endpointing_ms,
            # Safety net: if speech_final never fires (noisy channel), deepgram
            # sends UtteranceEnd after this much silence — we flush on it.
            "utterance_end_ms": 1200,
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
                    self._segments.clear()  # don't stitch across reconnects
                    self._last_interim = ""
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
        now = time.monotonic()
        duration = float(msg.get("duration") or 0.0)

        def event(full_text: str, *, final: bool) -> TranscriptEvent:
            return TranscriptEvent(
                source=self.source,
                text=full_text,
                is_final=final,
                started_at=now - duration,
                ended_at=now,
            )

        if msg.get("type") == "UtteranceEnd":
            # speech_final never fired for this utterance — flush what we have,
            # falling back to interim-only words so nothing hangs grey forever.
            full = " ".join(self._segments) or self._last_interim
            self._segments.clear()
            self._last_interim = ""
            return event(full, final=True) if full else None
        if msg.get("type") != "Results":
            return None
        alt = (msg.get("channel") or {}).get("alternatives") or [{}]
        text = (alt[0].get("transcript") or "").strip()

        if msg.get("speech_final"):
            # End of utterance: stitch buffered segments + this message's tail.
            full = " ".join((*self._segments, text)).strip()
            had_interim = bool(self._last_interim)
            self._segments.clear()
            self._last_interim = ""
            if full:
                return event(full, final=True)
            # Empty endpoint after interim-only words: deepgram retracted them —
            # an empty final clears the stale grey line in the UI.
            return event("", final=True) if had_interim else None
        if msg.get("is_final"):
            if text:
                # Segment finalized but the utterance continues — buffer it.
                self._segments.append(text)
                self._last_interim = ""
                return event(" ".join(self._segments), final=False)
            # Finalized as silence: any pending interim words were retracted.
            had_interim = bool(self._last_interim)
            self._last_interim = ""
            if not self._segments and had_interim:
                return event("", final=True)
            return None
        # Plain interim: live preview = buffered segments + in-progress text.
        if not text:
            return None
        self._last_interim = text
        return event(" ".join((*self._segments, text)), final=False)
