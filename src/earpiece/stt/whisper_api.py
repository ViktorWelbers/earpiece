"""Whisper over any OpenAI-compatible /audio/transcriptions endpoint.

Nothing is hardcoded into the binary: endpointing (when an utterance starts
and ends) happens locally with an energy VAD; each finalized speech segment is
posted as a small WAV to STT_BASE_URL. Works with vLLM serving a Whisper
model, speaches / faster-whisper-server, LocalAI, or OpenAI's hosted
whisper-1 — fully local operation just means pointing STT_BASE_URL at
localhost.

Long utterances get interim transcriptions (~every 2.5 s of ongoing speech) so
the UI stays live; the final segment is transcribed once more as a whole.
"""

from __future__ import annotations

import io
import logging
import wave
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable

from openai import AsyncOpenAI

from ..audio.vad import EnergyVAD
from ..config import SAMPLE_RATE, Settings
from ..events import AudioChunk, Speaker, TranscriptEvent
from .base import register

log = logging.getLogger(__name__)

_PREROLL_BLOCKS = 10  # ~200 ms kept before VAD attack so word onsets aren't clipped
_RELEASE_SECS = 0.5  # must match EnergyVAD release_blocks (25 * 20 ms)
_MIN_SPEECH_SECS = 0.25  # drop blips shorter than this
_INTERIM_EVERY_SECS = 2.5


def _to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return buf.getvalue()


@register("whisper")
class WhisperAPISTT:
    def __init__(self, settings: Settings, source: Speaker) -> None:
        assert settings.stt_base_url and settings.stt_model
        self.source = source
        self.model = settings.stt_model
        self._client = AsyncOpenAI(base_url=settings.stt_base_url, api_key=settings.stt_api_key)
        self.connected = False  # surfaced in the status bar
        # injectable for tests: async (pcm) -> text
        self.transcribe_fn: Callable[[bytes], Awaitable[str]] | None = None

    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptEvent]:
        vad = EnergyVAD()
        preroll: deque[bytes] = deque(maxlen=_PREROLL_BLOCKS)
        buffer: list[bytes] = []
        seg_start = 0.0
        last_interim = 0.0

        async for chunk in audio:
            active = vad.feed(chunk.pcm)

            if not active and not buffer:
                preroll.append(chunk.pcm)
                continue

            if active and not buffer:
                # speech onset: prepend pre-roll so the first word isn't clipped
                buffer.extend(preroll)
                preroll.clear()
                seg_start = chunk.ts
                last_interim = chunk.ts

            buffer.append(chunk.pcm)

            if active:
                if chunk.ts - last_interim >= _INTERIM_EVERY_SECS:
                    last_interim = chunk.ts
                    text = await self._transcribe(b"".join(buffer))
                    if text:
                        yield TranscriptEvent(
                            source=self.source,
                            text=text,
                            is_final=False,
                            started_at=seg_start,
                            ended_at=chunk.ts,
                        )
            else:
                # VAD released: finalize the segment
                pcm = b"".join(buffer)
                buffer.clear()
                duration = len(pcm) / 2 / SAMPLE_RATE
                if duration - _RELEASE_SECS >= _MIN_SPEECH_SECS:
                    text = await self._transcribe(pcm)
                    if text:
                        yield TranscriptEvent(
                            source=self.source,
                            text=text,
                            is_final=True,
                            started_at=seg_start,
                            ended_at=chunk.ts,
                        )

    async def _transcribe(self, pcm: bytes) -> str:
        if self.transcribe_fn is not None:
            return await self.transcribe_fn(pcm)
        try:
            result = await self._client.audio.transcriptions.create(
                model=self.model,
                file=("segment.wav", _to_wav(pcm), "audio/wav"),
            )
            self.connected = True
            return (result.text or "").strip()
        except Exception:  # noqa: BLE001 — a failed segment must not kill the stream
            log.exception("whisper[%s] transcription request failed", self.source)
            self.connected = False
            return ""
