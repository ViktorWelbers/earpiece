"""Whisper over any OpenAI-compatible /audio/transcriptions endpoint.

Nothing is hardcoded into the binary: endpointing (when an utterance starts
and ends) happens locally with an energy VAD; each finalized speech segment is
posted as a small WAV to STT_BASE_URL. Works with vLLM serving a Whisper
model, speaches / faster-whisper-server, LocalAI, or OpenAI's hosted
whisper-1 — fully local operation just means pointing STT_BASE_URL at
localhost.

Liveness invariants (the transcript must never silently lose speech):
- The segmenter task consumes audio and NEVER awaits the network, so slow
  transcription can't back-pressure the capture queue into dropping chunks.
  Transcription runs in a separate worker task fed through a job queue.
- Segments are force-finalized at _MAX_SEGMENT_SECS even if speech continues,
  so continuous talk becomes a steady series of utterances instead of one
  ever-growing grey interim.
- Every segment that showed an interim resolves to a final: if the final
  transcription fails, the last interim text is used; if there is no text at
  all, an empty final is emitted so the UI clears the stale interim.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import wave
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

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
_MAX_SEGMENT_SECS = 10.0  # force-finalize mid-speech so monologues stream steadily


def _to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return buf.getvalue()


@dataclass(frozen=True)
class _Job:
    kind: str  # "interim" | "final"
    pcm: bytes
    started_at: float
    ended_at: float


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
        jobs: asyncio.Queue[_Job | None] = asyncio.Queue()
        out: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        segmenter = asyncio.create_task(self._segment(audio, jobs), name=f"seg-{self.source}")
        worker = asyncio.create_task(self._worker(jobs, out), name=f"stt-{self.source}")
        try:
            while True:
                event = await out.get()
                if event is None:  # worker drained after audio ended
                    break
                yield event
        finally:
            segmenter.cancel()
            worker.cancel()
            for task in (segmenter, worker):
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # -- segmenter: audio -> jobs (no network awaits in here) ----------------

    async def _segment(self, audio: AsyncIterator[AudioChunk], jobs: asyncio.Queue) -> None:
        vad = EnergyVAD()
        preroll: deque[bytes] = deque(maxlen=_PREROLL_BLOCKS)
        buffer: list[bytes] = []
        seg_start = 0.0
        last_interim = 0.0
        try:
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
                    if chunk.ts - seg_start >= _MAX_SEGMENT_SECS:
                        # cap hit: finalize and roll over, carrying a little
                        # context so the boundary word isn't fully clipped
                        jobs.put_nowait(_Job("final", b"".join(buffer), seg_start, chunk.ts))
                        buffer = buffer[-_PREROLL_BLOCKS:]
                        seg_start = chunk.ts
                        last_interim = chunk.ts
                    elif chunk.ts - last_interim >= _INTERIM_EVERY_SECS:
                        last_interim = chunk.ts
                        jobs.put_nowait(_Job("interim", b"".join(buffer), seg_start, chunk.ts))
                else:
                    # VAD released: finalize the segment
                    pcm = b"".join(buffer)
                    buffer.clear()
                    duration = len(pcm) / 2 / SAMPLE_RATE
                    if duration - _RELEASE_SECS >= _MIN_SPEECH_SECS:
                        jobs.put_nowait(_Job("final", pcm, seg_start, chunk.ts))
        finally:
            jobs.put_nowait(None)  # audio ended (or cancelled): let the worker drain

    # -- worker: jobs -> transcript events ------------------------------------

    async def _worker(self, jobs: asyncio.Queue, out: asyncio.Queue) -> None:
        interim_text = ""  # last interim emitted for the in-progress segment
        while True:
            job = await jobs.get()
            if job is None:
                out.put_nowait(None)
                return
            if job.kind == "interim":
                if not jobs.empty():
                    continue  # stale: a newer job is already waiting
                text = await self._transcribe(job.pcm)
                if text:
                    interim_text = text
                    out.put_nowait(self._event(text, job, final=False))
                continue
            # final: guaranteed resolution — fall back to the interim text,
            # and emit an (empty) final even then so a shown interim never hangs
            text = await self._transcribe(job.pcm) or interim_text
            had_interim = bool(interim_text)
            interim_text = ""
            if text or had_interim:
                out.put_nowait(self._event(text, job, final=True))

    def _event(self, text: str, job: _Job, *, final: bool) -> TranscriptEvent:
        return TranscriptEvent(
            source=self.source,
            text=text,
            is_final=final,
            started_at=job.started_at,
            ended_at=job.ended_at,
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
