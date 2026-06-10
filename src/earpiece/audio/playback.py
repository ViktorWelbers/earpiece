"""Cancellable, gateable TTS playback routed to a specific output device.

Routing invariant (feedback-loop guard #1): this plays on the configured
earpiece output device, which must NOT be part of the Multi-Output Device that
feeds system-audio capture — the assistant must never transcribe itself.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from ..config import SAMPLE_RATE

log = logging.getLogger(__name__)

_BLOCK = 1024  # playback granularity ~64 ms; cancel/gate latency is one block


class PlaybackQueue:
    def __init__(self, device: int | None) -> None:
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._gate = asyncio.Event()
        self._gate.set()  # ungated by default
        self._task: asyncio.Task | None = None
        self.playing = False  # observed by the system-audio STT gate

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="playback")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def enqueue(self, pcm: bytes) -> None:
        """Queue 16 kHz mono s16le PCM for playback."""
        self._queue.put_nowait(pcm)

    def cancel(self) -> None:
        """Drop everything queued; current ~64 ms block finishes (word boundary)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def gate(self, blocked: bool) -> None:
        """Barge-in: blocked=True pauses playback instantly until released."""
        if blocked:
            self._gate.clear()
        else:
            self._gate.set()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        stream = sd.OutputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        stream.start()
        try:
            while True:
                pcm = await self._queue.get()
                buf = np.frombuffer(pcm, dtype=np.int16)
                self.playing = True
                try:
                    for i in range(0, len(buf), _BLOCK):
                        await self._gate.wait()
                        block = buf[i : i + _BLOCK]
                        # sd.OutputStream.write blocks; keep the loop responsive.
                        await loop.run_in_executor(None, stream.write, block.reshape(-1, 1))
                        if self._queue.qsize() == 0 and i + _BLOCK >= len(buf):
                            break
                finally:
                    self.playing = not self._queue.empty()
        finally:
            stream.stop()
            stream.close()
