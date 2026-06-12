"""Microphone and system-audio capture.

Invariant: capture is SHARED MODE — we never request exclusive/hog access, so
other applications (Zoom, Meet, games) keep full use of the same devices.
PortAudio on macOS opens Core Audio devices shared by default; nothing in this
module may change that.
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from ..config import BLOCK_SAMPLES, SAMPLE_RATE
from ..events import AudioChunk, Speaker

log = logging.getLogger(__name__)


def resolve_device(spec: int | str | None, *, kind: str) -> int | None:
    """Resolve a device index/name-substring to a device index.

    None means: system default (mic/output) — except system-audio capture,
    where we auto-detect a BlackHole-style loopback device.
    """
    if spec is None:
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    matches = [
        i
        for i, d in enumerate(sd.query_devices())
        if spec.lower() in d["name"].lower()
        and d["max_input_channels" if kind == "input" else "max_output_channels"] > 0
    ]
    if not matches:
        raise DeviceError(f"No {kind} device matching {spec!r}. Run `earpiece devices`.")
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {sd.query_devices(i)['name']}" for i in matches)
        raise DeviceError(f"Ambiguous {kind} device {spec!r}: {names}")
    return matches[0]


def find_loopback_device() -> int:
    """Auto-detect a BlackHole-style loopback input for system audio."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "blackhole" in d["name"].lower():
            return i
    raise DeviceError(
        "No BlackHole loopback device found for system-audio capture.\n"
        "Install it with:  brew install blackhole-2ch\n"
        "then create a Multi-Output Device (see README) — or pass --system-device."
    )


class DeviceError(RuntimeError):
    pass


class ChannelCapture:
    """One input device -> asyncio.Queue[AudioChunk], tagged with a speaker."""

    def __init__(
        self,
        source: Speaker,
        device: int | None,
        out: asyncio.Queue[AudioChunk],
        *,
        dump_wav_to: Path | None = None,
    ) -> None:
        self.source = source
        self.device = device
        self.out = out
        self.dropped = 0  # chunks lost to backpressure — surfaced in the status bar
        self._last_drop_log = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._wav: wave.Wave_write | None = None
        if dump_wav_to is not None:
            dump_wav_to.parent.mkdir(parents=True, exist_ok=True)
            self._wav = wave.open(str(dump_wav_to), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(SAMPLE_RATE)

    def _callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            log.warning("%s capture status: %s", self.source, status)
        pcm = indata[:, 0].tobytes() if indata.ndim > 1 else indata.tobytes()
        if self._wav is not None:
            self._wav.writeframes(pcm)
        chunk = AudioChunk(source=self.source, pcm=pcm, ts=time.monotonic())
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._put_nowait_dropping, chunk)

    def _put_nowait_dropping(self, chunk: AudioChunk) -> None:
        try:
            self.out.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop oldest under backpressure — realtime beats completeness.
            # This means a consumer is too slow; it MUST show up in the log.
            self.dropped += 1
            if chunk.ts - self._last_drop_log > 5.0:
                self._last_drop_log = chunk.ts
                log.warning(
                    "%s capture dropping audio under backpressure (%d chunks lost so far) — "
                    "the STT consumer is falling behind",
                    self.source,
                    self.dropped,
                )
            try:
                self.out.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.out.put_nowait(chunk)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()
        name = sd.query_devices(self.device)["name"] if self.device is not None else "default"
        log.info("%s capture started on %s (shared mode)", self.source, name)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._wav is not None:
            self._wav.close()
            self._wav = None


def validate_system_device(device: int) -> None:
    """Warn when the system-audio device looks like a physical mic, not a loopback."""
    name = sd.query_devices(device)["name"]
    if not any(hint in name.lower() for hint in ("blackhole", "loopback", "soundflower")):
        log.warning(
            "System-audio device %r doesn't look like a loopback device — "
            "you may be capturing a physical microphone instead of system audio.",
            name,
        )
