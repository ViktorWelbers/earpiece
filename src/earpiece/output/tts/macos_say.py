"""Zero-setup local TTS via the macOS `say` command.

Synthesizes each sentence chunk to a temp WAVE-compatible file, then converts
to 16 kHz mono s16le PCM with afconvert. Fully local; latency is acceptable
for short cues. Upgrade path: elevenlabs streaming engine, same interface.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from ...config import SAMPLE_RATE, Settings
from .base import register

log = logging.getLogger(__name__)


@register("say")
class MacosSayTTS:
    def __init__(self, settings: Settings) -> None:
        self.voice = settings.extra.get("say_voice")  # None = system default voice

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        with tempfile.TemporaryDirectory(prefix="earpiece-tts-") as tmp:
            aiff = Path(tmp) / "cue.aiff"
            wav = Path(tmp) / "cue.wav"
            say_cmd = ["say", "-o", str(aiff)]
            if self.voice:
                say_cmd += ["-v", self.voice]
            say_cmd.append(text)
            await _run(say_cmd)
            await _run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    f"LEI16@{SAMPLE_RATE}",
                    "-c",
                    "1",
                    str(aiff),
                    str(wav),
                ]
            )
            with wave.open(str(wav), "rb") as f:
                while True:
                    frames = f.readframes(SAMPLE_RATE // 4)  # 250 ms chunks
                    if not frames:
                        break
                    yield frames


async def _run(cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {stderr.decode().strip()}")
