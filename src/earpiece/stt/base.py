"""Pluggable STT engine interface + registry.

One engine instance is created PER CHANNEL (ME and THEM each get their own
stream) — speaker attribution comes from the channel, not diarization.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from ..config import ConfigError, Settings
from ..events import AudioChunk, Speaker, TranscriptEvent


class STTEngine(Protocol):
    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[TranscriptEvent]:
        """Consume audio chunks, yield interim + final transcript events."""
        ...


_REGISTRY: dict[str, Callable[[Settings, Speaker], STTEngine]] = {}

# engine name -> (module, install hint for missing optional deps)
_MODULES: dict[str, tuple[str, str | None]] = {
    "deepgram": ("deepgram", None),
    "whisper": ("whisper_api", None),
}


def register(name: str) -> Callable:
    def deco(factory: Callable[[Settings, Speaker], STTEngine]) -> Callable:
        _REGISTRY[name] = factory
        return factory

    return deco


def create(name: str, settings: Settings, source: Speaker) -> STTEngine:
    if name not in _REGISTRY:
        if name not in _MODULES:
            raise ConfigError(f"Unknown STT engine {name!r}. Available: {sorted(_MODULES)}")
        module, hint = _MODULES[name]
        try:
            importlib.import_module(f"{__package__}.{module}")
        except ImportError as exc:
            extra = f" Install with: {hint}" if hint else ""
            raise ConfigError(
                f"STT engine {name!r} is missing dependencies ({exc}).{extra}"
            ) from exc
    return _REGISTRY[name](settings, source)
