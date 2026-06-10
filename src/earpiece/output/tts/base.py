"""Pluggable TTS engine interface + registry.

Engines synthesize text into 16 kHz mono s16le PCM chunks; playback (device
routing, cancellation, barge-in gating) lives in audio/playback.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Protocol

from ...config import Settings


class TTSEngine(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks (16 kHz mono s16le) for the given text."""
        ...


_REGISTRY: dict[str, Callable[[Settings], TTSEngine]] = {}


def register(name: str) -> Callable:
    def deco(factory: Callable[[Settings], TTSEngine]) -> Callable:
        _REGISTRY[name] = factory
        return factory

    return deco


def create(name: str, settings: Settings) -> TTSEngine:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown TTS engine {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](settings)
