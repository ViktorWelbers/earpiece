"""Pipeline vocabulary: every queue between stages carries exactly one of these."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

Speaker = Literal["ME", "THEM"]


@dataclass(frozen=True)
class AudioChunk:
    """capture -> STT"""

    source: Speaker
    pcm: bytes  # 16 kHz mono s16le
    ts: float  # monotonic capture time


@dataclass(frozen=True)
class TranscriptEvent:
    """STT -> transcript store. Interim results update the UI only; finals drive the brain."""

    source: Speaker
    text: str
    is_final: bool
    started_at: float
    ended_at: float


class Action(StrEnum):
    STAY_SILENT = "stay_silent"
    RESPOND = "respond"
    INTERRUPT_AND_RESPOND = "interrupt_and_respond"


class Decision(BaseModel):
    """Watcher output, parsed from (structured) JSON."""

    action: Action
    reason: str
    urgency: Literal["low", "normal", "high"] = "normal"


@dataclass(frozen=True)
class AnswerDelta:
    """responder -> console + TTS"""

    answer_id: str
    text: str


@dataclass(frozen=True)
class AnswerEnd:
    answer_id: str
    interrupted: bool
