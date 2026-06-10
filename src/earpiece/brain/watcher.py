"""Per-utterance decision: should the assistant speak right now?

Fail-safe: any LLM or parse error degrades to STAY_SILENT — a missed cue is
better than a broken pipeline.
"""

from __future__ import annotations

import logging

from ..events import Action, Decision
from ..llm import LLMHandle
from .prompts import watcher_system
from .transcript import TranscriptStore

log = logging.getLogger(__name__)

SILENT = Decision(action=Action.STAY_SILENT, reason="watcher error — failing safe", urgency="low")


class Watcher:
    def __init__(self, llm: LLMHandle, mission: str) -> None:
        self.llm = llm
        self.mission = mission

    async def decide(
        self,
        transcript: TranscriptStore,
        partial_answer: str | None,
    ) -> Decision:
        in_flight = (
            f"Assistant answer currently being spoken (partial):\n{partial_answer}"
            if partial_answer
            else "No assistant answer is currently in flight."
        )
        user = (
            f"MISSION: {self.mission}\n\n"
            f"RECENT TRANSCRIPT:\n{transcript.recent_lines()}\n\n"
            f"{in_flight}"
        )
        try:
            decision = await self.llm.structured(
                [
                    {"role": "system", "content": watcher_system()},
                    {"role": "user", "content": user},
                ],
                Decision,
            )
        except Exception:  # noqa: BLE001 — fail safe to silence
            log.exception("watcher call failed")
            return SILENT
        if decision.action is Action.INTERRUPT_AND_RESPOND and partial_answer is None:
            # Contract violation by the model; downgrade.
            return Decision(action=Action.RESPOND, reason=decision.reason, urgency=decision.urgency)
        return decision
