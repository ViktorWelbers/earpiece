"""The asyncio task graph. The Orchestrator is the only component that owns
tasks and cancellation authority.

    queues:  mic_audio, sys_audio -> stt tasks -> transcript_events
    tasks:   captures, stt_me, stt_them, brain_loop, playback, tts, console
    state:   responder.current (the one in-flight answer, if any)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from .audio.capture import (
    ChannelCapture,
    find_loopback_device,
    resolve_device,
    validate_system_device,
)
from .audio.playback import PlaybackQueue
from .audio.vad import EnergyVAD
from .brain.responder import Responder
from .brain.transcript import TranscriptStore
from .brain.watcher import Watcher
from .config import Settings
from .events import Action, AudioChunk, Decision, TranscriptEvent
from .llm import LLMClient
from .output.console import ConsoleView, StatusState
from .output.tts import base as tts_base
from .output.tts import macos_say  # noqa: F401 — registers the engine
from .stt import base as stt_base  # engines are lazy-loaded by stt_base.create

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings.responder, settings.watcher)
        self.transcript = TranscriptStore(
            mission=settings.mission, max_context_tokens=settings.max_context_tokens
        )
        mode = "eager" if settings.eager else "normal"
        self.console = ConsoleView(self.transcript, StatusState(mode=mode))
        self.watcher = Watcher(self.llm.watcher, settings.mission)

        # audio plumbing
        self.mic_q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=512)
        self.sys_q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=512)
        self.transcript_q: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self.vad = EnergyVAD()
        self.mic_muted = False

        # voice output (optional)
        self.playback: PlaybackQueue | None = None
        self.tts = None
        if settings.tts_engine:
            out_dev = resolve_device(settings.output_device, kind="output")
            self.playback = PlaybackQueue(out_dev)
            self.tts = tts_base.create(settings.tts_engine, settings)
        self._tts_q: asyncio.Queue[str] = asyncio.Queue()

        self.responder = Responder(
            self.llm.responder,
            self.transcript,
            on_delta=self.console.on_delta,
            on_sentence=(self._enqueue_tts if self.tts else None),
            on_end=self.console.on_end,
            on_error=self._on_llm_error,
        )

        self._tasks: list[asyncio.Task] = []
        self._push_to_ask = asyncio.Event()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        mic_dev = resolve_device(self.settings.mic_device, kind="input")
        sys_dev = (
            resolve_device(self.settings.system_device, kind="input")
            if self.settings.system_device is not None
            else find_loopback_device()
        )
        validate_system_device(sys_dev)

        dump_dir = Path("debug_audio") if self.settings.debug_dump_wav else None
        mic_cap = ChannelCapture(
            "ME", mic_dev, self.mic_q,
            dump_wav_to=(dump_dir / "me.wav") if dump_dir else None,
        )
        sys_cap = ChannelCapture(
            "THEM", sys_dev, self.sys_q,
            dump_wav_to=(dump_dir / "them.wav") if dump_dir else None,
        )

        stt_me = stt_base.create(self.settings.stt_engine, self.settings, "ME")
        stt_them = stt_base.create(self.settings.stt_engine, self.settings, "THEM")

        self.console.start()
        mic_cap.start()
        sys_cap.start()
        self.console.status.mic = True
        self.console.status.system_audio = True
        if self.playback is not None:
            self.playback.start()
            self.console.status.voice = True

        mic_audio = self._audio_iter(self.mic_q, mic=True)
        sys_audio = self._audio_iter(self.sys_q)
        self._tasks = [
            asyncio.create_task(self._pump_stt(stt_me, mic_audio), name="stt-me"),
            asyncio.create_task(self._pump_stt(stt_them, sys_audio), name="stt-them"),
            asyncio.create_task(self._brain_loop(), name="brain"),
            asyncio.create_task(self._status_loop(), name="status"),
        ]
        if self.tts is not None:
            self._tasks.append(asyncio.create_task(self._tts_loop(), name="tts"))

        try:
            await self._stop.wait()
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.responder.interrupt()
            if self.playback is not None:
                await self.playback.stop()
            mic_cap.stop()
            sys_cap.stop()
            self.console.stop()

    def shutdown(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- hotkeys

    def push_to_ask(self) -> None:
        """Force a response now (spacebar)."""
        self._push_to_ask.set()

    def toggle_mute(self) -> None:
        self.mic_muted = not self.mic_muted
        self.console.status.mic_muted = self.mic_muted
        self.console.refresh()

    # --------------------------------------------------------------- audio

    async def _audio_iter(
        self, queue: asyncio.Queue[AudioChunk], *, mic: bool = False
    ) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await queue.get()
            if mic:
                speaking = self.vad.feed(chunk.pcm)
                if self.playback is not None:
                    # Barge-in: never talk over the operator.
                    self.playback.gate(speaking)
                if self.mic_muted:
                    continue
            yield chunk

    async def _pump_stt(self, engine, audio: AsyncIterator[AudioChunk]) -> None:
        async for event in engine.stream(audio):
            self.console.status.stt = True
            # Feedback-loop guard #2: drop system-audio STT while TTS is playing.
            if (
                event.source == "THEM"
                and self.playback is not None
                and self.playback.playing
            ):
                continue
            await self.transcript_q.put(event)

    # --------------------------------------------------------------- brain

    async def _brain_loop(self) -> None:
        """Sequential decision point. Watcher calls are serialized; utterances
        arriving during a call are coalesced into the next one."""
        while True:
            event = await self._next_brain_event()
            if event is not None:
                utterance = self.transcript.add(event)
                self.console.refresh()
                if utterance is None:  # interim — UI only
                    continue

            forced = self._push_to_ask.is_set()
            self._push_to_ask.clear()

            decision = await self._decide(event, forced)
            self.console.status.last_decision = decision.action.value
            self.console.status.decision_reason = decision.reason
            self.console.refresh()

            match decision.action:
                case Action.STAY_SILENT:
                    pass
                case Action.RESPOND:
                    if self.responder.partial_answer is None:
                        await self._start_answer()
                case Action.INTERRUPT_AND_RESPOND:
                    await self._interrupt_current()
                    await self._start_answer()

            if self.transcript.needs_compaction():
                await self.transcript.compact(self.llm.watcher)

    async def _next_brain_event(self) -> TranscriptEvent | None:
        """Wait for a transcript event OR the push-to-ask hotkey."""
        getter = asyncio.create_task(self.transcript_q.get())
        pusher = asyncio.create_task(self._push_to_ask.wait())
        done, pending = await asyncio.wait({getter, pusher}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if getter in done:
            return getter.result()
        return None  # push-to-ask fired

    async def _decide(self, event: TranscriptEvent | None, forced: bool) -> Decision:
        in_flight = self.responder.partial_answer is not None
        if forced:
            action = Action.INTERRUPT_AND_RESPOND if in_flight else Action.RESPOND
            return Decision(action=action, reason="push-to-ask", urgency="high")
        if event is None:
            return Decision(action=Action.STAY_SILENT, reason="no new input", urgency="low")
        if self.settings.eager:
            if event.source != "THEM" and self.settings.eager_source != "both":
                return Decision(
                    action=Action.STAY_SILENT, reason="eager: ME utterance", urgency="low"
                )
            action = Action.INTERRUPT_AND_RESPOND if in_flight else Action.RESPOND
            return Decision(action=action, reason="eager mode", urgency="normal")
        return await self.watcher.decide(self.transcript, self.responder.partial_answer)

    async def _start_answer(self) -> None:
        self.console.on_answer_start()
        self.responder.start()

    def _on_llm_error(self, exc: Exception) -> None:
        self.console.status.notice = f"LLM error: {type(exc).__name__}: {exc} (see earpiece.log)"
        self.console.refresh()

    async def _interrupt_current(self) -> None:
        if self.playback is not None:
            self.playback.cancel()
        # Drop queued-but-unspoken sentences.
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self.responder.interrupt()

    # ----------------------------------------------------------------- tts

    def _enqueue_tts(self, sentence: str) -> None:
        self._tts_q.put_nowait(sentence)

    async def _tts_loop(self) -> None:
        assert self.tts is not None and self.playback is not None
        while True:
            sentence = await self._tts_q.get()
            try:
                async for pcm in self.tts.synthesize(sentence):
                    self.playback.enqueue(pcm)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a failed cue must not kill the loop
                log.exception("TTS synthesis failed")

    # -------------------------------------------------------------- status

    async def _status_loop(self) -> None:
        while True:
            usage = self.llm.responder.last_usage
            if usage:
                self.console.status.prompt_tokens = usage.get("prompt_tokens", 0)
                details = usage.get("prompt_tokens_details") or {}
                self.console.status.cached_tokens = details.get("cached_tokens") or 0
            self.console.refresh()
            await asyncio.sleep(1.0)
