"""The asyncio task graph. The Orchestrator is the only component that owns
tasks and cancellation authority.

    queues:  mic_audio, sys_audio -> stt tasks -> transcript_events
    tasks:   captures, stt_me, stt_them, brain_loop, playback, tts, console
    state:   responder.current (the one in-flight answer, if any)
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from .config import Settings
from .events import Action, AudioChunk, Decision, TranscriptEvent
from .llm import LLMHandle
from .output.console import ConsoleView, StatusState
from .output.tts import base as tts_base
from .output.tts import macos_say  # noqa: F401 — registers the engine
from .stt import base as stt_base  # engines are lazy-loaded by stt_base.create

log = logging.getLogger(__name__)

# An interim that hasn't changed for this long will never be finalized by its
# engine (stalled transcription, retraction bug, dead connection) — promote it.
_STALE_INTERIM_SECS = 6.0


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # only the responder (the ACP agent) talks to a model now; this plain
        # OpenAI-compatible handle is kept solely to summarize a long transcript
        self.summarizer = LLMHandle(settings.watcher)
        self.transcript = TranscriptStore(
            mission=settings.mission, max_context_tokens=settings.max_context_tokens
        )
        self.console = ConsoleView(self.transcript, StatusState())

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
            settings,
            self.transcript,
            on_delta=self.console.on_delta,
            on_sentence=(self._enqueue_tts if self.tts else None),
            on_end=self.console.on_end,
            on_error=self._on_llm_error,
            on_action=self.console.on_action,
        )

        self._tasks: list[asyncio.Task] = []
        self._push_to_ask = asyncio.Event()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        # spawn the ACP harness before anything else: a broken AGENT_CMD should
        # be one clear startup error, not a failure on the first answer
        await self.responder.start_agent()

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
        self._captures = [mic_cap, sys_cap]  # read by the status loop (drop counts)

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
            asyncio.create_task(self._interim_watchdog(), name="interim-watchdog"),
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
            await self.responder.close()
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

    def resolve_action(self, approved: bool) -> None:
        """y/n hotkey verdict on a tool call awaiting confirmation."""
        if self.responder.resolve_action(approved):
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
            elif self.playback is not None and self.playback.suppress_capture:
                # Feedback-loop guard: our own TTS loops back through the
                # Multi-Output Device — keep it out of the THEM channel.
                continue
            yield chunk

    async def _pump_stt(self, engine, audio: AsyncIterator[AudioChunk]) -> None:
        async for event in engine.stream(audio):
            self.console.status.stt = True
            # Belt-and-suspenders: the chunk-level guard in _audio_iter should
            # make this unreachable, but never transcribe ourselves.
            if (
                event.source == "THEM"
                and self.playback is not None
                and self.playback.suppress_capture
            ):
                continue
            await self.transcript_q.put(event)

    # ------------------------------------------------------------ watchdog

    async def _interim_watchdog(self) -> None:
        """Liveness guarantee, independent of any STT engine's quirks: a grey
        interim line that stops updating is promoted to a final utterance so
        its text always reaches the transcript and the LLM."""
        while True:
            await asyncio.sleep(1.0)
            for event in self._stale_interim_events():
                await self.transcript_q.put(event)

    def _stale_interim_events(self) -> list[TranscriptEvent]:
        now = time.monotonic()
        events: list[TranscriptEvent] = []
        for speaker, updated in list(self.transcript.interim_updated.items()):
            if now - updated < _STALE_INTERIM_SECS:
                continue
            text = self.transcript.interim.get(speaker, "").strip()
            if not text:
                self.transcript.interim.pop(speaker, None)
                self.transcript.interim_updated.pop(speaker, None)
                continue
            log.warning("promoting stale %s interim to final: %r", speaker, text)
            events.append(
                TranscriptEvent(
                    source=speaker, text=text, is_final=True, started_at=updated, ended_at=now
                )
            )
            # the final flows through the brain loop and clears the interim;
            # bump the timestamp so the next tick doesn't promote it again
            self.transcript.interim_updated[speaker] = now
        return events

    # --------------------------------------------------------------- brain

    async def _brain_loop(self) -> None:
        """Sequential decision point: each finalized utterance triggers an
        answer turn; utterances arriving during a turn are coalesced."""
        while True:
            event = await self._next_brain_event()
            if event is not None:
                utterance = self.transcript.add(event)
                self.console.refresh()
                if utterance is None:  # interim — UI only
                    continue

            forced = self._push_to_ask.is_set()
            self._push_to_ask.clear()

            decision = self._decide(event, forced)
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
                await self.transcript.compact(self.summarizer)

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

    def _decide(self, event: TranscriptEvent | None, forced: bool) -> Decision:
        """Always answer: every finalized utterance (either speaker) gets a
        response; an in-flight answer is interrupted so the latest line wins."""
        in_flight = self.responder.partial_answer is not None
        action = Action.INTERRUPT_AND_RESPOND if in_flight else Action.RESPOND
        if forced:
            return Decision(action=action, reason="push-to-ask", urgency="high")
        if event is None:
            return Decision(action=Action.STAY_SILENT, reason="no new input", urgency="low")
        return Decision(action=action, reason="new utterance", urgency="normal")

    async def _start_answer(self, prompt_text: str | None = None) -> None:
        self.console.on_answer_start()
        self.responder.start(prompt_text)

    async def send_chat(self, text: str) -> None:
        """Operator typed a message in the chat box — forward it straight to the
        ACP agent, pre-empting any in-flight answer."""
        if self.responder.partial_answer is not None:
            await self._interrupt_current()
        self.console.on_chat(text)
        await self._start_answer(prompt_text=text)

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
            usage = self.responder.last_usage
            if usage:
                self.console.status.prompt_tokens = usage.get("prompt_tokens", 0)
                details = usage.get("prompt_tokens_details") or {}
                self.console.status.cached_tokens = details.get("cached_tokens") or 0
            self.console.status.dropped_chunks = sum(
                cap.dropped for cap in getattr(self, "_captures", [])
            )
            self.console.refresh()
            await asyncio.sleep(1.0)
