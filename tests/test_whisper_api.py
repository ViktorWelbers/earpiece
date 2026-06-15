"""Segmentation tests for the whisper API engine — no network, no audio hardware.

The transcription call is injected via `transcribe_fn`; tests feed synthetic
PCM (constant-amplitude "speech" blocks vs zero "silence" blocks) through the
energy-VAD endpointing and assert on the emitted TranscriptEvents.
"""

from collections.abc import AsyncIterator

import numpy as np

from earpiece.config import BLOCK_SAMPLES, SAMPLE_RATE, Settings
from earpiece.events import AudioChunk
from earpiece.stt.whisper_api import WhisperAPISTT

LOUD = np.full(BLOCK_SAMPLES, 6000, dtype=np.int16).tobytes()  # ~ -15 dBFS
SILENCE = np.zeros(BLOCK_SAMPLES, dtype=np.int16).tobytes()


def make_engine() -> WhisperAPISTT:
    settings = Settings(
        mission="m",
        agent_cmd="fake --acp",
        stt_base_url="http://localhost:8001/v1",
        stt_model="whisper-test",
    )
    engine = WhisperAPISTT(settings, "THEM")
    engine.received: list[bytes] = []  # type: ignore[attr-defined]

    async def fake_transcribe(pcm: bytes) -> str:
        engine.received.append(pcm)
        return "hello world"

    engine.transcribe_fn = fake_transcribe
    return engine


async def feed(blocks: list[bytes]) -> AsyncIterator[AudioChunk]:
    import asyncio

    for i, pcm in enumerate(blocks):
        await asyncio.sleep(0)  # yield to the event loop, like real capture does
        yield AudioChunk(source="THEM", pcm=pcm, ts=i * 0.02)


async def collect(engine: WhisperAPISTT, blocks: list[bytes]) -> list:
    return [event async for event in engine.stream(feed(blocks))]


async def test_utterance_is_finalized_after_silence():
    engine = make_engine()
    # 5 leading silence, 400 ms speech, 800 ms trailing silence (> 500 ms release)
    events = await collect(engine, [SILENCE] * 5 + [LOUD] * 20 + [SILENCE] * 40)
    finals = [e for e in events if e.is_final]
    assert len(finals) == 1
    assert finals[0].text == "hello world"
    assert finals[0].source == "THEM"
    assert finals[0].started_at < finals[0].ended_at


async def test_preroll_is_included_in_segment():
    engine = make_engine()
    await collect(engine, [SILENCE] * 5 + [LOUD] * 20 + [SILENCE] * 40)
    # transcribed segment must contain more than just the post-attack speech
    # blocks: pre-roll + speech + release tail
    segment_secs = len(engine.received[-1]) / 2 / SAMPLE_RATE
    assert segment_secs > 20 * 0.02

async def test_single_block_blip_is_ignored():
    # a single loud block never reaches the VAD attack threshold
    engine = make_engine()
    events = await collect(engine, [SILENCE] * 5 + [LOUD] * 1 + [SILENCE] * 40)
    assert events == []
    assert engine.received == []


async def test_too_short_speech_is_dropped():
    engine = make_engine()
    # 60 ms of speech: VAD triggers, but segment is below the 250 ms minimum
    events = await collect(engine, [SILENCE] * 5 + [LOUD] * 3 + [SILENCE] * 40)
    assert events == []
    assert engine.received == []


async def test_long_speech_emits_interims_then_final():
    engine = make_engine()
    # ~4 s of continuous speech -> at least one interim before the final
    events = await collect(engine, [LOUD] * 200 + [SILENCE] * 40)
    interims = [e for e in events if not e.is_final]
    finals = [e for e in events if e.is_final]
    assert len(interims) >= 1
    assert len(finals) == 1


async def test_failed_transcription_yields_no_event():
    engine = make_engine()

    async def failing(pcm: bytes) -> str:
        return ""  # engine maps request errors to empty string

    engine.transcribe_fn = failing
    events = await collect(engine, [LOUD] * 20 + [SILENCE] * 40)
    assert events == []


async def test_continuous_speech_is_capped_into_multiple_finals():
    engine = make_engine()
    # ~12 s of continuous speech (600 blocks): the 10 s cap must force a final
    # mid-speech, then the VAD release produces the closing one.
    events = await collect(engine, [LOUD] * 600 + [SILENCE] * 40)
    finals = [e for e in events if e.is_final]
    assert len(finals) == 2
    assert finals[0].ended_at < 10.5  # first final arrives at the cap, not at the end


async def test_failed_final_falls_back_to_interim_text():
    engine = make_engine()
    calls = {"n": 0}

    async def flaky(pcm: bytes) -> str:
        calls["n"] += 1
        return "partial words" if calls["n"] == 1 else ""  # interim ok, final fails

    engine.transcribe_fn = flaky
    # ~4 s speech: one interim succeeds, then the final transcription fails
    events = await collect(engine, [LOUD] * 200 + [SILENCE] * 40)
    finals = [e for e in events if e.is_final]
    assert len(finals) == 1
    assert finals[0].text == "partial words"  # shown interim never hangs grey
