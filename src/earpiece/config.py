"""Settings resolved from environment variables + CLI flags (flags win)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SAMPLE_RATE = 16_000
BLOCK_MS = 20
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000  # 320 samples per chunk


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMSlot:
    """One configured model endpoint (responder or watcher)."""

    base_url: str
    api_key: str
    model: str
    supports_json_schema: bool = True
    verify_tls: bool = True  # LLM_VERIFY_TLS=false for self-signed internal endpoints


@dataclass(frozen=True)
class Settings:
    mission: str
    responder: LLMSlot
    watcher: LLMSlot
    deepgram_api_key: str | None = None
    # devices: None = system default / auto-detect; int index or name substring otherwise
    mic_device: int | str | None = None
    system_device: int | str | None = None
    output_device: int | str | None = None
    stt_engine: str = "deepgram"
    tts_engine: str | None = None  # None = text only; "say" | "elevenlabs"
    eager: bool = False
    eager_source: str = "them"  # "them" | "both" — which channels trigger eager answers
    endpointing_ms: int = 300
    # --stt whisper: any OpenAI-compatible /audio/transcriptions endpoint
    # (vLLM serving a Whisper model, speaches, LocalAI, OpenAI itself, ...)
    stt_base_url: str | None = None
    stt_api_key: str = "local"
    stt_model: str | None = None
    # transcript sliding window (approx tokens, estimated len/4)
    max_context_tokens: int = 60_000
    debug_dump_wav: bool = False
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_env(
        mission: str,
        *,
        mic_device: int | str | None = None,
        system_device: int | str | None = None,
        output_device: int | str | None = None,
        stt_engine: str | None = None,
        tts_engine: str | None = None,
        eager: bool = False,
        eager_source: str = "them",
        debug_dump_wav: bool = False,
    ) -> Settings:
        if eager_source not in ("them", "both"):
            raise ConfigError("--eager-source must be 'them' or 'both'")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("LLM_API_KEY", "")
        responder_model = os.environ.get("LLM_RESPONDER_MODEL", "")
        if not api_key:
            raise ConfigError("LLM_API_KEY is required")
        if not responder_model:
            raise ConfigError("LLM_RESPONDER_MODEL is required (e.g. gpt-4.1, claude-opus-4-8)")
        supports_schema = _env_bool("LLM_JSON_SCHEMA", True)
        verify_tls = _env_bool("LLM_VERIFY_TLS", True)

        responder = LLMSlot(
            base_url=base_url,
            api_key=api_key,
            model=responder_model,
            supports_json_schema=supports_schema,
            verify_tls=verify_tls,
        )
        watcher = LLMSlot(
            base_url=os.environ.get("LLM_WATCHER_BASE_URL", base_url),
            api_key=os.environ.get("LLM_WATCHER_API_KEY", api_key),
            model=os.environ.get("LLM_WATCHER_MODEL", responder_model),
            supports_json_schema=supports_schema,
            verify_tls=verify_tls,
        )

        stt = stt_engine or os.environ.get("EARPIECE_STT", "deepgram")
        deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
        if stt == "deepgram" and not deepgram_key:
            raise ConfigError("DEEPGRAM_API_KEY is required for --stt deepgram")
        stt_base_url = os.environ.get("STT_BASE_URL")
        stt_model = os.environ.get("STT_MODEL")
        if stt == "whisper" and not (stt_base_url and stt_model):
            raise ConfigError(
                "--stt whisper needs STT_BASE_URL and STT_MODEL — any OpenAI-compatible "
                "/audio/transcriptions endpoint works (vLLM serving a Whisper model, "
                "speaches, LocalAI, or https://api.openai.com/v1 with model whisper-1)"
            )

        return Settings(
            mission=mission,
            responder=responder,
            watcher=watcher,
            deepgram_api_key=deepgram_key,
            stt_base_url=stt_base_url,
            stt_api_key=os.environ.get("STT_API_KEY", "local"),
            stt_model=stt_model,
            mic_device=mic_device,
            system_device=system_device,
            output_device=output_device,
            stt_engine=stt,
            tts_engine=tts_engine,
            eager=eager,
            eager_source=eager_source,
            debug_dump_wav=debug_dump_wav,
        )


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration; CLI prints these without a traceback."""
