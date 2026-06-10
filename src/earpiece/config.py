"""Settings resolved from a config file + environment variables + CLI flags.

Precedence (lowest to highest): ~/.config/earpiece/config.toml, environment
variables, CLI flags. The config file uses the same key names as the env vars.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
BLOCK_MS = 20
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000  # 320 samples per chunk


def config_path() -> Path:
    return Path(os.environ.get("EARPIECE_CONFIG", "~/.config/earpiece/config.toml")).expanduser()


def _load_config_file() -> dict[str, str]:
    """Config-file values, normalized to the strings the env vars would hold."""
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None
    return {
        key: ("true" if value else "false") if isinstance(value, bool) else str(value)
        for key, value in data.items()
    }


def save_config(values: dict[str, str | bool]) -> Path:
    """Write the config file (created by `earpiece configure`)."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# earpiece configuration — same keys as the environment variables;",
        "# env vars override anything set here. Regenerate: earpiece configure",
    ]
    for key, value in values.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n")
    return path


def _as_bool(raw: str | None, default: bool) -> bool:
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
        cfg = _load_config_file()

        def get(name: str, default: str | None = None) -> str | None:
            return os.environ.get(name, cfg.get(name, default))

        base_url = get("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = get("LLM_API_KEY", "")
        responder_model = get("LLM_RESPONDER_MODEL", "")
        if not api_key:
            raise ConfigError("LLM_API_KEY is not set — run `earpiece configure` (or export it)")
        if not responder_model:
            raise ConfigError(
                "LLM_RESPONDER_MODEL is not set — run `earpiece configure` (or export it)"
            )
        supports_schema = _as_bool(get("LLM_JSON_SCHEMA"), True)
        verify_tls = _as_bool(get("LLM_VERIFY_TLS"), True)

        responder = LLMSlot(
            base_url=base_url,
            api_key=api_key,
            model=responder_model,
            supports_json_schema=supports_schema,
            verify_tls=verify_tls,
        )
        watcher = LLMSlot(
            base_url=get("LLM_WATCHER_BASE_URL", base_url),
            api_key=get("LLM_WATCHER_API_KEY", api_key),
            model=get("LLM_WATCHER_MODEL", responder_model),
            supports_json_schema=supports_schema,
            verify_tls=verify_tls,
        )

        stt = stt_engine or get("EARPIECE_STT", "deepgram")
        deepgram_key = get("DEEPGRAM_API_KEY")
        if stt == "deepgram" and not deepgram_key:
            raise ConfigError("DEEPGRAM_API_KEY is required for --stt deepgram")
        stt_base_url = get("STT_BASE_URL")
        stt_model = get("STT_MODEL")
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
            stt_api_key=get("STT_API_KEY", "local"),
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
