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


def _read_config_toml() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None


def _load_config_file() -> dict[str, str]:
    """Flat config-file values, normalized to the strings the env vars would hold.

    Tables ([mcp_servers.*]) are structural, not env-var-shaped — see
    `load_mcp_servers()`.
    """
    return {
        key: ("true" if value else "false") if isinstance(value, bool) else str(value)
        for key, value in _read_config_toml().items()
        if not isinstance(value, dict)
    }


def load_mcp_servers() -> dict[str, dict]:
    """The [mcp_servers.<name>] tables: command/args/env (stdio) or url (HTTP/SSE)."""
    servers = _read_config_toml().get("mcp_servers", {})
    if not isinstance(servers, dict) or not all(
        isinstance(v, dict) for v in servers.values()
    ):
        raise ConfigError("mcp_servers must contain [mcp_servers.<name>] tables")
    return servers


def _toml_value(value: str | bool | list) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_config(
    values: dict[str, str | bool], mcp_servers: dict[str, dict] | None = None
) -> Path:
    """Write the config file (created by `earpiece configure`)."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# earpiece configuration — same keys as the environment variables;",
        "# env vars override anything set here. Regenerate: earpiece configure",
    ]
    lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
    for name, server in (mcp_servers or {}).items():
        lines.append(f"\n[mcp_servers.{name}]")
        env = None
        for key, value in server.items():
            if key == "env":
                env = value  # subtable must come after the plain keys
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        if env:
            lines.append(f"\n[mcp_servers.{name}.env]")
            lines.extend(f"{key} = {_toml_value(value)}" for key, value in env.items())
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass(frozen=True)
class Settings:
    mission: str
    deepgram_api_key: str | None = None
    # devices: None = system default / auto-detect; int index or name substring otherwise
    mic_device: int | str | None = None
    system_device: int | str | None = None
    output_device: int | str | None = None
    stt_engine: str = "deepgram"
    tts_engine: str | None = None  # None = text only; "say" | "elevenlabs"
    endpointing_ms: int = 300
    # --stt whisper: any OpenAI-compatible /audio/transcriptions endpoint
    # (vLLM serving a Whisper model, speaches, LocalAI, OpenAI itself, ...)
    stt_base_url: str | None = None
    stt_api_key: str = "local"
    stt_model: str | None = None
    debug_dump_wav: bool = False
    # the ACP agent harness that answers (and acts): a shell-ish command line,
    # e.g. "opencode acp" | "npx @zed-industries/claude-code-acp" | "npx pi-acp"
    agent_cmd: str | None = None
    agent_cwd: str | None = None  # workspace the harness operates in (default: cwd)
    # comma-separated fnmatch globs of tool names that run without confirmation
    # (read-only tool kinds — read/search/fetch/think — always auto-run)
    agent_auto_tools: str = ""
    # turns before the harness session is compressed and reopened; 0 disables.
    # A long-lived session degenerates: past a few dozen micro-turns the agent
    # stops answering the transcript and starts continuing it, inventing the
    # next "[hh:mm:ss] THEM:" line and replying to itself. Observed onset was
    # gradual — a stray bad answer from ~turn 12, half of them wrong by ~turn 45.
    # 25 keeps well clear of that while limiting how many times the conversation
    # is re-summarized from its own summary.
    agent_session_turns: int = 25
    # {name: {command, args, env} | {url, headers}} — from [mcp_servers.*] tables,
    # forwarded to the harness at session setup
    mcp_servers: dict = field(default_factory=dict)
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
        debug_dump_wav: bool = False,
    ) -> Settings:
        cfg = _load_config_file()

        def get(name: str, default: str | None = None) -> str | None:
            return os.environ.get(name, cfg.get(name, default))

        agent_cmd = get("AGENT_CMD")
        if not agent_cmd:
            raise ConfigError(
                "AGENT_CMD is not set — earpiece forwards answers to an ACP agent "
                "harness. Run `earpiece configure` or set it to e.g. "
                '"opencode acp" | "npx @zed-industries/claude-code-acp" | "npx pi-acp"'
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
            deepgram_api_key=deepgram_key,
            stt_base_url=stt_base_url,
            stt_api_key=get("STT_API_KEY", "local"),
            stt_model=stt_model,
            # devices: CLI flags win, then config file / env, then auto-detect
            mic_device=mic_device if mic_device is not None else get("EARPIECE_MIC_DEVICE"),
            system_device=(
                system_device if system_device is not None else get("EARPIECE_SYSTEM_DEVICE")
            ),
            output_device=(
                output_device if output_device is not None else get("EARPIECE_OUTPUT_DEVICE")
            ),
            stt_engine=stt,
            tts_engine=tts_engine,
            debug_dump_wav=debug_dump_wav,
            agent_cmd=agent_cmd,
            agent_cwd=get("AGENT_CWD"),
            agent_auto_tools=get("AGENT_AUTO_TOOLS", ""),
            agent_session_turns=_int(get("AGENT_SESSION_TURNS"), 25),
            mcp_servers=load_mcp_servers(),
        )


def _int(value: str | None, default: int) -> int:
    """Config/env ints are strings; a malformed one falls back to the default."""
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration; CLI prints these without a traceback."""
