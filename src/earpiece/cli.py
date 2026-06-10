"""CLI entry point.

    earpiece configure
    earpiece run "help me in this sales discussion" --voice say
    earpiece devices
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import termios
import tty

import typer
from rich.console import Console
from rich.table import Table

from .config import ConfigError, Settings, _load_config_file, config_path, save_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
err_console = Console(stderr=True)

configure_app = typer.Typer(add_completion=False)
app.add_typer(configure_app, name="configure")


@app.command()
def devices() -> None:
    """List audio devices (use with --mic-device / --system-device / --output-device)."""
    import sounddevice as sd

    table = Table(title="Audio devices")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Notes")

    default_in, default_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        notes = []
        if i == default_in:
            notes.append("default input")
        if i == default_out:
            notes.append("default output")
        if "blackhole" in d["name"].lower():
            notes.append("loopback (system-audio candidate)")
        table.add_row(
            str(i), d["name"], str(d["max_input_channels"]), str(d["max_output_channels"]),
            ", ".join(notes),
        )
    Console().print(table)


@configure_app.callback(invoke_without_command=True)
def configure(ctx: typer.Context) -> None:
    """Interactive setup — answers are stored in ~/.config/earpiece/config.toml."""
    if ctx.invoked_subcommand is None:
        _wizard()


_KNOWN_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_RESPONDER_MODEL",
    "LLM_WATCHER_MODEL",
    "LLM_WATCHER_BASE_URL",
    "LLM_WATCHER_API_KEY",
    "LLM_VERIFY_TLS",
    "LLM_JSON_SCHEMA",
    "EARPIECE_STT",
    "EARPIECE_MIC_DEVICE",
    "EARPIECE_SYSTEM_DEVICE",
    "EARPIECE_OUTPUT_DEVICE",
    "STT_BASE_URL",
    "STT_MODEL",
    "STT_API_KEY",
    "DEEPGRAM_API_KEY",
)


@configure_app.command()
def show() -> None:
    """Print the effective configuration (config file + env-var overrides)."""
    console = Console()
    path = config_path()
    try:
        file_values = _load_config_file()
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from None
    if not path.is_file():
        console.print(f"[dim]no config file at {path} — run `earpiece configure`[/dim]")

    keys = list(_KNOWN_KEYS) + sorted(set(file_values) - set(_KNOWN_KEYS))
    table = Table(title=str(path))
    table.add_column("key")
    table.add_column("value")
    table.add_column("source")
    for key in keys:
        env_value, file_value = os.environ.get(key), file_values.get(key)
        if env_value is None and file_value is None:
            continue
        if env_value is not None:
            source = "env (overrides file)" if file_value is not None else "env"
        else:
            source = "file"
        table.add_row(key, _mask(key, env_value if env_value is not None else file_value), source)
    console.print(table)


def _mask(key: str, value: str) -> str:
    if key.endswith("API_KEY") and value not in ("", "local") and len(value) > 8:
        return f"{value[:4]}…{value[-2:]}"
    return value


def _wizard() -> None:
    console = Console()
    console.print(
        f"[bold]earpiece setup[/bold] — answers are stored in [cyan]{config_path()}[/cyan]"
    )
    console.print("[dim]Environment variables override the file; rerun anytime.[/dim]\n")

    values: dict[str, str | bool] = {}
    values["LLM_BASE_URL"] = typer.prompt(
        "LLM base URL (any OpenAI-compatible endpoint)",
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    values["LLM_API_KEY"] = typer.prompt(
        "LLM API key ('local' works for most self-hosted servers)",
        default=os.environ.get("LLM_API_KEY", "local"),
    )
    values["LLM_VERIFY_TLS"] = typer.confirm(
        "Verify TLS certificates? (answer n for self-signed certs)", default=True
    )

    models = _discover_models(
        values["LLM_BASE_URL"], values["LLM_API_KEY"], values["LLM_VERIFY_TLS"]
    )
    if models:
        console.print(f"[dim]models on this endpoint: {', '.join(models[:10])}[/dim]")
    values["LLM_RESPONDER_MODEL"] = typer.prompt(
        "Responder model (the smart one, streams answers)",
        default=models[0] if models else None,
    )
    values["LLM_WATCHER_MODEL"] = typer.prompt(
        "Watcher model (fast/cheap, decides when to speak)",
        default=values["LLM_RESPONDER_MODEL"],
    )

    stt = ""
    while stt not in ("whisper", "deepgram"):
        stt = typer.prompt("STT engine (whisper | deepgram)", default="whisper").strip().lower()
    values["EARPIECE_STT"] = stt
    if stt == "whisper":
        values["STT_BASE_URL"] = typer.prompt(
            "Whisper endpoint (/v1 of any OpenAI-compatible transcription server)",
            default=os.environ.get("STT_BASE_URL", "http://localhost:8001/v1"),
        )
        values["STT_MODEL"] = typer.prompt(
            "Whisper model",
            default=os.environ.get("STT_MODEL", "Systran/faster-whisper-small"),
        )
    else:
        values["DEEPGRAM_API_KEY"] = typer.prompt("Deepgram API key")

    path = save_config(values)
    console.print(f"\n[green]saved[/green] {path}")
    console.print('try it:  [bold]earpiece run "help me with tech trivia" --eager[/bold]')


def _discover_models(base_url: str, api_key: str, verify: bool) -> list[str]:
    """Best-effort model listing for wizard defaults; empty on any failure."""
    import httpx

    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            verify=verify,
            timeout=5.0,
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:  # noqa: BLE001 — discovery is a convenience, never fatal
        return []


@app.command()
def run(
    mission: str = typer.Argument(..., help='e.g. "help me in this sales discussion"'),
    mic_device: str | None = typer.Option(None, help="Mic device index or name substring"),
    system_device: str | None = typer.Option(None, help="System-audio (loopback) device"),
    output_device: str | None = typer.Option(None, help="TTS output device (your earpiece)"),
    stt: str | None = typer.Option(
        None, help="STT engine: deepgram | whisper (default: from config file, else deepgram)"
    ),
    voice: str | None = typer.Option(None, help="TTS engine: say | elevenlabs (default: text)"),
    eager: bool = typer.Option(False, "--eager", help="Respond on every THEM utterance"),
    eager_source: str = typer.Option(
        "them", help="Eager trigger channel: them | both (both = answer your own mic too)"
    ),
    debug_dump_wav: bool = typer.Option(False, help="Dump captured audio to debug_audio/*.wav"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging to earpiece.log"),
) -> None:
    """Start the assistant with a mission."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        filename="earpiece.log",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    def make_settings() -> Settings:
        return Settings.from_env(
            mission,
            mic_device=mic_device,
            system_device=system_device,
            output_device=output_device,
            stt_engine=stt,
            tts_engine=voice,
            eager=eager,
            eager_source=eager_source,
            debug_dump_wav=debug_dump_wav,
        )

    try:
        settings = make_settings()
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        if not (sys.stdin.isatty() and typer.confirm("Run interactive setup now?", default=True)):
            raise typer.Exit(1) from None
        _wizard()
        try:
            settings = make_settings()
        except ConfigError as exc2:
            err_console.print(f"[red]config error:[/red] {exc2}")
            raise typer.Exit(1) from None

    try:
        asyncio.run(_main(settings))
    except KeyboardInterrupt:
        pass


async def _main(settings: Settings) -> None:
    from .audio.capture import DeviceError
    from .orchestrator import Orchestrator

    try:
        orch = Orchestrator(settings)
    except (DeviceError, ConfigError, KeyError) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    hotkeys = asyncio.create_task(_hotkey_loop(orch))
    try:
        await orch.run()
    finally:
        hotkeys.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hotkeys


async def _hotkey_loop(orch) -> None:
    """Raw-tty hotkeys: space=push-to-ask, m=mute mic, v=toggle voice gate, q=quit."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return  # not a tty (tests, pipes) — hotkeys disabled
    tty.setcbreak(fd)
    try:
        while True:
            key = await loop.run_in_executor(None, sys.stdin.read, 1)
            match key:
                case " ":
                    orch.push_to_ask()
                case "m":
                    orch.toggle_mute()
                case "v":
                    if orch.playback is not None:
                        orch.console.status.voice = not orch.console.status.voice
                        orch.playback.gate(not orch.console.status.voice)
                case "q":
                    orch.shutdown()
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:  # console-script shim
    app()


if __name__ == "__main__":
    main()
