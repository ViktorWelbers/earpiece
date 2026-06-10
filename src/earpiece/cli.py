"""CLI entry point.

    earpiece "help me in this sales discussion" --voice say
    earpiece devices
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import termios
import tty

import typer
from rich.console import Console
from rich.table import Table

from .config import ConfigError, Settings

app = typer.Typer(add_completion=False, no_args_is_help=True)
err_console = Console(stderr=True)


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


@app.command()
def run(
    mission: str = typer.Argument(..., help='e.g. "help me in this sales discussion"'),
    mic_device: str | None = typer.Option(None, help="Mic device index or name substring"),
    system_device: str | None = typer.Option(None, help="System-audio (loopback) device"),
    output_device: str | None = typer.Option(None, help="TTS output device (your earpiece)"),
    stt: str = typer.Option("deepgram", help="STT engine: deepgram | whisper"),
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
    try:
        settings = Settings.from_env(
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
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
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
