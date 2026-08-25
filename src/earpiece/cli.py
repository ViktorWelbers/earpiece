"""CLI entry point.

    earpiece configure
    earpiece run "help me in this sales discussion" --voice say
    earpiece devices
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import os
import re
import sys
import termios
import tty

import typer
from rich.console import Console
from rich.table import Table

from .brain import session_store
from .brain.session_store import SessionRecord
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
    "AGENT_CMD",
    "AGENT_CWD",
    "AGENT_AUTO_TOOLS",
    "AGENT_SESSION_TURNS",
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

    from .config import load_mcp_servers

    servers = load_mcp_servers()
    if servers:
        mcp_table = Table(title="mcp servers (forwarded to the agent harness)")
        mcp_table.add_column("name")
        mcp_table.add_column("target")
        for name, cfg in servers.items():
            target = cfg.get("url") or " ".join([cfg.get("command", "?"), *cfg.get("args", [])])
            mcp_table.add_row(name, target)
        console.print(mcp_table)


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
    console.print(
        "[bold]agent harness[/bold] — answers (and tool use) are forwarded to an ACP agent;"
        " the harness brings its own model + tool configuration"
    )
    values["AGENT_CMD"] = typer.prompt(
        'Agent command ("opencode acp" | "npx @zed-industries/claude-code-acp" | '
        '"gemini --experimental-acp" | "npx pi-acp")',
        default=os.environ.get("AGENT_CMD", "opencode acp"),
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
    console.print(
        "[dim]optional: give the agent extra MCP tools by adding tables to the config "
        "file, e.g.\n  [mcp_servers.jira]\n  command = \"uvx\"\n  "
        "args = [\"mcp-atlassian\"][/dim]"
    )
    console.print('try it:  [bold]earpiece run "help me with tech trivia"[/bold]')


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


@app.command()
def resume(
    mic_device: str | None = typer.Option(None, help="Mic device index or name substring"),
    system_device: str | None = typer.Option(None, help="System-audio (loopback) device"),
    output_device: str | None = typer.Option(None, help="TTS output device (your earpiece)"),
    stt: str | None = typer.Option(None, help="STT engine: deepgram | whisper"),
    voice: str | None = typer.Option(None, help="TTS engine: say | elevenlabs (default: text)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging to earpiece.log"),
) -> None:
    """Pick a past session to resume (restores the transcript + the agent's memory)."""
    console = Console()
    records = session_store.list_sessions()
    if not records:
        console.print("[yellow]no saved sessions yet[/yellow] — start one with "
                      '[bold]earpiece run "..."[/bold]')
        raise typer.Exit(0)

    table = Table(title="resume a session", show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("mission")
    table.add_column("last active", style="dim")
    table.add_column("turns", justify="right")
    table.add_column("workspace", style="dim")
    for i, rec in enumerate(records, 1):
        mission = rec.mission if len(rec.mission) <= 50 else rec.mission[:49] + "…"
        table.add_row(str(i), mission, session_store.ago(rec.updated_at),
                      str(rec.turns), rec.agent_cwd or "—")
    console.print(table)

    choice = typer.prompt("resume which? (number, or q to quit)", default="1")
    if choice.strip().lower() in ("q", "quit"):
        raise typer.Exit(0)
    try:
        record = records[int(choice) - 1]
    except (ValueError, IndexError):
        err_console.print(f"[red]not a valid choice:[/red] {choice}")
        raise typer.Exit(1) from None

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        filename="earpiece.log",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        settings = Settings.from_env(
            record.mission,
            mic_device=mic_device,
            system_device=system_device,
            output_device=output_device,
            stt_engine=stt,
            tts_engine=voice,
        )
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(f'[green]resuming[/green] "{record.mission}" ({record.turns} turns)')
    try:
        asyncio.run(_main(settings, resume=record))
    except KeyboardInterrupt:
        pass


async def _main(settings: Settings, resume: SessionRecord | None = None) -> None:
    from .audio.capture import DeviceError
    from .brain.acp import ACPError
    from .orchestrator import Orchestrator

    try:
        orch = Orchestrator(settings, resume=resume)
    except (DeviceError, ConfigError, KeyError) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from None

    hotkeys = asyncio.create_task(_hotkey_loop(orch))
    try:
        await orch.run()
    except ACPError as exc:
        err_console.print(f"[red]agent harness error:[/red] {exc}")
        err_console.print(
            "[dim]check AGENT_CMD (earpiece configure show) and that the harness is "
            "installed and configured — e.g. `opencode acp` needs opencode set up with "
            "a model/provider.[/dim]"
        )
        raise typer.Exit(1) from None
    finally:
        hotkeys.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hotkeys


# ANSI escape sequences (arrow/function keys, bare ESC) — stripped before the
# bytes are decoded so they never land in the chat buffer.
_ESC_SEQ = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1bO.|\x1b")

# Scrollback navigation keys → (method name, step). Matched against the escape
# sequences above before they're stripped; PgUp/PgDn page, arrows nudge.
_SCROLL_PAGE = 10
_NAV_KEYS: dict[bytes, tuple[str, int]] = {
    b"\x1b[5~": ("scroll_up", _SCROLL_PAGE),   # PageUp
    b"\x1b[6~": ("scroll_down", _SCROLL_PAGE),  # PageDown
    b"\x1b[A": ("scroll_up", 1), b"\x1bOA": ("scroll_up", 1),       # ↑
    b"\x1b[B": ("scroll_down", 1), b"\x1bOB": ("scroll_down", 1),   # ↓
    b"\x1b[H": ("scroll_to_top", 0), b"\x1b[1~": ("scroll_to_top", 0),
    b"\x1bOH": ("scroll_to_top", 0),
    b"\x1b[F": ("scroll_to_bottom", 0), b"\x1b[4~": ("scroll_to_bottom", 0),
    b"\x1bOF": ("scroll_to_bottom", 0),
}


async def _hotkey_loop(orch) -> None:
    """Always-on chat composer over the raw tty: every key types into the bar,
    enter sends the message straight to the ACP agent (an empty line is
    push-to-ask), backspace edits. PgUp/PgDn (and ↑/↓, Home/End) scroll the
    history; submitting snaps back to live. Slash commands run actions instead
    of sending: /quit (or /q), /mute, /voice. While a tool call is awaiting
    confirmation, y/n approve/deny it (the modal case wins over composing)."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return  # not a tty (tests, pipes) — input disabled
    tty.setcbreak(fd)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    try:
        while True:
            data = await loop.run_in_executor(None, os.read, fd, 256)
            if not data:  # EOF on stdin
                return
            for seq in _ESC_SEQ.findall(data):  # scrollback keys (before they're stripped)
                nav = _NAV_KEYS.get(seq)
                if nav is not None:
                    method, step = nav
                    fn = getattr(orch.console, method)
                    fn(step) if step else fn()
            for ch in decoder.decode(_ESC_SEQ.sub(b"", data)):
                # A pending tool call is modal: y/n resolve it, nothing else.
                if orch.console.status.pending_action and ch in ("y", "n"):
                    orch.resolve_action(ch == "y")
                    continue
                if ch in ("\r", "\n"):  # submit
                    text, buffer = buffer.strip(), ""
                    orch.console.set_input("")
                    orch.console.scroll_to_bottom()  # snap back to live on submit
                    if _dispatch_command(orch, text):
                        if text in ("/quit", "/q"):
                            return
                    elif text:
                        await orch.send_chat(text)
                    else:
                        orch.push_to_ask()
                elif ch in ("\x7f", "\x08"):  # backspace
                    buffer = buffer[:-1]
                    orch.console.set_input(buffer)
                elif ch.isprintable():
                    buffer += ch
                    orch.console.set_input(buffer)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _dispatch_command(orch, text: str) -> bool:
    """Run a /slash command; return True if `text` was a recognized command."""
    match text:
        case "/quit" | "/q":
            orch.shutdown()
        case "/mute":
            orch.toggle_mute()
        case "/voice":
            if orch.playback is not None:
                orch.console.status.voice = not orch.console.status.voice
                orch.playback.gate(not orch.console.status.voice)
        case _:
            return False
    return True


def main() -> None:  # console-script shim
    app()


if __name__ == "__main__":
    main()
