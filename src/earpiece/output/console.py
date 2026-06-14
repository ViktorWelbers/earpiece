"""rich.Live three-pane console: transcript | answer timeline | status bar."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..brain.responder import NOTHING, summarize_args
from ..brain.transcript import TranscriptStore

_SPEAKER_STYLE = {"ME": "bold cyan", "THEM": "bold magenta"}
_ACTION_ICON = {"pending": "⏸", "running": "⚙", "done": "✓", "denied": "✗", "failed": "✗"}
_ACTION_STYLE = {
    "pending": "yellow",
    "running": "cyan",
    "done": "green",
    "denied": "red",
    "failed": "red",
}


@dataclass
class StatusState:
    mic: bool = False
    system_audio: bool = False
    stt: bool = False
    voice: bool = False
    mic_muted: bool = False
    mode: str = "normal"  # normal | eager
    last_decision: str = "—"
    decision_reason: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    dropped_chunks: int = 0  # capture backpressure losses — should stay 0
    notice: str = ""
    pending_action: str = ""  # tool name awaiting the operator's y/n


@dataclass
class AnswerEntry:
    """One finished answer in the timeline."""

    wall_time: str
    text: str
    interrupted: bool


@dataclass
class ActionEntry:
    """One tool call in the timeline (pending → running → done/denied)."""

    wall_time: str
    tool: str
    args_summary: str
    status: str


@dataclass
class ConsoleView:
    transcript: TranscriptStore
    status: StatusState = field(default_factory=StatusState)
    answers: list[AnswerEntry | ActionEntry] = field(default_factory=list)
    answer_text: str = ""  # the one in-flight (streaming) answer
    _live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=Console(),
            refresh_per_second=10,
            screen=True,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # -- answer pane callbacks (wired to Responder) ----------------------

    def on_answer_start(self) -> None:
        self.answer_text = ""
        self.status.pending_action = ""
        self.refresh()

    def on_delta(self, _answer_id: str, delta: str) -> None:
        self.answer_text += delta
        self.refresh()

    def on_end(self, _answer_id: str, interrupted: bool) -> None:
        text = self.answer_text.strip()
        self.answer_text = ""
        self.status.pending_action = ""  # the turn is over; nothing to confirm
        if text and text != NOTHING:
            self.answers.append(AnswerEntry(time.strftime("%H:%M:%S"), text, interrupted))
        self.refresh()

    def on_action(self, tool: str, args: dict, status: str) -> None:
        """Tool-call lifecycle from the responder: pending → running → done/denied."""
        entry = next(
            (
                e
                for e in reversed(self.answers)
                if isinstance(e, ActionEntry)
                and e.tool == tool
                and e.status in ("pending", "running")
            ),
            None,
        )
        if entry is None:
            self.answers.append(
                ActionEntry(time.strftime("%H:%M:%S"), tool, summarize_args(args), status)
            )
        else:
            entry.status = status
            if args and not entry.args_summary:  # later updates may carry the input
                entry.args_summary = summarize_args(args)
        self.status.pending_action = tool if status == "pending" else ""
        self.refresh()

    # -- rendering --------------------------------------------------------

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="main", ratio=1),
            Layout(name="status", size=4),
        )
        layout["main"].split_row(
            Layout(self._transcript_panel(), name="transcript", ratio=1),
            Layout(self._answer_panel(), name="answer", ratio=1),
        )
        layout["status"].update(self._status_panel())
        return layout

    def _transcript_panel(self) -> Panel:
        lines: list[Text] = []
        for utt in self.transcript.utterances[-30:]:
            text = Text()
            text.append(f"[{utt.wall_time}] ", style="dim")
            text.append(f"{utt.speaker}: ", style=_SPEAKER_STYLE[utt.speaker])
            text.append(utt.text)
            lines.append(text)
        for speaker, interim in self.transcript.interim.items():
            text = Text()
            text.append("▌ ", style="dim")
            text.append(f"{speaker}: {interim}", style="dim italic")
            lines.append(text)
        return Panel(Group(*lines) if lines else Text("listening…", style="dim"),
                     title="transcript", border_style="blue")

    def _answer_panel(self) -> Panel:
        blocks: list[Text] = []
        for entry in self.answers[-8:]:
            if isinstance(entry, ActionEntry):
                line = Text()
                line.append(f"[{entry.wall_time}] ", style="dim")
                label = f"{_ACTION_ICON[entry.status]} {entry.tool}({entry.args_summary})"
                if entry.status in ("pending", "denied"):
                    label += f" — {entry.status}"
                line.append(label, style=_ACTION_STYLE[entry.status])
                blocks.append(line)
                continue
            text = Text()
            text.append(f"[{entry.wall_time}] ", style="dim")
            if entry.interrupted:
                text.append("[interrupted] ", style="yellow")
            text.append(entry.text)
            blocks.append(text)
            blocks.append(Text(""))
        if self.answer_text:
            live = Text()
            live.append("▌ ", style="green")
            live.append(self.answer_text)
            blocks.append(live)
        elif blocks and not blocks[-1].plain:
            blocks.pop()  # drop trailing spacer
        body = Group(*blocks) if blocks else Text("—", style="dim")
        return Panel(body, title="answers", border_style="green")

    def _status_panel(self) -> Panel:
        s = self.status

        def dot(ok: bool) -> str:
            return "[green]●[/green]" if ok else "[red]●[/red]"

        mic_label = "mic[dim](muted)[/dim]" if s.mic_muted else "mic"
        line1 = (
            f"{dot(s.mic)} {mic_label}  {dot(s.system_audio)} sys  {dot(s.stt)} stt  "
            f"{dot(s.voice)} voice  |  mode: {s.mode}  |  decision: [bold]{s.last_decision}[/bold]"
        )
        if s.pending_action:  # a tool call is waiting on the operator — nothing matters more
            line2 = (
                f"[bold yellow]confirm: {s.pending_action} — press y to run, n to deny"
                f"[/bold yellow]"
            )
        else:
            line2 = f"[dim]{s.notice or s.decision_reason}[/dim]"  # errors win over decisions
        line3 = (
            f"[dim]prompt {s.prompt_tokens / 1000:.1f}k tok "
            f"(cached {s.cached_tokens / 1000:.1f}k)[/dim]"
        )
        if s.dropped_chunks:
            line3 += f"  [red]audio dropped: {s.dropped_chunks} chunks[/red]"
        lines = Group(*(Text.from_markup(line) for line in (line1, line2, line3)))
        return Panel(lines, border_style="white")
