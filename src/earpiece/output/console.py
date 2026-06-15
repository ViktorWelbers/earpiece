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
    call_id: str  # harness toolCallId — identity for status updates
    tool: str
    args_summary: str
    status: str


@dataclass
class ChatEntry:
    """One operator message typed into the chat box and forwarded to the agent."""

    wall_time: str
    text: str


@dataclass
class ConsoleView:
    transcript: TranscriptStore
    status: StatusState = field(default_factory=StatusState)
    answers: list[AnswerEntry | ActionEntry | ChatEntry] = field(default_factory=list)
    answer_text: str = ""  # the one in-flight (streaming) answer
    input_buffer: str = ""  # the always-on chat composer
    scroll: int = 0  # display lines scrolled up from the live bottom (0 = follow)
    _console: Console | None = None
    _live: Live | None = None

    def start(self) -> None:
        self._console = Console()
        self._live = Live(
            self._render(),
            console=self._console,
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

    # -- scrollback -------------------------------------------------------

    def scroll_up(self, n: int = 1) -> None:
        # upper bound is loose; _window clamps the actual view to the content top
        cap = len(self.transcript.utterances) + len(self.answers) + 50
        self.scroll = min(cap, self.scroll + n)
        self.refresh()

    def scroll_down(self, n: int = 1) -> None:
        self.scroll = max(0, self.scroll - n)
        self.refresh()

    def scroll_to_top(self) -> None:
        self.scroll_up(len(self.transcript.utterances) + len(self.answers) + 50)

    def scroll_to_bottom(self) -> None:
        self.scroll = 0
        self.refresh()

    def _pane_height(self) -> int:
        """Interior rows of a main pane: total − input(3) − status(4) − borders(2)."""
        height = self._console.size.height if self._console is not None else 24
        return max(3, height - 3 - 4 - 2)

    @staticmethod
    def _window(lines: list[Text], height: int, scroll: int) -> list[Text]:
        """The visible slice of `lines`: the bottom `height`, shifted up by `scroll`."""
        if len(lines) <= height:
            return lines
        end = max(height, min(len(lines) - scroll, len(lines)))
        return lines[end - height : end]

    def _scroll_hint(self, total: int) -> str:
        if self.scroll > 0 and total > self._pane_height():
            return "  [dim]↑ scrolled · End=live[/dim]"
        return ""

    # -- answer pane callbacks (wired to Responder) ----------------------

    def on_answer_start(self) -> None:
        self.answer_text = ""
        self.status.pending_action = ""
        self.refresh()

    def on_chat(self, text: str) -> None:
        """Echo an operator chat message into the timeline before the agent replies."""
        self.answers.append(ChatEntry(time.strftime("%H:%M:%S"), text))
        self.refresh()

    def set_input(self, buffer: str) -> None:
        """Update the always-on chat composer (driven by the input loop)."""
        self.input_buffer = buffer
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

    def on_action(self, call_id: str, tool: str, args: dict, status: str) -> None:
        """Tool-call lifecycle from the responder: pending → running → done/denied.

        Matched by the harness toolCallId, so repeated calls to the same tool
        each get their own timeline entry instead of folding onto the first."""
        entry = next(
            (e for e in self.answers if isinstance(e, ActionEntry) and e.call_id == call_id),
            None,
        )
        if entry is None:
            self.answers.append(
                ActionEntry(time.strftime("%H:%M:%S"), call_id, tool, summarize_args(args), status)
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
            Layout(self._input_panel(), name="input", size=3),
            Layout(self._status_panel(), name="status", size=4),
        )
        layout["main"].split_row(
            Layout(self._transcript_panel(), name="transcript", ratio=1),
            Layout(self._answer_panel(), name="answer", ratio=1),
        )
        return layout

    def _input_panel(self) -> Panel:
        text = Text()
        text.append("› ", style="bold green")
        text.append(self.input_buffer)
        text.append("▌", style="green")  # cursor
        return Panel(text, title="chat → agent  [dim](enter=send · empty=respond now)[/dim]",
                     border_style="green")

    def _transcript_panel(self) -> Panel:
        lines: list[Text] = []
        for utt in self.transcript.utterances:
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
        visible = self._window(lines, self._pane_height(), self.scroll)
        return Panel(Group(*visible) if visible else Text("listening…", style="dim"),
                     title=f"transcript{self._scroll_hint(len(lines))}", border_style="blue")

    def _answer_panel(self) -> Panel:
        blocks: list[Text] = []
        for entry in self.answers:
            if isinstance(entry, ActionEntry):
                line = Text()
                line.append(f"[{entry.wall_time}] ", style="dim")
                label = f"{_ACTION_ICON[entry.status]} {entry.tool}({entry.args_summary})"
                if entry.status in ("pending", "denied"):
                    label += f" — {entry.status}"
                line.append(label, style=_ACTION_STYLE[entry.status])
                blocks.append(line)
                continue
            if isinstance(entry, ChatEntry):
                line = Text()
                line.append(f"[{entry.wall_time}] ", style="dim")
                line.append("you: ", style="bold green")
                line.append(entry.text)
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
        visible = self._window(blocks, self._pane_height(), self.scroll)
        body = Group(*visible) if visible else Text("—", style="dim")
        return Panel(body, title=f"answers{self._scroll_hint(len(blocks))}", border_style="green")

    def _status_panel(self) -> Panel:
        s = self.status

        def dot(ok: bool) -> str:
            return "[green]●[/green]" if ok else "[red]●[/red]"

        mic_label = "mic[dim](muted)[/dim]" if s.mic_muted else "mic"
        line1 = (
            f"{dot(s.mic)} {mic_label}  {dot(s.system_audio)} sys  {dot(s.stt)} stt  "
            f"{dot(s.voice)} voice  |  decision: [bold]{s.last_decision}[/bold]"
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
            f"(cached {s.cached_tokens / 1000:.1f}k)  ·  "
            "PgUp/PgDn scroll  ·  /mute /voice /quit[/dim]"
        )
        if s.dropped_chunks:
            line3 += f"  [red]audio dropped: {s.dropped_chunks} chunks[/red]"
        lines = Group(*(Text.from_markup(line) for line in (line1, line2, line3)))
        return Panel(lines, border_style="white")
