"""rich.Live three-pane console: transcript | answer timeline | status bar."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..brain.responder import NOTHING
from ..brain.transcript import TranscriptStore

_SPEAKER_STYLE = {"ME": "bold cyan", "THEM": "bold magenta"}


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


@dataclass
class AnswerEntry:
    """One finished answer in the timeline."""

    wall_time: str
    text: str
    interrupted: bool


@dataclass
class ConsoleView:
    transcript: TranscriptStore
    status: StatusState = field(default_factory=StatusState)
    answers: list[AnswerEntry] = field(default_factory=list)
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
        self.refresh()

    def on_delta(self, _answer_id: str, delta: str) -> None:
        self.answer_text += delta
        self.refresh()

    def on_end(self, _answer_id: str, interrupted: bool) -> None:
        text = self.answer_text.strip()
        self.answer_text = ""
        if text and text != NOTHING:
            self.answers.append(AnswerEntry(time.strftime("%H:%M:%S"), text, interrupted))
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
        elif blocks:
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
        line2 = f"[dim]{s.notice or s.decision_reason}[/dim]"  # errors win over decisions
        line3 = (
            f"[dim]prompt {s.prompt_tokens / 1000:.1f}k tok "
            f"(cached {s.cached_tokens / 1000:.1f}k)[/dim]"
        )
        if s.dropped_chunks:
            line3 += f"  [red]audio dropped: {s.dropped_chunks} chunks[/red]"
        lines = Group(*(Text.from_markup(line) for line in (line1, line2, line3)))
        return Panel(lines, border_style="white")
