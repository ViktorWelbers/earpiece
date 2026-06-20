"""Persist a session to disk so it can be resumed later.

One JSON file per session under `~/.config/earpiece/sessions/<id>.json`,
auto-saved as the session runs. Each record holds earpiece's own view (the
transcript + the answers timeline) plus the harness session id, so a resume can
both restore the UI and reattach the agent's memory. Conversation text only —
never API keys or internal hostnames; the file lives next to (not inside) the
repo, under the user's config dir.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..config import Settings, config_path

log = logging.getLogger(__name__)


def sessions_dir() -> Path:
    path = config_path().parent / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class SessionRecord:
    id: str
    mission: str
    agent_cmd: str | None
    agent_cwd: str | None
    created_at: float
    updated_at: float
    harness_session_id: str | None = None
    agent_name: str | None = None
    transcript: dict = field(default_factory=dict)
    answers: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, settings: Settings) -> SessionRecord:
        now = time.time()
        return cls(
            id=uuid.uuid4().hex,
            mission=settings.mission,
            agent_cmd=settings.agent_cmd,
            agent_cwd=settings.agent_cwd or os.getcwd(),
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_dict(cls, data: dict) -> SessionRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def turns(self) -> int:
        """Assistant answers in the saved timeline — the picker's 'turns' count."""
        return sum(1 for e in self.answers if e.get("kind") == "answer")


def save(record: SessionRecord) -> Path:
    """Write the record atomically (temp file + rename) so a crash mid-write
    never leaves a half-written session behind."""
    record.updated_at = time.time()
    path = sessions_dir() / f"{record.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2))
    os.replace(tmp, path)
    return path


def load(session_id: str) -> SessionRecord:
    path = sessions_dir() / f"{session_id}.json"
    return SessionRecord.from_dict(json.loads(path.read_text()))


def list_sessions() -> list[SessionRecord]:
    """Every saved session, newest first; unreadable files are skipped."""
    records: list[SessionRecord] = []
    for path in sessions_dir().glob("*.json"):
        try:
            records.append(SessionRecord.from_dict(json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError, TypeError, KeyError) as exc:
            log.warning("skipping unreadable session %s: %s", path.name, exc)
    records.sort(key=lambda r: r.updated_at, reverse=True)
    return records


def ago(ts: float) -> str:
    """Coarse 'how long ago' for the picker."""
    secs = max(0.0, time.time() - ts)
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if secs >= size:
            return f"{int(secs // size)}{unit} ago"
    return "just now"
