from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import MEETMEET_DIR

TRANSCRIPT_FILENAME = "transcript.md"
SUMMARY_FILENAME = "summary.md"
META_FILENAME = "meta.json"


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-")


def timestamp_for_path(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def make_meeting_dir(title: str, started_at: datetime, default_title: str) -> Path:
    """Create and return a new meeting directory. If title equals the default
    timestamp string, the slug suffix is omitted."""
    base = timestamp_for_path(started_at)
    if title.strip() == default_title:
        name = base
    else:
        slug = slugify(title)
        name = f"{base}_{slug}" if slug else base
    path = MEETMEET_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_transcript_header(transcript_path: Path, title: str, started_at: datetime) -> None:
    header = f"# {title}\n_Started {started_at.strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
    transcript_path.write_text(header)


def append_chunk(transcript_path: Path, chunk_time: datetime, text: str) -> None:
    line = f"[{chunk_time.strftime('%H:%M:%S')}] {text.strip()}\n"
    with transcript_path.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def write_meta(meeting_dir: Path, meta: dict) -> None:
    (meeting_dir / META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(meeting_dir: Path) -> dict | None:
    p = meeting_dir / META_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write_summary(meeting_dir: Path, summary_text: str) -> Path:
    p = meeting_dir / SUMMARY_FILENAME
    p.write_text(summary_text)
    return p


def has_summary(meeting_dir: Path) -> bool:
    return (meeting_dir / SUMMARY_FILENAME).exists()


def read_transcript(meeting_dir: Path) -> str:
    return (meeting_dir / TRANSCRIPT_FILENAME).read_text()


@dataclass
class MeetingInfo:
    path: Path
    name: str
    title: str
    started_at: datetime | None
    has_summary: bool


def _parse_started_from_dirname(name: str) -> datetime | None:
    m = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def list_meetings(limit: int | None = None) -> list[MeetingInfo]:
    if not MEETMEET_DIR.exists():
        return []
    out: list[MeetingInfo] = []
    for p in MEETMEET_DIR.iterdir():
        if not p.is_dir():
            continue
        if not (p / TRANSCRIPT_FILENAME).exists():
            continue
        meta = read_meta(p) or {}
        started_at = None
        if "started_at" in meta:
            try:
                started_at = datetime.fromisoformat(meta["started_at"])
            except ValueError:
                started_at = None
        if started_at is None:
            started_at = _parse_started_from_dirname(p.name)
        title = meta.get("title") or p.name
        out.append(
            MeetingInfo(
                path=p,
                name=p.name,
                title=title,
                started_at=started_at,
                has_summary=(p / SUMMARY_FILENAME).exists(),
            )
        )
    out.sort(key=lambda m: m.started_at or datetime.min, reverse=True)
    if limit is not None:
        return out[:limit]
    return out
