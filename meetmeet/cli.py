from __future__ import annotations

import os
import pydoc
import queue
import re
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from datetime import datetime
from pathlib import Path

import questionary
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import storage
from .audio import AudioPipeline, TranscribeChunk, WHISPER_RATE
from .config import Config, load as load_config
from .setup_check import PreflightResult, run as run_preflight
from .stt import STTBackend, get_backend as get_stt_backend
from .summary import get_backend as get_summary_backend


console = Console()

_TIMESTAMP_TITLE_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def _ask_with_back(question):
    """Bind escape to exit the prompt with None (same outcome as Ctrl-C),
    so callers can treat None as 'go back'."""
    app = question.application
    # ttimeoutlen guards bare-ESC vs. escape-sequence disambiguation; set low
    # so arrow keys still parse but bare ESC fires fast.
    app.ttimeoutlen = 0.01

    # eager=True is the critical bit: without it, prompt_toolkit waits
    # `timeoutlen` (~1s) to see if a longer Meta-prefixed binding will match.
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event):
        event.app.exit(result=None)

    # questionary.text/confirm hand back a `_MergedKeyBindings` (read-only)
    # while questionary.select hands back a mutable `KeyBindings`. Merging
    # works for both shapes.
    app.key_bindings = merge_key_bindings(
        [app.key_bindings, kb] if app.key_bindings else [kb]
    )
    return question.ask()


def ask_select(message, choices, **kwargs):
    return _ask_with_back(questionary.select(message, choices=choices, **kwargs))


def ask_text(message, **kwargs):
    return _ask_with_back(questionary.text(message, **kwargs))


def ask_confirm(message, **kwargs):
    return _ask_with_back(questionary.confirm(message, **kwargs))


def main() -> None:
    cfg, created_default = load_config()
    if created_default:
        console.print(f"[dim]Created default config at ~/.meetmeet/config.toml[/dim]")

    pre = run_preflight(cfg, console)
    if not pre.ok:
        for err in pre.errors:
            console.print(f"[red]ERROR:[/red] {err}")
        sys.exit(1)

    if not pre.blackhole_present and cfg.audio.prefer_blackhole:
        cont = ask_confirm("Continue with mic-only mode for this session?", default=True)
        if not cont:
            return

    while True:
        choice = ask_select(
            "What would you like to do?",
            choices=["Start a new meeting", "Browse past meetings", "Quit"],
        )
        if choice is None or choice == "Quit":
            return
        if choice == "Start a new meeting":
            run_new_meeting(cfg, pre)
        elif choice == "Browse past meetings":
            browse_past_meetings(cfg)


# --- new meeting ----------------------------------------------------------------

def run_new_meeting(cfg: Config, pre: PreflightResult) -> None:
    started_at = datetime.now()
    default_title = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    title = ask_text("Meeting title?", placeholder=default_title)
    if title is None:
        return
    title = title.strip() or default_title

    meeting_dir = storage.make_meeting_dir(title, started_at, default_title)
    transcript_path = meeting_dir / storage.TRANSCRIPT_FILENAME
    storage.write_transcript_header(transcript_path, title, started_at)

    # Initialize STT backend with a spinner (model may download on first run).
    stt_backend = _init_stt_backend(cfg)

    # Audio pipeline.
    transcribe_q: queue.Queue[TranscribeChunk] = queue.Queue(maxsize=8)
    overflow_count = {"n": 0}

    def on_overflow() -> None:
        overflow_count["n"] += 1

    pipeline = AudioPipeline(
        mic_index=pre.devices.mic_index,
        blackhole_index=pre.devices.blackhole_index if cfg.audio.prefer_blackhole else None,
        capture_rate=cfg.audio.sample_rate_capture,
        chunk_seconds=cfg.stt.chunk_seconds,
        chunk_overlap_seconds=cfg.stt.chunk_overlap_seconds,
        out_queue=transcribe_q,
        on_overflow=on_overflow,
    )

    stop_event = threading.Event()
    ui_lines: deque[str] = deque(maxlen=12)
    ui_lock = threading.Lock()

    def transcriber_loop() -> None:
        while not (stop_event.is_set() and transcribe_q.empty()):
            try:
                item = transcribe_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                text = stt_backend.transcribe(item.audio, WHISPER_RATE)
            except Exception as e:
                with ui_lock:
                    ui_lines.append(f"[transcribe error: {e}]")
                continue
            if not text.strip():
                continue
            storage.append_chunk(transcript_path, item.start_wall, text)
            line = f"[{item.start_wall.strftime('%H:%M:%S')}] {text.strip()}"
            with ui_lock:
                ui_lines.append(line)

    transcriber_thread = threading.Thread(target=transcriber_loop, daemon=True)

    # SIGINT handler (graceful)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    pipeline.start()
    transcriber_thread.start()

    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    key_thread = threading.Thread(
        target=_listen_for_q, args=(stop_event, fd), daemon=True
    )

    end_at = None
    try:
        tty.setcbreak(fd)
        key_thread.start()
        with Live(
            _render_live(title, started_at, ui_lines, ui_lock, pipeline.sources),
            console=console,
            refresh_per_second=10,
            transient=False,
        ) as live:
            while not stop_event.is_set():
                live.update(_render_live(title, started_at, ui_lines, ui_lock, pipeline.sources))
                time.sleep(0.1)
            live.update(_render_live(title, started_at, ui_lines, ui_lock, pipeline.sources, stopping=True))
        end_at = datetime.now()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
        signal.signal(signal.SIGINT, prev_sigint)

    # Stop pipeline (flushes a final chunk if buffered) and drain transcriber.
    with console.status("[bold]Finishing transcription...[/bold]", spinner="dots"):
        pipeline.stop(flush=True)
        transcriber_thread.join(timeout=60)

    if end_at is None:
        end_at = datetime.now()

    audio_sources = pipeline.sources

    if overflow_count["n"]:
        console.print(
            f"[yellow]warning:[/yellow] {overflow_count['n']} chunk(s) dropped due to backpressure."
        )

    # Write meta.json (without summary info initially).
    meta = {
        "title": title,
        "started_at": started_at.isoformat(),
        "ended_at": end_at.isoformat(),
        "stt": {"backend": cfg.stt.backend, "model": cfg.stt.model},
        "audio_sources": audio_sources,
    }
    storage.write_meta(meeting_dir, meta)

    console.print(
        f"[green]Saved transcript:[/green] {transcript_path}"
    )

    # Offer summary.
    if not pre.ollama_up:
        console.print("[dim]Ollama unreachable; skipping summary offer.[/dim]")
        return
    if ask_confirm("Generate summary now?", default=True):
        _run_summary(cfg, meeting_dir, meta)


def _init_stt_backend(cfg: Config) -> STTBackend:
    with console.status(
        f"[bold]Loading STT backend ({cfg.stt.backend}: {cfg.stt.model})...[/bold]",
        spinner="dots",
    ):
        return get_stt_backend(cfg.stt)


def _render_live(
    title: str,
    started_at: datetime,
    ui_lines: deque[str],
    ui_lock: threading.Lock,
    sources: list[str],
    stopping: bool = False,
) -> Panel:
    elapsed = datetime.now() - started_at
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    with ui_lock:
        body_lines = list(ui_lines)
    body = "\n".join(body_lines) if body_lines else "[dim]Listening...[/dim]"
    sources_str = "+".join(sources) if sources else "(none)"
    subtitle = (
        "stopping..."
        if stopping
        else "press q to stop, Ctrl-C also works"
    )
    return Panel(
        Text.from_markup(body),
        title=f"meetmeet - {title}  [{sources_str}]  {elapsed_str}",
        subtitle=subtitle,
        border_style="cyan" if not stopping else "yellow",
    )


def _listen_for_q(stop_event: threading.Event, fd: int) -> None:
    while not stop_event.is_set():
        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not r:
            continue
        try:
            ch = os.read(fd, 1).decode("utf-8", errors="ignore")
        except OSError:
            return
        if ch.lower() == "q":
            stop_event.set()
            return


# --- summary --------------------------------------------------------------------

def _run_summary(cfg: Config, meeting_dir: Path, meta: dict) -> None:
    transcript = storage.read_transcript(meeting_dir)
    title = meta.get("title", meeting_dir.name)

    backend = get_summary_backend(cfg.summary)

    accumulated: list[str] = []
    try:
        with Live(
            Panel(Text("(generating...)"), title="Summary", border_style="magenta"),
            console=console,
            refresh_per_second=10,
        ) as live:
            for chunk in backend.summarize(transcript, title):
                accumulated.append(chunk)
                live.update(
                    Panel(
                        Markdown("".join(accumulated)),
                        title=f"Summary - {title}",
                        border_style="magenta",
                    )
                )
    except Exception as e:
        console.print(f"[red]Summary failed:[/red] {e}")
        return

    summary_text = "".join(accumulated).strip() + "\n"
    summary_path = storage.write_summary(meeting_dir, summary_text)

    meta = dict(meta)
    meta["summary"] = {"backend": cfg.summary.backend, "model": cfg.summary.model}
    storage.write_meta(meeting_dir, meta)

    console.print(f"[green]Saved summary:[/green] {summary_path}")


# --- browse ---------------------------------------------------------------------

def browse_past_meetings(cfg: Config) -> None:
    show_n = 5
    while True:
        meetings = storage.list_meetings(limit=20)
        if not meetings:
            console.print("[dim]No meetings yet.[/dim]")
            return
        visible = meetings[:show_n]
        choices = []
        for m in visible:
            tag = "[summary]" if m.has_summary else "[no summary]"
            when = m.started_at.strftime("%Y-%m-%d %H:%M") if m.started_at else "?"
            name = "" if _TIMESTAMP_TITLE_RE.fullmatch(m.title) else m.title
            label = f"{when}  {name}  {tag}" if name else f"{when}  {tag}"
            choices.append(
                questionary.Choice(
                    title=label,
                    value=("meeting", m),
                )
            )
        if show_n < len(meetings):
            choices.append(questionary.Choice(title="[Show more]", value=("more", None)))
        choices.append(questionary.Choice(title="[Back]", value=("back", None)))

        ans = ask_select("Past meetings", choices=choices)
        if ans is None:
            return
        action, payload = ans
        if action == "back":
            return
        if action == "more":
            show_n = len(meetings)
            continue
        meeting = payload
        _meeting_actions(cfg, meeting)


def _meeting_actions(cfg: Config, meeting: storage.MeetingInfo) -> None:
    while True:
        meta = storage.read_meta(meeting.path) or {}
        has_sum = (meeting.path / storage.SUMMARY_FILENAME).exists()
        when = meeting.started_at.strftime("%Y-%m-%d %H:%M") if meeting.started_at else "?"
        info_lines = [
            f"[bold]{meeting.title}[/bold]",
            f"[dim]Started {when}  -  {meeting.path}[/dim]",
        ]
        if meta.get("stt"):
            info_lines.append(f"STT: {meta['stt'].get('backend')} / {meta['stt'].get('model')}")
        if meta.get("summary"):
            info_lines.append(f"Summary: {meta['summary'].get('backend')} / {meta['summary'].get('model')}")
        console.print(Panel("\n".join(info_lines), border_style="blue"))

        choices = ["Open transcript"]
        if has_sum:
            choices.append("Open summary")
        else:
            choices.append("Generate summary")
        choices.append("Back")

        ans = ask_select("Action", choices=choices)
        if ans is None or ans == "Back":
            return
        if ans == "Open transcript":
            pydoc.pager(storage.read_transcript(meeting.path))
        elif ans == "Open summary":
            pydoc.pager((meeting.path / storage.SUMMARY_FILENAME).read_text())
        elif ans == "Generate summary":
            meta = storage.read_meta(meeting.path) or {
                "title": meeting.title,
                "started_at": meeting.started_at.isoformat() if meeting.started_at else "",
            }
            _run_summary(cfg, meeting.path, meta)


if __name__ == "__main__":
    main()
