from __future__ import annotations

import dataclasses
import os
import queue
import re
import select
import signal
import sys
import termios
import textwrap
import threading
import time
import tty
from collections import deque
from datetime import datetime
from pathlib import Path

import httpx
import questionary
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from questionary.prompts.common import InquirerControl
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import storage
from .audio import AudioPipeline, TranscribeChunk, WHISPER_RATE
from .clipboard import copy_to_clipboard
from .config import Config, load as load_config, save_summary_model
from .setup_check import PreflightResult, run as run_preflight
from .stt import STTBackend, get_backend as get_stt_backend
from .summary import get_backend as get_summary_backend
from .summary.ollama_backend import list_local_models


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


_EXIT_CHOICE_TITLES = {"back", "[back]", "quit", "[quit]"}


def _decorate_menu_choices(choices):
    """Add a blank Separator above the first option, and another above any
    trailing Back/Quit option, for visual breathing room."""
    def title_of(c):
        raw = c if isinstance(c, str) else getattr(c, "title", None)
        return raw.strip().lower() if isinstance(raw, str) else None

    decorated = [questionary.Separator(" "), *choices]
    if decorated and title_of(decorated[-1]) in _EXIT_CHOICE_TITLES:
        decorated.insert(-1, questionary.Separator(" "))
    return decorated


def ask_select(message, choices, **kwargs):
    choices = _decorate_menu_choices(choices)
    return _ask_with_back(questionary.select(message, choices=choices, **kwargs))


def ask_select_with_keys(message, choices, extra_keys, **kwargs):
    """Like ask_select, but binds extra single-key handlers that receive the
    currently focused choice value. Each handler returns an optional string
    to print as transient feedback (rendered via rich)."""
    decorated = _decorate_menu_choices(choices)
    q = questionary.select(message, choices=decorated, **kwargs)
    app = q.application
    app.ttimeoutlen = 0.01

    control = next(
        (c for c in app.layout.find_all_controls() if isinstance(c, InquirerControl)),
        None,
    )

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event):
        event.app.exit(result=None)

    def _make_handler(handler):
        def _on_key(event):
            if control is None:
                return
            choice = control.get_pointed_at()
            value = getattr(choice, "value", None) or getattr(choice, "title", None)
            msg = handler(value)
            if msg:
                run_in_terminal(lambda: console.print(msg))
        return _on_key

    for key, handler in extra_keys.items():
        kb.add(key, eager=True)(_make_handler(handler))

    app.key_bindings = merge_key_bindings(
        [app.key_bindings, kb] if app.key_bindings else [kb]
    )
    return q.ask()


def _view_text(text: str) -> None:
    """Scrollable plain-text viewer. ESC, q, or Ctrl-C exits."""
    buf = Buffer(document=Document(text, 0), read_only=True)

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    @kb.add("q")
    @kb.add("c-c")
    def _exit(event):
        event.app.exit()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        buf.cursor_up()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        buf.cursor_down()

    @kb.add("pageup")
    @kb.add("c-b")
    def _pgup(event):
        buf.cursor_up(count=20)

    @kb.add("pagedown")
    @kb.add("c-f")
    @kb.add("space")
    def _pgdn(event):
        buf.cursor_down(count=20)

    @kb.add("home")
    @kb.add("g")
    def _home(event):
        buf.cursor_position = 0

    @kb.add("end")
    @kb.add("G")
    def _end(event):
        buf.cursor_position = len(text)

    app = Application(
        layout=Layout(Window(BufferControl(buffer=buf), wrap_lines=True)),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
    )
    app.ttimeoutlen = 0.01
    app.run()


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
            choices=["Start a new meeting", "Browse past meetings", "Config", "Quit"],
        )
        if choice is None or choice == "Quit":
            return
        if choice == "Start a new meeting":
            run_new_meeting(cfg, pre)
        elif choice == "Browse past meetings":
            browse_past_meetings(cfg)
        elif choice == "Config":
            cfg = run_config_menu(cfg)


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

    generated_at = datetime.now()
    header = (
        f"_Generated by {cfg.summary.model} on "
        f"{generated_at.strftime('%Y-%m-%d %H:%M')}_\n\n"
    )
    body = "".join(accumulated).strip() + "\n"
    summary_text = header + body
    summary_path = storage.write_summary(meeting_dir, summary_text)

    meta = dict(meta)
    meta["summary"] = {
        "backend": cfg.summary.backend,
        "model": cfg.summary.model,
        "generated_at": generated_at.isoformat(),
    }
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
        when_w = 16  # YYYY-MM-DD HH:MM
        tag_w = len("(no summary)")
        sep = "  "
        # questionary prepends a 3-char indicator (" » " / "   ") to every row.
        name_w = max(10, console.size.width - 3 - tag_w - len(sep) * 2 - when_w)
        cont_pad = " " * (tag_w + len(sep) + when_w + len(sep))
        for m in visible:
            tag = ("(summary)" if m.has_summary else "(no summary)").rjust(tag_w)
            when = (
                m.started_at.strftime("%Y-%m-%d %H:%M") if m.started_at else "?"
            ).ljust(when_w)
            raw_name = "" if _TIMESTAMP_TITLE_RE.fullmatch(m.title) else m.title
            name_lines = textwrap.wrap(raw_name, width=name_w) or [""]
            first = f"{tag}{sep}{when}{sep}{name_lines[0]}"
            rest = [f"{cont_pad}{ln}" for ln in name_lines[1:]]
            choices.append(
                questionary.Choice(
                    title="\n".join([first, *rest]),
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
            choices.append("Re-summarize")
        else:
            choices.append("Generate summary")
        choices.append("Back")

        def _copy(focused):
            if focused == "Open transcript":
                ok = copy_to_clipboard(storage.read_transcript(meeting.path))
                return (
                    "[green]Transcript copied to clipboard[/green]"
                    if ok
                    else "[red]Copy failed (pbcopy unavailable)[/red]"
                )
            if focused in ("Open summary", "Re-summarize"):
                text = (meeting.path / storage.SUMMARY_FILENAME).read_text()
                ok = copy_to_clipboard(text)
                return (
                    "[green]Summary copied to clipboard[/green]"
                    if ok
                    else "[red]Copy failed (pbcopy unavailable)[/red]"
                )
            return None

        ans = ask_select_with_keys(
            "Action  [c to copy focused]",
            choices=choices,
            extra_keys={"c": _copy},
        )
        if ans is None or ans == "Back":
            return
        if ans == "Open transcript":
            _view_text(storage.read_transcript(meeting.path))
        elif ans == "Open summary":
            _view_text((meeting.path / storage.SUMMARY_FILENAME).read_text())
        elif ans in ("Generate summary", "Re-summarize"):
            meta = storage.read_meta(meeting.path) or {
                "title": meeting.title,
                "started_at": meeting.started_at.isoformat() if meeting.started_at else "",
            }
            _run_summary(cfg, meeting.path, meta)


# --- config -------------------------------------------------------------------

def run_config_menu(cfg: Config) -> Config:
    while True:
        info = (
            f"[bold]Summary[/bold]\n"
            f"  model: {cfg.summary.model}\n"
            f"  ollama_url: {cfg.summary.ollama_url}"
        )
        console.print(Panel(info, title="Config", border_style="blue"))

        choice = ask_select(
            "Config",
            choices=["Change summary model", "Back"],
        )
        if choice is None or choice == "Back":
            return cfg
        if choice == "Change summary model":
            cfg = _pick_summary_model(cfg)


def _pick_summary_model(cfg: Config) -> Config:
    try:
        models = list_local_models(cfg.summary.ollama_url)
    except httpx.HTTPError as e:
        console.print(
            f"[red]Could not reach Ollama at {cfg.summary.ollama_url}: {e}[/red]"
        )
        return cfg

    if not models:
        console.print(
            "[yellow]No local Ollama models found.[/yellow] "
            "Pull one first, e.g. [cyan]ollama pull qwen3.5:4b[/cyan]"
        )
        return cfg

    current = cfg.summary.model
    choices = []
    default_choice = None
    for name in models:
        title = f"{name} (current)" if name == current else name
        c = questionary.Choice(title=title, value=name)
        choices.append(c)
        if name == current:
            default_choice = c
    choices.append(questionary.Choice(title="[Back]", value=None))

    picked = ask_select(
        "Summary model",
        choices=choices,
        default=default_choice,
    )
    if picked is None or picked == current:
        return cfg

    try:
        save_summary_model(picked)
    except Exception as e:
        console.print(f"[red]Failed to save config:[/red] {e}")
        return cfg

    console.print(f"[green]Summary model set to {picked}[/green]")
    return dataclasses.replace(
        cfg,
        summary=dataclasses.replace(cfg.summary, model=picked),
    )


if __name__ == "__main__":
    main()
