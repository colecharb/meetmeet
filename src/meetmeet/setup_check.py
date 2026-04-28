from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audio import find_devices, DeviceSelection
from .config import Config
from .summary.ollama_backend import OllamaBackend


@dataclass
class PreflightResult:
    ok: bool
    devices: DeviceSelection
    blackhole_present: bool
    ollama_up: bool
    ollama_model_present: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


BLACKHOLE_INSTRUCTIONS = """[bold]BlackHole 2ch not detected.[/bold] System audio capture will be unavailable.

To enable system-audio capture:
  1. [cyan]brew install blackhole-2ch[/cyan]
  2. Open [cyan]Audio MIDI Setup[/cyan] -> [cyan]+[/cyan] -> [cyan]Create Multi-Output Device[/cyan]
  3. Check both your built-in output and [cyan]BlackHole 2ch[/cyan]
  4. [cyan]System Settings -> Sound -> Output[/cyan] -> select the Multi-Output Device
  5. Restart [bold]meetmeet[/bold]

Tip: avoid Bluetooth (e.g. AirPods) as the master in a Multi-Output device."""


def run(cfg: Config, console: Console) -> PreflightResult:
    table = Table(title="meetmeet preflight", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")

    warnings: list[str] = []
    errors: list[str] = []

    # Python version
    py_ok = sys.version_info >= (3, 11)
    table.add_row(
        "Python >= 3.11",
        "[green]OK[/green]" if py_ok else f"[red]FAIL[/red] (have {sys.version.split()[0]})",
    )
    if not py_ok:
        errors.append("Python 3.11+ required (uses tomllib).")

    # Architecture
    arch = platform.machine()
    is_arm = arch == "arm64"
    table.add_row(
        "Apple Silicon (arm64)",
        "[green]OK[/green]" if is_arm else f"[yellow]WARN[/yellow] ({arch})",
    )
    if not is_arm and cfg.stt.backend == "mlx":
        warnings.append(
            "Running x86_64 Python; mlx-whisper requires arm64. "
            "Switch backend to 'faster' in config or use an arm64 Python."
        )

    # sounddevice import + device discovery
    try:
        devices = find_devices()
        sd_ok = True
    except Exception as e:
        devices = DeviceSelection(None, None, None, None)
        sd_ok = False
        errors.append(f"sounddevice failed to enumerate devices: {e}")
    table.add_row(
        "sounddevice",
        "[green]OK[/green]" if sd_ok else "[red]FAIL[/red]",
    )

    # Mic
    if devices.mic_index is not None:
        table.add_row("Default microphone", f"[green]OK[/green] ({devices.mic_name})")
    else:
        table.add_row("Default microphone", "[yellow]WARN[/yellow] (none detected)")
        warnings.append("No default input device detected.")

    # BlackHole
    if devices.blackhole_index is not None:
        table.add_row("BlackHole 2ch", f"[green]OK[/green] ({devices.blackhole_name})")
    else:
        table.add_row("BlackHole 2ch", "[yellow]WARN[/yellow] (not detected)")

    # Ollama
    backend = OllamaBackend(cfg.summary.ollama_url, cfg.summary.model, cfg.summary.system_prompt)
    ollama_up = backend.is_available()
    table.add_row(
        "Ollama HTTP",
        f"[green]OK[/green] ({cfg.summary.ollama_url})"
        if ollama_up
        else f"[yellow]WARN[/yellow] ({cfg.summary.ollama_url} unreachable)",
    )
    if not ollama_up:
        warnings.append("Ollama not reachable; summary generation will be unavailable.")

    ollama_model_present = False
    if ollama_up:
        ollama_model_present = backend.has_model()
        table.add_row(
            f"Ollama model: {cfg.summary.model}",
            "[green]OK[/green]"
            if ollama_model_present
            else f"[yellow]WARN[/yellow] (run: ollama pull {cfg.summary.model})",
        )
        if not ollama_model_present:
            warnings.append(f"Configured Ollama model not pulled: ollama pull {cfg.summary.model}")

    # mlx model cache (informational only)
    table.add_row(
        "mlx-whisper model",
        "[blue]INFO[/blue] (downloads on first use to ~/.cache/huggingface)",
    )

    console.print(table)

    if devices.blackhole_index is None:
        console.print(Panel(BLACKHOLE_INSTRUCTIONS, title="System audio", border_style="yellow"))

    return PreflightResult(
        ok=not errors,
        devices=devices,
        blackhole_present=devices.blackhole_index is not None,
        ollama_up=ollama_up,
        ollama_model_present=ollama_model_present,
        warnings=warnings,
        errors=errors,
    )
