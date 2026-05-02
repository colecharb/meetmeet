from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MEETMEET_DIR = Path.home() / ".meetmeet"
CONFIG_PATH = MEETMEET_DIR / "config.toml"

DEFAULT_SYSTEM_PROMPT = """You are a meeting-notes assistant. Given a raw transcript that may contain noise, filler, and unclear speakers, produce concise, well-structured Markdown notes with these sections:
- **Summary** (2-4 sentences)
- **Key Topics** (bulleted)
- **Decisions** (bulleted)
- **Action Items** (bulleted, with owner if mentioned)
- **Open Questions** (bulleted)
Omit any section that has no content. Do not invent details."""

DEFAULT_CONFIG_TOML = f'''[stt]
backend = "mlx"
model = "mlx-community/whisper-large-v3-turbo"
chunk_seconds = 8.0
chunk_overlap_seconds = 0.5

[summary]
backend = "ollama"
model = "qwen3.5:4b"
ollama_url = "http://localhost:11434"
system_prompt = """{DEFAULT_SYSTEM_PROMPT}"""

[audio]
prefer_blackhole = true
sample_rate_capture = 48000
'''


@dataclass(frozen=True)
class STTConfig:
    backend: str
    model: str
    chunk_seconds: float
    chunk_overlap_seconds: float


@dataclass(frozen=True)
class SummaryConfig:
    backend: str
    model: str
    ollama_url: str
    system_prompt: str


@dataclass(frozen=True)
class AudioConfig:
    prefer_blackhole: bool
    sample_rate_capture: int


@dataclass(frozen=True)
class Config:
    stt: STTConfig
    summary: SummaryConfig
    audio: AudioConfig


def _ensure_default_config() -> bool:
    """Create ~/.meetmeet/ and config.toml with defaults if missing. Returns True if created."""
    MEETMEET_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return False
    CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return True


def load() -> tuple[Config, bool]:
    """Load config from disk, creating defaults if missing. Returns (config, created_default)."""
    created = _ensure_default_config()
    with CONFIG_PATH.open("rb") as f:
        data = tomllib.load(f)

    stt = data.get("stt", {})
    summary = data.get("summary", {})
    audio = data.get("audio", {})

    cfg = Config(
        stt=STTConfig(
            backend=stt.get("backend", "mlx"),
            model=stt.get("model", "mlx-community/whisper-large-v3-turbo"),
            chunk_seconds=float(stt.get("chunk_seconds", 8.0)),
            chunk_overlap_seconds=float(stt.get("chunk_overlap_seconds", 0.5)),
        ),
        summary=SummaryConfig(
            backend=summary.get("backend", "ollama"),
            model=summary.get("model", "qwen3.5:4b"),
            ollama_url=summary.get("ollama_url", "http://localhost:11434"),
            system_prompt=summary.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        ),
        audio=AudioConfig(
            prefer_blackhole=bool(audio.get("prefer_blackhole", True)),
            sample_rate_capture=int(audio.get("sample_rate_capture", 48000)),
        ),
    )
    return cfg, created


_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_MODEL_LINE_RE = re.compile(r'^(\s*model\s*=\s*)("[^"]*"|\'[^\']*\')(\s*)$')


def save_summary_model(model: str) -> None:
    """Rewrite only the `model = "..."` line under [summary] in config.toml.

    Preserves all other content (comments, formatting, triple-quoted strings)
    by walking lines and replacing the first model assignment found inside
    the [summary] section. Atomic via tmp-file + os.replace.
    """
    text = CONFIG_PATH.read_text()
    lines = text.splitlines(keepends=True)
    in_summary = False
    replaced = False
    for i, line in enumerate(lines):
        section = _SECTION_RE.match(line)
        if section:
            in_summary = section.group(1).strip() == "summary"
            continue
        if in_summary and not replaced:
            m = _MODEL_LINE_RE.match(line)
            if m:
                escaped = model.replace('"', '\\"')
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = f'{m.group(1)}"{escaped}"{m.group(3).rstrip()}{newline}'
                replaced = True
                break

    if not replaced:
        raise RuntimeError(
            f"Could not find `model = \"...\"` under [summary] in {CONFIG_PATH}; "
            "edit it manually."
        )

    tmp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    tmp.write_text("".join(lines))
    os.replace(tmp, CONFIG_PATH)
