from __future__ import annotations

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
