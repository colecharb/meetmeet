# meetmeet

Lightweight macOS terminal app for capturing meeting audio (system + microphone) and producing a live transcript, with optional post-meeting summary via Ollama.

## Setup

```bash
brew install blackhole-2ch ffmpeg
# Audio MIDI Setup -> + -> Multi-Output Device
#   -> check Built-in Output + BlackHole 2ch
# System Settings -> Sound -> Output -> select Multi-Output Device

cd /path/to/meetmeet
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Make sure Ollama is running and the configured summary model is pulled:

```bash
ollama pull qwen3.5:4b
```

## Usage

```bash
meetmeet
```

Opens a menu:
- **Start a new meeting** — prompts for a title, then captures + transcribes live until you press `q` (or Ctrl-C).
- **Browse past meetings** — lists recent meetings; offers to generate a summary for any meeting that doesn't yet have one.

Transcripts and summaries land under `~/.meetmeet/<slug>/`.

## Config

`~/.meetmeet/config.toml` is created with defaults on first run. STT and summary backends are pluggable via the `[stt]` and `[summary]` sections.
