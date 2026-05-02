# meetmeet

Lightweight macOS terminal app for capturing meeting audio (system + microphone) and producing a live transcript, with optional post-meeting summary via Ollama.

## Prerequisites

Install BlackHole (a virtual audio driver that lets meetmeet capture system audio) and ffmpeg:

```bash
brew install blackhole-2ch ffmpeg
```

> **Note:** A restart may be required before BlackHole shows up as an audio device.

Then route system audio through BlackHole *and* your speakers simultaneously, so you can still hear the meeting while it's being captured:

1. Open **Audio MIDI Setup** (in `/Applications/Utilities/`).
2. Click the **+** button in the bottom-left corner and choose **Create Multi-Output Device**.
3. In the device list on the right, check both **Built-in Output** (or whichever speakers/headphones you use) and **BlackHole 2ch**.
4. Open **System Settings → Sound → Output** and select the new **Multi-Output Device** as your output.

When meetmeet runs, it records from BlackHole (system audio) plus your microphone, so both sides of the call end up in the transcript.

Install [Ollama](https://ollama.com/download) (the GUI app or `brew install ollama` both work), then make sure it's running and pull the configured summary model:

```bash
ollama pull qwen3.5:4b
```

## Install

```bash
pip install meetmeet
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

## Running from source

```bash
git clone https://github.com/colecharb/meetmeet.git
cd meetmeet
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
