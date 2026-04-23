# MeetingGenie

### AI-powered Meeting Transcription & Summarization

> Transform your meetings into structured, searchable transcripts with speaker identification, AI summaries, and real-time live transcription.

---

## Features

| Feature | Description |
|---------|-------------|
| **Transcription** | High-accuracy audio/video transcription using Whisper AI |
| **Speaker Detection** | Automatic speaker identification with voice signature database |
| **AI Summaries** | One-click meeting summaries via Claude API (executive summary, action items, decisions) |
| **Live Mode** | Real-time transcription from microphone or system audio |
| **Multiple Formats** | Export as TXT, SRT, VTT, or HTML |
| **60+ Languages** | Auto-detection or manual language selection |
| **Offline** | Runs completely locally — no cloud, no data leaves your machine |
| **Dark/Light Mode** | Modern Material Design 3 interface |

## Quick Start

### macOS (Apple Silicon)

1. **Download** `MeetingGenie.dmg` from [Releases](https://github.com/phyrexia/noScribe/releases)
2. **Drag** MeetingGenie.app to Applications
3. **Open** and start transcribing

### From Source

```bash
# Clone
git clone https://github.com/phyrexia/noScribe.git
cd noScribe

# Create venv (outside OneDrive/cloud sync)
python3 -m venv ~/.meetinggenie-venv
~/.meetinggenie-venv/bin/pip install -r environments/requirements_macOS_arm64.txt flet

# Run
~/.meetinggenie-venv/bin/python3 main.py
```

## Models

MeetingGenie uses Whisper models for transcription. Choose based on your needs:

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **Small** | 246 MB | Fastest | Good | Live mode, quick drafts |
| **Fast** | 785 MB | Fast | Very good | Most meetings |
| **Precise** | 1.5 GB | Slower | Best | Important recordings |

Models download automatically on first use from [GitHub Releases](https://github.com/phyrexia/noScribe/releases/tag/models-v1).

## Usage

### Transcribe a Recording

1. Select an audio/video file (supports WAV, MP3, M4A, MP4, MOV, MKV, and more)
2. Choose output format and model
3. Enable speaker detection if needed
4. Click **Start Transcription**
5. After diarization, identify and name speakers in the popup dialog
6. Review transcript in the built-in editor

### Live Transcription

1. Select your audio input device (microphone, BlackHole, Teams Audio, etc.)
2. Click **Live Mode**
3. Speak — text appears in real-time
4. Click **Stop Live** when done

### AI Summary

1. Add your Anthropic API key in Settings
2. After transcription, click **Generar Resumen**
3. Summary saves as `{filename}_summary.md` next to the transcript

## Architecture

```
main.py                    # Flet app entry point
views/
  shell.py                 # NavigationRail + header layout
  transcribe_page.py       # Main transcription UI
  queue_page.py            # Job queue management
  editor_page.py           # Transcript editor
  settings_page.py         # Configuration
  dialogs/
    speaker_naming.py      # Speaker identification dialog
transcription_runner.py    # ffmpeg -> pyannote -> whisper pipeline
live_mp_worker.py          # Real-time audio capture + transcription
whisper_mp_worker.py       # Whisper subprocess worker
pyannote_mp_worker.py      # Speaker diarization subprocess
model_manager.py           # Model download + management
speaker_db.py              # Voice signature database
config.py                  # Configuration management
```

## Configuration

Config file: `~/.config/MeetingGenie/config.yml`

Key settings:
- `anthropic_api_key` — For AI summaries
- `proxy_url` — Corporate proxy (e.g., `http://proxy.corp.com:8080`)
- `ignore_ssl` — Bypass SSL verification for corporate environments
- `whisper_compute_type` — `int8` (CPU) or `float32`
- `force_pyannote_cpu` — Force CPU for diarization (disable MPS/GPU)

Speaker signatures: `~/.config/MeetingGenie/speaker_signatures.json`

## Building

```bash
# Build macOS .app + DMG
sh scripts/build_meetinggenie.sh

# Skip signing and DMG
MG_SKIP_SIGN=1 MG_SKIP_DMG=1 sh scripts/build_meetinggenie.sh

# With code signing
MG_IDENTITY="Developer ID" MG_APPLE_ID="you@email.com" \
MG_TEAM_ID="XXXXXXXXXX" MG_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
sh scripts/build_meetinggenie.sh
```

## Tech Stack

- **UI**: [Flet](https://flet.dev/) (Flutter/Material Design 3 for Python)
- **Transcription**: [faster-whisper](https://github.com/guillaumekln/faster-whisper) (CTranslate2)
- **Speaker Diarization**: [pyannote.audio](https://github.com/pyannote/pyannote-audio) v4
- **AI Summary**: [Anthropic Claude API](https://docs.anthropic.com/)
- **Audio**: FFmpeg, sounddevice
- **Packaging**: PyInstaller

## Credits

Built on top of [noScribe](https://github.com/kaixxx/noScribe) by Kai Droge, with macOS port by [gernophil](https://github.com/gernophil).

Powered by:
- [Whisper](https://github.com/openai/whisper) (OpenAI)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) (Guillaume Klein)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (Herve Bredin)

## License

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)
