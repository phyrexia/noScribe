# MeetingGenie Changelog

## Unreleased
- Migrate UI from Tkinter to Flet (`main.py` + `views/`)
- Apple Speech worker (on-device transcription on macOS 14+)
- Live mode with VAD-driven streaming
- Anthropic-powered meeting summarizer
- Batch transcription queue, speaker DB, editor UI
- Packaging pipeline (PyInstaller + DMG) for macOS

## Heritage

MeetingGenie was forked from a Tkinter-based ancestor; refer to the
original project for pre-fork history. The legacy UI has been removed
in favour of the Flet rewrite.
