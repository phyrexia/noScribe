# MeetingGenie - Application State
# Singleton shared between all views

import os
from models import TranscriptionQueue
from config import get_config, set_config, save_config, load_languages, config_dir
from event_bus import EventBus


class AppState:
    """Centralised application state shared across all Flet views."""

    def __init__(self, app_dir: str):
        self.app_dir = app_dir

        # Event bus for worker ↔ UI communication
        self.bus = EventBus()

        # Transcription queue
        self.queue = TranscriptionQueue()

        # Languages (filtered by languages.yml)
        self.languages: dict[str, str] = load_languages(app_dir)

        # Cancel flags
        self.cancel: bool = False
        self.cancel_job_only: bool = False

        # Live mode state
        self.live_process_running: bool = False

        # Current audio file selections
        self.audio_files: list[str] = []
        self.transcript_files: list[str] = []

        # Model paths cache
        self.whisper_model_paths: dict[str, str] = {}

    def get_language_names(self) -> list[str]:
        return list(self.languages.keys())

    def get_language_code(self, name: str) -> str:
        return self.languages.get(name, 'auto')

    def get_config(self, key: str, default=None):
        return get_config(key, default)

    def set_config(self, key: str, value):
        set_config(key, value)

    def save_config(self):
        save_config()
