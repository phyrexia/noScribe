# MeetingGenie - Application State
# Singleton shared between all views

import os
from models import TranscriptionQueue
from config import get_config, set_config, save_config, load_languages, config_dir
from event_bus import EventBus, EventType


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

        # Last generated summary
        self.last_summary_path: str = ""

        # Compute device indicator — updated by workers at runtime.
        # Initial value is a safe default; the actual backend is reported
        # by the whisper worker once it starts.
        self.compute_device: str = self._detect_device()
        self.compute_device_label: str = "CPU"
        self.compute_backend: str = ""

    def _detect_device(self) -> str:
        """Initial compute device placeholder.

        Workers will report the real backend once they start. We do NOT
        claim Metal here just because the host is Apple Silicon — the
        CT2 whisper backend runs on CPU regardless.
        """
        return "cpu"

    def update_compute_device(self, backend: str, label: str):
        """Called when a worker reports its actual compute backend.

        backend: 'ct2-cpu' | 'mlx-metal' | 'cuda' | ...
        label: human-readable string for the badge.
        """
        self.compute_backend = backend or ""
        self.compute_device_label = label or "CPU"
        if backend == "cuda":
            self.compute_device = "cuda"
        elif backend == "mlx-metal":
            self.compute_device = "metal"
        else:
            self.compute_device = "cpu"
        try:
            self.bus.publish(EventType.DEVICE_UPDATE, {
                "backend": self.compute_backend,
                "label": self.compute_device_label,
            })
        except Exception:
            pass

    def get_device_label(self) -> str:
        return self.compute_device_label

    def get_device_icon_name(self) -> str:
        if self.compute_backend in ("mlx-metal", "cuda"):
            return "memory"  # chip icon
        return "computer"

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
