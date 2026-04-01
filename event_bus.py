# MeetingGenie - Event Bus
# Thread-safe pub/sub system to decouple workers from UI

import threading
from enum import Enum
from typing import Callable, Any


class EventType(Enum):
    LOG = "log"                                 # {text, level, where}
    PROGRESS = "progress"                       # {step, value, speaker_detection}
    QUEUE_UPDATED = "queue_updated"             # {}
    JOB_STARTED = "job_started"                 # {job_id}
    JOB_FINISHED = "job_finished"               # {job_id}
    JOB_ERROR = "job_error"                     # {job_id, message}
    LIVE_SEGMENT = "live_segment"               # {text}
    LIVE_FINISHED = "live_finished"             # {}
    MODEL_DOWNLOAD_PROGRESS = "model_dl_progress"  # {bytes, total}
    SPEAKER_NAMING_REQUEST = "speaker_naming"   # {speakers_data, future}


class EventBus:
    """Simple thread-safe pub/sub event bus.

    Subscribers are called synchronously on the publishing thread.
    For UI frameworks that require main-thread updates (Flet, tkinter),
    wrap the subscriber callback to dispatch to the UI thread.
    """

    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable[[dict], Any]):
        """Register a callback for an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Remove a callback for an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event_type: EventType, payload: dict = None):
        """Publish an event to all subscribers."""
        if payload is None:
            payload = {}
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, []))
        for cb in callbacks:
            try:
                cb(payload)
            except Exception:
                pass  # Don't let one subscriber break others

    def clear(self):
        """Remove all subscriptions."""
        with self._lock:
            self._subscribers.clear()
