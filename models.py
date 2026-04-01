# MeetingGenie - Transcription Job Models
# Extracted from noScribe.py for clean architecture

import os
import datetime
import platform
from enum import Enum
from typing import Optional, List

import utils


class JobStatus(Enum):
    WAITING = "waiting"
    AUDIO_CONVERSION = "audio_conversion"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    TRANSCRIPTION = "transcription"
    CANCELING = "canceling"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"


class TranscriptionJob:
    """Represents a single transcription job with all its parameters and status"""

    def __init__(self):
        # Status tracking
        self.status: JobStatus = JobStatus.WAITING
        self.error_message: Optional[str] = None
        self.error_tb: Optional[str] = None
        self.created_at: datetime.datetime = datetime.datetime.now()
        self.started_at: Optional[datetime.datetime] = None
        self.finished_at: Optional[datetime.datetime] = None

        # Progress tracking
        self.progress: float = 0.0  # Progress from 0.0 to 1.0

        # File paths
        self.audio_file: str = ''
        self.transcript_file: str = ''
        self.has_partial_transcript: bool = False

        # Time range
        self.start: int = 0  # milliseconds
        self.stop: int = 0   # milliseconds (0 means until end)

        # Language and model settings
        self.language_name: str = 'Auto'
        self.whisper_model: str = ''  # path to the model

        # Processing options
        self.speaker_detection: str = 'auto'
        self.overlapping: bool = True
        self.timestamps: bool = False
        self.disfluencies: bool = True
        self.pause: int = 0  # index value (0=none, 1=1sec+, etc.)

        # Config-based options
        self.whisper_beam_size: int = 1
        self.whisper_temperature: float = 0.0
        self.whisper_compute_type: str = 'default'
        self.timestamp_interval: int = 60_000
        self.timestamp_color: str = '#78909C'
        self.pause_marker: str = '.'
        self.auto_save: bool = True
        self.whisper_xpu: str = 'cpu'
        self.vad_threshold: float = 0.5

        # Derived properties
        self.file_ext: str = ''

    def set_running(self):
        """Mark job as running and record start time"""
        self.status = JobStatus.AUDIO_CONVERSION
        self.started_at = datetime.datetime.now()

    def set_finished(self):
        """Mark job as finished and record completion time"""
        self.status = JobStatus.FINISHED
        self.finished_at = datetime.datetime.now()

    def set_error(self, error_message: str, error_tb: str = ''):
        """Mark job as failed and store error message"""
        self.status = JobStatus.ERROR
        self.error_message = error_message
        self.error_tb = error_tb
        self.finished_at = datetime.datetime.now()

    def set_canceled(self, message: Optional[str] = None):
        """Mark job as canceled by the user"""
        self.status = JobStatus.CANCELED
        self.error_message = message
        self.finished_at = datetime.datetime.now()

    def get_duration(self) -> Optional[datetime.timedelta]:
        """Get processing duration if job is completed"""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def format_summary(self, t_func=None) -> str:
        """Build a concise, multi-line summary for tooltips.

        Uses localized UI labels where available and simple symbols for booleans.
        t_func: translation function (i18n.t). If None, uses raw keys.
        """
        if t_func is None:
            t_func = lambda key, **kw: key

        lines = []

        def yn(v: bool) -> str:
            return '✓' if bool(v) else '✗'

        try:
            out_name = os.path.basename(self.transcript_file) if self.transcript_file else ''
            lines.append(f"{t_func('job_tt_transcript_file')} {out_name}")
        except Exception:
            pass

        try:
            start_ms = getattr(self, 'start', 0) or 0
            stop_ms = getattr(self, 'stop', 0) or 0
            start_txt = utils.ms_to_str(start_ms) if start_ms > 0 else '00:00:00'
            stop_txt = utils.ms_to_str(stop_ms) if stop_ms > 0 else 'end'
            lines.append(f"{t_func('label_start')} {start_txt}")
            lines.append(f"{t_func('label_stop')} {stop_txt}")
        except Exception:
            pass

        try:
            lines.append(f"{t_func('label_language')} {self.language_name}")
        except Exception:
            pass

        try:
            model_disp = os.path.basename(self.whisper_model) if self.whisper_model else ''
            if not model_disp:
                model_disp = str(self.whisper_model)
            lines.append(f"{t_func('label_whisper_model')} {model_disp}")
        except Exception:
            pass

        try:
            pause_opts = ['none', '1sec+', '2sec+', '3sec+']
            pause_disp = pause_opts[self.pause] if isinstance(self.pause, int) and 0 <= self.pause < len(pause_opts) else str(self.pause)
            lines.append(f"{t_func('label_pause')} {pause_disp}")
        except Exception:
            pass

        try:
            lines.append(f"{t_func('label_speaker')} {self.speaker_detection}")
        except Exception:
            pass

        try:
            lines.append(f"{t_func('label_overlapping')} {yn(self.overlapping)}")
        except Exception:
            pass

        try:
            lines.append(f"{t_func('label_disfluencies')} {yn(self.disfluencies)}")
        except Exception:
            pass

        try:
            lines.append(f"{t_func('label_timestamps')} {yn(self.timestamps)}")
        except Exception:
            pass

        return "\n".join([ln for ln in lines if ln])


class TranscriptionQueue:
    """Manages a queue of transcription jobs"""

    def __init__(self):
        self.jobs: List[TranscriptionJob] = []
        self.current_job: Optional[TranscriptionJob] = None

    def add_job(self, job: TranscriptionJob):
        """Add a job to the queue"""
        self.jobs.append(job)

    def get_waiting_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.WAITING]

    def get_running_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.jobs if job.status in [
            JobStatus.AUDIO_CONVERSION, JobStatus.SPEAKER_IDENTIFICATION,
            JobStatus.TRANSCRIPTION, JobStatus.CANCELING
        ]]

    def get_finished_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.FINISHED]

    def get_failed_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.ERROR]

    def get_canceled_jobs(self) -> List[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.CANCELED]

    def has_pending_jobs(self) -> bool:
        return len(self.get_waiting_jobs()) > 0

    def is_running(self) -> bool:
        return len(self.get_running_jobs()) > 0

    def get_next_waiting_job(self) -> Optional[TranscriptionJob]:
        waiting_jobs = self.get_waiting_jobs()
        return waiting_jobs[0] if waiting_jobs else None

    def get_queue_summary(self) -> dict:
        return {
            'total': len(self.jobs),
            'waiting': len(self.get_waiting_jobs()),
            'running': len(self.get_running_jobs()),
            'finished': len(self.get_finished_jobs()),
            'errors': len(self.get_failed_jobs()),
            'canceled': len(self.get_canceled_jobs()),
        }

    def is_empty(self) -> bool:
        return len(self.jobs) == 0

    def has_output_conflict(self, transcript_file: str, ignore_job: Optional[TranscriptionJob] = None) -> bool:
        """Check if another queue job uses the same output file."""
        try:
            target = os.path.abspath(transcript_file)
        except Exception:
            return False
        try:
            for j in self.jobs:
                try:
                    if not j or j is ignore_job:
                        continue
                    tf = getattr(j, 'transcript_file', None)
                    if not tf:
                        continue
                    if os.path.abspath(tf) == target and j.status not in [
                        JobStatus.ERROR, JobStatus.CANCELING, JobStatus.CANCELED
                    ]:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False


def create_transcription_job(
    audio_file=None, transcript_file=None, start_time=None, stop_time=None,
    language_name=None, whisper_model_name=None, speaker_detection=None,
    overlapping=None, timestamps=None, disfluencies=None, pause=None,
    languages=None, get_config=None, cli_mode=False
) -> TranscriptionJob:
    """Create a TranscriptionJob with all default values.

    languages: dict mapping language names to codes.
    get_config: callable(key, default) to read config values.
    """
    if languages is None:
        languages = {"Auto": "auto"}
    if get_config is None:
        get_config = lambda key, default: default

    job = TranscriptionJob()

    # File paths
    job.audio_file = audio_file or ''
    job.transcript_file = transcript_file or ''
    if job.transcript_file:
        job.file_ext = os.path.splitext(job.transcript_file)[1][1:]
        if job.file_ext not in ['html', 'txt', 'vtt', 'srt']:
            raise Exception(f'Unsupported output format: {job.file_ext}')

    # Time range
    job.start = start_time if start_time is not None else 0
    job.stop = stop_time if stop_time is not None else 0

    # Language - handle both language names and codes
    if language_name:
        if language_name in languages.values():
            job.language_name = next(name for name, code in languages.items() if code == language_name)
        elif language_name in languages.keys():
            job.language_name = language_name
        else:
            raise ValueError(f"Unknown language: {language_name}")
    else:
        job.language_name = 'Auto'

    # Model
    job.whisper_model = whisper_model_name or 'precise'

    # Processing options with defaults
    job.speaker_detection = speaker_detection if speaker_detection is not None else 'auto'
    job.overlapping = overlapping if overlapping is not None else True
    job.timestamps = timestamps if timestamps is not None else False
    job.disfluencies = disfluencies if disfluencies is not None else True

    # Pause setting
    if pause is not None:
        if isinstance(pause, str):
            pause_options = ['none', '1sec+', '2sec+', '3sec+']
            if pause in pause_options:
                job.pause = pause_options.index(pause)
            else:
                job.pause = 1
        else:
            job.pause = pause
    else:
        job.pause = 1

    # Config-based options
    job.whisper_beam_size = get_config('whisper_beam_size', 1)
    job.whisper_temperature = get_config('whisper_temperature', 0.0)
    job.whisper_compute_type = get_config('whisper_compute_type', 'default')

    # Optimize compute type for macOS (Apple Silicon)
    if platform.system() == "Darwin":
        model_str = str(job.whisper_model).lower()
        if 'int8' in model_str:
            if job.whisper_compute_type in ['default', 'float16']:
                job.whisper_compute_type = 'int8'
        else:
            if job.whisper_compute_type in ['default', 'float16']:
                job.whisper_compute_type = 'float32'

    job.timestamp_interval = get_config('timestamp_interval', 60_000)
    job.timestamp_color = get_config('timestamp_color', '#78909C')
    job.pause_marker = get_config('pause_seconds_marker', '.')
    job.auto_save = False if get_config('auto_save', 'True') == 'False' else True
    job.vad_threshold = float(get_config('voice_activity_detection_threshold', '0.5'))

    # Check for invalid VTT options
    if job.file_ext == 'vtt' and (job.pause > 0 or job.overlapping or job.timestamps):
        if cli_mode:
            print("Warning: VTT format doesn't support pause markers, overlapping speech, or timestamps. These options will be disabled.")
        job.pause = 0
        job.overlapping = False
        job.timestamps = False

    return job


def create_job_from_cli_args(args, languages=None, get_config=None) -> TranscriptionJob:
    """Create a TranscriptionJob from command line arguments"""
    start_time = utils.str_to_ms(args.start) if args.start else None
    stop_time = utils.str_to_ms(args.stop) if args.stop else None

    return create_transcription_job(
        audio_file=args.audio_file,
        transcript_file=args.output_file,
        start_time=start_time,
        stop_time=stop_time,
        language_name=args.language,
        whisper_model_name=args.model,
        speaker_detection=args.speaker_detection,
        overlapping=args.overlapping,
        timestamps=args.timestamps,
        disfluencies=args.disfluencies,
        pause=args.pause,
        languages=languages,
        get_config=get_config,
        cli_mode=True
    )
