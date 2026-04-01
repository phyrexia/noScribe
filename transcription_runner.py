# MeetingGenie - Transcription Runner
# Standalone pipeline that runs ffmpeg → pyannote → whisper
# Communicates via callbacks (log_fn, progress_fn) instead of self.logn()

import os
import platform
import shlex
import ssl
import time
import datetime
import multiprocessing as mp
import queue as pyqueue
from subprocess import Popen, DEVNULL, STDOUT
from tempfile import TemporaryDirectory
from pathlib import Path

# Force offline mode for HuggingFace/pyannote — models are bundled locally
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
# Bypass corporate SSL
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Also patch ssl globally for this process
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from models import TranscriptionJob, JobStatus
from config import get_config
import utils
import model_manager


def _pyannote_target(args, q):
    """Wrapper that imports pyannote only inside the subprocess."""
    from pyannote_mp_worker import pyannote_proc_entrypoint
    pyannote_proc_entrypoint(args, q)


def _whisper_target(args, q):
    """Wrapper that imports whisper only inside the subprocess."""
    from whisper_mp_worker import whisper_proc_entrypoint
    whisper_proc_entrypoint(args, q)


def _find_ffmpeg(app_dir: str) -> str:
    """Locate the ffmpeg binary for the current platform."""
    if platform.system() == "Darwin":
        arm64 = os.path.join(app_dir, 'ffmpeg-arm64')
        if platform.machine() == "arm64" and os.path.exists(arm64):
            return arm64
        return os.path.join(app_dir, 'ffmpeg')
    elif platform.system() == "Windows":
        return os.path.join(app_dir, 'ffmpeg.exe')
    elif platform.system() == "Linux":
        return os.path.join(app_dir, 'ffmpeg-linux-x86_64')
    raise Exception('Platform not supported.')


def run_transcription(
    job: TranscriptionJob,
    app_dir: str,
    log_fn=None,
    progress_fn=None,
    cancel_check=None,
):
    """Run a full transcription pipeline for one job.

    log_fn(text, level='info')  — 'info', 'highlight', 'error'
    progress_fn(pct)            — 0-100
    cancel_check()              — returns True if user requested cancel
    """
    if log_fn is None:
        log_fn = lambda text, level='info': print(f"[{level}] {text}")
    if progress_fn is None:
        progress_fn = lambda pct: None
    if cancel_check is None:
        cancel_check = lambda: False

    tmpdir = TemporaryDirectory('MeetingGenie')
    tmp_audio = os.path.join(tmpdir.name, 'tmp_audio.wav')

    try:
        job.set_running()

        # ── 1. Audio conversion (ffmpeg) ─────────────────────────────
        log_fn("Converting audio...", 'highlight')

        end_pos = f'-t {int(job.stop) - int(job.start)}ms' if int(job.stop) > 0 else ''
        arguments = f' -loglevel warning -hwaccel auto -y -ss {job.start}ms -i "{job.audio_file}" {end_pos} -ar 16000 -ac 1 -c:a pcm_s16le "{tmp_audio}"'

        ffmpeg_path = _find_ffmpeg(app_dir)
        if platform.system() == 'Windows':
            ffmpeg_cmd = ffmpeg_path + arguments
        else:
            ffmpeg_cmd = shlex.split(ffmpeg_path + arguments)

        proc = Popen(ffmpeg_cmd, stdout=DEVNULL, stderr=STDOUT,
                      universal_newlines=True, encoding='utf-8')
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if cancel_check():
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                raise Exception("Canceled by user")
            time.sleep(0.1)

        if proc.returncode and proc.returncode > 0:
            raise Exception("FFmpeg conversion failed")

        log_fn("Audio conversion complete.", 'info')
        progress_fn(5)

        # ── 2. Speaker diarization (pyannote) ────────────────────────
        diarization = []
        if job.speaker_detection != 'none' and job.speaker_detection != 'off':
            log_fn("Identifying speakers...", 'highlight')
            job.status = JobStatus.SPEAKER_IDENTIFICATION

            args = {
                "device": 'cpu',
                "audio_path": tmp_audio,
                "num_speakers": (int(job.speaker_detection) if str(job.speaker_detection).isdigit() else None),
                "ignore_ssl": True,
                "proxy_url": get_config('proxy_url', ''),
            }

            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=_pyannote_target, args=(args, q))
            p.start()

            try:
                while True:
                    try:
                        msg = q.get(timeout=0.2)
                    except pyqueue.Empty:
                        if cancel_check():
                            p.terminate()
                            raise Exception("Canceled by user")
                        if not p.is_alive():
                            raise Exception(f"Diarization worker exited (code {p.exitcode})")
                        continue

                    mtype = msg.get("type")
                    if mtype == "log":
                        lvl = msg.get("level", "info")
                        log_fn(f"[pyannote] {msg.get('msg', '')}", 'error' if lvl == 'error' else 'info')
                    elif mtype == "progress":
                        pct = msg.get("pct", 0)
                        progress_fn(5 + int(pct * 0.45))  # 5-50%
                    elif mtype == "result":
                        if msg.get("ok"):
                            diarization = msg.get("segments", [])
                        else:
                            err = msg.get("error", "Diarization failed")
                            trace = msg.get("trace", "")
                            log_fn(f"[pyannote] Error: {err}", 'error')
                            if trace:
                                log_fn(f"[pyannote] Traceback:\n{trace}", 'error')
                            raise Exception(f"[PYANNOTE] {err}")
                        break
            finally:
                try:
                    p.join(timeout=0.5)
                except Exception:
                    pass
                if p.is_alive():
                    p.terminate()

            log_fn(f"Found {len(set(s['label'] for s in diarization))} speakers.", 'info')
            progress_fn(50)
        else:
            progress_fn(50)

        # ── 3. Transcription (whisper) ───────────────────────────────
        log_fn("Transcribing...", 'highlight')
        job.status = JobStatus.TRANSCRIPTION

        force_cpu = get_config('force_whisper_cpu', '').lower() == 'true'
        number_threads = int(get_config('threads', 4))

        language_code = None
        from config import ALL_LANGUAGES
        if job.language_name not in ('Auto', 'Multilingual'):
            language_code = ALL_LANGUAGES.get(job.language_name)

        w_args = {
            "model_name_or_path": job.whisper_model,
            "device": 'cpu' if force_cpu else 'auto',
            "compute_type": job.whisper_compute_type,
            "cpu_threads": number_threads,
            "local_files_only": True,
            "audio_path": tmp_audio,
            "language_name": job.language_name,
            "language_code": language_code,
            "disfluencies": job.disfluencies,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_threshold": job.vad_threshold,
            "locale": get_config('locale', 'en'),
        }

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_whisper_target, args=(w_args, q))
        p.start()

        segments_text = []
        try:
            while True:
                try:
                    msg = q.get(timeout=0.2)
                except pyqueue.Empty:
                    if cancel_check():
                        p.terminate()
                        raise Exception("Canceled by user")
                    if not p.is_alive():
                        raise Exception(f"Whisper worker exited (code {p.exitcode})")
                    continue

                mtype = msg.get("type")
                if mtype == "log":
                    lvl = msg.get("level", "info")
                    log_fn(f"[whisper] {msg.get('msg', '')}", 'error' if lvl == 'error' else 'info')
                elif mtype == "progress":
                    pct = msg.get("pct", 0)
                    progress_fn(50 + int(pct * 0.45))  # 50-95%
                elif mtype == "segment":
                    seg = msg.get("segment", {})
                    text = seg.get("text", "").strip()
                    start_ms = round(seg.get("start", 0) * 1000)
                    end_ms = round(seg.get("end", 0) * 1000)

                    # Find speaker if diarization available
                    speaker = ""
                    if diarization:
                        from transcription_service import find_speaker
                        speaker = find_speaker(
                            diarization, start_ms, end_ms,
                            overlapping_enabled=job.overlapping
                        )

                    if speaker:
                        line = f"{speaker}: {text}"
                    else:
                        line = text

                    # Add timestamp if enabled
                    if job.timestamps:
                        ts = utils.ms_to_str(job.start + start_ms)
                        line = f"[{ts}] {line}"

                    segments_text.append(line)
                    log_fn(text, 'info')

                elif mtype == "result":
                    if not msg.get("ok"):
                        err = msg.get("error", "Transcription failed")
                        trace = msg.get("trace", "")
                        log_fn(f"[whisper] Error: {err}", 'error')
                        if trace:
                            log_fn(f"[whisper] Traceback:\n{trace}", 'error')
                        raise Exception(f"[WHISPER] {err}")
                    break
        finally:
            try:
                p.join(timeout=0.5)
            except Exception:
                pass
            if p.is_alive():
                p.terminate()

        progress_fn(95)

        # ── 4. Save output ───────────────────────────────────────────
        log_fn("Saving transcript...", 'highlight')

        output_text = "\n".join(segments_text)

        ext = job.file_ext or os.path.splitext(job.transcript_file)[1][1:]
        if ext == 'txt' or ext == '':
            with open(job.transcript_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
        elif ext == 'srt':
            srt = ""
            for i, seg in enumerate(segments_text):
                # Simple SRT: sequential numbering
                srt += f"{i+1}\n00:00:00,000 --> 00:00:00,000\n{seg}\n\n"
            with open(job.transcript_file, 'w', encoding='utf-8') as f:
                f.write(srt)
        elif ext == 'vtt':
            vtt = "WEBVTT\n\n"
            for seg in segments_text:
                vtt += f"{seg}\n\n"
            with open(job.transcript_file, 'w', encoding='utf-8') as f:
                f.write(vtt)
        else:
            # Default: plain text
            with open(job.transcript_file, 'w', encoding='utf-8') as f:
                f.write(output_text)

        job.set_finished()
        progress_fn(100)
        log_fn(f"Saved: {job.transcript_file}", 'highlight')

        duration = job.get_duration()
        if duration:
            secs = int(duration.total_seconds())
            log_fn(f"Completed in {secs // 60}:{secs % 60:02d}", 'info')

    except Exception as e:
        msg = str(e)
        if "Canceled" in msg:
            job.set_canceled(msg)
            log_fn("Transcription canceled.", 'error')
        else:
            import traceback as tb
            full_trace = tb.format_exc()
            job.set_error(msg)
            log_fn(f"Error: {msg}", 'error')
            log_fn(f"Full traceback:\n{full_trace}", 'error')
            print(full_trace)  # also to terminal
    finally:
        try:
            tmpdir.cleanup()
        except Exception:
            pass
