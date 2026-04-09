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
    speaker_naming_fn=None,
):
    """Run a full transcription pipeline for one job.

    log_fn(text, level='info')  — 'info', 'highlight', 'error'
    progress_fn(pct)            — 0-100
    cancel_check()              — returns True if user requested cancel
    speaker_naming_fn(speakers_data, audio_path) — returns {label: name} map, or None to skip
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
        embeddings = {}
        if job.speaker_detection != 'none' and job.speaker_detection != 'off':
            # Check for cached diarization result (avoids re-processing on crash)
            import json as _json
            cache_path = os.path.splitext(job.transcript_file)[0] + "_diarization.json"
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = _json.load(f)
                    diarization = cached.get("segments", [])
                    embeddings = cached.get("embeddings", {})
                    if diarization:
                        log_fn(f"Loaded cached diarization ({len(diarization)} segments)", 'highlight')
                        progress_fn(50)
                except Exception:
                    diarization = []
                    embeddings = {}

            if not diarization:
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
                                embeddings = msg.get("embeddings", {})
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

            unique_speakers = sorted(set(s['label'] for s in diarization))
            log_fn(f"Found {len(unique_speakers)} speakers.", 'info')

            # Cache pyannote result so we don't need to reprocess on crash
            import json
            cache_path = os.path.splitext(job.transcript_file)[0] + "_diarization.json"
            try:
                # Convert embeddings values to lists for JSON serialization
                serializable_emb = {}
                for k, v in embeddings.items():
                    if hasattr(v, 'tolist'):
                        serializable_emb[k] = v.tolist()
                    elif isinstance(v, list):
                        serializable_emb[k] = v
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"segments": diarization, "embeddings": serializable_emb}, f)
                log_fn(f"Diarization cached: {os.path.basename(cache_path)}", 'info')
            except Exception as ce:
                log_fn(f"Cache save warning: {ce}", 'info')

            # Speaker naming — ask the user to assign names
            speaker_name_map = {}
            log_fn(f"Embeddings: {len(embeddings)} speakers, callback: {'yes' if speaker_naming_fn else 'no'}", 'info')
            if speaker_naming_fn and unique_speakers:
                # Build speaker data for the dialog
                speaker_segs_map = {}
                for seg in diarization:
                    speaker_segs_map.setdefault(seg["label"], []).append(seg)

                speakers_data = []
                for lbl in unique_speakers:
                    short = f'S{lbl[8:]}' if len(lbl) > 8 else lbl
                    emb = embeddings.get(lbl)
                    matched_name, sim = None, 0.0
                    if emb:
                        try:
                            import speaker_db
                            matched_name, sim = speaker_db.find_match(emb)
                        except Exception:
                            pass
                    segs = speaker_segs_map.get(lbl, [])
                    sorted_segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
                    samples = [{"start": s["start"], "end": s["end"]} for s in sorted_segs[:2]]
                    speakers_data.append({
                        'label': lbl,
                        'short_label': short,
                        'matched_name': matched_name,
                        'similarity': sim,
                        'embedding': emb,
                        'samples': samples,
                    })

                try:
                    result = speaker_naming_fn(speakers_data, tmp_audio)
                    if result:
                        speaker_name_map = result
                        named = [f"{v} ({k})" for k, v in result.items()]
                        log_fn(f"Speakers identified: {', '.join(named)}", 'highlight')
                    else:
                        log_fn("Speaker naming skipped — using S01, S02...", 'info')
                except Exception as e:
                    log_fn(f"Speaker naming error: {e}", 'info')

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

        # Each segment: {"text": str, "start_ms": int, "end_ms": int, "speaker": str}
        segments_data = []
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
                            speaker_name_map=speaker_name_map,
                            overlapping_enabled=job.overlapping
                        )

                    segments_data.append({
                        "text": text,
                        "start_ms": job.start + start_ms,
                        "end_ms": job.start + end_ms,
                        "speaker": speaker,
                    })
                    log_fn(f"{speaker + ': ' if speaker else ''}{text}", 'info')

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

        def _ms_to_srt_ts(ms):
            s, ms_r = divmod(ms, 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d},{ms_r:03d}"

        def _ms_to_vtt_ts(ms):
            s, ms_r = divmod(ms, 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms_r:03d}"

        def _seg_line(seg, include_ts=False):
            spk = seg["speaker"]
            txt = seg["text"]
            line = f"{spk}: {txt}" if spk else txt
            if include_ts:
                ts = utils.ms_to_str(seg["start_ms"])
                line = f"[{ts}] {line}"
            return line

        ext = job.file_ext or os.path.splitext(job.transcript_file)[1][1:]

        if ext == 'srt':
            out = ""
            for i, seg in enumerate(segments_data):
                start = _ms_to_srt_ts(seg["start_ms"])
                end = _ms_to_srt_ts(seg["end_ms"])
                spk = seg["speaker"]
                txt = f"[{spk}] {seg['text']}" if spk else seg["text"]
                out += f"{i+1}\n{start} --> {end}\n{txt}\n\n"
        elif ext == 'vtt':
            out = "WEBVTT\n\n"
            for i, seg in enumerate(segments_data):
                start = _ms_to_vtt_ts(seg["start_ms"])
                end = _ms_to_vtt_ts(seg["end_ms"])
                spk = seg["speaker"]
                txt = f"<v {spk}>{seg['text']}" if spk else seg["text"]
                out += f"{i+1}\n{start} --> {end}\n{txt}\n\n"
        elif ext == 'html':
            # Simple HTML with speaker names and timestamps
            lines = [f"<html><body><h2>{Path(job.audio_file).stem}</h2>"]
            for seg in segments_data:
                spk = seg["speaker"]
                ts = utils.ms_to_str(seg["start_ms"])
                txt = seg["text"]
                if spk:
                    lines.append(f"<p><b>{spk}</b> <span style='color:#78909C'>[{ts}]</span> {txt}</p>")
                else:
                    lines.append(f"<p><span style='color:#78909C'>[{ts}]</span> {txt}</p>")
            lines.append("</body></html>")
            out = "\n".join(lines)
        else:
            # txt or unknown — plain text
            out = "\n".join(_seg_line(seg, include_ts=job.timestamps) for seg in segments_data)

        with open(job.transcript_file, 'w', encoding='utf-8') as f:
            f.write(out)

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
