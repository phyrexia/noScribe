# MeetingGenie - Transcription Runner
# Standalone pipeline that runs audio conversion → pyannote → whisper
# Communicates via callbacks (log_fn, progress_fn) instead of self.logn()

import os
import platform
import ssl
import time
import datetime
import multiprocessing as mp
import queue as pyqueue
import threading
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


def _mlx_whisper_target(args, q):
    """Wrapper that imports mlx-whisper only inside the subprocess."""
    from mlx_whisper_worker import mlx_whisper_proc_entrypoint
    mlx_whisper_proc_entrypoint(args, q)


def _apple_speech_target(args, q):
    """Wrapper that imports the Apple Speech worker only in the subprocess."""
    from apple_speech_worker import apple_speech_proc_entrypoint
    apple_speech_proc_entrypoint(args, q)


def _resolve_whisper_backend(job, app_dir: str) -> str:
    """Pick which whisper backend to spawn.

    Resolution order:
      1. Explicit config setting `whisper_backend` ∈ {'ct2', 'mlx',
         'apple-speech'}.
      2. 'auto' (or missing): look up `job.whisper_model`'s tier in
         `model_manager.MODELS`. If the tier reports `backend == 'mlx'`
         use MLX; `apple-speech` for the Apple Neural Engine tier;
         otherwise CT2. On Apple Silicon, an unknown tier
         (e.g. raw filesystem path) defaults to CT2 for compatibility.
    """
    cfg = (get_config('whisper_backend', 'auto') or 'auto').lower()
    if cfg in ('ct2', 'mlx', 'apple-speech'):
        return cfg

    # Try to map job.whisper_model back to a tier — it may be either a
    # tier key, an HF repo id, or a resolved filesystem path.
    candidate = str(job.whisper_model or '')
    for tier, entry in model_manager.MODELS.items():
        if candidate == tier:
            return entry.get('backend', 'ct2')
        if entry.get('backend') == 'mlx' and candidate == entry.get('hf_repo'):
            return 'mlx'
        # Path may end with the tier name (CT2 local dir).
        if entry.get('backend') == 'ct2' and candidate.rstrip('/').endswith(f"/{tier}"):
            return 'ct2'

    return 'ct2'


def _finalize_diarization_and_naming(
    diarization,
    embeddings,
    job: TranscriptionJob,
    tmp_audio: str,
    speaker_naming_fn,
    log_fn,
):
    """Persist diarization cache + prompt user for speaker names.

    Shared between the sequential and parallel orchestration paths.
    Returns a `speaker_name_map` dict (may be empty).
    """
    unique_speakers = sorted(set(s['label'] for s in diarization))
    log_fn(f"Found {len(unique_speakers)} speakers.", 'info')

    # Cache pyannote result so we don't need to reprocess on crash
    import json
    cache_path = os.path.splitext(job.transcript_file)[0] + "_diarization.json"
    try:
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

    speaker_name_map = {}
    log_fn(
        f"Embeddings: {len(embeddings)} speakers, callback: "
        f"{'yes' if speaker_naming_fn else 'no'}",
        'info',
    )
    if speaker_naming_fn and unique_speakers:
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

    return speaker_name_map


def _diarize_and_transcribe_parallel(
    job: TranscriptionJob,
    app_dir: str,
    tmp_audio: str,
    log_fn,
    progress_fn,
    cancel_check,
    speaker_naming_fn,
    device_fn,
    backend: str = 'mlx',
):
    """Run pyannote and whisper subprocesses concurrently.

    Used for backends whose CPU footprint is light: MLX (Metal GPU) and
    Apple Speech (ANE). Pyannote runs on MPS + minimal CPU, so the two
    can coexist without thrashing. CT2 stays sequential because it
    saturates the P-cores via `cpu_threads`.

    Whisper segments are buffered until pyannote completes (we need the
    speaker timeline before we can label them), then flushed in order.

    Returns:
      (diarization, embeddings, speaker_name_map, segments_data,
       diar_seconds, trans_seconds)
    """
    diar_start = time.time()
    trans_start = time.time()  # adjusted below; whisper drain starts now

    # ── 1. Spawn pyannote subprocess ────────────────────────────────
    force_pyannote_cpu = get_config('force_pyannote_cpu', '').lower() == 'true'
    py_args = {
        "device": 'cpu' if force_pyannote_cpu else '',
        "audio_path": tmp_audio,
        "num_speakers": (int(job.speaker_detection) if str(job.speaker_detection).isdigit() else None),
        "ignore_ssl": True,
        "proxy_url": get_config('proxy_url', ''),
    }
    ctx = mp.get_context("spawn")
    py_q = ctx.Queue()
    py_p = ctx.Process(target=_pyannote_target, args=(py_args, py_q))
    py_p.start()

    # Shared state between the pyannote drain thread and the main thread.
    py_state = {
        "diarization": [],
        "embeddings": {},
        "error": None,
    }
    py_done = threading.Event()
    cancel_flag = threading.Event()

    def _pyannote_drain():
        try:
            while True:
                if cancel_flag.is_set():
                    return
                try:
                    msg = py_q.get(timeout=0.2)
                except pyqueue.Empty:
                    if not py_p.is_alive():
                        py_state["error"] = f"Diarization worker exited (code {py_p.exitcode})"
                        return
                    continue

                mtype = msg.get("type")
                if mtype == "log":
                    lvl = msg.get("level", "info")
                    log_fn(f"[pyannote] {msg.get('msg', '')}", 'error' if lvl == 'error' else 'info')
                elif mtype == "progress":
                    pct = msg.get("pct", 0)
                    # Parallel: pyannote drives 5-50%; whisper drives 50-95.
                    progress_fn(5 + int(pct * 0.45))
                elif mtype == "device":
                    try:
                        device_fn(
                            msg.get("backend", ""),
                            msg.get("label", "CPU"),
                            role=msg.get("role", "diarization"),
                        )
                    except Exception:
                        pass
                elif mtype == "result":
                    if msg.get("ok"):
                        py_state["diarization"] = msg.get("segments", [])
                        py_state["embeddings"] = msg.get("embeddings", {})
                    else:
                        err = msg.get("error", "Diarization failed")
                        trace = msg.get("trace", "")
                        log_fn(f"[pyannote] Error: {err}", 'error')
                        if trace:
                            log_fn(f"[pyannote] Traceback:\n{trace}", 'error')
                        py_state["error"] = err
                    return
        finally:
            py_done.set()

    py_thread = threading.Thread(target=_pyannote_drain, daemon=True)
    py_thread.start()

    # ── 2. Spawn MLX whisper subprocess ─────────────────────────────
    force_cpu = get_config('force_whisper_cpu', '').lower() == 'true'
    from config import detect_thread_count, ALL_LANGUAGES
    number_threads = int(get_config('threads', detect_thread_count()))
    language_code = None
    if job.language_name not in ('Auto', 'Multilingual'):
        language_code = ALL_LANGUAGES.get(job.language_name)
    w_args = {
        "model_name_or_path": job.whisper_model,
        "device": 'cpu' if force_cpu else 'auto',
        "compute_type": job.whisper_compute_type,
        "cpu_threads": number_threads,
        "local_files_only": False,  # MLX downloads via HF hub
        "audio_path": tmp_audio,
        "language_name": job.language_name,
        "language_code": language_code,
        "disfluencies": job.disfluencies,
        "beam_size": job.whisper_beam_size,
        "word_timestamps": False,
        "vad_filter": True,
        "vad_threshold": job.vad_threshold,
        "locale": get_config('locale', 'en'),
    }
    if backend == 'mlx':
        target = _mlx_whisper_target
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        log_fn("Using MLX (Metal GPU) whisper backend.", 'info')
    elif backend == 'apple-speech':
        target = _apple_speech_target
        log_fn("Using Apple Neural Engine whisper backend.", 'info')
    else:
        # Defensive fallback — should never hit because the dispatcher only
        # picks the parallel path for ANE-friendly backends.
        target = _whisper_target

    w_q = ctx.Queue()
    w_p = ctx.Process(target=target, args=(w_args, w_q))
    w_p.start()
    trans_start = time.time()

    # ── 3. Drain whisper on the main thread; buffer until labels ready ──
    pending_segments = []  # tuples (start_ms, end_ms, text)
    segments_data = []
    speaker_name_map = {}
    diarization_for_lookup = []
    labels_ready = False
    diar_seconds = 0.0
    trans_seconds = 0.0

    def _flush_pending():
        """Drain `pending_segments` into `segments_data` using diarization labels."""
        if not pending_segments:
            return
        from transcription_service import find_speaker
        for (s_ms, e_ms, txt) in pending_segments:
            speaker = ""
            if diarization_for_lookup:
                speaker = find_speaker(
                    diarization_for_lookup, s_ms, e_ms,
                    speaker_name_map=speaker_name_map,
                    overlapping_enabled=job.overlapping,
                )
            segments_data.append({
                "text": txt,
                "start_ms": job.start + s_ms,
                "end_ms": job.start + e_ms,
                "speaker": speaker,
            })
            log_fn(f"{speaker + ': ' if speaker else ''}{txt}", 'info')
        pending_segments.clear()

    def _try_finalize_labels():
        """If pyannote is done, materialize speaker_name_map and unlock labels."""
        nonlocal labels_ready, diar_seconds, diarization_for_lookup, speaker_name_map
        if labels_ready:
            return
        if not py_done.is_set():
            return
        diar_seconds = time.time() - diar_start
        if py_state["error"]:
            log_fn(f"Diarization failed — continuing without speaker labels.", 'error')
            labels_ready = True
            return
        diarization_for_lookup = py_state["diarization"]
        # Cache write + speaker-naming dialog (may block the main thread —
        # but whisper continues running in its own process).
        try:
            speaker_name_map = _finalize_diarization_and_naming(
                py_state["diarization"], py_state["embeddings"],
                job, tmp_audio, speaker_naming_fn, log_fn,
            )
        except Exception as e:
            log_fn(f"Speaker naming error: {e}", 'info')
            speaker_name_map = {}
        log_fn(f"Diarization complete. ({diar_seconds:.0f}s)", 'info')
        labels_ready = True
        _flush_pending()

    try:
        while True:
            # Opportunistically finalize labels mid-stream.
            _try_finalize_labels()

            try:
                msg = w_q.get(timeout=0.2)
            except pyqueue.Empty:
                if cancel_check():
                    cancel_flag.set()
                    try:
                        w_p.terminate()
                    except Exception:
                        pass
                    try:
                        py_p.terminate()
                    except Exception:
                        pass
                    raise Exception("Canceled by user")
                if not w_p.is_alive():
                    raise Exception(f"Whisper worker exited (code {w_p.exitcode})")
                continue

            mtype = msg.get("type")
            if mtype == "device":
                try:
                    device_fn(
                        msg.get("backend", ""),
                        msg.get("label", "CPU"),
                        role=msg.get("role", "transcription"),
                    )
                except Exception:
                    pass
            elif mtype == "log":
                lvl = msg.get("level", "info")
                log_fn(f"[whisper] {msg.get('msg', '')}", 'error' if lvl == 'error' else 'info')
            elif mtype == "progress":
                pct = msg.get("pct", 0)
                progress_fn(50 + int(pct * 0.45))
            elif mtype == "finished":
                continue
            elif mtype == "segment":
                seg = msg.get("segment", {})
                text = (seg.get("text") or "").strip()
                start_ms = round((seg.get("start") or 0) * 1000)
                end_ms = round((seg.get("end") or 0) * 1000)

                if labels_ready:
                    from transcription_service import find_speaker
                    speaker = ""
                    if diarization_for_lookup:
                        speaker = find_speaker(
                            diarization_for_lookup, start_ms, end_ms,
                            speaker_name_map=speaker_name_map,
                            overlapping_enabled=job.overlapping,
                        )
                    segments_data.append({
                        "text": text,
                        "start_ms": job.start + start_ms,
                        "end_ms": job.start + end_ms,
                        "speaker": speaker,
                    })
                    log_fn(f"{speaker + ': ' if speaker else ''}{text}", 'info')
                else:
                    pending_segments.append((start_ms, end_ms, text))
            elif mtype == "result":
                if not msg.get("ok"):
                    err = msg.get("error", "Transcription failed")
                    trace = msg.get("trace", "")
                    log_fn(f"[whisper] Error: {err}", 'error')
                    if trace:
                        log_fn(f"[whisper] Traceback:\n{trace}", 'error')
                    cancel_flag.set()
                    try:
                        py_p.terminate()
                    except Exception:
                        pass
                    raise Exception(f"[WHISPER] {err}")
                trans_seconds = time.time() - trans_start
                break
    finally:
        try:
            w_p.join(timeout=0.5)
        except Exception:
            pass
        if w_p.is_alive():
            w_p.terminate()

    # Whisper finished. If pyannote is still running, wait for it now.
    if not py_done.is_set():
        log_fn("Whisper finished; waiting for diarization to complete...", 'info')
        while not py_done.is_set():
            if cancel_check():
                cancel_flag.set()
                try:
                    py_p.terminate()
                except Exception:
                    pass
                raise Exception("Canceled by user")
            py_done.wait(timeout=0.5)

    try:
        py_p.join(timeout=0.5)
    except Exception:
        pass
    if py_p.is_alive():
        py_p.terminate()
    py_thread.join(timeout=1.0)

    # Final attempt to flush any buffered segments after both processes done.
    _try_finalize_labels()
    if not labels_ready:
        # Defensive: labels never readied (shouldn't happen, but flush raw).
        labels_ready = True
        _flush_pending()
    else:
        _flush_pending()

    if diar_seconds == 0.0:
        diar_seconds = time.time() - diar_start

    return (
        py_state["diarization"],
        py_state["embeddings"],
        speaker_name_map,
        segments_data,
        diar_seconds,
        trans_seconds,
    )


def run_transcription(
    job: TranscriptionJob,
    app_dir: str,
    log_fn=None,
    progress_fn=None,
    cancel_check=None,
    speaker_naming_fn=None,
    device_fn=None,
    whisper_pool=None,
):
    """Run a full transcription pipeline for one job.

    log_fn(text, level='info')  — 'info', 'highlight', 'error'
    progress_fn(pct)            — 0-100
    cancel_check()              — returns True if user requested cancel
    speaker_naming_fn(speakers_data, audio_path) — returns {label: name} map, or None to skip
    device_fn(backend, label)   — notified when the worker publishes its compute device
    """
    if log_fn is None:
        log_fn = lambda text, level='info': print(f"[{level}] {text}")
    if progress_fn is None:
        progress_fn = lambda pct: None
    if cancel_check is None:
        cancel_check = lambda: False
    if device_fn is None:
        device_fn = lambda backend, label, role=None: None

    tmpdir = TemporaryDirectory('MeetingGenie')
    tmp_audio = os.path.join(tmpdir.name, 'tmp_audio.wav')

    try:
        job.set_running()
        timings = {}

        # ── 1. Audio conversion (PyAV) ───────────────────────────────
        t_step = time.time()
        log_fn("Converting audio...", 'highlight')

        stop_ms = int(job.stop) if int(job.stop) > 0 else None
        from audio import ToWav, AudioConversionCanceled, AudioConversionError
        try:
            with ToWav(
                job.audio_file,
                tmp_audio,
                start_ms=int(job.start),
                stop_ms=stop_ms,
                should_cancel=cancel_check,
            ) as conv:
                conv.run()
        except AudioConversionCanceled:
            raise Exception("Canceled by user")
        except AudioConversionError as e:
            raise Exception(f"Audio conversion failed: {e}")

        timings['audio_conversion'] = time.time() - t_step
        log_fn(f"Audio conversion complete. ({timings['audio_conversion']:.0f}s)", 'info')
        progress_fn(5)

        # Resolve whisper backend early — needed to choose between the
        # sequential (CT2) and parallel (MLX on Apple Silicon) orchestrations.
        backend = _resolve_whisper_backend(job, app_dir)

        need_pyannote = job.speaker_detection not in ('none', 'off')

        # Check for cached diarization first (works for both paths).
        cached_diarization = []
        cached_embeddings = {}
        cache_path = os.path.splitext(job.transcript_file)[0] + "_diarization.json"
        if need_pyannote and os.path.exists(cache_path):
            try:
                import json as _json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = _json.load(f)
                cached_diarization = cached.get("segments", [])
                cached_embeddings = cached.get("embeddings", {})
            except Exception:
                cached_diarization = []
                cached_embeddings = {}

        # Decide orchestration mode. MLX (Metal GPU) and Apple Speech (ANE)
        # both have light CPU footprints, so they can run concurrently with
        # pyannote (MPS + light CPU). CT2 (CPU) is sequential because it
        # saturates the P-cores via `cpu_threads`.
        run_parallel = (
            backend in ('mlx', 'apple-speech')
            and need_pyannote
            and not cached_diarization
        )

        diarization = []
        embeddings = {}
        speaker_name_map = {}
        segments_data = []

        if run_parallel:
            _backend_label = {
                'mlx': 'MLX',
                'apple-speech': 'Apple ANE',
            }.get(backend, backend.upper())
            log_fn(
                f"Running diarization and transcription concurrently "
                f"({_backend_label} + pyannote).",
                'highlight',
            )
            job.status = JobStatus.SPEAKER_IDENTIFICATION
            (diarization, embeddings, speaker_name_map, segments_data,
             diar_seconds, trans_seconds) = _diarize_and_transcribe_parallel(
                job=job,
                app_dir=app_dir,
                tmp_audio=tmp_audio,
                log_fn=log_fn,
                progress_fn=progress_fn,
                cancel_check=cancel_check,
                speaker_naming_fn=speaker_naming_fn,
                device_fn=device_fn,
                backend=backend,
            )
            timings['diarization'] = diar_seconds
            timings['transcription'] = trans_seconds
            log_fn(f"Transcription complete. ({trans_seconds:.0f}s)", 'info')
            progress_fn(95)

        # ── 2. Speaker diarization (pyannote) — sequential path ──────
        if not run_parallel:
            t_step = time.time()
            if need_pyannote:
                if cached_diarization:
                    diarization = cached_diarization
                    embeddings = cached_embeddings
                    log_fn(f"Loaded cached diarization ({len(diarization)} segments)", 'highlight')
                    progress_fn(50)

                if not diarization:
                    log_fn("Identifying speakers...", 'highlight')
                    job.status = JobStatus.SPEAKER_IDENTIFICATION

                    force_pyannote_cpu = get_config('force_pyannote_cpu', '').lower() == 'true'
                    args = {
                        "device": 'cpu' if force_pyannote_cpu else '',  # '' = auto-detect MPS/CUDA
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
                            elif mtype == "device":
                                try:
                                    device_fn(
                                        msg.get("backend", ""),
                                        msg.get("label", "CPU"),
                                        role=msg.get("role", "diarization"),
                                    )
                                except Exception:
                                    pass
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

                speaker_name_map = _finalize_diarization_and_naming(
                    diarization, embeddings, job, tmp_audio,
                    speaker_naming_fn, log_fn,
                )

                timings['diarization'] = time.time() - t_step
                log_fn(f"Diarization complete. ({timings['diarization']:.0f}s)", 'info')
                progress_fn(50)
            else:
                progress_fn(50)

        # ── 3. Transcription (whisper) ───────────────────────────────
        if not run_parallel:
            t_step = time.time()
            log_fn("Transcribing...", 'highlight')
            job.status = JobStatus.TRANSCRIPTION

            force_cpu = get_config('force_whisper_cpu', '').lower() == 'true'
            from config import detect_thread_count
            number_threads = int(get_config('threads', detect_thread_count()))

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
                "beam_size": job.whisper_beam_size,
                "word_timestamps": False,
                "vad_filter": True,
                "vad_threshold": job.vad_threshold,
                "locale": get_config('locale', 'en'),
            }

            if backend == 'mlx':
                target = _mlx_whisper_target
                # MLX backend does its own model resolution via HF hub; local-only
                # flag is irrelevant.
                w_args["local_files_only"] = False
                # MLX downloads from HF on first use — disable the offline gate the
                # parent process set for pyannote/CT2.
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                log_fn("Using MLX (Metal GPU) whisper backend.", 'info')
            elif backend == 'apple-speech':
                target = _apple_speech_target
                log_fn("Using Apple Neural Engine (on-device Speech) backend.", 'info')
            else:
                target = _whisper_target

            # Persistent worker pool path (CT2 only). MLX uses spawn-per-job.
            use_pool = (
                whisper_pool is not None
                and backend == 'ct2'
                and getattr(whisper_pool, 'is_alive', lambda: False)()
            )

            p = None
            if use_pool:
                try:
                    q = whisper_pool.run_job(w_args)
                    log_fn("Using persistent whisper worker pool.", 'info')
                except Exception as e:
                    log_fn(f"Whisper pool unavailable ({e}); falling back to spawn.", 'info')
                    use_pool = False

            if not use_pool:
                ctx = mp.get_context("spawn")
                q = ctx.Queue()
                p = ctx.Process(target=target, args=(w_args, q))
                p.start()

            try:
                while True:
                    try:
                        msg = q.get(timeout=0.2)
                    except pyqueue.Empty:
                        if cancel_check():
                            if use_pool:
                                try:
                                    whisper_pool.cancel_current()
                                except Exception:
                                    pass
                                try:
                                    whisper_pool.shutdown(timeout=0.5)
                                except Exception:
                                    pass
                            elif p is not None:
                                p.terminate()
                            raise Exception("Canceled by user")
                        if use_pool:
                            if not whisper_pool.is_alive():
                                raise Exception("Whisper pool worker died unexpectedly")
                        elif p is not None and not p.is_alive():
                            raise Exception(f"Whisper worker exited (code {p.exitcode})")
                        continue

                    mtype = msg.get("type")
                    if mtype == "device":
                        try:
                            device_fn(
                                msg.get("backend", ""),
                                msg.get("label", "CPU"),
                                role=msg.get("role", "transcription"),
                            )
                        except Exception:
                            pass
                    elif mtype == "log":
                        lvl = msg.get("level", "info")
                        log_fn(f"[whisper] {msg.get('msg', '')}", 'error' if lvl == 'error' else 'info')
                    elif mtype == "progress":
                        pct = msg.get("pct", 0)
                        progress_fn(50 + int(pct * 0.45))  # 50-95%
                    elif mtype == "finished":
                        continue
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
                if p is not None:
                    try:
                        p.join(timeout=0.5)
                    except Exception:
                        pass
                    if p.is_alive():
                        p.terminate()
                # Pool worker stays alive across jobs — no cleanup here.

            timings['transcription'] = time.time() - t_step
            log_fn(f"Transcription complete. ({timings['transcription']:.0f}s)", 'info')
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
            total_secs = int(duration.total_seconds())
            total_str = f"{total_secs // 60}:{total_secs % 60:02d}"
            breakdown = []
            for step, label in [('audio_conversion', 'Audio'), ('diarization', 'Speakers'), ('transcription', 'Whisper')]:
                if step in timings:
                    s = int(timings[step])
                    breakdown.append(f"{label}: {s // 60}:{s % 60:02d}")
            log_fn(f"━━━ Completed in {total_str} ━━━", 'highlight')
            if breakdown:
                log_fn(f"  {' | '.join(breakdown)}", 'info')

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
