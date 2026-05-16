import gc
import os
import platform
import ssl
import traceback
from dataclasses import asdict, is_dataclass

# Offline + SSL bypass for corporate environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from i18n import t


def whisper_proc_entrypoint(args: dict, q):
    """
    Runs in a child process. Streams progress/logs to parent via `q`.
    Messages put on `q` are dicts with one of the following shapes:
      {"type": "log", "level": "info"|"warn"|"error"|"debug", "msg": "..."}
      {"type": "progress", "pct": float, "detail": "..."}   # optional
      {"type": "result", "ok": True, "segments": [...], "info": {...}}
      {"type": "result", "ok": False, "error": str, "trace": str}
    """
    try:
        # Import heavy libs only in the child process

        from faster_whisper import WhisperModel, BatchedInferencePipeline
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        import torch
        import yaml
        import i18n

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        # Initialize i18n in child process (PyInstaller uses spawn; no globals shared)
        try:
            app_dir = os.path.abspath(os.path.dirname(__file__))
            i18n.set('filename_format', '{locale}.{format}')
            # Ensure translations directory is available to python-i18n
            i18n.load_path.append(os.path.join(app_dir, 'trans'))
            i18n.set('fallback', 'en')
            # Use locale passed by parent when available
            child_locale = args.get('locale') or 'en'
            i18n.set('locale', child_locale)
        except Exception:
            # Safe fallback: leave i18n defaults; keys may pass through
            pass
        
        # determine device
        device = args.get("device", "")
        if device != 'cpu':
            if platform.system() == "Darwin":  # MAC
                device = 'auto'
            elif platform.system() in ('Windows', 'Linux'):
                try:
                    device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
                except:
                    device = 'cpu'
            else:
                raise Exception('Platform not supported yet.')

        # Report actual compute device to parent so the UI can show a
        # truthful badge. The CT2 backend is CPU-only on macOS.
        try:
            if device == 'cuda':
                q.put({"type": "device", "role": "transcription", "backend": "cuda", "label": "NVIDIA GPU"})
            else:
                q.put({"type": "device", "role": "transcription", "backend": "ct2-cpu", "label": "CPU"})
        except Exception:
            pass

        # Build model in child using provided options
        model = WhisperModel(
            args["model_name_or_path"],
            device=device,
            compute_type=args.get("compute_type", "float16"),
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )
        # Wrap with BatchedInferencePipeline: VAD-segmented chunks processed
        # in parallel (batch_size at a time) for ~4x throughput on CPU.
        batched = BatchedInferencePipeline(model=model)

        # Define callbacks that forward to parent via queue (not used by faster-whisper directly, but kept for parity)
        def log_cb(level, msg):
            plog(level, msg)

        # Prepare audio and VAD
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

        sampling_rate = model.feature_extractor.sampling_rate
        audio = decode_audio(audio_path, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate
        log_cb("info", t('vad'))

        # VAD options
        vad_threshold = float(args.get("vad_threshold", 0.5))
        try:
            vad_parameters = VadOptions(min_silence_duration_ms=500, threshold=vad_threshold, speech_pad_ms=50)
        except TypeError:
            vad_parameters = VadOptions(min_silence_duration_ms=500, onset=vad_threshold, speech_pad_ms=50)

        # Language handling
        language_name = args.get("language_name")
        language_code = args.get("language_code")
        multilingual = False
        whisper_lang = None
        
        if not model.model.is_multilingual and language_code != 'en':
            language_name = 'English'
            language_code = 'en'
            log_cb("info", t('language_en_only'))
        
        if language_name == "Multilingual":
            multilingual = True
            whisper_lang = None
        elif language_name == "Auto":
            whisper_lang = None
        else:
            whisper_lang = language_code

        # Detect language if requested (Auto)
        if language_name == "Auto":
            whisper_lang, language_probability, _ = model.detect_language(
                audio, vad_filter=True, vad_parameters=vad_parameters
            )
            log_cb("info", t('language_detect', lang=whisper_lang, prob=f'{language_probability:.2f}'))

        # Build prompt/hotwords if disfluencies suppression is requested
        prompt = ""
        if args.get("disfluencies", False):
            prompt_file = 'prompt.yml'
        else:
            prompt_file = 'prompt_nd.yml'         
        try:
            with open(os.path.join(app_dir, prompt_file), 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f) or {}
            prompt = prompts.get(whisper_lang, '')
        except Exception:
            log_cb('error', t('err_loading_prompt') + '\n')
            prompt = ""

        # Perform transcription (streaming)
        segments, info = batched.transcribe(
            audio_path,
            language=whisper_lang,
            multilingual=multilingual,
            beam_size=args.get("beam_size", 5),
            word_timestamps=args.get("word_timestamps", False),
            hotwords=prompt,
            vad_filter=args.get("vad_filter", True),
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,
            batch_size=args.get("batch_size", 8),
        )
        
        log_cb('info', t('start_transcription') + '\n')
        
        # Stream segments to parent as they arrive
        for s in segments:
            try:
                seg_d = {
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "text": getattr(s, "text", None),
                }
                words = getattr(s, "words", None)
                if words:
                    seg_d["words"] = [
                        {
                            "word": getattr(w, "word", None),
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None),
                            "prob": getattr(w, "probability", None),
                        }
                        for w in words
                    ]
                q.put({"type": "segment", "segment": seg_d})
            except Exception:
                # Best-effort; continue on serialization issues
                pass

        # info into dict
        if is_dataclass(info):
            info_dict = asdict(info)
        else:
            info_dict = {}
            for k in ("language", "language_probability", "duration", "sample_rate"):
                if hasattr(info, k):
                    info_dict[k] = getattr(info, k)
        # Ensure duration is available
        info_dict.setdefault("duration", duration)

        try:
            q.put({"type": "result", "ok": True, "info": info_dict})
        except Exception:
            pass

        # Cleanup VRAM (harmless on CPU)
        try:
            del model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        plog("debug", "Subprocess finished cleanly.")

    except Exception as e:
        try:
            q.put({
                "type": "result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Persistent worker pool entrypoint
#
# `whisper_pool_entrypoint(in_q)` runs the WorkerPool loop. The model is
# loaded once on INIT and re-used for every subsequent job. Designed so
# the existing one-shot `whisper_proc_entrypoint` remains unchanged for
# the no-pool code path.
# ─────────────────────────────────────────────────────────────────────

def _pool_init(args):
    """One-time init: load WhisperModel + BatchedInferencePipeline."""
    # Heavy imports stay inside the subprocess.
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    import platform as _platform
    import torch

    # i18n bootstrap (one-time)
    try:
        import i18n as _i18n
        app_dir = os.path.abspath(os.path.dirname(__file__))
        _i18n.set('filename_format', '{locale}.{format}')
        _i18n.load_path.append(os.path.join(app_dir, 'trans'))
        _i18n.set('fallback', 'en')
        _i18n.set('locale', args.get('locale') or 'en')
    except Exception:
        pass

    device = args.get("device", "")
    if device != 'cpu':
        if _platform.system() == "Darwin":
            device = 'auto'
        elif _platform.system() in ('Windows', 'Linux'):
            try:
                device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
            except Exception:
                device = 'cpu'
        else:
            raise Exception('Platform not supported yet.')

    model = WhisperModel(
        args["model_name_or_path"],
        device=device,
        compute_type=args.get("compute_type", "float16"),
        cpu_threads=args.get("cpu_threads", 4),
        local_files_only=args.get("local_files_only", True),
    )
    batched = BatchedInferencePipeline(model=model)

    return {
        "model": model,
        "batched": batched,
        "device": device,
        "sampling_rate": model.feature_extractor.sampling_rate,
        "app_dir": os.path.abspath(os.path.dirname(__file__)),
    }


def _pool_job(state, args, q, cancel_flag):
    """Run one transcription job using the already-loaded model."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions
    from i18n import t
    import yaml

    model = state["model"]
    batched = state["batched"]
    device = state["device"]
    sampling_rate = state["sampling_rate"]
    app_dir = state["app_dir"]

    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    # Re-apply locale per job in case it changed
    try:
        import i18n as _i18n
        _i18n.set('locale', args.get('locale') or 'en')
    except Exception:
        pass

    # Report device label
    try:
        if device == 'cuda':
            q.put({"type": "device", "role": "transcription", "backend": "cuda", "label": "NVIDIA GPU"})
        else:
            q.put({"type": "device", "role": "transcription", "backend": "ct2-cpu", "label": "CPU"})
    except Exception:
        pass

    audio_path = args.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        q.put({
            "type": "result",
            "ok": False,
            "error": f"Audio path does not exist: {audio_path}",
            "trace": "",
        })
        return

    audio = decode_audio(audio_path, sampling_rate=sampling_rate)
    duration = audio.shape[0] / sampling_rate
    plog("info", t('vad'))

    vad_threshold = float(args.get("vad_threshold", 0.5))
    try:
        vad_parameters = VadOptions(min_silence_duration_ms=500, threshold=vad_threshold, speech_pad_ms=50)
    except TypeError:
        vad_parameters = VadOptions(min_silence_duration_ms=500, onset=vad_threshold, speech_pad_ms=50)

    language_name = args.get("language_name")
    language_code = args.get("language_code")
    multilingual = False
    whisper_lang = None

    if not model.model.is_multilingual and language_code != 'en':
        language_name = 'English'
        language_code = 'en'
        plog("info", t('language_en_only'))

    if language_name == "Multilingual":
        multilingual = True
        whisper_lang = None
    elif language_name == "Auto":
        whisper_lang = None
    else:
        whisper_lang = language_code

    if language_name == "Auto":
        whisper_lang, language_probability, _ = model.detect_language(
            audio, vad_filter=True, vad_parameters=vad_parameters
        )
        plog("info", t('language_detect', lang=whisper_lang, prob=f'{language_probability:.2f}'))

    prompt = ""
    prompt_file = 'prompt.yml' if args.get("disfluencies", False) else 'prompt_nd.yml'
    try:
        with open(os.path.join(app_dir, prompt_file), 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f) or {}
        prompt = prompts.get(whisper_lang, '')
    except Exception:
        plog('error', t('err_loading_prompt') + '\n')
        prompt = ""

    segments, info = batched.transcribe(
        audio_path,
        language=whisper_lang,
        multilingual=multilingual,
        beam_size=args.get("beam_size", 5),
        word_timestamps=args.get("word_timestamps", False),
        hotwords=prompt,
        vad_filter=args.get("vad_filter", True),
        vad_parameters=vad_parameters,
        condition_on_previous_text=False,
        batch_size=args.get("batch_size", 8),
    )

    plog('info', t('start_transcription') + '\n')

    for s in segments:
        if cancel_flag.get("set"):
            # Best-effort soft cancel: stop streaming segments. The
            # generator may still be running C++ code; the orchestrator
            # is expected to terminate the worker if this matters.
            q.put({
                "type": "result",
                "ok": False,
                "error": "Canceled by user",
                "trace": "",
            })
            return
        try:
            seg_d = {
                "start": getattr(s, "start", None),
                "end": getattr(s, "end", None),
                "text": getattr(s, "text", None),
            }
            words = getattr(s, "words", None)
            if words:
                seg_d["words"] = [
                    {
                        "word": getattr(w, "word", None),
                        "start": getattr(w, "start", None),
                        "end": getattr(w, "end", None),
                        "prob": getattr(w, "probability", None),
                    }
                    for w in words
                ]
            q.put({"type": "segment", "segment": seg_d})
        except Exception:
            pass

    if is_dataclass(info):
        info_dict = asdict(info)
    else:
        info_dict = {}
        for k in ("language", "language_probability", "duration", "sample_rate"):
            if hasattr(info, k):
                info_dict[k] = getattr(info, k)
    info_dict.setdefault("duration", duration)

    try:
        q.put({"type": "result", "ok": True, "info": info_dict})
    except Exception:
        pass

    # Soft cleanup between jobs — keep the model in memory, drop temp arrays.
    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass


def whisper_pool_entrypoint(in_q, out_q):
    """Persistent worker loop. Imports the pool helper inside the
    subprocess so it lands on the spawn import path.
    """
    from worker_pool import run_worker_loop
    run_worker_loop(in_q, out_q, _pool_init, _pool_job)
