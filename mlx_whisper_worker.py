import gc
import os
import ssl
import traceback


def mlx_whisper_proc_entrypoint(args: dict, q):
    """
    Runs Whisper transcription via mlx-whisper in a child process.
    Streams progress/logs/segments to parent via `q`.

    Messages put on `q` are dicts with one of the following shapes:
      {"type": "device", "backend": "mlx-metal", "label": "Metal GPU (MLX)"}
      {"type": "log", "level": "info"|"warn"|"error"|"debug", "msg": "..."}
      {"type": "progress", "pct": float, "detail": "..."}
      {"type": "segment", "segment": {...}}
      {"type": "result", "ok": True, "info": {...}}
      {"type": "result", "ok": False, "error": str, "trace": str}
      {"type": "finished"}
    """
    # MLX downloads from HuggingFace on first run. Override any offline gates
    # the spawning module set at import-time.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Publish the active backend immediately so the UI device badge reflects
    # the real whisper backend before model loading begins.
    try:
        q.put({"type": "device", "role": "transcription", "backend": "mlx-metal", "label": "Metal GPU (MLX)"})
    except Exception:
        pass

    # SSL bypass for corporate environments (matches whisper_mp_worker behaviour).
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    try:
        # Import heavy libs only in the child process.
        import mlx_whisper
        import yaml
        import i18n
        from i18n import t

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        # Initialize i18n in child process (spawn-safe; no shared globals).
        app_dir = os.path.abspath(os.path.dirname(__file__))
        try:
            i18n.set('filename_format', '{locale}.{format}')
            i18n.load_path.append(os.path.join(app_dir, 'trans'))
            i18n.set('fallback', 'en')
            child_locale = args.get('locale') or 'en'
            i18n.set('locale', child_locale)
        except Exception:
            pass

        # Resolve audio + model.
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

        path_or_hf_repo = args.get("model_name_or_path")
        if not path_or_hf_repo:
            raise ValueError("model_name_or_path is required for MLX worker")

        # Language handling.
        language_name = args.get("language_name")
        language_code = args.get("language_code")
        whisper_lang = None
        if language_name == "Multilingual":
            whisper_lang = None
        elif language_name == "Auto":
            whisper_lang = None
        else:
            whisper_lang = language_code

        # Prompt / hotwords for disfluency suppression.
        prompt_file = 'prompt.yml' if args.get("disfluencies", False) else 'prompt_nd.yml'
        initial_prompt = None
        try:
            with open(os.path.join(app_dir, prompt_file), 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f) or {}
            initial_prompt = prompts.get(whisper_lang or 'en', '') or None
        except Exception:
            plog('error', t('err_loading_prompt') + '\n')
            initial_prompt = None

        plog("info", t('start_transcription') + '\n')

        # Cancellation: drain the queue for a sentinel message between segments.
        # mlx-whisper.transcribe runs synchronously, so we cannot interrupt mid-call;
        # cancellation is checked before invocation and after, and the parent process
        # can always terminate this worker.
        cancel_requested = False

        def _check_cancel():
            nonlocal cancel_requested
            # We don't read from `q` (single-direction); rely on parent terminate.
            return cancel_requested

        if _check_cancel():
            raise Exception("Canceled by user")

        # Run transcription. mlx-whisper does not stream — it returns the full
        # result dict at the end. We emit segments after the call completes.
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=path_or_hf_repo,
            language=whisper_lang,
            initial_prompt=initial_prompt,
            word_timestamps=False,
            condition_on_previous_text=False,
            verbose=None,
        )

        segments = result.get("segments", []) if isinstance(result, dict) else []
        detected_language = result.get("language") if isinstance(result, dict) else None

        # Emit segments in whisper_mp_worker-compatible shape.
        for s in segments:
            if _check_cancel():
                break
            try:
                seg_d = {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text"),
                }
                words = s.get("words")
                if words:
                    seg_d["words"] = [
                        {
                            "word": w.get("word"),
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "prob": w.get("probability"),
                        }
                        for w in words
                    ]
                q.put({"type": "segment", "segment": seg_d})
            except Exception:
                # Best-effort; continue on serialization issues.
                pass

        # Build info payload.
        info_dict = {
            "language": detected_language,
            "language_probability": 1.0,
        }
        if segments:
            try:
                info_dict["duration"] = float(segments[-1].get("end") or 0.0)
            except Exception:
                pass

        try:
            q.put({"type": "result", "ok": True, "info": info_dict})
        except Exception:
            pass

        try:
            q.put({"type": "finished"})
        except Exception:
            pass

        # Cleanup.
        try:
            del result
        except Exception:
            pass
        gc.collect()
        plog("debug", "MLX subprocess finished cleanly.")

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
        try:
            q.put({"type": "finished"})
        except Exception:
            pass
