import gc
import os
import ssl
import traceback

# Whisper expects 16 kHz mono float32.
SAMPLE_RATE = 16000

# Chunk window for streaming transcription. 30 s mirrors Whisper's own
# log-mel context; 1 s of overlap reduces the chance of a word being clipped
# at the boundary.
CHUNK_LENGTH_S = 30.0
OVERLAP_S = 1.0


def _load_audio_float32(path: str):
    """Decode any media file to a 1-D float32 numpy array at 16 kHz mono.

    Uses PyAV (already a dependency via audio/convert.py); no ffmpeg CLI
    requirement and no temporary WAV files on disk.
    """
    import numpy as np
    import av
    from av.audio.resampler import AudioResampler

    container = av.open(path)
    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError(f"No audio stream in {path}")
        in_stream = audio_streams[0]
        for s in container.streams:
            s.thread_type = "AUTO" if s is in_stream else "NONE"

        resampler = AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(in_stream):
            for resampled in resampler.resample(frame):
                if resampled is None:
                    continue
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            if resampled is None:
                continue
            chunks.append(resampled.to_ndarray().reshape(-1))
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


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

        # Decode the input file once into a 16 kHz mono float32 buffer so we
        # can feed mlx-whisper fixed-size windows. Streaming windows yields a
        # progressive UX: segments arrive as each chunk is decoded rather than
        # all at once after the full file is processed.
        import numpy as np

        audio = _load_audio_float32(audio_path)
        total_samples = int(audio.shape[0])
        total_duration_s = total_samples / float(SAMPLE_RATE)
        plog("debug", f"Loaded audio: {total_duration_s:.1f}s @ {SAMPLE_RATE} Hz mono")

        chunk_samples = int(CHUNK_LENGTH_S * SAMPLE_RATE)
        overlap_samples = int(OVERLAP_S * SAMPLE_RATE)
        step_samples = max(1, chunk_samples - overlap_samples)
        if total_samples <= 0:
            total_chunks = 0
        else:
            total_chunks = max(1, 1 + (max(0, total_samples - chunk_samples) + step_samples - 1) // step_samples)

        all_segments = []
        detected_language = None
        prev_chunk_end_global = 0.0  # absolute (file-relative) seconds

        if total_chunks == 0:
            try:
                q.put({"type": "progress", "pct": 100})
            except Exception:
                pass
        else:
            chunk_idx = 0
            offset = 0
            while offset < total_samples:
                if _check_cancel():
                    raise Exception("Canceled by user")

                end_offset = min(offset + chunk_samples, total_samples)
                chunk = audio[offset:end_offset]
                offset_s = offset / float(SAMPLE_RATE)

                # mlx-whisper accepts numpy arrays directly (Union[str, ndarray, mx.array]).
                # condition_on_previous_text=False because we are not feeding cross-chunk context.
                result = mlx_whisper.transcribe(
                    chunk,
                    path_or_hf_repo=path_or_hf_repo,
                    language=whisper_lang,
                    initial_prompt=initial_prompt,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    verbose=None,
                )

                if detected_language is None and isinstance(result, dict):
                    detected_language = result.get("language")

                segs = result.get("segments", []) if isinstance(result, dict) else []

                # Emit chunk segments with timestamps shifted into the global
                # file timeline. Drop the leading overlap zone from chunks
                # after the first to avoid duplicates with the previous chunk's tail.
                emit_floor_global = prev_chunk_end_global - (OVERLAP_S * 0.5) if chunk_idx > 0 else -1.0
                last_seg_end_global = offset_s
                for s in segs:
                    if _check_cancel():
                        break
                    try:
                        s_start = float(s.get("start") or 0.0)
                        s_end = float(s.get("end") or s_start)
                    except Exception:
                        continue
                    global_start = s_start + offset_s
                    global_end = s_end + offset_s
                    last_seg_end_global = max(last_seg_end_global, global_end)
                    # Suppress overlap-zone duplicates from non-initial chunks.
                    if chunk_idx > 0 and global_start < emit_floor_global:
                        continue
                    try:
                        seg_d = {
                            "start": global_start,
                            "end": global_end,
                            "text": s.get("text"),
                        }
                        q.put({"type": "segment", "segment": seg_d})
                        all_segments.append(seg_d)
                    except Exception:
                        pass

                prev_chunk_end_global = last_seg_end_global
                chunk_idx += 1
                try:
                    pct = int(min(100, (chunk_idx / total_chunks) * 100))
                    q.put({"type": "progress", "pct": pct})
                except Exception:
                    pass

                # Free chunk-local references before next iteration.
                del result, chunk
                if offset + chunk_samples >= total_samples:
                    break
                offset += step_samples

        segments = all_segments

        # Build info payload.
        info_dict = {
            "language": detected_language,
            "language_probability": 1.0,
            "duration": total_duration_s,
        }

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
            del audio, segments
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
