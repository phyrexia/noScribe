import gc
import os
import sys

# Cosmetic: suppress symlink warnings from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

# NOTE: SSL and offline mode are configured inside live_proc_entrypoint()
# *before* any ML imports, using values passed via the args dict.

import time
import queue
import platform
import traceback
def live_proc_entrypoint(args: dict, q: queue.Queue, stop_event=None):
    """
    Runs in a child process for Live Audio Capture, Transcription and Speaker Diarization.
    """
    try:
        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

        plog("info", "Initializing Live Worker and loading ML libraries... This may take several seconds.")

        # ── SSL / proxy / offline configuration ──────────────────────────────
        # Must be applied BEFORE any ML library import (torch, faster_whisper,
        # pyannote, huggingface_hub all read these env vars at import time).
        #
        # HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 tell HuggingFace/Transformers
        # to never attempt a network check — models are already local.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if args.get("ignore_ssl"):
            # Corporate MITM proxies intercept TLS; bypass certificate validation
            # only for this worker process (does not affect the main UI process).
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            os.environ["SSL_CERT_FILE"] = ""

        if args.get("proxy_url"):
            os.environ["HTTPS_PROXY"] = args["proxy_url"]
            os.environ["HTTP_PROXY"] = args["proxy_url"]
        # ─────────────────────────────────────────────────────────────────────

        import numpy as np
        import sounddevice as sd
        import torch
        import torchaudio

        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        from pyannote.audio import Model as PyannoteModel
        from pyannote.audio import Inference
        import speaker_db

        # I18N Initialization
        try:
            import i18n
            app_dir = os.path.abspath(os.path.dirname(__file__))
            i18n.set('filename_format', '{locale}.{format}')
            i18n.load_path.append(os.path.join(app_dir, 'trans'))
            i18n.set('fallback', 'en')
            child_locale = args.get('locale') or 'en'
            i18n.set('locale', child_locale)
        except Exception:
            pass

        cpu_threads = args.get("cpu_threads", 4)
        input_device_id = args.get("input_device_id", None)
        vad_threshold = float(args.get("vad_threshold", 0.5))
        model_size_or_path = args.get("model_name_or_path", "base")

        # Set threading env vars here, using the actual cpu_threads value
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        # CTranslate2 (faster-whisper backend) does NOT support MPS.
        # Auto-detect: use CUDA if available, otherwise CPU.
        requested_device = args.get("device", "auto")
        if requested_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = requested_device

        plog("info", f"Loading Whisper model from '{model_size_or_path}' on '{device}' for Live Transcription...")
        compute_type = args.get("compute_type", "float16" if device == "cuda" else "int8")

        try:
            whisper_model = WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                local_files_only=args.get("local_files_only", True),
            )
        except Exception as e:
            raise e

        plog("info", f"Loading Pyannote Embedding model for Live Diarization...")
        try:
            # Always load from the bundled local path — never contact HuggingFace Hub.
            # HF_HUB_OFFLINE=1 is already set above, but loading from the explicit
            # local path is more robust and faster (no cache lookup overhead).
            pyannote_dir = os.path.join(os.path.dirname(__file__), 'pyannote', 'embedding')
            embedding_model = PyannoteModel.from_pretrained(
                pyannote_dir, local_files_only=True
            )
            embedding_model.eval()
            
            torch_device = device
            if device == "auto":
                torch_device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
                
            embedding_model.to(torch.device(torch_device))
            inference = Inference(embedding_model, window="whole")
        except Exception as e:
            plog("error", f"Failed to load embedding model: {e}")
            inference = None

        # Load known speakers once into memory at startup (in-session cache)
        try:
            _db = speaker_db.load_db()
            known_speakers = _db.get("speakers", [])
        except Exception:
            known_speakers = []

        def identify_speaker(audio_chunk_np, sample_rate):
            if inference is None or len(known_speakers) == 0:
                return "Unknown Speaker"
            
            # Convert to torch tensor for pyannote
            waveform = torch.from_numpy(audio_chunk_np).unsqueeze(0).float()
            
            try:
                # pyannote inference expects Dict with waveform and sample_rate
                emb = inference({"waveform": waveform, "sample_rate": sample_rate})
                
                best_match = None
                best_sim = -1.0
                
                # Check against known speakers
                for spk in known_speakers:
                    # In speaker_db get_all_speakers returns a dictionary per record
                    spk_emb = spk.get('embedding')
                    if spk_emb is None: continue
                    
                    sim = speaker_db._cosine_similarity(emb, spk_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = spk.get('name', 'Unknown')
                
                if best_match and best_sim >= speaker_db.SIMILARITY_THRESHOLD: # Threshold configured in speaker_db
                    return f"{best_match} ({int(best_sim*100)}%)"
                return "Unknown Speaker"
                
            except Exception as e:
                plog("debug", f"Diarization error: {e}")
                return "Error Speaker"

        SAMPLE_RATE = 16000
        CHUNK_DURATION = 0.5  # Read from audio source in 0.5 second chunks (lower latency)
        VAD_WINDOW = 2.0      # Minimum seconds of audio before checking for silence
        SILENCE_TRIGGER = 0.8 # Seconds of trailing silence that trigger transcription
        MAX_BUFFER_SIZE = 15.0 # Force transcription if buffer grows beyond this

        audio_buffer = np.array([], dtype=np.float32)
        
        try:
            vad_parameters = VadOptions(min_silence_duration_ms=500, threshold=vad_threshold, speech_pad_ms=50)
        except TypeError:
            vad_parameters = VadOptions(min_silence_duration_ms=500, onset=vad_threshold, speech_pad_ms=50)

        audio_stream_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                plog("warn", f"Audio stream status: {status}")
            audio_stream_queue.put(indata.copy()[:, 0]) # Mono

        plog("info", "Starting audio stream...")
        try:
            # Try to start the stream. In some systems string input device ID might fail
            input_device = input_device_id
            if input_device and input_device.isdigit():
                input_device = int(input_device)
                
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', 
                                device=input_device, blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
                                callback=audio_callback):

                while True:
                    if stop_event is not None and stop_event.is_set():
                        plog("info", "Stop signal received via Event.")
                        break

                    try:
                        chunk = audio_stream_queue.get(timeout=0.1)
                        audio_buffer = np.concatenate((audio_buffer, chunk))
                    except queue.Empty:
                        continue

                    buffer_duration = len(audio_buffer) / SAMPLE_RATE

                    if buffer_duration >= VAD_WINDOW:
                        # Detect trailing silence using VAD to decide when to transcribe
                        should_transcribe = buffer_duration >= MAX_BUFFER_SIZE
                        if not should_transcribe:
                            try:
                                speech_ts = get_speech_timestamps(audio_buffer, vad_parameters)
                                if not speech_ts:
                                    # Entire buffer is silence — skip without transcribing
                                    audio_buffer = np.array([], dtype=np.float32)
                                    continue
                                last_speech_end_sec = speech_ts[-1]["end"] / SAMPLE_RATE
                                trailing_silence = buffer_duration - last_speech_end_sec
                                should_transcribe = trailing_silence >= SILENCE_TRIGGER
                            except Exception:
                                # Fallback: transcribe every 5s if VAD check fails
                                should_transcribe = buffer_duration >= 5.0

                        if should_transcribe:
                            segments, _ = whisper_model.transcribe(
                                audio_buffer,
                                language=args.get("language_code"),
                                vad_filter=args.get("vad_filter", True),
                                vad_parameters=vad_parameters,
                            )

                            full_text = " ".join([s.text for s in segments]).strip()

                            if full_text:
                                speaker_name = identify_speaker(audio_buffer, SAMPLE_RATE)
                                q.put({"type": "live_segment", "text": f"[{speaker_name}]: {full_text}"})

                            audio_buffer = np.array([], dtype=np.float32)

        except sd.PortAudioError as pae:
            plog("error", f"PortAudioError (Input Device issue?): {pae}")
        except Exception as e:
            plog("error", f"Stream error: {e}")

        # Cleanup
        del whisper_model
        if inference:
            del embedding_model
        try:
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

        q.put({"type": "live_finished"})

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
