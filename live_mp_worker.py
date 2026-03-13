import gc
import os
import time
import queue
import platform
import traceback
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from i18n import t

def live_proc_entrypoint(args: dict, q: queue.Queue):
    """
    Runs in a child process for Live Audio Capture, Transcription and Speaker Diarization.
    """
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions
        from pyannote.audio import Model as PyannoteModel
        from pyannote.audio import Inference
        import speaker_db

        def plog(level, msg):
            try:
                q.put({"type": "log", "level": level, "msg": str(msg)})
            except Exception:
                pass

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

        device = args.get("device", "cpu")
        model_size = args.get("model_name_or_path", "base")
        input_device_id = args.get("input_device_id", None)
        vad_threshold = float(args.get("vad_threshold", 0.5))

        plog("info", f"Loading Whisper model '{model_size}' on '{device}' for Live Transcription...")
        compute_type = args.get("compute_type", "float16" if device != "cpu" else "int8")
        whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=args.get("cpu_threads", 4),
            local_files_only=args.get("local_files_only", True),
        )

        plog("info", f"Loading Pyannote Embedding model for Live Diarization...")
        try:
            # Load the embbedding model
            embedding_model = PyannoteModel.from_pretrained(
                "pyannote/embedding", use_auth_token="HF_TOKEN_NOT_NEEDED_FOR_LOCAL" # We use local
            )
            # if we have it locally in pyannote dir
            pyannote_dir = os.path.join(os.path.dirname(__file__), 'pyannote', 'embedding')
            if os.path.exists(os.path.join(pyannote_dir, "pytorch_model.bin")):
                embedding_model = PyannoteModel.from_pretrained(pyannote_dir)
            embedding_model.eval()
            embedding_model.to(torch.device(device))
            inference = Inference(embedding_model, window="whole")
        except Exception as e:
            plog("error", f"Failed to load embedding model: {e}")
            inference = None

        # Load known speakers
        try:
            speaker_db.init_db()
            known_speakers = speaker_db.get_all_speakers()
        except:
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
        CHUNK_DURATION = 1.0 # Read from audio source in 1 second chunks
        VAD_WINDOW = 3.0 # Minimum seconds of audio to keep in buffer to check for VAD
        MAX_BUFFER_SIZE = 15.0 # Max seconds to hold in buffer before forcing transcription

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
                    try:
                        cmd = q.get_nowait()
                        if isinstance(cmd, dict) and cmd.get("action") == "stop":
                            plog("info", "Stop signal received.")
                            break
                    except queue.Empty:
                        pass

                    try:
                        chunk = audio_stream_queue.get(timeout=0.1)
                        audio_buffer = np.concatenate((audio_buffer, chunk))
                    except queue.Empty:
                        continue

                    buffer_duration = len(audio_buffer) / SAMPLE_RATE

                    if buffer_duration >= VAD_WINDOW:
                        # Only transcribe if we pass max buffer or (todo: silence detected)
                        if buffer_duration >= MAX_BUFFER_SIZE or buffer_duration >= 5.0:
                            
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
