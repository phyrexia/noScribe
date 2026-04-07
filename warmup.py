# MeetingGenie - Model Warmup
# Pre-loads ML libraries and models in a background subprocess
# so subsequent transcription starts faster (OS page cache).

import os
import sys
import ssl
import time

# SSL bypass for corporate environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass


def warmup():
    """Import heavy ML libraries to warm the OS page cache."""
    t0 = time.time()
    app_dir = os.path.abspath(os.path.dirname(__file__))

    try:
        import torch
        print(f"[warmup] torch loaded ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[warmup] torch failed: {e}")

    try:
        import torchaudio
        print(f"[warmup] torchaudio loaded ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[warmup] torchaudio failed: {e}")

    try:
        from faster_whisper import WhisperModel
        print(f"[warmup] faster_whisper loaded ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[warmup] faster_whisper failed: {e}")

    try:
        from pyannote.audio import Pipeline
        from pathlib import Path
        pyannote_dir = os.path.join(app_dir, 'pyannote')
        if os.path.isdir(pyannote_dir):
            Pipeline.from_pretrained(Path(pyannote_dir))
            print(f"[warmup] pyannote pipeline loaded ({time.time()-t0:.1f}s)")
        else:
            print(f"[warmup] pyannote dir not found: {pyannote_dir}")
    except Exception as e:
        print(f"[warmup] pyannote failed: {e}")

    # Also warm whisper model if available
    try:
        import model_manager
        for quality in ('fast', 'precise'):
            if model_manager.model_is_ready(quality):
                path = model_manager.get_model_path_for_app(quality)
                if path:
                    WhisperModel(path, device='cpu', compute_type='int8')
                    print(f"[warmup] whisper '{quality}' loaded ({time.time()-t0:.1f}s)")
                    break  # Only warm one model
    except Exception as e:
        print(f"[warmup] whisper model failed: {e}")

    print(f"[warmup] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    warmup()
