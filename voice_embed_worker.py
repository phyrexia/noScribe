# MeetingGenie - Voice embedding worker (single-speaker fast path).
#
# Skips the full pyannote diarization pipeline (segmentation + clustering)
# and runs only the speaker-embedding model on a short clean window of the
# audio. The resulting embedding is matched against speaker_db so a single
# stored signature can be applied to every transcription segment.
#
# Time budget: ~1-2s on M-series vs ~14s for the full diarization path.

import os
import ssl
import traceback


def voice_embed_proc_entrypoint(args: dict, q):
    """
    Args:
      audio_path : str — 16kHz mono wav from the audio conversion step
      app_dir    : str — repo root (used to find pyannote/embedding/)
      window_s   : float — seconds of audio to embed (default 10.0)
      offset_s   : float — seconds to skip from start (default 1.0)

    Queue messages (mirrors other workers):
      {"type": "device", "role": "voice-embed", "backend": str, "label": str}
      {"type": "log", "level": "info"|"error", "msg": str}
      {"type": "result", "ok": True, "embedding": [...], "info": {...}}
      {"type": "result", "ok": False, "error": str, "trace": str}
    """
    # SSL / offline gates inherited from runner; keep pyannote happy.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    try:
        import torch
        import torchaudio
        import numpy as np
        import platform
        from pyannote.audio import Inference, Model

        audio_path = args["audio_path"]
        app_dir = args["app_dir"]
        window_s = float(args.get("window_s", 10.0))
        offset_s = float(args.get("offset_s", 1.0))

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Device: MPS where available, else CPU.
        if (platform.system() == "Darwin"
                and platform.mac_ver()[0] >= "12.3"
                and torch.backends.mps.is_available()):
            device = "mps"
            label = "Metal/GPU (MPS)"
        elif torch.cuda.is_available():
            device = "cuda"
            label = "NVIDIA GPU"
        else:
            device = "cpu"
            label = "CPU"

        try:
            q.put({"type": "device", "role": "voice-embed",
                   "backend": device, "label": label})
        except Exception:
            pass

        # Load embedding model standalone (no segmentation/clustering).
        embed_dir = os.path.join(app_dir, "pyannote", "embedding")
        model_path = os.path.join(embed_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model missing: {model_path}")

        model = Model.from_pretrained(model_path).to(torch.device(device))
        inference = Inference(model, window="whole")

        # Load + crop audio.
        waveform, sample_rate = torchaudio.load(audio_path)
        total_samples = waveform.shape[1]
        total_seconds = total_samples / float(sample_rate)

        start = max(0.0, min(offset_s, max(0.0, total_seconds - 2.0)))
        end = min(total_seconds, start + window_s)
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if end_idx - start_idx < int(2.0 * sample_rate):
            # Fall back to the entire clip if it's shorter than the window.
            start_idx, end_idx = 0, total_samples

        seg_wav = waveform[:, start_idx:end_idx]
        # Mono mix if needed (pyannote embedding expects mono).
        if seg_wav.shape[0] > 1:
            seg_wav = seg_wav.mean(dim=0, keepdim=True)

        plog("info",
             f"Embedding window {start:.1f}-{end:.1f}s "
             f"({(end - start):.1f}s of {total_seconds:.1f}s total)")

        raw = inference({"waveform": seg_wav, "sample_rate": sample_rate})
        # Inference(window="whole") returns a 1-D ndarray (or torch tensor).
        if hasattr(raw, "numpy"):
            arr = raw.numpy()
        else:
            arr = np.asarray(raw)
        arr = np.array(arr, dtype=np.float32).reshape(-1)

        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise RuntimeError("Embedding extraction returned empty or non-finite vector")

        q.put({
            "type": "result",
            "ok": True,
            "embedding": arr.tolist(),
            "info": {
                "dim": int(arr.size),
                "window_start_s": start,
                "window_end_s": end,
            },
        })

    except Exception as e:
        try:
            q.put({
                "type": "result",
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass
