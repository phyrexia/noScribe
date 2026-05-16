# MeetingGenie - sherpa-onnx diarization worker
#
# Optional alternative to `pyannote_mp_worker.py`. Uses sherpa-onnx's
# OfflineSpeakerDiarization, which wraps the same upstream
# pyannote-segmentation-3.0 model exported to ONNX plus a 3D-Speaker
# embedding extractor. Runs on the macOS CoreML execution provider (or CPU)
# without requiring PyTorch.
#
# Queue protocol mirrors `pyannote_mp_worker.pyannote_proc_entrypoint` so the
# runner can dispatch on `diarization_backend` without further changes:
#
#   {"type": "device", "role": "diarization", "backend": str, "label": str}
#   {"type": "log",      "level": "info|warn|error|debug", "msg": str}
#   {"type": "progress", "step": str, "pct": int}
#   {"type": "result",   "ok": True,  "segments": [...], "embeddings": {...}}
#   {"type": "result",   "ok": False, "error": str, "trace": str}
#
# Segment `start`/`end` are integer milliseconds — matching the pyannote
# worker so `transcription_runner` can treat the two outputs interchangeably.

from __future__ import annotations

import os
import ssl
import time
import shutil
import tarfile
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model assets — sherpa-onnx publishes prebuilt ONNX bundles on GitHub.
# ---------------------------------------------------------------------------

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_FILE = "model.onnx"

EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
EMBEDDING_FILE = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def _user_models_dir() -> Path:
    """Where downloaded sherpa-onnx assets live (next to whisper models)."""
    d = Path.home() / ".meetinggenie" / "sherpa_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_model_paths(app_dir: str) -> tuple[Path, Path]:
    """Return absolute paths to segmentation and embedding ONNX files.

    Search order:
      1. Bundled with the app:        {app_dir}/models/sherpa/
      2. User download directory:     ~/.meetinggenie/sherpa_models/
    """
    bundled = Path(app_dir) / "models" / "sherpa"
    user = _user_models_dir()

    seg_candidates = [
        bundled / SEGMENTATION_DIR / SEGMENTATION_FILE,
        user / SEGMENTATION_DIR / SEGMENTATION_FILE,
    ]
    emb_candidates = [
        bundled / EMBEDDING_FILE,
        user / EMBEDDING_FILE,
    ]

    seg = next((p for p in seg_candidates if p.exists()), seg_candidates[-1])
    emb = next((p for p in emb_candidates if p.exists()), emb_candidates[-1])
    return seg, emb


def _build_opener(proxy_url: Optional[str], ignore_ssl: bool):
    """Build a urllib opener honoring proxy + SSL bypass (corporate networks)."""
    handlers = []
    if proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        handlers.append(urllib.request.ProxyHandler())
    if ignore_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def _download(url: str, dest: Path, plog, proxy_url: Optional[str], ignore_ssl: bool) -> None:
    """Stream `url` to `dest` (atomic-ish: writes to a `.tmp` then renames)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    opener = _build_opener(proxy_url, ignore_ssl)
    try:
        with opener.open(url) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            last_pct = -1
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(262144)  # 256 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct and pct % 10 == 0:
                            plog("info",
                                 f"Downloading {dest.name}: {pct}% "
                                 f"({downloaded // (1024 * 1024)} / {total // (1024 * 1024)} MB)")
                            last_pct = pct
        shutil.move(str(tmp), str(dest))
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise


def _ensure_segmentation(target_dir: Path, plog, proxy_url, ignore_ssl) -> Path:
    seg_file = target_dir / SEGMENTATION_DIR / SEGMENTATION_FILE
    if seg_file.exists():
        return seg_file

    plog("info",
         f"Fetching sherpa-onnx segmentation model "
         f"(~6 MB) from {SEGMENTATION_URL}")
    archive = target_dir / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    _download(SEGMENTATION_URL, archive, plog, proxy_url, ignore_ssl)
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=target_dir)
    finally:
        try:
            archive.unlink()
        except Exception:
            pass

    if not seg_file.exists():
        raise FileNotFoundError(
            f"Segmentation model missing after extract: {seg_file}"
        )
    return seg_file


def _ensure_embedding(target_dir: Path, plog, proxy_url, ignore_ssl) -> Path:
    emb_file = target_dir / EMBEDDING_FILE
    if emb_file.exists():
        return emb_file

    plog("info",
         f"Fetching sherpa-onnx speaker-embedding model "
         f"(~37 MB) from {EMBEDDING_URL}")
    _download(EMBEDDING_URL, emb_file, plog, proxy_url, ignore_ssl)
    return emb_file


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------

def sherpa_diar_proc_entrypoint(args: dict, q):
    """Run sherpa-onnx diarization in a child process.

    Args (dict):
      audio_path     : str    — 16kHz mono WAV from the conversion step.
      app_dir        : str    — repository root (for bundled models lookup).
      num_speakers   : Optional[int] — if known; otherwise auto-cluster.
      provider       : str    — 'cpu' or 'coreml' (default 'cpu').
                                CoreML can be slower for these small models on
                                M-series due to ANE compile/transfer overhead;
                                CPU with 4 threads is typically faster.
      num_threads    : int    — ONNX Runtime intra-op threads (default 4).
      cluster_threshold : float — agglomerative clustering threshold
                                   (default 0.5; lower = more speakers).
      min_duration_on  : float — min on-segment seconds (default 0.2).
      min_duration_off : float — min off-segment seconds (default 0.5).
      proxy_url      : str    — HTTP proxy for model download.
      ignore_ssl     : bool   — disable SSL verification for download.
    """
    started_at = time.time()

    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    def pprogress(step, pct):
        try:
            q.put({"type": "progress", "step": str(step), "pct": int(pct)})
        except Exception:
            pass

    device_label = "sherpa-onnx"
    backend_id = "sherpa"
    try:
        audio_path = args["audio_path"]
        app_dir = args.get("app_dir") or os.path.abspath(os.path.dirname(__file__))
        provider = (args.get("provider") or "cpu").lower()
        if provider not in ("cpu", "coreml"):
            provider = "cpu"
        num_threads = int(args.get("num_threads", 4))
        num_speakers = args.get("num_speakers")
        cluster_threshold = float(args.get("cluster_threshold", 0.5))
        min_duration_on = float(args.get("min_duration_on", 0.2))
        min_duration_off = float(args.get("min_duration_off", 0.5))
        proxy_url = args.get("proxy_url") or None
        ignore_ssl = bool(args.get("ignore_ssl", True))

        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        plog("debug", "Subprocess (sherpa-onnx diarize) started.")

        # Resolve / fetch model assets.
        seg_path, emb_path = _resolve_model_paths(app_dir)
        if not seg_path.exists() or not emb_path.exists():
            target = _user_models_dir()
            if not seg_path.exists():
                seg_path = _ensure_segmentation(target, plog, proxy_url, ignore_ssl)
            if not emb_path.exists():
                emb_path = _ensure_embedding(target, plog, proxy_url, ignore_ssl)

        plog("info", f"Segmentation: {seg_path.name}")
        plog("info", f"Embedding:    {emb_path.name}")
        pprogress("loading_model", 5)

        # Surface backend metadata for the UI before the heavy import work.
        device_label = ("Apple Neural Engine (sherpa-onnx)"
                        if provider == "coreml"
                        else "CPU (sherpa-onnx)")
        backend_id = "coreml" if provider == "coreml" else "cpu"
        try:
            q.put({"type": "device", "role": "diarization",
                   "backend": backend_id, "label": device_label})
        except Exception:
            pass

        import wave
        import numpy as np
        try:
            import sherpa_onnx
        except ImportError as ie:
            raise RuntimeError(
                "sherpa-onnx is not installed. Install with "
                "`pip install sherpa-onnx` to use this backend."
            ) from ie

        # Build the OfflineSpeakerDiarization pipeline.
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_path),
                ),
                provider=provider,
                num_threads=num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_path),
                provider=provider,
                num_threads=num_threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=(int(num_speakers) if num_speakers else -1),
                threshold=cluster_threshold,
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        expected_sr = sd.sample_rate

        # Load mono 16kHz PCM samples as float32 in [-1, 1].
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(n_frames)
        if sr != expected_sr:
            raise RuntimeError(
                f"Audio sample rate {sr} != expected {expected_sr}. "
                "ToWav should produce 16 kHz mono."
            )
        if sample_width != 2:
            raise RuntimeError(
                f"Audio must be 16-bit PCM (got sample_width={sample_width})."
            )
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        total_secs = float(samples.shape[0]) / float(expected_sr)
        plog("info",
             f"Diarizing {total_secs:.1f}s of audio "
             f"(provider={provider}, threads={num_threads}).")

        # Progress callback fires per chunk in [0, num_chunks].
        progress_state = {"last_pct": 0}

        def _on_progress(num_processed_chunks: int, num_total_chunks: int) -> int:
            try:
                if num_total_chunks > 0:
                    pct = int(num_processed_chunks * 100 / num_total_chunks)
                else:
                    pct = 0
                if pct != progress_state["last_pct"] and pct % 5 == 0:
                    pprogress("segmentation", pct)
                    progress_state["last_pct"] = pct
            except Exception:
                pass
            return 0  # 0 = continue, non-zero = abort (not used here)

        t0 = time.time()
        result = sd.process(samples, callback=_on_progress)
        proc_seconds = time.time() - t0
        rtf = total_secs / proc_seconds if proc_seconds > 0 else 0.0
        plog("info",
             f"sherpa-onnx diarization finished in {proc_seconds:.2f}s "
             f"({rtf:.1f}x RTF). Speakers: {result.num_speakers}, "
             f"segments: {result.num_segments}.")

        # Emit segments matching pyannote_mp_worker shape: ms ints + SPEAKER_NN.
        seg_list = []
        for seg in result.sort_by_start_time():
            seg_list.append({
                "start": int(seg.start * 1000),
                "end": int(seg.end * 1000),
                "label": f"SPEAKER_{int(seg.speaker):02d}",
            })

        # ------------------------------------------------------------------
        # Per-speaker embeddings: sherpa-onnx OfflineSpeakerDiarizationResult
        # does not expose the underlying embeddings, so we re-extract one
        # vector per speaker from a clean segment using the same ONNX
        # embedding model. Wrapped in try/except — failure does not block
        # diarization.
        # ------------------------------------------------------------------
        speaker_embeddings = {}
        try:
            extractor_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_path),
                provider=provider,
                num_threads=num_threads,
            )
            extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_cfg)

            # Group segments by label, sort longest-first, pick top-2 clean
            # windows (>= 1.5s, no overlap with other speakers).
            by_label: dict[str, list] = {}
            for s in seg_list:
                by_label.setdefault(s["label"], []).append(s)

            def _has_overlap(seg, skip_label):
                s0, e0 = seg["start"], seg["end"]
                for other in seg_list:
                    if other["label"] == skip_label:
                        continue
                    if min(e0, other["end"]) - max(s0, other["start"]) > 0:
                        return True
                return False

            for label, segs in by_label.items():
                segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"],
                                     reverse=True)
                vectors = []
                for s in segs_sorted:
                    if len(vectors) >= 3:
                        break
                    duration_s = (s["end"] - s["start"]) / 1000.0
                    if duration_s < 1.5:
                        continue
                    if _has_overlap(s, label):
                        continue
                    start_idx = int(s["start"] / 1000.0 * expected_sr)
                    end_idx = int(s["end"] / 1000.0 * expected_sr)
                    clip = samples[start_idx:end_idx]
                    if clip.size < int(0.5 * expected_sr):
                        continue
                    try:
                        stream = extractor.create_stream()
                        stream.accept_waveform(sample_rate=expected_sr,
                                               waveform=clip.tolist())
                        stream.input_finished()
                        if extractor.is_ready(stream):
                            vec = np.asarray(extractor.compute(stream), dtype=np.float32)
                            if vec.size and np.all(np.isfinite(vec)):
                                vectors.append(vec)
                    except Exception as ex:
                        plog("debug", f"{label}: embedding extract failed ({ex})")

                if vectors:
                    avg = np.mean(np.stack(vectors), axis=0)
                    n = float(np.linalg.norm(avg))
                    if n > 1e-6:
                        avg = avg / n
                    speaker_embeddings[label] = avg.astype(np.float32).tolist()
                else:
                    plog("debug",
                         f"{label}: no clean segment for embedding, omitted.")
        except Exception as emb_err:
            plog("debug", f"Speaker embedding extraction skipped: {emb_err}")

        wall = time.time() - started_at
        plog("info", f"sherpa-onnx total wall-clock: {wall:.2f}s")

        try:
            q.put({
                "type": "result",
                "ok": True,
                "segments": seg_list,
                "embeddings": speaker_embeddings,
            })
        except Exception:
            pass

    except Exception as e:
        try:
            error_str = f"{type(e).__name__}: {e} (device_{backend_id[:3]})"
            q.put({
                "type": "result",
                "ok": False,
                "error": error_str,
                "trace": traceback.format_exc(),
            })
        except Exception:
            pass
