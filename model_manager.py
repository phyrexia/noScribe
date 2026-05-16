# MeetingGenie - Model Manager
# Handles on-demand download of Whisper models with corporate proxy support.
# Models are searched in 3 locations: bundled → user dir → download from GitHub.

import os
import sys
import json
import shutil
import tarfile
import platform
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable

APP_NAME = 'MeetingGenie'

# User models directory (outside app bundle)
_USER_MODELS_DIR = Path.home() / '.meetinggenie' / 'models'

# GitHub releases URL for model downloads
GH_RELEASE_URL = "https://github.com/phyrexia/noScribe/releases/download/models-v1/{quality}.tar.gz"

# ---------------------------------------------------------------------------
# Model registry — 3 tiers
# ---------------------------------------------------------------------------

MODELS = {
    "small": {
        "backend": "ct2",
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json"],
        "size_mb": 246,
        "label": "Small (246 MB) – fastest, for live mode",
    },
    "fast": {
        "backend": "ct2",
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"],
        "size_mb": 785,
        "label": "Fast (785 MB) – good for most meetings",
    },
    "precise": {
        "backend": "ct2",
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"],
        "size_mb": 1500,
        "label": "Precise (1.5 GB) – highest accuracy",
    },
    # MLX tiers — Metal-accelerated. Downloaded by mlx-whisper from HuggingFace
    # on first use to ~/.cache/huggingface/hub/. Path resolution returns the
    # HF repo id, not a local filesystem path.
    "mlx-fast": {
        "backend": "mlx",
        "hf_repo": "mlx-community/whisper-large-v3-turbo",
        "size_mb": 1600,
        "label": "MLX Fast (1.6 GB) – Metal GPU, turbo",
    },
    "mlx-precise": {
        "backend": "mlx",
        "hf_repo": "mlx-community/whisper-large-v3-mlx",
        "size_mb": 3000,
        "label": "MLX Precise (3 GB) – Metal GPU, highest accuracy",
    },
}


def model_backend(quality: str) -> Optional[str]:
    """Return the backend identifier for a model tier ('ct2' or 'mlx')."""
    entry = MODELS.get(quality)
    return entry.get("backend") if entry else None


def _app_dir() -> Path:
    """App directory (where the executable lives)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(os.path.abspath(os.path.dirname(__file__)))


def models_dir() -> Path:
    """User models directory (~/.meetinggenie/models/)."""
    _USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _USER_MODELS_DIR


def _mlx_hf_cache_dir(hf_repo: str) -> Path:
    """Return the HuggingFace hub cache directory path for an MLX repo.

    HF cache layout: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/...
    """
    org, name = hf_repo.split('/', 1) if '/' in hf_repo else ('', hf_repo)
    cache_root = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'
    return cache_root / f"models--{org}--{name}"


def model_path(quality: str) -> Optional[Path]:
    """Return the path where the model is found, checking bundled first.

    For CT2 models, returns a local directory path. For MLX models, returns
    the HF cache directory if a snapshot exists locally (None otherwise).

    Note: callers that pass the model identifier to mlx-whisper should use
    `get_model_path_for_app()` which returns the HF repo id string for MLX
    tiers (mlx-whisper resolves the cache itself).

    Search order (CT2):
      1. Bundled: {app_dir}/models/{quality}/
      2. User:    ~/.meetinggenie/models/{quality}/
    Returns None if not found anywhere.
    """
    entry = MODELS.get(quality)
    if not entry:
        return None

    if entry.get("backend") == "mlx":
        hf_repo = entry.get("hf_repo")
        if not hf_repo:
            return None
        cache_dir = _mlx_hf_cache_dir(hf_repo)
        snapshots = cache_dir / 'snapshots'
        if snapshots.is_dir():
            for snap in snapshots.iterdir():
                if snap.is_dir() and any(snap.iterdir()):
                    return snap
        return None

    # 1. Bundled with app
    bundled = _app_dir() / 'models' / quality
    if bundled.is_dir() and all((bundled / f).exists() for f in entry["files"]):
        return bundled

    # 2. User directory
    user = _USER_MODELS_DIR / quality
    if user.is_dir() and all((user / f).exists() for f in entry["files"]):
        return user

    return None


def model_is_ready(quality: str) -> bool:
    """Return True if the model is available (bundled or downloaded)."""
    return model_path(quality) is not None


def get_model_path_for_app(quality: str) -> Optional[str]:
    """Return the identifier to pass to the whisper backend.

    - CT2 tiers: absolute filesystem path (faster-whisper).
    - MLX tiers: HuggingFace repo id (mlx-whisper resolves/downloads itself).

    Returns None only if the tier is unknown.
    """
    entry = MODELS.get(quality)
    if not entry:
        return None
    if entry.get("backend") == "mlx":
        # mlx-whisper accepts an HF repo id and handles download/cache.
        return entry.get("hf_repo")
    p = model_path(quality)
    return str(p) if p else None


def list_available_models() -> dict:
    """Return dict of {quality: {"ready": bool, "location": str, ...}}."""
    result = {}
    for q, entry in MODELS.items():
        p = model_path(q)
        if entry.get("backend") == "mlx":
            location = str(p) if p else (entry.get("hf_repo", "") + " (not downloaded)")
        else:
            location = str(p) if p else "not downloaded"
        result[q] = {
            "ready": p is not None,
            "backend": entry.get("backend", "ct2"),
            "location": location,
            "size_mb": entry["size_mb"],
            "label": entry["label"],
        }
    return result


# ---------------------------------------------------------------------------
# Download from GitHub Releases
# ---------------------------------------------------------------------------

def _build_opener(proxy_url: Optional[str] = None, ignore_ssl: bool = False):
    """Build a urllib opener that respects corporate proxies and optional SSL bypass."""
    handlers = []
    if proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        handlers.append(urllib.request.ProxyHandler())
    if ignore_ssl:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def download_model(
    quality: str,
    proxy_url: Optional[str] = None,
    ignore_ssl: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Download a model from GitHub Releases as .tar.gz.

    Downloads to ~/.meetinggenie/models/{quality}/
    """
    if quality not in MODELS:
        raise ValueError(f"Unknown model: {quality}. Available: {list(MODELS.keys())}")

    # MLX models are fetched lazily by mlx-whisper from HuggingFace on first use.
    # We do not pre-download them here — that is delegated to the worker.
    if MODELS[quality].get("backend") == "mlx":
        return

    dest_dir = _USER_MODELS_DIR / quality
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = GH_RELEASE_URL.format(quality=quality)
    tmp_file = _USER_MODELS_DIR / f"{quality}.tar.gz.tmp"

    opener = _build_opener(proxy_url=proxy_url, ignore_ssl=ignore_ssl)

    try:
        with opener.open(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 131072  # 128 KB

            with open(tmp_file, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)

        # Extract tar.gz
        with tarfile.open(tmp_file, "r:gz") as tar:
            tar.extractall(path=dest_dir)

        tmp_file.unlink()

    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise


def delete_model(quality: str) -> None:
    """Remove a downloaded model to free disk space. Won't delete bundled models."""
    entry = MODELS.get(quality)
    if entry and entry.get("backend") == "mlx":
        hf_repo = entry.get("hf_repo")
        if hf_repo:
            cache_dir = _mlx_hf_cache_dir(hf_repo)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        return

    user_path = _USER_MODELS_DIR / quality
    if user_path.exists():
        shutil.rmtree(user_path)


def ensure_model_available(
    quality: str,
    proxy_url: Optional[str] = None,
    ignore_ssl: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Download the model if needed, then return its identifier.

    For CT2 tiers this returns a filesystem path. For MLX tiers it returns
    the HuggingFace repo id (mlx-whisper handles the download lazily).
    """
    entry = MODELS.get(quality)
    if entry and entry.get("backend") == "mlx":
        # Defer download to mlx-whisper at transcribe time.
        return entry.get("hf_repo")

    if not model_is_ready(quality):
        download_model(quality, proxy_url=proxy_url, ignore_ssl=ignore_ssl, progress_cb=progress_cb)
    path = get_model_path_for_app(quality)
    if path is None:
        raise RuntimeError(f"Model '{quality}' could not be made available.")
    return path


def get_proxy_from_config(config: dict) -> tuple[Optional[str], bool]:
    """Read proxy settings from config dict. Returns (proxy_url, ignore_ssl)."""
    proxy_url = config.get("proxy_url", "").strip() or None
    ignore_ssl = str(config.get("ignore_ssl", "false")).lower() == "true"
    return proxy_url, ignore_ssl
