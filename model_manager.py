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
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json"],
        "size_mb": 246,
        "label": "Small (246 MB) – fastest, for live mode",
    },
    "fast": {
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"],
        "size_mb": 785,
        "label": "Fast (785 MB) – good for most meetings",
    },
    "precise": {
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"],
        "size_mb": 1500,
        "label": "Precise (1.5 GB) – highest accuracy",
    },
}


def _app_dir() -> Path:
    """App directory (where the executable lives)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(os.path.abspath(os.path.dirname(__file__)))


def models_dir() -> Path:
    """User models directory (~/.meetinggenie/models/)."""
    _USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _USER_MODELS_DIR


def model_path(quality: str) -> Optional[Path]:
    """Return the path where the model is found, checking bundled first.

    Search order:
      1. Bundled: {app_dir}/models/{quality}/
      2. User:    ~/.meetinggenie/models/{quality}/
    Returns None if not found anywhere.
    """
    entry = MODELS.get(quality)
    if not entry:
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
    """Return absolute path for faster-whisper, or None if not available."""
    p = model_path(quality)
    return str(p) if p else None


def list_available_models() -> dict:
    """Return dict of {quality: {"ready": bool, "location": str, ...}}."""
    result = {}
    for q, entry in MODELS.items():
        p = model_path(q)
        result[q] = {
            "ready": p is not None,
            "location": str(p) if p else "not downloaded",
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
    user_path = _USER_MODELS_DIR / quality
    if user_path.exists():
        shutil.rmtree(user_path)


def ensure_model_available(
    quality: str,
    proxy_url: Optional[str] = None,
    ignore_ssl: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Download the model if needed, then return its path."""
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
