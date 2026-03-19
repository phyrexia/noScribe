# MeetingGenie - Model Manager
# Handles on-demand download of Whisper models with corporate proxy support.

import os
import sys
import json
import shutil
import hashlib
import platform
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable

import appdirs

APP_NAME = 'MeetingGenie'

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each model entry defines:
#   hf_repo  – HuggingFace repo slug (used to build download URLs)
#   files    – list of filenames to download from the repo
#   size_mb  – approximate total size (for UI display)
#   quality  – 'fast' | 'precise'
#
# We ship ZERO model files in the installer.
# On first launch the user is prompted to download the 'fast' model.
# 'precise' is downloaded only if the user explicitly selects it.
# ---------------------------------------------------------------------------

MODELS = {
    "fast": {
        "hf_repo": "mukowaty/faster-whisper-int8",
        "files": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
            "preprocessor_config.json",
        ],
        "size_mb": 310,
        "label": "Fast (310 MB) – good for most meetings",
    },
    "precise": {
        "hf_repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "files": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
            "preprocessor_config.json",
        ],
        "size_mb": 1500,
        "label": "Precise (1.5 GB) – highest accuracy",
    },
}

HF_BASE = "https://huggingface.co/{repo}/resolve/main/{file}"


def models_dir() -> Path:
    """Root directory where models are stored (user data, not inside the .app)."""
    base = Path(appdirs.user_data_dir(APP_NAME))
    base.mkdir(parents=True, exist_ok=True)
    return base


def model_path(quality: str) -> Path:
    """Return the directory for a given model quality."""
    return models_dir() / quality


def model_is_ready(quality: str) -> bool:
    """Return True if all required files for the model are present."""
    entry = MODELS.get(quality)
    if not entry:
        return False
    mpath = model_path(quality)
    return all((mpath / f).exists() for f in entry["files"])


def _build_opener(proxy_url: Optional[str] = None, ignore_ssl: bool = False):
    """Build a urllib opener that respects corporate proxies and optional SSL bypass."""
    handlers = []

    if proxy_url:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_url,
            "https": proxy_url,
        })
        handlers.append(proxy_handler)
    else:
        # Honour system proxy settings (set by corporate IT via macOS Network Preferences)
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
    """
    Download a model from HuggingFace Hub.

    Args:
        quality:      'fast' or 'precise'
        proxy_url:    Optional explicit proxy, e.g. 'http://proxy.corp.com:8080'
        ignore_ssl:   Set True to bypass SSL verification (corporate proxy interception)
        progress_cb:  Called with (bytes_downloaded, total_bytes) during download
    """
    entry = MODELS[quality]
    dest_dir = model_path(quality)
    dest_dir.mkdir(parents=True, exist_ok=True)

    opener = _build_opener(proxy_url=proxy_url, ignore_ssl=ignore_ssl)

    for filename in entry["files"]:
        dest_file = dest_dir / filename
        if dest_file.exists():
            continue  # already downloaded

        url = HF_BASE.format(repo=entry["hf_repo"], file=filename)
        tmp_file = dest_dir / (filename + ".tmp")

        try:
            with opener.open(url) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536  # 64 KB

                with open(tmp_file, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total)

            tmp_file.rename(dest_file)

        except Exception:
            if tmp_file.exists():
                tmp_file.unlink()
            raise


def delete_model(quality: str) -> None:
    """Remove a downloaded model to free disk space."""
    mpath = model_path(quality)
    if mpath.exists():
        shutil.rmtree(mpath)


def get_model_path_for_app(quality: str) -> Optional[str]:
    """
    Return the absolute path to use as 'model_size_or_path' in faster-whisper,
    or None if the model is not downloaded yet.
    """
    if model_is_ready(quality):
        return str(model_path(quality))
    return None


def ensure_model_available(
    quality: str,
    proxy_url: Optional[str] = None,
    ignore_ssl: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Download the model if needed, then return its path.
    Raises RuntimeError if download fails.
    """
    if not model_is_ready(quality):
        download_model(
            quality,
            proxy_url=proxy_url,
            ignore_ssl=ignore_ssl,
            progress_cb=progress_cb,
        )
    path = get_model_path_for_app(quality)
    if path is None:
        raise RuntimeError(f"Model '{quality}' could not be downloaded.")
    return path


def get_proxy_from_config(config: dict) -> tuple[Optional[str], bool]:
    """
    Read proxy settings from the app config dict.

    Config keys:
        proxy_url   – e.g. 'http://proxy.corp.com:8080', or '' for system proxy
        ignore_ssl  – 'true'/'false' to bypass SSL verification (corporate MITM proxy)

    Returns (proxy_url_or_None, ignore_ssl_bool)
    """
    proxy_url = config.get("proxy_url", "").strip() or None
    ignore_ssl = str(config.get("ignore_ssl", "false")).lower() == "true"
    return proxy_url, ignore_ssl
