# noScribe - Model Manager
# Handles on-demand download of Whisper models from GitHub Releases.

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

APP_NAME = 'noScribe'

# Default models directory: ~/.noscribe/models/
_DEFAULT_MODELS_DIR = Path.home() / '.noscribe' / 'models'

# ---------------------------------------------------------------------------
# GitHub Release configuration
# ---------------------------------------------------------------------------
GH_REPO = "phyrexia/noScribe"
GH_RELEASE_TAG = "models-v1"
GH_ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{file}"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "small": {
        "asset": "small.tar.gz",
        "files": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
        ],
        "size_mb": 205,
        "label": "Small (205 MB) – lightweight, quick transcriptions",
    },
    "fast": {
        "asset": "fast.tar.gz",
        "files": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
            "preprocessor_config.json",
        ],
        "size_mb": 656,
        "label": "Fast (656 MB) – best speed/quality balance",
    },
    "precise": {
        "asset": "precise.tar.gz",
        "files": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
            "preprocessor_config.json",
        ],
        "size_mb": 1400,
        "label": "Precise (1.4 GB) – highest accuracy",
    },
}


def models_dir() -> Path:
    """Root directory where models are stored (~/.noscribe/models/)."""
    base = _DEFAULT_MODELS_DIR
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
    Download a model from GitHub Releases and extract it.

    Args:
        quality:      'small', 'fast', or 'precise'
        proxy_url:    Optional explicit proxy, e.g. 'http://proxy.corp.com:8080'
        ignore_ssl:   Set True to bypass SSL verification (corporate proxy interception)
        progress_cb:  Called with (bytes_downloaded, total_bytes) during download
    """
    entry = MODELS[quality]
    dest_dir = model_path(quality)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if model_is_ready(quality):
        return

    url = GH_ASSET_URL.format(repo=GH_REPO, tag=GH_RELEASE_TAG, file=entry["asset"])
    tmp_file = dest_dir / entry["asset"]

    opener = _build_opener(proxy_url=proxy_url, ignore_ssl=ignore_ssl)

    try:
        # Follow redirects (GitHub redirects to CDN)
        req = urllib.request.Request(url)
        with opener.open(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 262144  # 256 KB

            with open(tmp_file, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)

        # Extract tar.gz into model directory
        with tarfile.open(tmp_file, "r:gz") as tar:
            tar.extractall(path=dest_dir)

    except Exception:
        # Clean up partial downloads
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    finally:
        # Remove the tar.gz after extraction
        if tmp_file.exists():
            tmp_file.unlink()


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
