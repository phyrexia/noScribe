# MeetingGenie - Flet Application Entry Point

import os
import sys
import ssl
import urllib.request

# Bypass corporate SSL proxy for any Flet runtime downloads
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
)

import flet as ft

# Ensure the app directory is on sys.path so modules resolve correctly
app_dir = os.path.abspath(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import multiprocessing as mp

from app_state import AppState
from views.shell import build_shell
from views.transcribe_page import build_transcribe_page
from views.queue_page import build_queue_page
from views.editor_page import build_editor_page
from views.settings_page import build_settings_page

mp.freeze_support()

BRAND_BLUE = "#0A84FF"


def _start_warmup():
    """Spawn a background subprocess that imports torch/pyannote/whisper
    to warm the OS page cache. Makes first transcription ~2-3x faster."""
    try:
        ctx = mp.get_context("spawn")
        from warmup import warmup
        proc = ctx.Process(target=warmup, daemon=True)
        proc.start()
        print("[main] Warmup subprocess started")
    except Exception as e:
        print(f"[main] Warmup failed to start: {e}")


def main(page: ft.Page):
    # --- Window setup --------------------------------------------------
    page.title = "MeetingGenie"
    page.window.width = 1200
    page.window.height = 780
    page.window.min_width = 900
    page.window.min_height = 600

    # --- Theme ---------------------------------------------------------
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme_seed=BRAND_BLUE,
        visual_density=ft.VisualDensity.COMPACT,
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=BRAND_BLUE,
        visual_density=ft.VisualDensity.COMPACT,
    )

    # --- App state -----------------------------------------------------
    state = AppState(app_dir)

    # --- Build pages ---------------------------------------------------
    pages = {
        "transcribe": build_transcribe_page(page, state),
        "queue": build_queue_page(page, state),
        "editor": build_editor_page(page, state),
        "settings": build_settings_page(page, state),
    }

    # --- Assemble shell ------------------------------------------------
    build_shell(page, state, pages)


if __name__ == "__main__":
    _start_warmup()
    ft.run(main)
