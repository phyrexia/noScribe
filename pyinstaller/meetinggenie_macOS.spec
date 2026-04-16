# -*- mode: python ; coding: utf-8 -*-

############################################
# MeetingGenie (Flet) – macOS build spec
# Run from /noScribe/pyinstaller subdir!
############################################

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

import os as _os, platform as _platform
_spec_dir = _os.path.dirname(_os.path.abspath(SPEC))
_root_dir = _os.path.dirname(_spec_dir)

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
mg_datas = [
    ('../trans/', './trans/'),
    ('../LICENSE.txt', '.'),
    ('../languages.yml', '.'),
    ('../views/', './views/'),
]

# Bundle fast model if available (on-demand download for others)
_fast_model = _os.path.join(_root_dir, 'models', 'fast')
if _os.path.isdir(_fast_model):
    mg_datas += [('../models/fast/', './models/fast/')]

# Pyannote models (bundled, ~200 MB)
mg_datas += [('../pyannote/', './pyannote/')]

# ML framework data
mg_datas += copy_metadata('AdvancedHTMLParser')
mg_datas += collect_data_files('faster_whisper')
mg_datas += collect_data_files('lightning')
mg_datas += collect_data_files('lightning_fabric')
mg_datas += collect_data_files('librosa')
mg_datas += collect_data_files('pyannote')
mg_datas += collect_data_files('torchcodec')
mg_datas += copy_metadata('filelock')
mg_datas += copy_metadata('tqdm')
mg_datas += copy_metadata('requests')
mg_datas += copy_metadata('packaging')
mg_datas += copy_metadata('numpy')
mg_datas += copy_metadata('scipy')
mg_datas += copy_metadata('tokenizers')
mg_datas += copy_metadata('pyannote.audio')
mg_datas += copy_metadata('pyannote.core')
mg_datas += copy_metadata('pyannote.database')
mg_datas += copy_metadata('pyannote.metrics')
mg_datas += copy_metadata('pyannote.pipeline')

# Flet runtime
mg_datas += collect_data_files('flet')
mg_datas += collect_data_files('flet_desktop')

# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------
_ffmpeg_arm64 = _os.path.join(_root_dir, 'ffmpeg-arm64')
_ffmpeg_x86   = _os.path.join(_root_dir, 'ffmpeg')
_is_arm64 = _platform.machine() == 'arm64'

mg_binaries = []
if _is_arm64 and _os.path.exists(_ffmpeg_arm64):
    mg_binaries += [('../ffmpeg-arm64', '.')]
elif _os.path.exists(_ffmpeg_x86):
    mg_binaries += [('../ffmpeg', '.')]

mg_binaries += collect_dynamic_libs('pyannote')
mg_binaries += collect_dynamic_libs('torchcodec')

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
mg_hiddenimports = [
    'flet', 'flet_desktop', 'flet_core',
    'views', 'views.shell', 'views.transcribe_page',
    'views.queue_page', 'views.editor_page', 'views.settings_page',
    'views.dialogs', 'views.dialogs.speaker_naming',
    'app_state', 'config', 'models', 'event_bus',
    'transcription_runner', 'transcription_service',
    'model_manager', 'utils', 'speaker_db',
    'anthropic_summarizer', 'warmup',
    'whisper_mp_worker', 'pyannote_mp_worker', 'live_mp_worker',
]
mg_hiddenimports += collect_submodules('pyannote')
mg_hiddenimports += collect_submodules('scipy')

# SpeechBrain (required by pyannote)
tmp_ret = collect_all('speechbrain')
mg_datas += tmp_ret[0]
mg_binaries += tmp_ret[1]
mg_hiddenimports += tmp_ret[2]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
mg_a = Analysis(
    ['../main.py'],
    pathex=[_root_dir],
    binaries=mg_binaries,
    datas=mg_datas,
    hiddenimports=mg_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'gradio', 'triton', 'IPython', 'jupyter', 'matplotlib',
        'PIL.ImageQt', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'customtkinter', 'CTkToolTip', 'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

mg_pyz = PYZ(mg_a.pure, mg_a.zipped_data, cipher=block_cipher)

mg_exe = EXE(
    mg_pyz,
    mg_a.scripts,
    [],
    exclude_binaries=True,
    name='MeetingGenie',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
    icon=['../noScribeLogo.ico'],
)

coll = COLLECT(
    mg_exe,
    mg_a.binaries,
    mg_a.zipfiles,
    mg_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeetingGenie',
)

app = BUNDLE(
    coll,
    name='MeetingGenie.app',
    icon='../noScribeLogo.ico',
    bundle_identifier='com.meetinggenie.app',
    info_plist={
        "CFBundleShortVersionString": "1.0",
        "CFBundleName": "MeetingGenie",
        "CFBundleDisplayName": "MeetingGenie",
        "NSMicrophoneUsageDescription": "MeetingGenie needs microphone access for live meeting transcription.",
        "NSAppleEventsUsageDescription": "MeetingGenie uses Apple Events for system integration.",
        "LSMinimumSystemVersion": "14.0",
    },
)
