# -*- mode: python ; coding: utf-8 -*-

############################################
# MeetingGenie – macOS build spec
# Run from /noScribe/pyinstaller subdir!
############################################

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# Models are NOT bundled – they are downloaded on first use.
# Only the pyannote embedding/segmentation models ship with the app (small, ~200 MB).

noScribe_datas = [
    # ('../models/precise/', './models/precise/'),  # on-demand download
    # ('../models/fast/', './models/fast/'),         # on-demand download
    ('../trans/', './trans/'),
    ('../graphic_sw.png', '.'),
    ('../LICENSE.txt', '.'),
    ('../noScribeLogo.ico', '.'),
    ('../prompt.yml', '.'),
    ('../prompt_nd.yml', '.'),
    ('../README.md', '.'),
]
noScribe_datas += collect_data_files('customtkinter')
noScribe_datas += copy_metadata('AdvancedHTMLParser')
noScribe_datas += collect_data_files('faster_whisper')

# pyannote integration
noScribe_datas += [('../pyannote/', './pyannote/')]
noScribe_datas += collect_data_files('lightning')
noScribe_datas += collect_data_files('lightning_fabric')
noScribe_datas += collect_data_files('librosa')
noScribe_datas += collect_data_files('pyannote')
noScribe_datas += copy_metadata('filelock')
noScribe_datas += copy_metadata('tqdm')
# noScribe_datas += copy_metadata('regex')
noScribe_datas += copy_metadata('requests')
noScribe_datas += copy_metadata('packaging')
noScribe_datas += copy_metadata('numpy')
noScribe_datas += copy_metadata('scipy')
noScribe_datas += copy_metadata('tokenizers')
noScribe_datas += copy_metadata('pyannote.audio')
noScribe_datas += copy_metadata('pyannote.core')
noScribe_datas += copy_metadata('pyannote.database')
noScribe_datas += copy_metadata('pyannote.metrics')
noScribe_datas += copy_metadata('pyannote.pipeline')

import os as _os, platform as _platform
_spec_dir = _os.path.dirname(_os.path.abspath(SPEC))
_root_dir = _os.path.dirname(_spec_dir)
_ffmpeg_arm64 = _os.path.join(_root_dir, 'ffmpeg-arm64')
_ffmpeg_x86   = _os.path.join(_root_dir, 'ffmpeg')
_is_arm64 = _platform.machine() == 'arm64'

# On arm64 Macs only bundle ffmpeg-arm64 (saves ~77 MB vs. bundling both).
# Fall back to x86 ffmpeg if arm64 binary is missing (Rosetta2 will run it).
noScribe_binaries = []
if _is_arm64 and _os.path.exists(_ffmpeg_arm64):
    noScribe_binaries += [('../ffmpeg-arm64', '.')]
elif _os.path.exists(_ffmpeg_x86):
    noScribe_binaries += [('../ffmpeg', '.')]

noScribe_binaries += collect_dynamic_libs('pyannote')
noScribe_binaries += collect_dynamic_libs('torchcodec')
noScribe_datas += collect_data_files('torchcodec')

noScribe_hiddenimports = ['tkinter']
noScribe_hiddenimports += collect_submodules('pyannote')
noScribe_hiddenimports += collect_submodules('scipy')
# noScribe_hiddenimports += ['scipy._lib.array_api_compat.numpy.fft']

tmp_ret = collect_all('speechbrain')
noScribe_datas += tmp_ret[0]
noScribe_binaries += tmp_ret[1]
noScribe_hiddenimports += tmp_ret[2]

noScribe_a = Analysis(
    ['../noScribe.py'],
    pathex=[],
    binaries=noScribe_binaries,
    datas=noScribe_datas,
    hiddenimports=noScribe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude heavyweight packages that are not needed at runtime:
    # - gradio: web UI framework pulled as transitive dep, never used
    # - triton: GPU compiler, not needed on macOS
    # - IPython / jupyter: dev tools
    # - matplotlib: plotting, not used in the UI
    # - pandas: data analysis, not used
    excludes=['gradio', 'triton', 'IPython', 'jupyter', 'matplotlib', 'PIL.ImageQt', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

noScribe_pyz = PYZ(noScribe_a.pure, noScribe_a.zipped_data, cipher=block_cipher)

noScribe_exe = EXE(
    noScribe_pyz,
    noScribe_a.scripts,
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
    target_arch='arm64',  # Apple Silicon native
    codesign_identity=None,
    entitlements_file=None,
    icon=['../noScribeLogo.ico'],
)

coll = COLLECT(
    noScribe_exe,
    noScribe_a.binaries,
    noScribe_a.zipfiles,
    noScribe_a.datas,
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

