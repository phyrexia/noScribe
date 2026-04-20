#!/bin/bash
# ==============================================================================
# MeetingGenie (Flet) – macOS Build Script
# Produces MeetingGenie.app and optionally a DMG
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${MG_VENV_DIR:-$HOME/.meetinggenie-venv}"
SPEC="$ROOT_DIR/pyinstaller/meetinggenie_macOS.spec"
DIST="$ROOT_DIR/pyinstaller/dist"

echo "========================================"
echo "  MeetingGenie Build (Flet)"
echo "========================================"
echo "Root: $ROOT_DIR"
echo "Venv: $VENV_DIR"
echo ""

# --- 1. Check venv -----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    echo "Create it with:"
    echo "  python3 -m venv $VENV_DIR"
    echo "  $VENV_DIR/bin/pip install -r $ROOT_DIR/environments/requirements_macOS_arm64.txt flet pyinstaller"
    exit 1
fi

PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

# Ensure pyinstaller is available
if ! "$VENV_DIR/bin/pyinstaller" --version &>/dev/null; then
    echo "Installing PyInstaller..."
    "$PIP" install pyinstaller
fi

# --- 2. Download ffmpeg-arm64 if needed --------------------------------------
FFMPEG="$ROOT_DIR/ffmpeg-arm64"
if [ ! -f "$FFMPEG" ]; then
    echo "--- Downloading ffmpeg-arm64 ---"
    FFMPEG_URL="https://evermeet.cx/ffmpeg/getrelease/zip"
    TMP_ZIP="/tmp/ffmpeg-arm64.zip"
    curl -fSL "$FFMPEG_URL" -o "$TMP_ZIP" || {
        echo "WARNING: ffmpeg download failed. Build will continue without native ffmpeg."
        echo "You can manually place ffmpeg-arm64 in $ROOT_DIR"
    }
    if [ -f "$TMP_ZIP" ]; then
        unzip -o "$TMP_ZIP" -d /tmp/ffmpeg-extract
        mv /tmp/ffmpeg-extract/ffmpeg "$FFMPEG"
        chmod +x "$FFMPEG"
        rm -rf "$TMP_ZIP" /tmp/ffmpeg-extract
        echo "✓ ffmpeg-arm64 downloaded"
    fi
fi

# --- 3. Download fast model if not present -----------------------------------
FAST_MODEL="$ROOT_DIR/models/fast"
if [ ! -d "$FAST_MODEL" ] || [ ! -f "$FAST_MODEL/model.bin" ]; then
    echo "--- Downloading fast model ---"
    mkdir -p "$FAST_MODEL"
    MODEL_URL="https://github.com/phyrexia/noScribe/releases/download/models-v1/fast.tar.gz"
    TMP_TAR="/tmp/meetinggenie-fast.tar.gz"
    curl -fSL "$MODEL_URL" -o "$TMP_TAR" || {
        echo "WARNING: Model download failed. App will download on first use."
    }
    if [ -f "$TMP_TAR" ]; then
        tar xzf "$TMP_TAR" -C "$FAST_MODEL"
        rm "$TMP_TAR"
        echo "✓ Fast model downloaded ($(du -sh "$FAST_MODEL" | cut -f1))"
    fi
fi

# --- 4. PyInstaller ----------------------------------------------------------
echo ""
echo "--- Building with PyInstaller ---"
cd "$ROOT_DIR/pyinstaller"
"$VENV_DIR/bin/pyinstaller" "$SPEC" --distpath dist --workpath build --noconfirm
echo "✓ Build complete: $DIST/MeetingGenie.app"

# --- 4b. Bundle Flet desktop client ------------------------------------------
echo ""
echo "--- Bundling Flet client ---"
FLET_VER=$("$PYTHON" -c "import flet; print(flet.__version__)" 2>/dev/null || echo "0.84.0")
FLET_CLIENT="$HOME/.flet/client/flet-desktop-full-$FLET_VER"
FLET_DEST="$DIST/MeetingGenie.app/Contents/Resources/flet_client/flet-desktop-full-$FLET_VER"

if [ -d "$FLET_CLIENT" ]; then
    mkdir -p "$(dirname "$FLET_DEST")"
    cp -a "$FLET_CLIENT" "$FLET_DEST"
    echo "✓ Flet client $FLET_VER bundled ($(du -sh "$FLET_DEST" | cut -f1))"
else
    echo "WARNING: Flet client not found at $FLET_CLIENT"
    echo "Run the app once with: $PYTHON main.py  (to download client)"
fi

# --- 5. Code signing (optional) ----------------------------------------------
if [ "${MG_SKIP_SIGN:-0}" != "1" ] && [ -n "${MG_IDENTITY:-}" ]; then
    echo ""
    echo "--- Code Signing ---"
    if [ -f "$SCRIPT_DIR/sign_and_notarize.sh" ]; then
        bash "$SCRIPT_DIR/sign_and_notarize.sh" "$DIST/MeetingGenie.app"
    else
        echo "sign_and_notarize.sh not found, skipping"
    fi
else
    echo ""
    echo "Skipping code signing (set MG_IDENTITY to enable)"
fi

# --- 6. DMG creation (optional) ----------------------------------------------
if [ "${MG_SKIP_DMG:-0}" != "1" ]; then
    echo ""
    echo "--- Creating DMG ---"
    DMG_NAME="MeetingGenie-$(date +%Y%m%d).dmg"
    DMG_PATH="$DIST/$DMG_NAME"

    if command -v create-dmg &>/dev/null; then
        create-dmg \
            --volname "MeetingGenie" \
            --window-pos 200 120 \
            --window-size 660 400 \
            --icon "MeetingGenie.app" 180 170 \
            --app-drop-link 480 170 \
            --no-internet-enable \
            "$DMG_PATH" \
            "$DIST/MeetingGenie.app" || true
    else
        hdiutil create -volname "MeetingGenie" \
            -srcfolder "$DIST/MeetingGenie.app" \
            -ov -format UDZO \
            "$DMG_PATH"
    fi
    echo "✓ DMG created: $DMG_PATH ($(du -sh "$DMG_PATH" | cut -f1))"
else
    echo "Skipping DMG creation"
fi

echo ""
echo "========================================"
echo "  Build complete!"
echo "  App: $DIST/MeetingGenie.app"
echo "========================================"
