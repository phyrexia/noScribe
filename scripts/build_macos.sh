#!/usr/bin/env bash
# ============================================================
#  MeetingGenie – macOS full build pipeline
#
#  Steps:
#    1. PyInstaller → dist/MeetingGenie.app
#    2. Download ffmpeg-arm64 if missing
#    3. Sign & notarize  (skipped if MG_SKIP_SIGN=1)
#    4. Build .dmg       (skipped if MG_SKIP_DMG=1)
#
#  Required env vars for signing (or set MG_SKIP_SIGN=1):
#    MG_IDENTITY, MG_APPLE_ID, MG_TEAM_ID, MG_APP_PASSWORD
#
#  Optional:
#    MG_SKIP_SIGN=1   – skip signing/notarization
#    MG_SKIP_DMG=1    – skip DMG creation
#    MG_VENV_DIR      – path to venv (default: ../venv)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP="$DIST_DIR/MeetingGenie.app"
DMG="$DIST_DIR/MeetingGenie.dmg"
SPEC="$ROOT_DIR/pyinstaller/noScribe_macOS.spec"
VENV_DIR="${MG_VENV_DIR:-$ROOT_DIR/venv}"
SKIP_SIGN="${MG_SKIP_SIGN:-0}"
SKIP_DMG="${MG_SKIP_DMG:-0}"

# ── ffmpeg arm64 ──────────────────────────────────────────────────────────────
FFMPEG_ARM64="$ROOT_DIR/ffmpeg-arm64"
FFMPEG_ARM64_URL="https://evermeet.cx/ffmpeg/getrelease/zip"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "▶ $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
hr()   { echo "────────────────────────────────────────────────────────"; }

# ── Preflight ─────────────────────────────────────────────────────────────────
log "MeetingGenie macOS build – $(date '+%Y-%m-%d %H:%M:%S')"
hr

[[ -d "$VENV_DIR" ]] || die "venv not found at $VENV_DIR – run: python3 -m venv $VENV_DIR && pip install -r environments/requirements_macOS.txt"
command -v pyinstaller >/dev/null 2>&1 || source "$VENV_DIR/bin/activate"
command -v pyinstaller >/dev/null 2>&1 || die "pyinstaller not found – activate the venv or install it"

# ── Download ffmpeg-arm64 if missing ─────────────────────────────────────────
if [[ ! -f "$FFMPEG_ARM64" ]]; then
  log "Downloading ffmpeg arm64..."
  TMP_ZIP=$(mktemp /tmp/ffmpeg-arm64-XXXX.zip)
  TMP_DIR=$(mktemp -d /tmp/ffmpeg-arm64-XXXX)
  if curl -fsSL "$FFMPEG_ARM64_URL" -o "$TMP_ZIP"; then
    unzip -o "$TMP_ZIP" -d "$TMP_DIR" >/dev/null 2>&1 || true
    FFMPEG_BIN=$(find "$TMP_DIR" -name "ffmpeg" -type f | head -1)
    if [[ -n "$FFMPEG_BIN" ]]; then
      cp "$FFMPEG_BIN" "$FFMPEG_ARM64"
      chmod +x "$FFMPEG_ARM64"
      log "ffmpeg-arm64 saved to $FFMPEG_ARM64"
    else
      log "WARNING: ffmpeg binary not found in zip – will use x86_64 ffmpeg (Rosetta2)"
    fi
  else
    log "WARNING: could not download ffmpeg-arm64 – will use x86_64 ffmpeg (Rosetta2)"
  fi
  rm -rf "$TMP_DIR" "$TMP_ZIP"
fi

# ── Clean previous build ──────────────────────────────────────────────────────
log "Cleaning previous build..."
rm -rf "$DIST_DIR/MeetingGenie.app" "$DIST_DIR/MeetingGenie" "$ROOT_DIR/build/MeetingGenie"

# ── PyInstaller ───────────────────────────────────────────────────────────────
log "Running PyInstaller..."
hr
cd "$ROOT_DIR/pyinstaller"
pyinstaller "$SPEC" --distpath "$DIST_DIR" --workpath "$ROOT_DIR/build" --noconfirm
hr
log "PyInstaller done → $APP"

[[ -d "$APP" ]] || die ".app not found at $APP"

# ── Sign & Notarize ───────────────────────────────────────────────────────────
if [[ "$SKIP_SIGN" == "1" ]]; then
  log "Skipping signing (MG_SKIP_SIGN=1)"
else
  log "Signing and notarizing..."
  bash "$SCRIPT_DIR/sign_and_notarize.sh" "$APP"
fi

# ── Build DMG ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_DMG" == "1" ]]; then
  log "Skipping DMG (MG_SKIP_DMG=1)"
else
  log "Building DMG..."
  rm -f "$DMG"

  # Use create-dmg if available, otherwise plain hdiutil
  if command -v create-dmg >/dev/null 2>&1; then
    create-dmg \
      --volname "MeetingGenie" \
      --volicon "$ROOT_DIR/noScribeLogo.ico" \
      --window-pos 200 120 \
      --window-size 660 400 \
      --icon-size 120 \
      --icon "MeetingGenie.app" 180 170 \
      --hide-extension "MeetingGenie.app" \
      --app-drop-link 480 170 \
      --no-internet-enable \
      "$DMG" \
      "$DIST_DIR/MeetingGenie.app"
  else
    log "create-dmg not found – using plain hdiutil (no fancy layout)"
    log "  Install with: brew install create-dmg"
    hdiutil create \
      -volname "MeetingGenie" \
      -srcfolder "$APP" \
      -ov \
      -format UDZO \
      "$DMG"
  fi

  # Sign the DMG too (if signing is enabled)
  if [[ "$SKIP_SIGN" != "1" && -n "${MG_IDENTITY:-}" ]]; then
    log "Signing DMG..."
    codesign --sign "$MG_IDENTITY" --timestamp "$DMG"
    log "Notarizing DMG..."
    xcrun notarytool submit "$DMG" \
      --apple-id "$MG_APPLE_ID" \
      --team-id "$MG_TEAM_ID" \
      --password "$MG_APP_PASSWORD" \
      --wait --timeout 600
    xcrun stapler staple "$DMG"
  fi

  log "DMG ready → $DMG"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
hr
echo ""
echo "Build complete!"
echo "  App : $APP"
[[ "$SKIP_DMG" != "1" ]] && echo "  DMG : $DMG"
echo ""
echo "Quick test:"
echo "  open \"$APP\""
echo ""
