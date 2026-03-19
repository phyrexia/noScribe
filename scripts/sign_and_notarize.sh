#!/usr/bin/env bash
# ============================================================
#  MeetingGenie – macOS Sign & Notarize
#  Usage:
#    ./scripts/sign_and_notarize.sh [path/to/MeetingGenie.app]
#
#  Required env vars (or hardcode below):
#    MG_IDENTITY       – Developer ID Application: Your Name (TEAMID)
#    MG_APPLE_ID       – your@apple.id (for notarytool)
#    MG_TEAM_ID        – 10-char Apple Team ID
#    MG_APP_PASSWORD   – app-specific password from appleid.apple.com
#
#  Optional:
#    MG_BUNDLE_ID      – override bundle identifier
#                        default: com.meetinggenie.app
# ============================================================
set -euo pipefail

APP="${1:-dist/MeetingGenie.app}"
IDENTITY="${MG_IDENTITY:-}"
APPLE_ID="${MG_APPLE_ID:-}"
TEAM_ID="${MG_TEAM_ID:-}"
APP_PASSWORD="${MG_APP_PASSWORD:-}"
BUNDLE_ID="${MG_BUNDLE_ID:-com.meetinggenie.app}"
ENTITLEMENTS="$(dirname "$0")/entitlements.plist"
ZIP_PATH="${APP%.app}.zip"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [[ -z "$IDENTITY" || -z "$APPLE_ID" || -z "$TEAM_ID" || -z "$APP_PASSWORD" ]]; then
  echo "ERROR: set MG_IDENTITY, MG_APPLE_ID, MG_TEAM_ID and MG_APP_PASSWORD" >&2
  exit 1
fi
if [[ ! -d "$APP" ]]; then
  echo "ERROR: .app not found at: $APP" >&2
  exit 1
fi
if [[ ! -f "$ENTITLEMENTS" ]]; then
  echo "ERROR: entitlements.plist not found at: $ENTITLEMENTS" >&2
  exit 1
fi
command -v codesign  >/dev/null || { echo "codesign not found"; exit 1; }
command -v xcrun     >/dev/null || { echo "xcrun not found";  exit 1; }
command -v ditto     >/dev/null || { echo "ditto not found";  exit 1; }

# ── 1. Remove quarantine & fix permissions ───────────────────────────────────
echo "▶ Removing quarantine attributes..."
xattr -cr "$APP"
find "$APP" -name "*.dylib" -o -name "*.so" | xargs chmod +x 2>/dev/null || true

# ── 2. Sign all nested binaries first (inside-out) ───────────────────────────
echo "▶ Signing nested binaries..."
find "$APP/Contents/Frameworks" \
     "$APP/Contents/MacOS" \
     \( -name "*.dylib" -o -name "*.so" -o -name "*.framework" \) \
     2>/dev/null | sort -r | while read -r item; do
  codesign --force --options runtime \
           --entitlements "$ENTITLEMENTS" \
           --sign "$IDENTITY" \
           --timestamp \
           "$item" 2>/dev/null || true
done

# ── 3. Sign the main executable ──────────────────────────────────────────────
echo "▶ Signing main executable..."
MAIN_BIN="$APP/Contents/MacOS/MeetingGenie"
if [[ -f "$MAIN_BIN" ]]; then
  codesign --force --options runtime \
           --entitlements "$ENTITLEMENTS" \
           --sign "$IDENTITY" \
           --timestamp \
           "$MAIN_BIN"
fi

# ── 4. Sign the .app bundle itself ───────────────────────────────────────────
echo "▶ Signing .app bundle..."
codesign --force --deep --options runtime \
         --entitlements "$ENTITLEMENTS" \
         --sign "$IDENTITY" \
         --timestamp \
         --identifier "$BUNDLE_ID" \
         "$APP"

# ── 5. Verify signature ──────────────────────────────────────────────────────
echo "▶ Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP"
spctl --assess --type execute --verbose "$APP" || {
  echo "WARN: spctl check failed (expected before notarization)" >&2
}

# ── 6. Package for notarization ──────────────────────────────────────────────
echo "▶ Creating ZIP for notarization..."
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP" "$ZIP_PATH"

# ── 7. Submit to Apple Notary Service ────────────────────────────────────────
echo "▶ Submitting to Apple Notary Service (this may take a few minutes)..."
xcrun notarytool submit "$ZIP_PATH" \
  --apple-id "$APPLE_ID" \
  --team-id "$TEAM_ID" \
  --password "$APP_PASSWORD" \
  --wait \
  --timeout 600

# ── 8. Staple the notarization ticket ────────────────────────────────────────
echo "▶ Stapling notarization ticket..."
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

# ── 9. Final Gatekeeper check ────────────────────────────────────────────────
echo "▶ Final Gatekeeper check..."
spctl --assess --type execute --verbose "$APP"

echo ""
echo "✓ Done! $APP is signed, notarized and stapled."
echo "  ZIP left at: $ZIP_PATH"
echo ""
echo "Next steps:"
echo "  • Wrap in a .dmg: hdiutil create -volname MeetingGenie -srcfolder dist/"
echo "    MeetingGenie.app -ov -format UDZO dist/MeetingGenie.dmg"
echo "  • Sign the .dmg:  codesign --sign \"\$MG_IDENTITY\" dist/MeetingGenie.dmg"
echo "  • Notarize .dmg:  xcrun notarytool submit dist/MeetingGenie.dmg ..."
