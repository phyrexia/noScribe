#!/usr/bin/env bash
# noScribe — Run script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Check that install.sh has been run
# ---------------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

if [ ! -d "models/fast" ] || [ ! -f "models/fast/model.bin" ]; then
    echo "Whisper models not found. Run ./install.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Activate venv and run
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source venv/bin/activate

echo "Starting noScribe..."
python3 noScribe.py "$@"
