#!/bin/bash

# ==============================================================================
# MeetingGenie (Flet UI) runner script
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   Iniciando MeetingGenie..."
echo "========================================"

# Use local venv (outside OneDrive) if available, else fallback to ./venv
LOCAL_VENV="$HOME/.meetinggenie-venv"
if [ -d "$LOCAL_VENV" ]; then
    PYTHON="$LOCAL_VENV/bin/python3"
elif [ -d "venv" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="python3"
fi

if [ ! -x "$PYTHON" ] && ! command -v "$PYTHON" &> /dev/null; then
    echo "Error: python3 no encontrado."
    echo "Instala dependencias con: python3 -m venv ~/.meetinggenie-venv && ~/.meetinggenie-venv/bin/pip install -r environments/requirements_macOS_arm64.txt flet"
    exit 1
fi

echo "Usando: $PYTHON"
"$PYTHON" main.py "$@"
