#!/bin/bash

# ==============================================================================
# MeetingGenie runner script
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   Iniciando MeetingGenie..."
echo "========================================"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 no está instalado."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "No se encontró el entorno virtual. Creándolo..."
    python3 -m venv venv

    echo "Activando entorno virtual..."
    source venv/bin/activate

    echo "Instalando dependencias (esto puede tardar unos minutos)..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            REQ_FILE="environments/requirements_macOS_arm64.txt"
        else
            REQ_FILE="environments/requirements_macOS_x86_64_NOT_WORKING.txt"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        REQ_FILE="environments/requirements_linux.txt"
    else
        echo "SO no detectado automáticamente."
        REQ_FILE="environments/requirements_linux.txt"
    fi

    if [ -f "$REQ_FILE" ]; then
        pip install --upgrade pip
        pip install -r "$REQ_FILE"
    else
        echo "Error: No se encontró el archivo de requisitos $REQ_FILE"
        exit 1
    fi
else
    source venv/bin/activate
fi

echo "Ejecutando MeetingGenie..."
python3 noScribe.py "$@"

deactivate
