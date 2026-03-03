#!/bin/bash

# ==============================================================================
# noScribe runner script
# ==============================================================================

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   Iniciando noScribe..."
echo "========================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 no está instalado."
    exit 1
fi

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "No se encontró el entorno virtual. Creándolo..."
    python3 -m venv venv
    
    echo "Activando entorno virtual..."
    source venv/bin/activate
    
    echo "Instalando dependencias (esto puede tardar unos minutos)..."
    # Determine the requirements file based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if [[ $(uname -m) == "arm64" ]]; then
            REQ_FILE="environments/requirements_macOS_arm64.txt"
        else
            REQ_FILE="environments/requirements_macOS_x86_64_NOT_WORKING.txt"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        REQ_FILE="environments/requirements_linux.txt"
    else
        echo "SO no detectado automáticamente. Intentando con requisitos generales..."
        REQ_FILE="environments/requirements_linux.txt"
    fi

    if [ -f "$REQ_FILE" ]; then
        pip install --upgrade pip
        pip install -r "$REQ_FILE"
        
        # Also install Editor requirements if folder exists
        if [ -d "noScribeEdit" ] && [ -f "noScribeEdit/environments/requirements.txt" ]; then
            echo "Instalando dependencias del Editor..."
            pip install -r noScribeEdit/environments/requirements.txt
        fi
    else
        echo "Error: No se encontró el archivo de requisitos $REQ_FILE"
        exit 1
    fi
else
    # venv exists, just activate it
    source venv/bin/activate
fi

# Run noScribe
echo "Ejecutando noScribe.py..."
python3 noScribe.py "$@"

# Deactivate venv on exit
deactivate
