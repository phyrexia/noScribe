#!/bin/bash

# Activar el entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Ejecutar noScribe
echo "Iniciando noScribe..."
python3 noScribe.py
