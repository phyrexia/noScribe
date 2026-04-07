#!/bin/bash

# Activar el entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Ejecutar MeetingGenie
echo "Iniciando MeetingGenie..."
python3 noScribe.py
