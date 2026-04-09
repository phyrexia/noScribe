#!/usr/bin/env bash
# noScribe — Installation script for running from source
# Supports: macOS (Apple Silicon), Linux. Windows users: use WSL or the pre-built installer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="venv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
fail()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Package manager detection
# ---------------------------------------------------------------------------
OS="$(uname -s)"

install_pkg() {
    local pkg="$1"
    info "Attempting to install $pkg..."
    case "$OS" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install "$pkg"
            else
                fail "$pkg not found and Homebrew is not installed. Install Homebrew first: https://brew.sh"
            fi
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg"
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y "$pkg"
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -S --noconfirm "$pkg"
            else
                fail "$pkg not found and no supported package manager detected. Install it manually."
            fi
            ;;
        *)
            fail "$pkg not found. Install it manually."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# 0. Install system dependencies if missing
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

# --- git ---
if ! command -v git >/dev/null 2>&1; then
    warn "git not found."
    install_pkg git
fi
ok "git: $(git --version)"

# --- Python 3 ---
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    warn "Python 3 not found."
    case "$OS" in
        Darwin) install_pkg python@3.12 ;;
        Linux)  install_pkg python3 ;;
    esac
    # Refresh path
    hash -r 2>/dev/null || true
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    fail "Python 3 still not found after install attempt. Install Python 3.12+ manually."
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    fail "Python >= 3.11 required (found $PY_VERSION). Upgrade manually."
fi
ok "Python $PY_VERSION"

# --- python3-venv (Linux often needs it separately) ---
if [ "$OS" = "Linux" ]; then
    if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
        warn "python3-venv not found."
        install_pkg "python${PY_VERSION}-venv" 2>/dev/null || install_pkg python3-venv
    fi
fi

# --- ffmpeg ---
if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg not found (required for audio processing)."
    install_pkg ffmpeg
fi
ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# --- git-lfs (optional but preferred for model downloads) ---
if ! command -v git-lfs >/dev/null 2>&1; then
    info "git-lfs not found. Installing (improves model downloads)..."
    install_pkg git-lfs 2>/dev/null || warn "git-lfs install failed — will use direct download fallback."
fi
if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --skip-repo >/dev/null 2>&1 || true
    ok "git-lfs available"
fi

# ---------------------------------------------------------------------------
# 1. Detect platform & pick requirements file
# ---------------------------------------------------------------------------
ARCH="$(uname -m)"

case "$OS" in
    Darwin)
        if [ "$ARCH" = "arm64" ]; then
            REQUIREMENTS="environments/requirements_macOS_arm64.txt"
        else
            warn "Intel Mac support is experimental and may not work with pyannote v4."
            REQUIREMENTS="environments/requirements_macOS_x86_64_NOT_WORKING.txt"
        fi
        ;;
    Linux)
        REQUIREMENTS="environments/requirements_linux.txt"
        ;;
    *)
        fail "Unsupported OS: $OS. On Windows, use WSL or download the pre-built installer."
        ;;
esac

[ -f "$REQUIREMENTS" ] || fail "Requirements file not found: $REQUIREMENTS"
ok "Platform: $OS/$ARCH → $REQUIREMENTS"

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Using pip from: $(which pip)"

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing dependencies from $REQUIREMENTS (this may take a while)..."
pip install -r "$REQUIREMENTS" --quiet
ok "Python dependencies installed"

# ---------------------------------------------------------------------------
# 4. Clone noScribeEdit (the editor component)
# ---------------------------------------------------------------------------
if [ -L "noScribeEdit" ]; then
    info "Removing existing noScribeEdit symlink (points to local dev path)"
    rm noScribeEdit
fi

if [ -d "noScribeEdit" ]; then
    info "noScribeEdit already present"
else
    info "Cloning noScribeEdit..."
    git clone https://github.com/kaixxx/noScribeEditor.git noScribeEdit
    ok "noScribeEdit cloned"
fi

# ---------------------------------------------------------------------------
# 5. Download Whisper models
# ---------------------------------------------------------------------------
MODELS_DIR="models"

if [ -L "$MODELS_DIR" ]; then
    info "Removing existing models symlink (points to local dev path)"
    rm "$MODELS_DIR"
fi
mkdir -p "$MODELS_DIR"

download_model() {
    local name="$1"
    local repo="$2"
    local dest="$MODELS_DIR/$name"

    if [ -d "$dest" ] && [ -f "$dest/model.bin" ]; then
        ok "Model '$name' already downloaded"
        return
    fi

    info "Downloading model '$name' from HuggingFace ($repo)..."
    info "  This may take several minutes depending on your connection."

    if command -v git-lfs >/dev/null 2>&1; then
        git clone "https://huggingface.co/$repo" "$dest" --depth 1
    else
        # Fallback: download individual files without git-lfs
        mkdir -p "$dest"
        for f in config.json model.bin tokenizer.json vocabulary.json preprocessor_config.json; do
            local url="https://huggingface.co/$repo/resolve/main/$f"
            info "  Downloading $f..."
            if command -v curl >/dev/null 2>&1; then
                curl -L --progress-bar -o "$dest/$f" "$url"
            elif command -v wget >/dev/null 2>&1; then
                wget -q --show-progress -O "$dest/$f" "$url"
            else
                fail "Neither curl nor wget found. Install one of them."
            fi
        done
    fi
    ok "Model '$name' ready"
}

echo ""
info "=== Downloading Whisper models ==="
info "The 'fast' model (~310 MB) is required. The 'precise' model (~1.5 GB) is optional."
echo ""

download_model "fast" "mukowaty/faster-whisper-int8"

read -r -p "$(printf '\033[1;34m[INFO]\033[0m  Download the precise model too? (~1.5 GB) [y/N]: ')" REPLY
if [[ "${REPLY,,}" =~ ^y ]]; then
    download_model "precise" "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
ok "noScribe installation complete!"
echo "============================================"
echo ""
info "To run noScribe:"
info "  ./run.sh"
echo ""
info "To activate the virtual environment manually:"
info "  source $VENV_DIR/bin/activate"
info "  python3 noScribe.py"
echo ""
