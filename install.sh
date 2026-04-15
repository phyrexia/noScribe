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
# Package manager detection & installer
# ---------------------------------------------------------------------------
OS="$(uname -s)"

ensure_brew() {
    if command -v brew >/dev/null 2>&1; then
        return 0
    fi
    info "Homebrew not found. Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for this session (Apple Silicon vs Intel path)
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    if ! command -v brew >/dev/null 2>&1; then
        fail "Homebrew installation failed. Install manually: https://brew.sh"
    fi
    ok "Homebrew installed"
}

install_pkg() {
    local pkg="$1"
    info "Attempting to install $pkg..."
    case "$OS" in
        Darwin)
            ensure_brew
            brew install "$pkg"
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
find_python() {
    # Try common Python binary names in order of preference
    for cmd in python3.12 python3.13 python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local ver
            ver=$("$cmd" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || continue
            local maj
            maj=$("$cmd" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || continue
            if [ "$maj" -eq 3 ] && [ "$ver" -ge 11 ]; then
                PYTHON="$cmd"
                return 0
            fi
        fi
    done
    return 1
}

if ! find_python; then
    warn "Python >= 3.11 not found. Installing..."
    case "$OS" in
        Darwin) install_pkg python@3.12 ;;
        Linux)  install_pkg python3 ;;
    esac
    hash -r 2>/dev/null || true
    # After brew install, the binary may be python3.12 not python3
    if ! find_python; then
        fail "Python >= 3.11 still not found after install. Install Python 3.12+ manually."
    fi
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python $PY_VERSION ($(command -v "$PYTHON"))"

# --- pip (some distros don't include it) ---
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    warn "pip not found."
    case "$OS" in
        Darwin) "$PYTHON" -m ensurepip --upgrade 2>/dev/null || install_pkg python@3.12 ;;
        Linux)  install_pkg python3-pip ;;
    esac
fi

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
# 5. Download Whisper models from GitHub Releases
# ---------------------------------------------------------------------------
GH_REPO="phyrexia/noScribe"
GH_TAG="models-v1"
GH_ASSET_URL="https://github.com/${GH_REPO}/releases/download/${GH_TAG}"
MODELS_DIR="models"

if [ -L "$MODELS_DIR" ]; then
    info "Removing existing models symlink (points to local dev path)"
    rm "$MODELS_DIR"
fi
mkdir -p "$MODELS_DIR"

download_model() {
    local name="$1"
    local asset="${name}.tar.gz"
    local dest="$MODELS_DIR/$name"

    if [ -d "$dest" ] && [ -f "$dest/model.bin" ]; then
        ok "Model '$name' already downloaded"
        return
    fi

    # Clean up incomplete previous download
    if [ -d "$dest" ]; then
        warn "Removing incomplete model directory: $dest"
        rm -rf "$dest"
    fi
    mkdir -p "$dest"

    local url="${GH_ASSET_URL}/${asset}"
    local tmp_file="$MODELS_DIR/${asset}"

    info "Downloading model '$name' from GitHub Releases..."

    if command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$tmp_file" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$tmp_file" "$url"
    else
        fail "Neither curl nor wget found."
    fi

    info "Extracting model '$name'..."
    tar xzf "$tmp_file" -C "$dest"
    rm -f "$tmp_file"

    if [ -f "$dest/model.bin" ]; then
        ok "Model '$name' ready"
    else
        fail "Model '$name' extraction failed — model.bin not found"
    fi
}

echo ""
info "=== Downloading Whisper models ==="
info "Models are downloaded from GitHub (no HuggingFace / no Zscaler issues)."
echo ""
info "Available models:"
info "  1) small   — 205 MB (lightweight, quick transcriptions)"
info "  2) fast    — 656 MB (best speed/quality balance) [default]"
info "  3) precise — 1.4 GB (highest accuracy)"
echo ""
read -r -p "$(printf '\033[1;34m[INFO]\033[0m  Which model to install? [1/2/3, default=2]: ')" MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-2}"

case "$MODEL_CHOICE" in
    1) download_model "small" ;;
    2) download_model "fast" ;;
    3) download_model "fast"; download_model "precise" ;;
    *) info "Invalid choice, installing 'fast'"; download_model "fast" ;;
esac

read -r -p "$(printf '\033[1;34m[INFO]\033[0m  Download additional models? [y/N]: ')" MORE
MORE_LOWER="$(echo "$MORE" | tr '[:upper:]' '[:lower:]')"
if [ "$MORE_LOWER" = "y" ] || [ "$MORE_LOWER" = "yes" ]; then
    download_model "small"
    download_model "fast"
    download_model "precise"
fi

# ---------------------------------------------------------------------------
# 6. Ensure scripts are executable
# ---------------------------------------------------------------------------
chmod +x "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR/install.sh" 2>/dev/null || true

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
