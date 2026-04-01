# MeetingGenie - Configuration Management
# Extracted from noScribe.py for clean architecture

import os
import platform
import locale
import yaml
import appdirs

APP_NAME = 'MeetingGenie'
APP_VERSION = '1.0'
APP_YEAR = '2025'

# --- Paths -----------------------------------------------------------

config_dir = appdirs.user_config_dir(APP_NAME)
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

config_file = os.path.join(config_dir, 'config.yml')

# --- All supported languages ------------------------------------------

ALL_LANGUAGES = {
    "Auto": "auto",
    "Multilingual": "multilingual",
    "Afrikaans": "af",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Marathi": "mr",
    "Maori": "mi",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Welsh": "cy",
}

# --- Config singleton -------------------------------------------------

_config: dict = {}


def _load_config() -> dict:
    """Load config from YAML file, return empty dict on failure."""
    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _ensure_loaded():
    """Lazy-load config on first access."""
    global _config
    if not _config:
        _config = _load_config()


def get_config(key: str, default=None):
    """Get a config value, set it if it doesn't exist."""
    _ensure_loaded()
    if key not in _config:
        _config[key] = default
    return _config[key]


def set_config(key: str, value):
    """Set a config value (does not save to disk)."""
    _ensure_loaded()
    _config[key] = value


def save_config():
    """Persist config to YAML file."""
    _ensure_loaded()
    with open(config_file, 'w') as f:
        yaml.safe_dump(_config, f)


def get_raw_config() -> dict:
    """Get the raw config dict (for backward compatibility)."""
    _ensure_loaded()
    return _config


# --- Languages filter -------------------------------------------------

_LANGUAGES_FILE_HEADER = """\
# MeetingGenie – Transcription Language List
# ----------------------------------------
# Each uncommented line enables that language in the dropdown menu.
# To hide a language, add '#' at the beginning of the line.
# This file is NEVER rewritten by the app, so your edits are safe.
#
# Tip: keep only the languages you actually use to shorten the list.
# 'Auto' lets MeetingGenie detect the language automatically (recommended).
# 'Multilingual' is an experimental mode for mixed-language recordings.

"""


def _build_languages_file_content() -> str:
    lines = [_LANGUAGES_FILE_HEADER]
    for name in ALL_LANGUAGES:
        lines.append(f"- {name}\n")
    return "".join(lines)


def load_languages(app_dir: str) -> dict:
    """Load the language filter from languages.yml.

    Creates the file with defaults if it doesn't exist.
    Returns the filtered languages dict (always includes 'Auto').
    """
    languages_file = os.path.join(app_dir, 'languages.yml')
    languages = dict(ALL_LANGUAGES)

    # Create default file if missing
    if not os.path.exists(languages_file):
        try:
            with open(languages_file, 'w', encoding='utf-8') as f:
                f.write(_build_languages_file_content())
        except Exception:
            pass  # Non-fatal

    # Load and filter
    try:
        with open(languages_file, 'r', encoding='utf-8') as f:
            lang_list = yaml.safe_load(f)
        if isinstance(lang_list, list) and lang_list:
            allowed = {str(x) for x in lang_list if x is not None}
            filtered = {k: v for k, v in ALL_LANGUAGES.items() if k in allowed}
            filtered.setdefault('Auto', 'auto')
            languages = filtered
    except Exception:
        pass  # Non-fatal

    return languages


# --- i18n setup -------------------------------------------------------

def setup_i18n(app_dir: str, config_locale: str = 'auto') -> str:
    """Initialize the i18n system. Returns the resolved locale string."""
    import i18n

    i18n.set('filename_format', '{locale}.{format}')
    i18n.load_path.append(os.path.join(app_dir, 'trans'))

    app_locale = config_locale

    if app_locale == 'auto':
        try:
            if platform.system() == 'Windows':
                app_locale = locale.getdefaultlocale()[0][0:2]
            elif platform.system() == "Darwin":
                import Foundation
                app_locale = Foundation.NSUserDefaults.standardUserDefaults().stringForKey_('AppleLocale')[0:2]
        except Exception:
            app_locale = 'en'

    i18n.set('fallback', 'en')

    try:
        i18n.set('locale', app_locale)
    except Exception:
        if app_locale != 'en':
            try:
                i18n.set('locale', 'en')
                app_locale = 'en'
            except Exception:
                raise SystemExit('Failed to load translations.')
        else:
            raise SystemExit('Failed to load translations.')

    return app_locale


# --- Thread count detection -------------------------------------------

def detect_thread_count() -> int:
    """Determine optimal number of threads for faster-whisper."""
    if platform.system() == 'Windows':
        import cpufeature
        return cpufeature.CPUFeature["num_physical_cores"]
    elif platform.system() == "Linux":
        return os.cpu_count() if os.cpu_count() is not None else 4
    elif platform.system() == "Darwin":
        from subprocess import check_output
        if platform.machine() == "arm64":
            cpu_count = int(check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu_max"]))
        elif platform.machine() == "x86_64":
            cpu_count = int(check_output(["sysctl", "-n", "hw.logicalcpu_max"]))
        else:
            raise Exception("Unsupported mac architecture")
        return int(cpu_count * 0.75)
    else:
        raise Exception('Platform not supported yet.')


# --- Helpers ----------------------------------------------------------

def version_higher(version1: str, version2: str, subversion_level: int = 99) -> int:
    """Compare two version strings.
    Returns 1 if version1 > version2, 2 if version2 > version1, 0 if equal.
    """
    v1 = version1.split('.')
    v2 = version2.split('.')
    elem_num = max(len(v1), len(v2))
    while len(v1) < elem_num:
        v1.append('0')
    while len(v2) < elem_num:
        v2.append('0')
    for i in range(elem_num):
        if int(v1[i]) > int(v2[i]):
            return 1
        elif int(v2[i]) > int(v1[i]):
            return 2
        if i >= subversion_level:
            break
    return 0


CUDA_ERROR_KEYWORDS = (
    'cuda', 'cublas', 'cudnn', 'cufft', 'device-side assert',
    'invalid device function', 'nccl', 'gpu driver',
    'compute capability', 'hip error',
)


def is_cuda_error_message(message: str) -> bool:
    if not message:
        return False
    if message.find('(device_cpu)') != -1:
        return False
    lower = message.lower()
    return any(kw in lower for kw in CUDA_ERROR_KEYWORDS)
