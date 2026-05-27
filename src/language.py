# src/language.py
"""Lightweight, offline language detection for resumes and job descriptions.

Uses `langdetect` (a pure-Python port of Google's language-detection library).
Chosen over translation-based approaches because résumés are sensitive
documents: we never send candidate data to an external translation API.
Detection is deterministic (fixed seed) to keep the evaluation harness
reproducible, matching the project's existing reproducibility goals.
"""
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
import config

# Make detection deterministic across runs (langdetect is non-deterministic by default).
DetectorFactory.seed = 0

# ISO 639-1 -> human-readable, for display in the API/dashboard.
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish",
}


def detect_language(text: str, default: str | None = None) -> str:
    """Return an ISO 639-1 code for `text`, or `default` if it can't be detected.

    Short or empty text is unreliable, so we fall back to the configured default
    rather than guessing.
    """
    default = default or config.DEFAULT_LANGUAGE
    if not text or len(text.strip()) < config.MIN_CHARS_FOR_DETECTION:
        return default
    try:
        return detect(text)
    except LangDetectException:
        return default


def detect_language_confident(text: str, default: str | None = None) -> tuple[str, float]:
    """Like detect_language but also returns the model's confidence in [0, 1]."""
    default = default or config.DEFAULT_LANGUAGE
    if not text or len(text.strip()) < config.MIN_CHARS_FOR_DETECTION:
        return default, 0.0
    try:
        top = detect_langs(text)[0]   # highest-probability candidate
        return top.lang, round(top.prob, 4)
    except (LangDetectException, IndexError):
        return default, 0.0


def language_name(code: str) -> str:
    """Human-readable name for an ISO code (falls back to the raw code)."""
    return LANGUAGE_NAMES.get(code, code)