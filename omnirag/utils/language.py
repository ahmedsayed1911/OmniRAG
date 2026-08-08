"""Language detection and Arabic-aware text utilities.

Deterministic Unicode-range counting rather than a probabilistic detector: it is
faster, dependency-free, and entirely sufficient for the Arabic/English/mixed
buckets OmniRAG needs. No LLM call is made for something parsing can decide.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from omnirag.core.enums import Language

# Arabic, Arabic Supplement, Extended-A, Presentation Forms A/B.
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_MIXED_THRESHOLD = 0.20


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def script_ratios(text: str) -> tuple[float, float]:
    """Return ``(arabic_ratio, latin_ratio)`` over alphabetic characters only."""
    arabic = latin = 0
    for ch in text:
        if is_arabic_char(ch):
            arabic += 1
        elif _LATIN_RE.match(ch):
            latin += 1
    total = arabic + latin
    if total == 0:
        return 0.0, 0.0
    return arabic / total, latin / total


def detect_language(text: str) -> Language:
    """Classify text into Arabic / English / mixed / unknown."""
    if not text or not text.strip():
        return Language.UNKNOWN
    arabic, latin = script_ratios(text)
    if arabic == 0.0 and latin == 0.0:
        return Language.UNKNOWN
    if arabic >= _MIXED_THRESHOLD and latin >= _MIXED_THRESHOLD:
        return Language.MIXED
    return Language.ARABIC if arabic > latin else Language.ENGLISH


def detect_languages(texts: Iterable[str]) -> Language:
    """Aggregate detection over many strings (document-level language)."""
    joined = " ".join(t for t in texts if t)[:20000]
    return detect_language(joined)


def normalize_arabic(text: str) -> str:
    """Light Arabic normalisation for *search keys only*.

    Removes diacritics/tatweel and unifies alef/yeh/teh-marbuta variants. This is
    applied to keyword-index tokens and queries, never to stored source text —
    displayed content keeps full fidelity.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _ARABIC_DIACRITICS.sub("", out)
    out = out.replace(_TATWEEL, "")
    out = re.sub("[آأإٱ]", "ا", out)  # آ أ إ ٱ -> ا
    out = out.replace("ى", "ي")                       # ى -> ي
    out = out.replace("ة", "ه")                       # ة -> ه
    out = out.replace("ؤ", "و").replace("ئ", "ي")
    return out


def normalize_for_search(text: str) -> str:
    """Full normalisation used by the keyword index and query pipeline."""
    return re.sub(r"\s+", " ", normalize_arabic(text).lower()).strip()


def contains_arabic(text: str) -> bool:
    return any(is_arabic_char(ch) for ch in text)


def language_name(language: Language) -> str:
    return {
        Language.ARABIC: "Arabic",
        Language.ENGLISH: "English",
        Language.MIXED: "mixed Arabic/English",
        Language.UNKNOWN: "the user's language",
    }[language]


def is_rtl(text: str) -> bool:
    """Whether a string should be rendered right-to-left."""
    arabic, latin = script_ratios(text)
    return arabic > latin
