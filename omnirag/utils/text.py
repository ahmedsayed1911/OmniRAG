"""Text normalisation, tokenisation and estimation helpers.

Normalisation policy: clean up *artifacts of extraction* (hyphenation across
line breaks, control characters, runaway whitespace, repeated headers/footers)
but never rewrite factual content. Numbers, units and wording are untouched.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence

from omnirag.utils.language import normalize_for_search

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.!?:;،؛])\n(?![\n\-\*•\d])")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.I)
_BULLET = re.compile(r"^\s*[•●▪·\-\*]\s+")
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)

# Rough characters-per-token; Arabic tokenises denser than English so we use a
# conservative value to avoid under-estimating context size.
_CHARS_PER_TOKEN = 3.2


def clean_text(text: str, *, join_soft_breaks: bool = True) -> str:
    """Normalise extracted text without altering its meaning."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _CONTROL.sub("", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _HYPHEN_BREAK.sub(r"\1\2", out)          # de-hyphenate line wraps
    if join_soft_breaks:
        out = _SOFT_BREAK.sub(" ", out)            # rejoin broken sentences
    out = _MULTI_SPACE.sub(" ", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def strip_page_artifacts(text: str) -> str:
    """Drop stand-alone page-number lines left over from PDF extraction."""
    kept = [ln for ln in text.split("\n") if not _PAGE_NUMBER_LINE.match(ln)]
    return "\n".join(kept).strip()


def detect_repeated_lines(
    page_texts: Sequence[str], *, min_pages: int = 3, ratio: float = 0.6, edge_lines: int = 3
) -> set[str]:
    """Find running headers/footers shared by most pages.

    Only the first/last few lines of each page are considered, so a sentence
    that legitimately repeats mid-body is never removed.
    """
    if len(page_texts) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        candidates = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(candidates):
            if 3 <= len(line) <= 120:
                counter[line] += 1

    threshold = max(min_pages, int(len(page_texts) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def remove_lines(text: str, blocked: set[str]) -> str:
    if not blocked:
        return text
    kept = [ln for ln in text.split("\n") if ln.strip() not in blocked]
    return "\n".join(kept).strip()


def normalize_bullets(text: str) -> str:
    return "\n".join(_BULLET.sub("- ", ln) for ln in text.split("\n"))


def tokenize(text: str) -> List[str]:
    """Language-aware tokens for BM25 (Arabic normalised, lower-cased)."""
    return _TOKEN.findall(normalize_for_search(text))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for context budgeting (no tokenizer dep)."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > max_chars * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + suffix


def snippet(text: str, max_chars: int = 320) -> str:
    return truncate(" ".join(text.split()), max_chars)


def split_sentences(text: str) -> List[str]:
    """Sentence split supporting Arabic punctuation (؟ ، ؛) and ellipses."""
    parts = re.split(r"(?<=[.!?؟])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p and p.strip()]


def is_meaningful(text: str, *, min_chars: int = 12, min_tokens: int = 2) -> bool:
    """Whether a fragment is worth indexing (filters OCR noise and page chrome)."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    tokens = _TOKEN.findall(stripped)
    if len(tokens) < min_tokens:
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    return alnum / max(1, len(stripped)) >= 0.35


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = normalize_for_search(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
