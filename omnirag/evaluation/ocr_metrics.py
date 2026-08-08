"""OCR accuracy metrics (CER / WER).

Requires ground-truth text, so this is an offline tool rather than something the
app runs on its own. Levenshtein distance is implemented here (two-row dynamic
programming) to avoid a dependency for ~20 lines of code.

Arabic note: comparison can optionally normalise diacritics and alef/yeh
variants, because a transcription that differs only in an optional diacritic is
usually correct for retrieval purposes — but the strict figure is reported too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from omnirag.utils.language import normalize_arabic


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (characters or words)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (item_a != item_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str, *, normalize: bool = False) -> float:
    """CER = edit distance / reference length. 0.0 is perfect."""
    ref = _prepare(reference, normalize)
    hyp = _prepare(hypothesis, normalize)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str, *, normalize: bool = False) -> float:
    """WER = word-level edit distance / reference word count."""
    ref = _prepare(reference, normalize).split()
    hyp = _prepare(hypothesis, normalize).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _prepare(text: str, normalize: bool) -> str:
    cleaned = " ".join((text or "").split())
    return normalize_arabic(cleaned) if normalize else cleaned


@dataclass
class OCRMetrics:
    samples: int = 0
    cer: float = 0.0
    wer: float = 0.0
    cer_normalized: float = 0.0
    wer_normalized: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "samples": self.samples,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "cer_normalized": round(self.cer_normalized, 4),
            "wer_normalized": round(self.wer_normalized, 4),
        }


def evaluate_ocr(pairs: Sequence[tuple[str, str]]) -> OCRMetrics:
    """Aggregate CER/WER over ``(reference, hypothesis)`` pairs."""
    if not pairs:
        return OCRMetrics()

    metrics = OCRMetrics(samples=len(pairs))
    metrics.cer = sum(character_error_rate(r, h) for r, h in pairs) / len(pairs)
    metrics.wer = sum(word_error_rate(r, h) for r, h in pairs) / len(pairs)
    metrics.cer_normalized = sum(
        character_error_rate(r, h, normalize=True) for r, h in pairs
    ) / len(pairs)
    metrics.wer_normalized = sum(
        word_error_rate(r, h, normalize=True) for r, h in pairs
    ) / len(pairs)
    return metrics


__all__ = [
    "OCRMetrics",
    "character_error_rate",
    "evaluate_ocr",
    "levenshtein",
    "word_error_rate",
]
