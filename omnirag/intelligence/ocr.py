"""OCR orchestration.

Thin engine on top of the OCR provider abstraction that adds the policies the
providers themselves should not own:

* **content-hash caching** — the same page image is never OCR'd twice;
* **confidence policy** — results below the threshold are marked uncertain
  rather than silently trusted;
* **failure containment** — a failed page returns an empty result with a reason,
  never an exception.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from omnirag.core.enums import Language
from omnirag.providers.ocr.base import BaseOCRProvider, OCRResult
from omnirag.utils.hashing import short_hash
from omnirag.utils.logging import get_logger
from omnirag.utils.text import clean_text, is_meaningful

logger = get_logger(__name__)


@dataclass
class OCRStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    low_confidence: int = 0


class OCREngine:
    """Caching, policy-aware wrapper around a :class:`BaseOCRProvider`."""

    def __init__(self, provider: BaseOCRProvider, *, min_confidence: float = 0.35):
        self.provider = provider
        self.min_confidence = min_confidence
        self._cache: Dict[str, OCRResult] = {}
        self._lock = threading.Lock()
        self.stats = OCRStats()

    @property
    def available(self) -> bool:
        return self.provider.is_available()

    @property
    def name(self) -> str:
        return self.provider.name

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        """OCR an image, using the cache when the exact bytes were seen before."""
        if not image:
            return OCRResult.empty(self.provider.name, error="empty image")

        key = f"{short_hash(image, 24)}|{hint}"
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        self.stats.calls += 1
        result = self.provider.recognize(image, hint=hint)

        # Normalise the recognised text (OCR output is full of broken lines).
        if result.text:
            result.text = clean_text(result.text)
            if not is_meaningful(result.text, min_chars=3, min_tokens=1):
                result.text = ""
                result.error = result.error or "no meaningful text detected"

        result.finalize(min_confidence=self.min_confidence)

        if not result.ok:
            self.stats.failures += 1
        elif result.uncertain:
            self.stats.low_confidence += 1

        with self._lock:
            self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


def build_ocr_engine(settings=None) -> OCREngine:
    from omnirag.config.settings import get_settings
    from omnirag.providers.ocr.factory import get_ocr_provider

    resolved = settings or get_settings()
    return OCREngine(
        get_ocr_provider(resolved), min_confidence=resolved.ocr.min_confidence
    )


__all__ = ["OCREngine", "OCRResult", "OCRStats", "Language", "build_ocr_engine"]
