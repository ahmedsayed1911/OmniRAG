"""OCR provider interface.

OCR is never allowed to abort ingestion: a failed page yields an empty
:class:`OCRResult` carrying the reason, and the document continues. Confidence
is always reported so downstream consumers (and the user) can see how much to
trust a recognised passage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from omnirag.core.enums import Language
from omnirag.utils.language import detect_language


@dataclass
class OCRResult:
    """Recognised text plus an honest confidence signal."""

    text: str = ""
    confidence: Optional[float] = None
    language: Language = Language.UNKNOWN
    engine: str = ""
    #: True when the engine itself flagged the reading as unreliable, or when
    #: the confidence fell below the configured threshold.
    uncertain: bool = False
    #: Populated only when OCR failed; ingestion turns it into a warning.
    error: Optional[str] = None
    #: Optional per-word/line confidences for future evaluation tooling.
    word_confidences: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None

    @classmethod
    def empty(cls, engine: str, error: Optional[str] = None) -> "OCRResult":
        return cls(engine=engine, error=error, uncertain=True)

    def finalize(self, *, min_confidence: float = 0.35) -> "OCRResult":
        """Fill in the detected language and the uncertainty flag."""
        if self.text and self.language == Language.UNKNOWN:
            self.language = detect_language(self.text)
        if self.confidence is not None and self.confidence < min_confidence:
            self.uncertain = True
        return self


class BaseOCRProvider(ABC):
    """Contract for every OCR backend."""

    name: str = "base"
    #: Whether the backend can read handwriting with any usefulness.
    supports_handwriting: bool = False
    #: Whether the backend supports Arabic script.
    supports_arabic: bool = True

    def __init__(self, *, languages: str = "ara+eng", min_confidence: float = 0.35):
        self.languages = languages
        self.min_confidence = min_confidence

    @abstractmethod
    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        """Extract text from an image. Must not raise — return an empty result."""

    def is_available(self) -> bool:
        """Whether this backend can actually run right now."""
        return True

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "languages": self.languages,
            "handwriting": self.supports_handwriting,
            "available": self.is_available(),
        }


class NullOCRProvider(BaseOCRProvider):
    """No-op backend used when OCR is disabled or nothing is configured.

    Returns an explanatory error so the UI can tell the user *why* a scanned
    page produced no text, instead of silently indexing an empty document.
    """

    name = "none"

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        return OCRResult.empty(
            self.name,
            error=(
                "No OCR provider is configured. Set an LLM API key to enable "
                "vision-based OCR, or install pytesseract for local OCR."
            ),
        )

    def is_available(self) -> bool:
        return False
