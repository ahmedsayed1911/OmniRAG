"""Handwriting extraction.

Deliberately honest module. Handwriting recognition — especially Arabic
handwriting, and especially mixed Arabic/English notes — is the least reliable
capability in this system. The design choices reflect that:

* every handwriting result is stored with a confidence and an ``uncertain``
  flag, which the UI surfaces on the source card;
* unclear words are preserved as ``[?]`` markers rather than guessed;
* the answer prompt is told to treat handwriting as low-confidence evidence;
* the original image is always kept, so the user can verify the transcription
  themselves from the source preview.

Detection is heuristic: printed-text OCR that comes back with low confidence, or
a vision classification of ``handwriting``, routes the image through a
handwriting-specialised prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omnirag.core.enums import BlockType, Language, SourceKind
from omnirag.intelligence.ocr import OCREngine
from omnirag.intelligence.vision import VisionAnalyzer, VisualAnalysis
from omnirag.utils.language import detect_language
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

#: Below this OCR confidence on a text-bearing image we suspect handwriting and
#: re-try with the handwriting-aware prompt.
HANDWRITING_SUSPICION_THRESHOLD = 0.55
#: Transcriptions with more than this share of `[?]` markers are flagged.
UNREADABLE_MARKER_RATIO = 0.25


@dataclass
class HandwritingResult:
    text: str = ""
    confidence: Optional[float] = None
    language: Language = Language.UNKNOWN
    uncertain: bool = True
    engine: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None

    @property
    def block_type(self) -> BlockType:
        return BlockType.HANDWRITING

    @property
    def source_kind(self) -> SourceKind:
        return SourceKind.OCR


class HandwritingExtractor:
    """Best-effort handwriting reader built on the OCR and vision engines."""

    def __init__(
        self,
        ocr: Optional[OCREngine] = None,
        vision: Optional[VisionAnalyzer] = None,
    ):
        self.ocr = ocr
        self.vision = vision

    @property
    def available(self) -> bool:
        if self.vision is not None and self.vision.available:
            return True
        return bool(
            self.ocr is not None
            and self.ocr.available
            and self.ocr.provider.supports_handwriting
        )

    # ------------------------------------------------------------------ #
    def looks_handwritten(self, analysis: Optional[VisualAnalysis], ocr_confidence: Optional[float]) -> bool:
        """Heuristic: trust an explicit vision classification, else low OCR confidence."""
        if analysis is not None and analysis.block_type == BlockType.HANDWRITING:
            return True
        if ocr_confidence is not None and ocr_confidence < HANDWRITING_SUSPICION_THRESHOLD:
            return True
        return False

    def extract(self, image: bytes) -> HandwritingResult:
        """Transcribe handwriting from an image. Never raises."""
        if not image:
            return HandwritingResult(error="empty image")

        # The vision path handles cursive and mixed scripts far better than a
        # classical OCR engine, so it is tried first when available.
        if self.ocr is not None and self.ocr.available and self.ocr.provider.supports_handwriting:
            result = self.ocr.recognize(image, hint="handwriting")
            if result.ok:
                return _from_ocr(result)
            error = result.error
        else:
            error = "no handwriting-capable OCR provider configured"

        if self.vision is not None and self.vision.available:
            analysis = self.vision.analyze(image, expect=BlockType.HANDWRITING)
            if analysis.ok:
                text = analysis.text or analysis.description
                return HandwritingResult(
                    text=text,
                    confidence=analysis.confidence,
                    language=detect_language(text),
                    uncertain=True,  # handwriting is never treated as certain
                    engine="vision",
                )
            error = analysis.error or error

        return HandwritingResult(error=error or "handwriting extraction unavailable")


def _from_ocr(result) -> HandwritingResult:
    text = result.text
    uncertain = True  # handwriting is always flagged, regardless of self-report
    if _unreadable_ratio(text) > UNREADABLE_MARKER_RATIO:
        logger.info("Handwriting transcription is mostly unreadable — flagging")
    return HandwritingResult(
        text=text,
        confidence=result.confidence,
        language=result.language if result.language != Language.UNKNOWN else detect_language(text),
        uncertain=uncertain,
        engine=result.engine,
    )


def _unreadable_ratio(text: str) -> float:
    if not text:
        return 1.0
    words = text.split()
    if not words:
        return 1.0
    return sum(1 for w in words if "[?]" in w) / len(words)


__all__ = ["HandwritingExtractor", "HandwritingResult"]
