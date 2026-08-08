"""Local Tesseract OCR (optional).

Not installed by default: it needs the ``tesseract`` binary plus the ``ara``/
``eng`` language data, which on Streamlit Community Cloud means adding a
``packages.txt``. When present it is faster and cheaper than the vision API for
clean printed scans, so it is preferred for bulk pages; handwriting still goes
to the vision provider.

Enable with::

    pip install pytesseract
    # packages.txt (Streamlit Cloud):
    #   tesseract-ocr
    #   tesseract-ocr-ara
    #   tesseract-ocr-eng
    OCR_PROVIDER=tesseract
"""

from __future__ import annotations

import io
from typing import Optional

from omnirag.providers.ocr.base import BaseOCRProvider, OCRResult
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import pytesseract
    from PIL import Image

    PYTESSERACT_AVAILABLE = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    PYTESSERACT_AVAILABLE = False


class TesseractOCRProvider(BaseOCRProvider):
    name = "tesseract"
    # Tesseract's handwriting accuracy is poor; we never claim otherwise.
    supports_handwriting = False
    supports_arabic = True

    def __init__(
        self,
        *,
        languages: str = "ara+eng",
        min_confidence: float = 0.35,
        tesseract_cmd: str = "",
        psm: int = 3,
    ):
        super().__init__(languages=languages, min_confidence=min_confidence)
        self.psm = psm
        if tesseract_cmd and PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not PYTESSERACT_AVAILABLE:
            self._available = False
            return False
        try:
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as exc:  # binary missing or not on PATH
            logger.info("Tesseract binary unavailable: %s", exc)
            self._available = False
        return self._available

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        if not image:
            return OCRResult.empty(self.name, error="empty image")
        if not self.is_available():
            return OCRResult.empty(
                self.name,
                error="Tesseract is not installed on this host.",
            )

        try:
            pil_image = Image.open(io.BytesIO(image))
            pil_image.load()
            config = f"--psm {self.psm}"
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.languages,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001 - never abort ingestion
            logger.warning("Tesseract OCR failed: %s", exc)
            return OCRResult.empty(self.name, error=f"OCR failed: {type(exc).__name__}")

        words, confidences = [], []
        for text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            token = (text or "").strip()
            if not token:
                continue
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < 0:
                continue
            words.append(token)
            confidences.append(confidence / 100.0)

        if not words:
            return OCRResult.empty(self.name, error="no text detected")

        mean_confidence = sum(confidences) / len(confidences)
        return OCRResult(
            text=" ".join(words),
            confidence=mean_confidence,
            engine=self.name,
            word_confidences=confidences,
        ).finalize(min_confidence=self.min_confidence)
