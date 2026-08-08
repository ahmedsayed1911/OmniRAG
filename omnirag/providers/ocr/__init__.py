"""OCR provider adapters."""

from omnirag.providers.ocr.base import BaseOCRProvider, NullOCRProvider, OCRResult
from omnirag.providers.ocr.factory import (
    SUPPORTED_PROVIDERS,
    build_ocr_provider,
    get_ocr_provider,
    reset_ocr_cache,
)
from omnirag.providers.ocr.tesseract import TesseractOCRProvider
from omnirag.providers.ocr.vision_llm import VisionOCRProvider

__all__ = [
    "BaseOCRProvider",
    "NullOCRProvider",
    "OCRResult",
    "SUPPORTED_PROVIDERS",
    "TesseractOCRProvider",
    "VisionOCRProvider",
    "build_ocr_provider",
    "get_ocr_provider",
    "reset_ocr_cache",
]
