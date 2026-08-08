"""OCR backend selection.

``OCR_PROVIDER=auto`` (default) resolves to:

1. local Tesseract when it is genuinely installed and usable — cheapest;
2. the vision LLM when an image-capable model is configured — works on
   Streamlit Cloud with no system packages;
3. :class:`NullOCRProvider`, which explains the gap instead of failing silently.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from omnirag.config.settings import AppSettings, get_settings
from omnirag.providers.ocr.base import BaseOCRProvider, NullOCRProvider, OCRResult
from omnirag.providers.ocr.tesseract import TesseractOCRProvider
from omnirag.providers.ocr.vision_llm import VisionOCRProvider
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

_cache: Dict[str, BaseOCRProvider] = {}
_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("auto", "vision", "tesseract", "none")


def build_ocr_provider(settings: Optional[AppSettings] = None) -> BaseOCRProvider:
    resolved = settings or get_settings()
    cfg = resolved.ocr
    provider = (cfg.provider or "auto").lower()

    if provider == "none":
        return NullOCRProvider(languages=cfg.languages, min_confidence=cfg.min_confidence)

    if provider in ("tesseract", "local"):
        tesseract = TesseractOCRProvider(
            languages=cfg.languages,
            min_confidence=cfg.min_confidence,
            tesseract_cmd=cfg.tesseract_cmd,
        )
        if tesseract.is_available():
            return tesseract
        logger.warning("OCR_PROVIDER=tesseract but it is unavailable — trying vision OCR")
        return _vision_or_null(resolved)

    if provider == "vision":
        return _vision_or_null(resolved)

    # --- auto ---------------------------------------------------------- #
    tesseract = TesseractOCRProvider(
        languages=cfg.languages,
        min_confidence=cfg.min_confidence,
        tesseract_cmd=cfg.tesseract_cmd,
    )
    if tesseract.is_available():
        logger.info("OCR: using local Tesseract (%s)", cfg.languages)
        return tesseract
    return _vision_or_null(resolved)


def _vision_or_null(settings: AppSettings) -> BaseOCRProvider:
    if not settings.llm.is_configured:
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )
    try:
        from omnirag.providers.llm.factory import get_llm_provider

        llm = get_llm_provider(settings)
    except Exception as exc:
        logger.warning("Vision OCR unavailable: %s", exc)
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )

    provider = VisionOCRProvider(
        llm,
        languages=settings.ocr.languages,
        min_confidence=settings.ocr.min_confidence,
    )
    if not provider.is_available():
        logger.warning(
            "Configured model cannot read images — OCR of scanned pages is disabled"
        )
        return NullOCRProvider(
            languages=settings.ocr.languages, min_confidence=settings.ocr.min_confidence
        )
    return provider


def get_ocr_provider(settings: Optional[AppSettings] = None) -> BaseOCRProvider:
    resolved = settings or get_settings()
    cfg = resolved.ocr
    key = f"{cfg.provider}|{cfg.languages}|{resolved.llm.chain_label}"
    with _lock:
        provider = _cache.get(key)
        if provider is None:
            provider = build_ocr_provider(resolved)
            _cache[key] = provider
            logger.info("OCR provider ready: %s", provider.name)
        return provider


def reset_ocr_cache() -> None:
    with _lock:
        _cache.clear()


__all__ = [
    "BaseOCRProvider",
    "NullOCRProvider",
    "OCRResult",
    "SUPPORTED_PROVIDERS",
    "build_ocr_provider",
    "get_ocr_provider",
    "reset_ocr_cache",
]
