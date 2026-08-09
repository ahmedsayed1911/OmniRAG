"""User-facing privacy boundary for technical diagnostics.

Backend messages retain provider details for logs and observability. These
helpers make sure normal UI rendering never repeats those details.
"""

from __future__ import annotations

import re
from typing import Optional

from omnirag.core.exceptions import ProviderCapabilityError, ProviderPolicyError

SERVICE_UNAVAILABLE = "The AI service is temporarily unavailable. Please try again shortly."
SERVICE_NOT_CONFIGURED = (
    "The AI service is not configured. Please contact the app administrator."
)

_CONFIG_MARKERS = (
    "api_key", "api key", "credential", "not configured", "streamlit secrets",
)
_PROVIDER_MARKERS = (
    "gemini", "groq", "openrouter", "provider", "fallback", "failover", "endpoint",
    "rate limit", "rate-limit", "quota", "credits", "payment required",
    "embedding", "reranker", "vector store", "qdrant", "faiss", "chroma",
    "http 4", "http 5", " 402", " 429", " 500", " 502", " 503", " 504",
)
_EXCEPTION_NAME = re.compile(r"\b[A-Z][A-Za-z]+(?:Error|Exception)\b")
_RAW_ERROR_BODY = re.compile(r"[\{\[]\s*[\"']?(?:error|status|code)[\"']?\s*:", re.I)
_HTTP_STATUS = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?)[\s:=_-]*[45]\d{2}\b",
    re.I,
)


def provider_error_message(exc: BaseException, *, debug: bool = False) -> str:
    """Return a neutral provider failure unless diagnostics were requested."""
    if debug:
        return str(getattr(exc, "user_message", "") or exc)
    if isinstance(exc, ProviderPolicyError):
        return "This request could not be completed because of safety restrictions."
    if isinstance(exc, ProviderCapabilityError):
        return (
            "The AI service cannot process the required visual evidence right now. "
            "Please try again shortly."
        )
    return SERVICE_UNAVAILABLE


def public_error_text(message: str, *, debug: bool = False) -> str:
    """Sanitize technical errors already present in UI session state."""
    text = str(message or "").strip()
    if debug or not text:
        return text
    lowered = text.lower()
    if any(marker in lowered for marker in _CONFIG_MARKERS):
        return SERVICE_NOT_CONFIGURED
    if (
        any(marker in lowered for marker in _PROVIDER_MARKERS)
        or _EXCEPTION_NAME.search(text)
        or _RAW_ERROR_BODY.search(text)
        or _HTTP_STATUS.search(text)
    ):
        return SERVICE_UNAVAILABLE
    return text


def public_generation_warning(message: str, *, debug: bool = False) -> Optional[str]:
    """Keep actionable answer notes while suppressing implementation details."""
    text = str(message or "").strip()
    if debug:
        return text or None
    lowered = text.lower()
    if "citation" in lowered or "source" in lowered:
        if "did not cite" in lowered:
            return "This answer did not include source references. Please verify it carefully."
        return "Some invalid source references were removed from the answer."
    if "output limit" in lowered or "continued" in lowered:
        return "The answer reached its length limit; the available response is shown."
    if any(word in lowered for word in ("visual", "image", "chart", "diagram")):
        return "Some visual evidence could not be analysed for this answer."
    return None


def public_processing_note(message: str, *, debug: bool = False) -> str:
    """Sanitize ingestion notes while retaining useful degradation context."""
    text = str(message or "").strip()
    if debug or not text:
        return text
    lowered = text.lower()
    if (
        any(marker in lowered for marker in _PROVIDER_MARKERS + _CONFIG_MARKERS)
        or "embedding" in lowered
        or "image-capable model" in lowered
        or _EXCEPTION_NAME.search(text)
        or _RAW_ERROR_BODY.search(text)
    ):
        return (
            "Some advanced document analysis was unavailable. "
            "The remaining document content is still usable."
        )
    return text


__all__ = [
    "SERVICE_NOT_CONFIGURED",
    "SERVICE_UNAVAILABLE",
    "provider_error_message",
    "public_error_text",
    "public_generation_warning",
    "public_processing_note",
]
