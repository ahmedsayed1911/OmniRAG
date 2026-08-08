"""Domain exceptions.

Every exception carries a ``user_message`` that is safe to render in the UI, and
keeps the technical detail in ``str(exc)``/``detail`` for the logs. The UI never
prints a traceback to end users (see ``omnirag.ui.components.render_error``).
"""

from __future__ import annotations

from typing import Optional


class OmniRAGError(Exception):
    """Base class for all OmniRAG errors."""

    default_user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None) -> None:
        super().__init__(detail or user_message or self.default_user_message)
        self.detail = detail
        self.user_message = user_message or self.default_user_message


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(OmniRAGError):
    default_user_message = (
        "OmniRAG is not configured correctly. Check your API keys and settings."
    )


class MissingCredentialError(ConfigurationError):
    def __init__(self, variable: str, purpose: str = "") -> None:
        suffix = f" ({purpose})" if purpose else ""
        super().__init__(
            f"Missing required credential: {variable}{suffix}",
            user_message=(
                f"`{variable}` is not configured{suffix}. "
                "Add it to your environment or Streamlit secrets."
            ),
        )
        self.variable = variable


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class IngestionError(OmniRAGError):
    default_user_message = "This document could not be processed."


class UnsupportedFileTypeError(IngestionError):
    def __init__(self, filename: str, extension: str, supported: str = "") -> None:
        super().__init__(
            f"Unsupported extension '{extension}' for file '{filename}'",
            user_message=(
                f"**{filename}** is not a supported file type."
                + (f" Supported types: {supported}." if supported else "")
            ),
        )


class FileTooLargeError(IngestionError):
    def __init__(self, filename: str, size_mb: float, limit_mb: float) -> None:
        super().__init__(
            f"'{filename}' is {size_mb:.1f} MB, limit is {limit_mb:.1f} MB",
            user_message=(
                f"**{filename}** is {size_mb:.1f} MB which exceeds the "
                f"{limit_mb:.0f} MB upload limit."
            ),
        )


class EmptyDocumentError(IngestionError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"'{filename}' produced no extractable content",
            user_message=(
                f"No readable content was found in **{filename}**. "
                "If it is a scan, enable an OCR or vision provider."
            ),
        )


class CorruptedDocumentError(IngestionError):
    def __init__(self, filename: str, detail: str = "") -> None:
        super().__init__(
            f"'{filename}' could not be opened: {detail}",
            user_message=f"**{filename}** appears to be corrupted or password protected.",
        )


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class ProviderError(OmniRAGError):
    """Base class for failures coming from an external provider."""

    default_user_message = "An external service failed. Please try again."

    def __init__(
        self,
        detail: str = "",
        *,
        user_message: Optional[str] = None,
        provider: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(detail, user_message=user_message)
        self.provider = provider
        self.retryable = retryable


class ProviderUnavailableError(ProviderError):
    """Provider-side outage: 5xx, connection failure, model temporarily down."""

    default_user_message = "The selected AI provider is unavailable right now."

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None,
                 provider: str = "", retryable: bool = True) -> None:
        super().__init__(detail, user_message=user_message, provider=provider, retryable=retryable)


class RateLimitError(ProviderError):
    """429 / quota exhaustion.

    ``quota_exhausted`` marks the *billing* kind (Gemini ``RESOURCE_EXHAUSTED``,
    OpenAI ``insufficient_quota``). Waiting will not help, so the retry policy
    skips backoff and the router switches provider immediately.
    """

    default_user_message = "Rate limit reached. Waiting before retrying."

    def __init__(
        self,
        detail: str = "",
        *,
        provider: str = "",
        retry_after: float | None = None,
        quota_exhausted: bool = False,
    ):
        super().__init__(detail, provider=provider, retryable=not quota_exhausted)
        self.retry_after = retry_after
        self.quota_exhausted = quota_exhausted


class ProviderAuthError(ProviderError):
    """401/403 — a bad or missing key. Never retried, never falls back."""

    default_user_message = "The AI provider rejected the configured API key."

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None, provider: str = ""):
        super().__init__(detail, user_message=user_message, provider=provider, retryable=False)


class ProviderBadRequestError(ProviderError):
    """400/404/422 — our request or configuration is wrong. Not retryable."""

    default_user_message = "The request sent to the AI provider was rejected as invalid."

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None, provider: str = ""):
        super().__init__(detail, user_message=user_message, provider=provider, retryable=False)


class ProviderPolicyError(ProviderError):
    """The model refused on safety/policy grounds.

    A deliberate, deterministic decision — retrying or switching vendor is not a
    fix, so this never triggers fallback.
    """

    default_user_message = (
        "The model declined to answer this request on safety grounds."
    )

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None,
                 provider: str = "", reason: str = "") -> None:
        super().__init__(detail, user_message=user_message, provider=provider, retryable=False)
        self.reason = reason


class ProviderCapabilityError(ProviderError):
    """The chosen model cannot do what the request needs (e.g. no image input).

    Raised instead of silently dropping visual evidence from a multimodal
    request — a silent drop would produce a confidently wrong answer.
    """

    default_user_message = (
        "The configured model cannot process images, so visual evidence "
        "could not be analysed."
    )

    def __init__(self, detail: str = "", *, user_message: Optional[str] = None,
                 provider: str = "", capability: str = "") -> None:
        super().__init__(detail, user_message=user_message, provider=provider, retryable=False)
        self.capability = capability


class AllProvidersFailedError(ProviderError):
    """Every provider in the chain failed. Carries the per-provider reasons."""

    default_user_message = "All configured AI providers failed to respond."

    def __init__(self, failures: "list[tuple[str, BaseException]]" | None = None) -> None:
        self.failures = failures or []
        detail = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in self.failures
        )
        first_user_message = next(
            (
                getattr(exc, "user_message", "")
                for _, exc in self.failures
                if getattr(exc, "user_message", "")
            ),
            "",
        )
        super().__init__(
            detail or "no providers attempted",
            user_message=(
                f"All configured AI providers failed. {first_user_message}".strip()
                if first_user_message
                else self.default_user_message
            ),
            retryable=False,
        )


class ProviderTimeoutError(ProviderError):
    default_user_message = "The AI provider took too long to respond."

    def __init__(self, detail: str = "", *, provider: str = "") -> None:
        super().__init__(detail, provider=provider, retryable=True)


class LLMError(ProviderError):
    default_user_message = "The language model could not generate an answer."


class EmbeddingError(ProviderError):
    default_user_message = "Embeddings could not be created for this content."


class OCRError(ProviderError):
    default_user_message = "Text recognition failed for this page."


class VisionError(ProviderError):
    default_user_message = "Visual analysis failed for this element."


class RerankError(ProviderError):
    default_user_message = "Reranking failed; falling back to retrieval order."


# --------------------------------------------------------------------------- #
# Storage / retrieval
# --------------------------------------------------------------------------- #
class VectorStoreError(OmniRAGError):
    default_user_message = "The vector database is unavailable."


class SessionIsolationError(OmniRAGError):
    """Raised when an operation would cross session boundaries.

    This is a hard invariant: it is never caught and downgraded to a warning.
    """

    default_user_message = "A data-isolation check failed; the request was blocked."


class RetrievalError(OmniRAGError):
    default_user_message = "Search over your documents failed."
