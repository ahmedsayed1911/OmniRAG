"""Explicit provider-error classification.

The fallback router must never "catch Exception and try the next vendor" — that
would hide programming bugs, mask malformed requests, and burn quota on errors a
different vendor would reject identically. Instead every failure is classified
into one of :class:`FailureClass`, and only :attr:`FailureClass.RECOVERABLE`
crosses the provider boundary.

Recoverable (switch provider):
    rate limit / 429, quota exhausted, 5xx, timeout, transient network error,
    temporary model unavailability.

Not recoverable (surface to the user as-is):
    invalid API key, malformed request, unsupported model, safety refusal,
    capability mismatch, and any non-provider Python exception (bugs).
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from omnirag.core.exceptions import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderCapabilityError,
    ProviderError,
    ProviderPolicyError,
    ProviderPaymentRequiredError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


class FailureClass(str, Enum):
    RECOVERABLE = "recoverable"        # try the next provider
    AUTH = "auth"                      # bad key — fix configuration
    BAD_REQUEST = "bad_request"        # our payload/model name is wrong
    PAYMENT = "payment_required"       # route needs credits
    POLICY = "policy"                  # deliberate safety refusal
    CAPABILITY = "capability"          # model cannot do this (e.g. images)
    BUG = "bug"                        # not a provider error at all


#: Exception types that mean "this vendor is having a bad moment".
RECOVERABLE_TYPES: Tuple[type, ...] = (
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def classify(exc: BaseException) -> FailureClass:
    """Map an exception to its :class:`FailureClass`. Never raises."""
    if isinstance(exc, ProviderPolicyError):
        return FailureClass.POLICY
    if isinstance(exc, ProviderCapabilityError):
        return FailureClass.CAPABILITY
    if isinstance(exc, ProviderAuthError):
        return FailureClass.AUTH
    if isinstance(exc, ProviderPaymentRequiredError):
        return FailureClass.PAYMENT
    if isinstance(exc, ProviderBadRequestError):
        return FailureClass.BAD_REQUEST
    if isinstance(exc, AllProvidersFailedError):
        return FailureClass.BAD_REQUEST
    if isinstance(exc, RECOVERABLE_TYPES):
        return FailureClass.RECOVERABLE
    if isinstance(exc, ProviderError):
        # Generic provider error: trust the explicit `retryable` flag rather
        # than guessing from the message text.
        return FailureClass.RECOVERABLE if exc.retryable else FailureClass.BAD_REQUEST
    return FailureClass.BUG


def should_failover(exc: BaseException) -> bool:
    """True only when trying a different provider could plausibly succeed."""
    return classify(exc) is FailureClass.RECOVERABLE


def should_retry_same_provider(exc: BaseException) -> bool:
    """True when re-issuing the *same* request to the *same* provider may work.

    Quota exhaustion is deliberately excluded: the quota will not refill within
    a Streamlit request, so retrying only makes the UI hang.
    """
    if isinstance(exc, RateLimitError) and exc.quota_exhausted:
        return False
    return classify(exc) is FailureClass.RECOVERABLE


def describe(exc: BaseException) -> str:
    """Short, log-safe description (never includes payloads or keys)."""
    provider = getattr(exc, "provider", "") or "provider"
    return f"{provider}/{type(exc).__name__}[{classify(exc).value}]"
