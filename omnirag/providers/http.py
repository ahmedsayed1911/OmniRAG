"""Shared HTTP plumbing for every external provider.

Centralises timeout handling, connection reuse, and — importantly — the mapping
from HTTP status codes to OmniRAG's typed exceptions, so the retry policy can
tell a transient 429/503 from a permanent 401.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, Optional

from omnirag.core.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderPaymentRequiredError,
    ProviderTimeoutError,
    ProviderTokenBudgetExceededError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard
    import httpx

    HTTPX_AVAILABLE = True
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

_clients: Dict[str, Any] = {}
_lock = threading.Lock()


def get_client(timeout_s: float = 60.0) -> "httpx.Client":
    """Pooled ``httpx.Client`` keyed by timeout (connection reuse across calls)."""
    if not HTTPX_AVAILABLE:
        raise ProviderUnavailableError(
            "httpx is not installed",
            user_message="The HTTP client library is missing. Run `pip install -r requirements.txt`.",
        )
    key = f"t{timeout_s:.0f}"
    with _lock:
        client = _clients.get(key)
        if client is None or client.is_closed:
            client = httpx.Client(
                timeout=httpx.Timeout(timeout_s, connect=min(15.0, timeout_s)),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                follow_redirects=True,
            )
            _clients[key] = client
        return client


def close_clients() -> None:
    with _lock:
        for client in _clients.values():
            try:
                client.close()
            except Exception:  # pragma: no cover
                pass
        _clients.clear()


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 60.0,
    provider: str = "provider",
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST JSON and return the decoded body, raising typed errors on failure."""
    client = get_client(timeout_s)
    try:
        response = client.post(url, json=payload, headers=headers or {})
    except httpx.TimeoutException as exc:  # type: ignore[union-attr]
        raise ProviderTimeoutError(f"{provider} timed out: {exc}", provider=provider) from exc
    except httpx.HTTPError as exc:  # type: ignore[union-attr]
        raise ProviderUnavailableError(
            f"{provider} connection failed: {exc}",
            provider=provider,
            retryable=True,
            user_message=f"Could not reach the {provider} API. Check your network and base URL.",
        ) from exc

    if diagnostics is not None:
        diagnostics.update(
            {
                "http_status": int(response.status_code),
                "content_length": response.headers.get("content-length", ""),
                "response_fully_received": True,
            }
        )
    body = _handle_response(response, provider=provider)
    if diagnostics is not None:
        diagnostics["json_parsed"] = True
    return body


def _handle_response(response: Any, *, provider: str) -> Dict[str, Any]:
    status = response.status_code
    if status < 300:
        try:
            return response.json()
        except Exception as exc:
            error = ProviderError(
                f"{provider} returned a non-JSON body",
                provider=provider,
                user_message=f"The {provider} API returned an unexpected response.",
            )
            raise attach_http_context(error, status, "<non-JSON response>") from exc

    body = _safe_body(response)

    if status == 429:
        retry_after = _retry_after_seconds(response.headers.get("retry-after"), body)
        remaining = response.headers.get("x-ratelimit-remaining", "")
        reset = response.headers.get("x-ratelimit-reset", "")
        exhausted = is_quota_exhausted(body) or str(remaining).strip() == "0"
        quota_scope = _quota_scope(body, exhausted=exhausted, reset=reset)
        raise attach_http_context(
            RateLimitError(
                f"{provider} rate limited: {body}",
                provider=provider,
                retry_after=retry_after,
                quota_exhausted=exhausted,
                quota_scope=quota_scope,
                reset_at=str(reset or ""),
            ),
            status,
            body,
            rate_limit_remaining=str(remaining or ""),
            rate_limit_reset=str(reset or ""),
        )
    if status in (408, 504):
        raise attach_http_context(
            ProviderTimeoutError(f"{provider} timeout ({status}): {body}", provider=provider),
            status,
            body,
        )
    if status == 413 and _is_token_budget_error(body):
        raise attach_http_context(
            ProviderTokenBudgetExceededError(
                f"{provider} token budget exceeded: {body}",
                provider=provider,
            ),
            status,
            body,
        )
    if status >= 500:
        raise attach_http_context(ProviderUnavailableError(
            f"{provider} server error {status}: {body}",
            provider=provider,
            user_message=f"The {provider} service is temporarily unavailable ({status}).",
        ), status, body)
    if status in (401, 403):
        raise attach_http_context(ProviderAuthError(
            f"{provider} authentication failed ({status}): {body}",
            provider=provider,
            user_message=(
                f"The {provider} API rejected your credentials. "
                "Check the API key in your secrets."
            ),
        ), status, body)
    if status == 402:
        raise attach_http_context(ProviderPaymentRequiredError(
            f"{provider} payment required: {body}",
            provider=provider,
            user_message=(
                "The configured OpenRouter route requires credits. "
                "Use `openrouter/free` or add credits to the OpenRouter account."
                if provider == "openrouter"
                else f"The {provider} API requires credits for this request."
            ),
        ), status, body)
    if status == 404:
        raise attach_http_context(ProviderBadRequestError(
            f"{provider} endpoint/model not found: {body}",
            provider=provider,
            user_message=(
                f"The {provider} model or endpoint was not found. "
                "Check the model name and base URL."
            ),
        ), status, body)
    raise attach_http_context(ProviderBadRequestError(
        f"{provider} request failed ({status}): {body}",
        provider=provider,
        user_message=f"The {provider} API rejected the request ({status}).",
    ), status, body)


def attach_http_context(
    exc: Exception,
    status: int,
    body: str,
    *,
    rate_limit_remaining: str = "",
    rate_limit_reset: str = "",
) -> Exception:
    """Attach bounded, credential-redacted-at-log-time HTTP diagnostics."""
    setattr(exc, "status_code", int(status))
    setattr(exc, "safe_body", " ".join((body or "").split())[:400])
    setattr(exc, "rate_limit_remaining", rate_limit_remaining[:40])
    setattr(exc, "rate_limit_reset", rate_limit_reset[:80])
    return exc


# Vendor-specific markers for "you are out of credit/quota" as opposed to
# "you are going too fast". Both arrive as HTTP 429.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "resource has been exhausted",
    "insufficient_quota",
    "quota exceeded",
    "exceeded your current quota",
    "out of credits",
    "credit balance",
    "billing",
)


def is_quota_exhausted(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _QUOTA_MARKERS)


def _retry_after_seconds(header: Optional[str], body: str) -> Optional[float]:
    if _is_number(header):
        return max(0.0, float(header))  # type: ignore[arg-type]
    match = re.search(
        r"(?:try again in|retry(?:ing)?(?: after| in)?)\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        body or "",
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _quota_scope(body: str, *, exhausted: bool, reset: str) -> str:
    lowered = (body or "").lower()
    if "tokens per minute" in lowered or " tpm" in lowered:
        return "minute_tpm"
    if "requests per minute" in lowered or " rpm" in lowered:
        return "minute_rpm"
    if reset and exhausted:
        return "daily_or_account"
    if exhausted:
        return "hard_quota"
    return "temporary_rate"


def _is_token_budget_error(body: str) -> bool:
    lowered = (body or "").lower()
    markers = (
        "type=tokens",
        '"type":"tokens"',
        '"type": "tokens"',
        "rate_limit_exceeded",
        "tpm limit",
        "token limit",
        "request too large",
        "requested tokens",
    )
    return any(marker in lowered for marker in markers)


def _safe_body(response: Any, limit: int = 400) -> str:
    try:
        return response.text[:limit]
    except Exception:  # pragma: no cover
        return "<unreadable body>"


def _is_number(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False
