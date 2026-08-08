"""Shared HTTP plumbing for every external provider.

Centralises timeout handling, connection reuse, and — importantly — the mapping
from HTTP status codes to OmniRAG's typed exceptions, so the retry policy can
tell a transient 429/503 from a permanent 401.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from omnirag.core.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderTimeoutError,
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

    return _handle_response(response, provider=provider)


def _handle_response(response: Any, *, provider: str) -> Dict[str, Any]:
    status = response.status_code
    if status < 300:
        try:
            return response.json()
        except Exception as exc:
            raise ProviderError(
                f"{provider} returned a non-JSON body",
                provider=provider,
                user_message=f"The {provider} API returned an unexpected response.",
            ) from exc

    body = _safe_body(response)

    if status == 429:
        retry_after = response.headers.get("retry-after")
        raise RateLimitError(
            f"{provider} rate limited: {body}",
            provider=provider,
            retry_after=float(retry_after) if _is_number(retry_after) else None,
            quota_exhausted=is_quota_exhausted(body),
        )
    if status in (408, 504):
        raise ProviderTimeoutError(f"{provider} timeout ({status}): {body}", provider=provider)
    if status >= 500:
        raise ProviderUnavailableError(
            f"{provider} server error {status}: {body}",
            provider=provider,
            user_message=f"The {provider} service is temporarily unavailable ({status}).",
        )
    if status in (401, 403):
        raise ProviderAuthError(
            f"{provider} authentication failed ({status}): {body}",
            provider=provider,
            user_message=(
                f"The {provider} API rejected your credentials. "
                "Check the API key in your secrets."
            ),
        )
    if status == 404:
        raise ProviderBadRequestError(
            f"{provider} endpoint/model not found: {body}",
            provider=provider,
            user_message=(
                f"The {provider} model or endpoint was not found. "
                "Check the model name and base URL."
            ),
        )
    raise ProviderBadRequestError(
        f"{provider} request failed ({status}): {body}",
        provider=provider,
        user_message=f"The {provider} API rejected the request ({status}).",
    )


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
