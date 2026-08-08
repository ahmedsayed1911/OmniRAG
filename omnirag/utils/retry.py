"""Conservative retry with exponential backoff and jitter.

Only *transient* failures are retried (rate limits, timeouts, 5xx). Permanent
validation errors (400/401/403/404) fail immediately — retrying them just burns
quota and delays the user-visible error.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, Iterable, Type, TypeVar

from omnirag.core.exceptions import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_RETRYABLE: tuple[Type[BaseException], ...] = (
    RateLimitError,
    ProviderTimeoutError,
)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, DEFAULT_RETRYABLE):
        return True
    if isinstance(exc, ProviderError):
        return exc.retryable
    return False


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter (attempt is 1-based)."""
    raw = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(base * 0.5, raw)


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 8.0,
    retry_on: Iterable[Type[BaseException]] | None = None,
    operation: str = "provider call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` retrying transient failures.

    ``sleep`` is injectable so tests do not spend real time.
    """
    extra = tuple(retry_on or ())
    last: BaseException | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            retryable = is_retryable(exc) or isinstance(exc, extra)
            if not retryable or attempt >= attempts:
                raise
            last = exc
            delay = getattr(exc, "retry_after", None) or backoff_delay(
                attempt, base_delay, max_delay
            )
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                operation,
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            sleep(float(delay))

    assert last is not None  # pragma: no cover - unreachable
    raise last


def with_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 8.0,
    operation: str = "",
):
    """Decorator form of :func:`retry_call`."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_call(
                lambda: func(*args, **kwargs),
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                operation=operation or func.__name__,
            )

        return wrapper

    return decorator
