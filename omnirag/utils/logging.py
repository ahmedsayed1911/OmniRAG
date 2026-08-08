"""Logging setup.

Rules enforced here:
* no ``print`` anywhere in the codebase — always ``get_logger(__name__)``;
* secrets are redacted by a filter before a record is emitted;
* document content is never logged at INFO (only lengths/counts/ids).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

_CONFIGURED = False
_LOGGER_NAME = "omnirag"

# Patterns that must never reach the log stream.
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(sk-ant-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
    re.compile(r"((?i:api[_-]?key)\"?\s*[:=]\s*\"?)([^\s\"',}]{6,})"),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{10,})"),
]


class RedactingFilter(logging.Filter):
    """Masks anything that looks like a credential in the formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - broken %-args
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def redact(text: str) -> str:
    """Replace credential-looking substrings with ``***``."""
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}***", out)
        else:
            out = pattern.sub("***", out)
    return out


def configure_logging(level: Optional[str] = None, *, force: bool = False) -> None:
    """Configure the ``omnirag`` logger tree exactly once."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, resolved, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)

    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a namespaced logger, configuring the tree on first use."""
    configure_logging()
    if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name.split('.')[-1]}")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Structured-ish one-line event log: ``event key=value key=value``."""
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
    logger.info("%s %s", event, parts)
