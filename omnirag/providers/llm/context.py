"""Operation labels for safe provider-chain diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_operation: ContextVar[str] = ContextVar("omnirag_llm_operation", default="unspecified")


def current_llm_operation() -> str:
    return _operation.get()


@contextmanager
def llm_operation(name: str) -> Iterator[None]:
    token = _operation.set(name or "unspecified")
    try:
        yield
    finally:
        _operation.reset(token)


__all__ = ["current_llm_operation", "llm_operation"]
