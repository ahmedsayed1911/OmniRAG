"""Operation labels for safe provider-chain diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_operation: ContextVar[str] = ContextVar("omnirag_llm_operation", default="unspecified")
_generation_id: ContextVar[str] = ContextVar("omnirag_generation_id", default="")


def current_llm_operation() -> str:
    return _operation.get()


def current_generation_id() -> str:
    return _generation_id.get()


@contextmanager
def llm_operation(name: str) -> Iterator[None]:
    token = _operation.set(name or "unspecified")
    try:
        yield
    finally:
        _operation.reset(token)


@contextmanager
def generation_context(generation_id: str) -> Iterator[None]:
    token = _generation_id.set(generation_id or "")
    try:
        yield
    finally:
        _generation_id.reset(token)


__all__ = [
    "current_generation_id",
    "current_llm_operation",
    "generation_context",
    "llm_operation",
]
