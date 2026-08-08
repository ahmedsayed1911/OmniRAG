"""Controlled runtime fallback for external embedding providers.

The fallback is sticky: once document vectors are generated in the hash
embedding space, queries must stay in that same space for the lifetime of the
engine. Only typed provider/configuration availability failures activate it;
malformed responses and ordinary Python exceptions remain visible.
"""

from __future__ import annotations

import threading
from typing import List, Sequence

from omnirag.core.exceptions import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.providers.embeddings.hashing import HashingEmbeddings
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

FALLBACK_ERRORS = (
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderAuthError,
    ProviderBadRequestError,
)


class ResilientEmbeddings(BaseEmbeddingProvider):
    """Use a real provider until a known availability failure, then hash."""

    def __init__(
        self,
        primary: BaseEmbeddingProvider,
        fallback: HashingEmbeddings | None = None,
    ) -> None:
        super().__init__(
            model=primary.model,
            batch_size=primary.batch_size,
            max_chars=primary.max_chars,
        )
        self.primary = primary
        self.fallback = fallback or HashingEmbeddings(
            max_chars=primary.max_chars, announce=False
        )
        self.name = primary.name
        self.dimensions = primary.dimensions
        self.supports_task_type = primary.supports_task_type
        self.fallback_reason = ""
        self._fallback_active = False
        self._lock = threading.RLock()

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def _activate_fallback(self, exc: BaseException) -> None:
        with self._lock:
            if self._fallback_active:
                return
            self._fallback_active = True
            self.fallback_reason = type(exc).__name__
            self.name = self.fallback.name
            self.model = self.fallback.model
            self.dimensions = self.fallback.dimensions
            self.batch_size = self.fallback.batch_size
            self.supports_task_type = self.fallback.supports_task_type
            logger.warning(
                "Embedding provider %s unavailable (%s); using offline hash embeddings",
                self.primary.name,
                self.fallback_reason,
            )

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if self._fallback_active:
            return self.fallback.embed_batch(texts, is_query=is_query)
        try:
            vectors = self.primary.embed_batch(texts, is_query=is_query)
            self.dimensions = self.primary.dimensions or (len(vectors[0]) if vectors else 0)
            return vectors
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            return self.fallback.embed_batch(texts, is_query=is_query)

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        if self._fallback_active:
            return self.fallback.embed_documents(texts)
        try:
            vectors = self.primary.embed_documents(texts)
            self.dimensions = self.primary.dimensions
            return vectors
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            # Re-embed the complete input; never mix vector spaces when a
            # failure happens after one or more successful primary batches.
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        if self._fallback_active:
            return self.fallback.embed_query(text)
        try:
            vector = self.primary.embed_query(text)
            self.dimensions = self.primary.dimensions
            return vector
        except FALLBACK_ERRORS as exc:
            self._activate_fallback(exc)
            return self.fallback.embed_query(text)


__all__ = ["FALLBACK_ERRORS", "ResilientEmbeddings"]
