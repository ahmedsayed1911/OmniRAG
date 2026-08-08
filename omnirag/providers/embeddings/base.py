"""Embedding provider interface.

Deliberately independent of the LLM failover chain: an LLM outage must never
invalidate vectors already written to Qdrant, and swapping the answering model
must not force a re-index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

Vector = List[float]


class BaseEmbeddingProvider(ABC):
    """Contract for every embedding backend."""

    name: str = "base"
    #: Set by the adapter once known; ``0`` means "probe on first call".
    dimensions: int = 0
    #: Whether the backend distinguishes query and document embeddings.
    supports_task_type: bool = False

    def __init__(self, *, model: str, batch_size: int = 64, max_chars: int = 8000):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_chars = max_chars

    # ------------------------------------------------------------------ #
    @abstractmethod
    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        """Embed a batch. Implementations must return one vector per input."""

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Embed many texts, chunked into provider-sized batches."""
        return self._embed_all(texts, is_query=False)

    def embed_query(self, text: str) -> Vector:
        vectors = self._embed_all([text], is_query=True)
        return vectors[0] if vectors else []

    def _embed_all(self, texts: Sequence[str], *, is_query: bool) -> List[Vector]:
        prepared = [self._prepare(t) for t in texts]
        out: List[Vector] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            vectors = self.embed_batch(batch, is_query=is_query)
            if len(vectors) != len(batch):
                from omnirag.core.exceptions import EmbeddingError

                raise EmbeddingError(
                    f"{self.name} returned {len(vectors)} vectors for {len(batch)} inputs",
                    provider=self.name,
                )
            out.extend(vectors)
        if out and not self.dimensions:
            self.dimensions = len(out[0])
        return out

    def _prepare(self, text: str) -> str:
        """Trim to the provider's input limit; empty strings get a placeholder.

        An all-whitespace input makes several APIs return HTTP 400, which would
        abort a whole batch because of one blank block.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return " "
        return cleaned[: self.max_chars]

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dimensions": self.dimensions}
