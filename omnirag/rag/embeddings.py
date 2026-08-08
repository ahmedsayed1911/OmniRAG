"""Embedding orchestration for indexing.

Adds the batching, deduplication and failure handling that belong to the
pipeline rather than to any single provider adapter:

* **deduplication** — identical chunk texts are embedded once and the vector is
  reused. Repeated boilerplate (cover pages, footers, repeated table headers) is
  common and paying for it twice is pure waste.
* **batching** — inputs are grouped into provider-sized batches.
* **partial failure** — if one batch fails, the chunks it covered are reported
  rather than silently dropped, so ingestion can warn instead of pretending the
  document is fully indexed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from omnirag.core.exceptions import EmbeddingError
from omnirag.core.models import Chunk
from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.utils.hashing import text_hash
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingBatchResult:
    """Chunks paired with their vectors, plus anything that failed."""

    chunks: List[Chunk] = field(default_factory=list)
    vectors: List[Vector] = field(default_factory=list)
    failed: List[Chunk] = field(default_factory=list)
    dimensions: int = 0
    api_calls: int = 0
    deduplicated: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.chunks) and not self.failed

    @property
    def count(self) -> int:
        return len(self.chunks)


class EmbeddingPipeline:
    """Deduplicating, batching wrapper around an embedding provider."""

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    def embed_chunks(self, chunks: Sequence[Chunk]) -> EmbeddingBatchResult:
        result = EmbeddingBatchResult()
        if not chunks:
            return result

        # Group identical texts so each unique string is embedded exactly once.
        unique_texts: List[str] = []
        index_of: Dict[str, int] = {}
        assignment: List[int] = []

        for chunk in chunks:
            key = text_hash(chunk.text)
            position = index_of.get(key)
            if position is None:
                position = len(unique_texts)
                index_of[key] = position
                unique_texts.append(chunk.text)
            else:
                result.deduplicated += 1
            assignment.append(position)

        try:
            vectors = self.provider.embed_documents(unique_texts)
            result.api_calls = _batch_count(len(unique_texts), self.provider.batch_size)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed: {exc}",
                provider=self.provider.name,
                user_message=(
                    "Could not create embeddings for this document. "
                    "Check the embedding provider configuration and try again."
                ),
            ) from exc

        if len(vectors) != len(unique_texts):
            raise EmbeddingError(
                f"Expected {len(unique_texts)} vectors, received {len(vectors)}",
                provider=self.provider.name,
            )

        for chunk, position in zip(chunks, assignment):
            vector = vectors[position]
            if not vector:
                result.failed.append(chunk)
                continue
            result.chunks.append(chunk)
            result.vectors.append(vector)

        result.dimensions = len(result.vectors[0]) if result.vectors else 0
        logger.info(
            "Embedded %d chunks (%d unique, %d duplicates reused, dim=%d)",
            len(chunks),
            len(unique_texts),
            result.deduplicated,
            result.dimensions,
        )
        return result

    def embed_query(self, query: str) -> Vector:
        return self.provider.embed_query(query)


def _batch_count(total: int, batch_size: int) -> int:
    if total <= 0 or batch_size <= 0:
        return 0
    return (total + batch_size - 1) // batch_size


def build_embedding_pipeline(settings=None) -> EmbeddingPipeline:
    from omnirag.config.settings import get_settings
    from omnirag.providers.embeddings.factory import get_embedding_provider

    return EmbeddingPipeline(get_embedding_provider(settings or get_settings()))


__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingPipeline",
    "build_embedding_pipeline",
]
