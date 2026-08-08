"""Vector storage with mandatory session isolation.

Two interchangeable backends behind one interface:

* :class:`QdrantVectorStore` — Qdrant Cloud or self-hosted, the production path;
* :class:`InMemoryVectorStore` — dependency-free NumPy fallback for local
  development, tests, and any deployment without a Qdrant URL.

**Isolation is enforced here, not by callers.** Every write requires a
``session_id``, every search requires a ``session_id``, and the filter is added
by the store itself — a caller cannot forget it or opt out. Requests without a
valid session raise :class:`SessionIsolationError` rather than returning
unfiltered results.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omnirag.config.settings import VectorStoreSettings
from omnirag.core.exceptions import SessionIsolationError, VectorStoreError
from omnirag.core.models import Chunk
from omnirag.storage.sessions import require_session_id
from omnirag.utils.hashing import stable_uuid
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class SearchFilter:
    """Metadata filter. ``session_id`` is applied by the store unconditionally."""

    document_ids: Optional[Sequence[str]] = None
    block_types: Optional[Sequence[str]] = None
    page_numbers: Optional[Sequence[int]] = None
    languages: Optional[Sequence[str]] = None
    source_kinds: Optional[Sequence[str]] = None

    def matches(self, payload: Dict[str, Any]) -> bool:
        """Python-side evaluation, used by the in-memory backend."""
        if self.document_ids and payload.get("document_id") not in set(self.document_ids):
            return False
        if self.block_types and payload.get("block_type") not in set(self.block_types):
            return False
        if self.page_numbers and payload.get("page_number") not in set(self.page_numbers):
            return False
        if self.languages and payload.get("language") not in set(self.languages):
            return False
        if self.source_kinds and payload.get("source_kind") not in set(self.source_kinds):
            return False
        return True


class BaseVectorStore(ABC):
    """Interface implemented by every vector backend."""

    name: str = "base"

    @abstractmethod
    def ensure_collection(self, dimensions: int) -> None:
        """Create the collection if needed (idempotent)."""

    @abstractmethod
    def upsert(self, session_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Store chunks with their vectors. Returns the number written."""

    @abstractmethod
    def search(
        self,
        session_id: str,
        vector: Sequence[float],
        *,
        top_k: int = 20,
        filters: Optional[SearchFilter] = None,
    ) -> List[ScoredChunk]:
        """Vector search, always restricted to ``session_id``."""

    @abstractmethod
    def list_chunks(
        self, session_id: str, *, document_ids: Optional[Sequence[str]] = None, limit: int = 10000
    ) -> List[Chunk]:
        """All chunks of a session — powers the BM25 index and evaluation."""

    @abstractmethod
    def delete_document(self, session_id: str, document_id: str) -> int:
        """Remove one document's vectors from a session."""

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Remove every vector belonging to a session."""

    @abstractmethod
    def count(self, session_id: str) -> int:
        ...

    def health(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #
class InMemoryVectorStore(BaseVectorStore):
    """NumPy-backed store. Data lives only for the life of the process."""

    name = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # session_id -> chunk_id -> (vector, chunk)
        self._data: Dict[str, Dict[str, Tuple[List[float], Chunk]]] = {}
        self._dimensions = 0

    def ensure_collection(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def upsert(
        self, session_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        session_id = require_session_id(session_id)
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(vectors)} vectors",
                user_message="Indexing failed due to an internal mismatch.",
            )
        with self._lock:
            bucket = self._data.setdefault(session_id, {})
            for chunk, vector in zip(chunks, vectors):
                _assert_same_session(chunk, session_id)
                bucket[chunk.chunk_id] = (list(vector), chunk)
        return len(chunks)

    def search(
        self,
        session_id: str,
        vector: Sequence[float],
        *,
        top_k: int = 20,
        filters: Optional[SearchFilter] = None,
    ) -> List[ScoredChunk]:
        session_id = require_session_id(session_id)
        with self._lock:
            bucket = list(self._data.get(session_id, {}).values())
        if not bucket or not vector:
            return []

        candidates = [
            (stored_vector, chunk)
            for stored_vector, chunk in bucket
            if filters is None or filters.matches(chunk.to_payload())
        ]
        if not candidates:
            return []

        scores = _cosine_scores([v for v, _ in candidates], list(vector))
        ranked = sorted(
            ((score, chunk) for score, (_, chunk) in zip(scores, candidates)),
            key=lambda item: item[0],
            reverse=True,
        )
        return [ScoredChunk(chunk=chunk, score=float(score)) for score, chunk in ranked[:top_k]]

    def list_chunks(
        self, session_id: str, *, document_ids: Optional[Sequence[str]] = None, limit: int = 10000
    ) -> List[Chunk]:
        session_id = require_session_id(session_id)
        with self._lock:
            chunks = [chunk for _, chunk in self._data.get(session_id, {}).values()]
        if document_ids:
            allowed = set(document_ids)
            chunks = [c for c in chunks if c.document_id in allowed]
        return chunks[:limit]

    def delete_document(self, session_id: str, document_id: str) -> int:
        session_id = require_session_id(session_id)
        with self._lock:
            bucket = self._data.get(session_id, {})
            targets = [cid for cid, (_, chunk) in bucket.items() if chunk.document_id == document_id]
            for chunk_id in targets:
                bucket.pop(chunk_id, None)
            return len(targets)

    def delete_session(self, session_id: str) -> int:
        session_id = require_session_id(session_id)
        with self._lock:
            removed = len(self._data.pop(session_id, {}))
        return removed

    def count(self, session_id: str) -> int:
        session_id = require_session_id(session_id)
        with self._lock:
            return len(self._data.get(session_id, {}))


def _cosine_scores(vectors: Sequence[Sequence[float]], query: Sequence[float]) -> List[float]:
    if NUMPY_AVAILABLE:
        matrix = np.asarray(vectors, dtype="float32")
        target = np.asarray(query, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != target.shape[0]:
            return [0.0] * len(vectors)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(target) or 1.0)
        norms[norms == 0] = 1.0
        return (matrix @ target / norms).tolist()

    # Pure-Python fallback keeps the store usable without NumPy.
    query_norm = math.sqrt(sum(q * q for q in query)) or 1.0
    out: List[float] = []
    for vector in vectors:
        if len(vector) != len(query):
            out.append(0.0)
            continue
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        out.append(sum(v * q for v, q in zip(vector, query)) / (norm * query_norm))
    return out


# --------------------------------------------------------------------------- #
# Qdrant backend
# --------------------------------------------------------------------------- #
class QdrantVectorStore(BaseVectorStore):
    """Qdrant Cloud / self-hosted backend.

    One collection holds every session; isolation comes from an indexed
    ``session_id`` payload field that is added to *every* query filter by this
    class. The field is explicitly indexed so filtering stays fast as the
    collection grows.
    """

    name = "qdrant"

    def __init__(self, settings: VectorStoreSettings):
        self.settings = settings
        self.collection = settings.collection
        self._client: Any = None
        self._lock = threading.RLock()
        self._ready = False

    # -- client ---------------------------------------------------------- #
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"qdrant-client is not installed: {exc}",
                user_message="The Qdrant client library is missing. Run `pip install -r requirements.txt`.",
            ) from exc

        url = self.settings.url
        if not url:
            raise VectorStoreError(
                "QDRANT_URL is empty",
                user_message="`QDRANT_URL` is not configured.",
            )
        try:
            if url.startswith(":memory:"):
                return QdrantClient(location=":memory:")
            return QdrantClient(
                url=url,
                api_key=self.settings.api_key or None,
                prefer_grpc=self.settings.prefer_grpc,
                timeout=int(self.settings.timeout_s),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not connect to Qdrant: {exc}",
                user_message=(
                    "Could not connect to the Qdrant vector database. "
                    "Check `QDRANT_URL` and `QDRANT_API_KEY`."
                ),
            ) from exc

    # -- schema ---------------------------------------------------------- #
    def ensure_collection(self, dimensions: int) -> None:
        if self._ready or dimensions <= 0:
            return
        from qdrant_client import models

        with self._lock:
            if self._ready:
                return
            try:
                exists = self.client.collection_exists(self.collection)
            except Exception as exc:
                raise VectorStoreError(
                    f"Qdrant is unreachable: {exc}",
                    user_message="The Qdrant vector database is unreachable.",
                ) from exc

            if not exists:
                distance = {
                    "cosine": models.Distance.COSINE,
                    "dot": models.Distance.DOT,
                    "euclid": models.Distance.EUCLID,
                }.get(self.settings.distance, models.Distance.COSINE)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=dimensions, distance=distance),
                )
                logger.info(
                    "Created Qdrant collection %s (dim=%d)", self.collection, dimensions
                )

            # Indexes that make session/document filtering fast.
            for field_name, schema in (
                ("session_id", models.PayloadSchemaType.KEYWORD),
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("block_type", models.PayloadSchemaType.KEYWORD),
                ("page_number", models.PayloadSchemaType.INTEGER),
            ):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field_name,
                        field_schema=schema,
                    )
                except Exception:
                    pass  # already exists
            self._ready = True

    # -- writes ---------------------------------------------------------- #
    def upsert(
        self, session_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        session_id = require_session_id(session_id)
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(vectors)} vectors",
                user_message="Indexing failed due to an internal mismatch.",
            )

        from qdrant_client import models

        self.ensure_collection(len(vectors[0]))
        points = []
        for chunk, vector in zip(chunks, vectors):
            _assert_same_session(chunk, session_id)
            points.append(
                models.PointStruct(
                    id=stable_uuid(chunk.chunk_id),
                    vector=list(vector),
                    payload=chunk.to_payload(),
                )
            )

        try:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant upsert failed: {exc}",
                user_message="Could not write to the vector database. Please retry.",
            ) from exc
        return len(points)

    # -- reads ----------------------------------------------------------- #
    def search(
        self,
        session_id: str,
        vector: Sequence[float],
        *,
        top_k: int = 20,
        filters: Optional[SearchFilter] = None,
    ) -> List[ScoredChunk]:
        session_id = require_session_id(session_id)
        if not vector:
            return []

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                query_filter=self._build_filter(session_id, filters),
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant search failed: {exc}",
                user_message="Search over your documents failed. Please retry.",
            ) from exc

        out: List[ScoredChunk] = []
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            if payload.get("session_id") != session_id:
                # Defence in depth: never surface a foreign-session payload.
                raise SessionIsolationError(
                    "Qdrant returned a point from a different session namespace"
                )
            try:
                out.append(
                    ScoredChunk(chunk=Chunk.from_payload(payload), score=float(point.score))
                )
            except Exception as exc:
                logger.warning("Skipping malformed payload: %s", exc)
        return out

    def list_chunks(
        self, session_id: str, *, document_ids: Optional[Sequence[str]] = None, limit: int = 10000
    ) -> List[Chunk]:
        session_id = require_session_id(session_id)
        filters = SearchFilter(document_ids=document_ids)
        chunks: List[Chunk] = []
        offset = None
        try:
            while len(chunks) < limit:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=self._build_filter(session_id, filters),
                    limit=min(256, limit - len(chunks)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points or []:
                    payload = getattr(point, "payload", None) or {}
                    if payload.get("session_id") != session_id:
                        raise SessionIsolationError(
                            "Qdrant scroll returned a foreign-session payload"
                        )
                    try:
                        chunks.append(Chunk.from_payload(payload))
                    except Exception:
                        continue
                if offset is None or not points:
                    break
        except SessionIsolationError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant scroll failed: {exc}",
                user_message="Could not read from the vector database.",
            ) from exc
        return chunks

    # -- deletes --------------------------------------------------------- #
    def delete_document(self, session_id: str, document_id: str) -> int:
        session_id = require_session_id(session_id)
        count = len(self.list_chunks(session_id, document_ids=[document_id]))
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=self._build_filter(
                    session_id, SearchFilter(document_ids=[document_id])
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant delete failed: {exc}",
                user_message="Could not remove the document from the vector database.",
            ) from exc
        return count

    def delete_session(self, session_id: str) -> int:
        session_id = require_session_id(session_id)
        count = self.count(session_id)
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=self._build_filter(session_id, None),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant delete failed: {exc}",
                user_message="Could not clear your documents from the vector database.",
            ) from exc
        return count

    def count(self, session_id: str) -> int:
        session_id = require_session_id(session_id)
        try:
            result = self.client.count(
                collection_name=self.collection,
                count_filter=self._build_filter(session_id, None),
                exact=True,
            )
            return int(getattr(result, "count", 0))
        except Exception:
            return 0

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False

    # -- internals ------------------------------------------------------- #
    def _build_filter(self, session_id: str, filters: Optional[SearchFilter]):
        """Build a Qdrant filter that ALWAYS pins ``session_id``."""
        from qdrant_client import models

        must: List[Any] = [
            models.FieldCondition(
                key="session_id", match=models.MatchValue(value=session_id)
            )
        ]
        if filters is not None:
            if filters.document_ids:
                must.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=list(filters.document_ids)),
                    )
                )
            if filters.block_types:
                must.append(
                    models.FieldCondition(
                        key="block_type", match=models.MatchAny(any=list(filters.block_types))
                    )
                )
            if filters.page_numbers:
                must.append(
                    models.FieldCondition(
                        key="page_number", match=models.MatchAny(any=list(filters.page_numbers))
                    )
                )
            if filters.languages:
                must.append(
                    models.FieldCondition(
                        key="language", match=models.MatchAny(any=list(filters.languages))
                    )
                )
            if filters.source_kinds:
                must.append(
                    models.FieldCondition(
                        key="source_kind", match=models.MatchAny(any=list(filters.source_kinds))
                    )
                )
        return models.Filter(must=must)


def _assert_same_session(chunk: Chunk, session_id: str) -> None:
    if chunk.session_id != session_id:
        raise SessionIsolationError(
            f"Chunk {chunk.chunk_id} belongs to another session and was not written"
        )


# --------------------------------------------------------------------------- #
_store: Optional[BaseVectorStore] = None
_store_lock = threading.Lock()


def build_vector_store(settings: Optional[VectorStoreSettings] = None) -> BaseVectorStore:
    """Create the configured backend, falling back to memory on failure."""
    from omnirag.config.settings import get_settings

    cfg = settings or get_settings().vector_store
    if not cfg.use_qdrant:
        logger.info("Vector store: in-memory (no QDRANT_URL configured)")
        return InMemoryVectorStore()

    store = QdrantVectorStore(cfg)
    try:
        if store.health():
            logger.info("Vector store: Qdrant at %s", _safe_url(cfg.url))
            return store
    except Exception as exc:
        logger.warning("Qdrant unavailable (%s)", exc)

    logger.warning(
        "Falling back to the in-memory vector store — the index will not persist"
    )
    return InMemoryVectorStore()


def get_vector_store() -> BaseVectorStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = build_vector_store()
        return _store


def set_vector_store(store: Optional[BaseVectorStore]) -> None:
    """Override the singleton (tests, alternative backends)."""
    global _store
    with _store_lock:
        _store = store


def _safe_url(url: str) -> str:
    """Strip credentials before logging a URL."""
    if "@" in url:
        scheme, _, rest = url.partition("://")
        return f"{scheme}://***@{rest.split('@', 1)[-1]}"
    return url


__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "ScoredChunk",
    "SearchFilter",
    "build_vector_store",
    "get_vector_store",
    "set_vector_store",
]
