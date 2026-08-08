"""Composition root.

Builds and holds the long-lived collaborators (providers, vector store, file
store, registry). The Streamlit layer caches one :class:`OmniRAGEngine` per
process with ``st.cache_resource``; a FastAPI deployment would build the same
object at startup. Nothing here imports Streamlit.

Everything is created lazily: opening a Qdrant connection or an HTTP client
before the user has uploaded anything would just slow down first paint.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omnirag.config.settings import AppSettings, get_settings
from omnirag.intelligence.handwriting import HandwritingExtractor
from omnirag.intelligence.ocr import OCREngine, build_ocr_engine
from omnirag.intelligence.vision import VisionAnalyzer, build_vision_analyzer
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.llm.base import BaseLLMProvider
from omnirag.rag.chunking import Chunker
from omnirag.rag.embeddings import EmbeddingPipeline
from omnirag.rag.vector_store import BaseVectorStore, build_vector_store
from omnirag.storage.files import FileStore, get_file_store
from omnirag.storage.sessions import DocumentRegistry, get_registry
from omnirag.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class EngineStatus:
    """Health snapshot rendered by the UI's status panel."""

    ready: bool
    issues: List[str]
    warnings: List[str]
    llm_chain: str
    llm_available: bool
    embedding_provider: str
    embedding_model: str
    vector_store: str
    reranker: str
    ocr_provider: str
    vision_available: bool


class OmniRAGEngine:
    """Holds every long-lived component; safe to share across sessions.

    Session isolation is *not* achieved by having one engine per user — it is
    enforced by the ``session_id`` filter inside the vector store and the
    registry, so a single shared engine is both correct and efficient.
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)

        self._lock = threading.RLock()
        self._llm: Optional[BaseLLMProvider] = None
        self._embeddings: Optional[BaseEmbeddingProvider] = None
        self._vector_store: Optional[BaseVectorStore] = None
        self._ocr: Optional[OCREngine] = None
        self._vision: Optional[VisionAnalyzer] = None
        self._handwriting: Optional[HandwritingExtractor] = None
        self._chunker: Optional[Chunker] = None
        self._file_store: Optional[FileStore] = None
        self._registry: Optional[DocumentRegistry] = None

    # -- lazily built components ---------------------------------------- #
    @property
    def file_store(self) -> FileStore:
        with self._lock:
            if self._file_store is None:
                self._file_store = get_file_store(self.settings.workspace_dir or None)
            return self._file_store

    @property
    def registry(self) -> DocumentRegistry:
        with self._lock:
            if self._registry is None:
                self._registry = get_registry()
            return self._registry

    @property
    def chunker(self) -> Chunker:
        with self._lock:
            if self._chunker is None:
                self._chunker = Chunker(self.settings.chunking)
            return self._chunker

    @property
    def vector_store(self) -> BaseVectorStore:
        with self._lock:
            if self._vector_store is None:
                self._vector_store = build_vector_store(self.settings.vector_store)
            return self._vector_store

    @property
    def llm(self) -> Optional[BaseLLMProvider]:
        """The provider router, or ``None`` when nothing is configured."""
        with self._lock:
            if self._llm is None and self.settings.llm.is_configured:
                try:
                    from omnirag.providers.llm.factory import get_llm_provider

                    self._llm = get_llm_provider(self.settings)
                except Exception as exc:
                    logger.warning("LLM unavailable: %s", exc)
            return self._llm

    @property
    def embeddings(self) -> BaseEmbeddingProvider:
        with self._lock:
            if self._embeddings is None:
                from omnirag.providers.embeddings.factory import get_embedding_provider

                self._embeddings = get_embedding_provider(self.settings)
            return self._embeddings

    @property
    def embedding_pipeline(self) -> EmbeddingPipeline:
        return EmbeddingPipeline(self.embeddings)

    @property
    def ocr(self) -> OCREngine:
        with self._lock:
            if self._ocr is None:
                self._ocr = build_ocr_engine(self.settings)
            return self._ocr

    @property
    def vision(self) -> VisionAnalyzer:
        with self._lock:
            if self._vision is None:
                self._vision = build_vision_analyzer(self.settings)
            return self._vision

    @property
    def handwriting(self) -> HandwritingExtractor:
        with self._lock:
            if self._handwriting is None:
                self._handwriting = HandwritingExtractor(self.ocr, self.vision)
            return self._handwriting

    # -- status ---------------------------------------------------------- #
    def status(self) -> EngineStatus:
        """Cheap health snapshot; never raises."""
        try:
            reranker_name = self._reranker_name()
        except Exception:
            reranker_name = "heuristic"

        try:
            ocr_name = self.ocr.name
        except Exception:
            ocr_name = "none"

        try:
            vision_available = self.vision.available
        except Exception:
            vision_available = False

        return EngineStatus(
            ready=self.settings.is_ready,
            issues=self.settings.validation_issues(),
            warnings=self.settings.warnings(),
            llm_chain=self.settings.llm.chain_label,
            llm_available=self.llm is not None,
            embedding_provider=self.settings.embedding.provider,
            embedding_model=self.settings.embedding.model,
            vector_store=self.vector_store.name,
            reranker=reranker_name,
            ocr_provider=ocr_name,
            vision_available=vision_available,
        )

    def _reranker_name(self) -> str:
        from omnirag.providers.rerank.factory import get_reranker

        return get_reranker(self.settings).name

    def provider_stats(self) -> Dict[str, Any]:
        """Router counters for the UI's provider indicator."""
        llm = self._llm
        stats = getattr(llm, "stats", None)
        if stats is None:
            return {}
        return {
            "calls": stats.calls,
            "failovers": stats.failovers,
            "by_provider": dict(stats.by_provider),
            "last_provider": stats.last_provider,
            "last_model": stats.last_model,
            "last_attempts": list(stats.last_attempts),
        }

    # -- maintenance ----------------------------------------------------- #
    def clear_session(self, session_id: str) -> Dict[str, int]:
        """Remove every trace of a session: vectors, files, registry entries."""
        from omnirag.rag.hybrid import get_bm25_cache

        removed_vectors = 0
        try:
            removed_vectors = self.vector_store.delete_session(session_id)
        except Exception as exc:
            logger.warning("Could not clear vectors for the session: %s", exc)

        removed_files = 0
        try:
            removed_files = self.file_store.delete_session(session_id)
        except Exception as exc:
            logger.warning("Could not clear files for the session: %s", exc)

        document_ids = self.registry.clear(session_id)
        get_bm25_cache().invalidate(session_id)

        logger.info(
            "Cleared session: %d documents, %d vectors, %d assets",
            len(document_ids),
            removed_vectors,
            removed_files,
        )
        return {
            "documents": len(document_ids),
            "vectors": removed_vectors,
            "assets": removed_files,
        }

    def cleanup_expired_sessions(self) -> int:
        """Best-effort reclamation of abandoned anonymous sessions."""
        expired = self.registry.expired_sessions(self.settings.session_ttl_minutes)
        for session_id in expired:
            try:
                self.clear_session(session_id)
                self.registry.drop_session(session_id)
            except Exception as exc:
                logger.warning("Cleanup failed for an expired session: %s", exc)
        return len(expired)


_engine: Optional[OmniRAGEngine] = None
_engine_lock = threading.Lock()


def get_engine(settings: Optional[AppSettings] = None) -> OmniRAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = OmniRAGEngine(settings)
        return _engine


def reset_engine() -> None:
    """Drop the cached engine (tests, configuration reload)."""
    global _engine
    with _engine_lock:
        _engine = None


__all__ = ["EngineStatus", "OmniRAGEngine", "get_engine", "reset_engine"]
