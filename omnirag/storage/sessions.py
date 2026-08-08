"""Session namespaces and the document registry.

Isolation model
---------------
Every uploaded document, every vector, and every query is tagged with a
``session_id``. The vector store *refuses* to search without one
(:class:`~omnirag.core.exceptions.SessionIsolationError`), so a bug elsewhere
cannot leak one visitor's documents into another's answers.

Anonymous-session limitations (see README "Security & privacy"): a Streamlit
session id lives as long as the browser tab. It is unguessable but not
authenticated — anyone who could replay the id could read that namespace.
Production multi-user deployments must replace :func:`new_session_id` with a
real authenticated user/tenant id.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from omnirag.core.enums import IngestionStatus
from omnirag.core.models import DocumentSummary, SessionInfo
from omnirag.utils.hashing import short_hash
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

SESSION_PREFIX = "s_"


def new_session_id() -> str:
    """Create an unguessable session namespace id."""
    return f"{SESSION_PREFIX}{uuid.uuid4().hex}"


def is_valid_session_id(session_id: Optional[str]) -> bool:
    return bool(session_id) and isinstance(session_id, str) and len(session_id) >= 8


def require_session_id(session_id: Optional[str]) -> str:
    """Guard used at every boundary that touches user data."""
    from omnirag.core.exceptions import SessionIsolationError

    if not is_valid_session_id(session_id):
        raise SessionIsolationError(
            f"Operation attempted without a valid session id (got {session_id!r})"
        )
    return str(session_id)


class DocumentRegistry:
    """In-process catalogue of the documents belonging to each session.

    Holds only :class:`DocumentSummary` objects (metadata) — never page content
    or image bytes, which live in the vector store and the file store. This
    keeps memory flat as documents grow and makes the registry trivially
    replaceable by a database in a FastAPI deployment.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionInfo] = {}
        self._documents: Dict[str, Dict[str, DocumentSummary]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}  # session -> hash -> doc id

    # -- sessions ---------------------------------------------------------- #
    def touch(self, session_id: str) -> SessionInfo:
        session_id = require_session_id(session_id)
        with self._lock:
            info = self._sessions.get(session_id)
            if info is None:
                info = SessionInfo(session_id=session_id)
                self._sessions[session_id] = info
                self._documents[session_id] = {}
                self._hashes[session_id] = {}
                logger.info("New session namespace %s", short_hash(session_id, 8))
            else:
                info.last_active = datetime.now(timezone.utc)
            return info

    def sessions(self) -> List[SessionInfo]:
        with self._lock:
            return list(self._sessions.values())

    # -- documents --------------------------------------------------------- #
    def add(self, summary: DocumentSummary) -> DocumentSummary:
        session_id = require_session_id(summary.session_id)
        self.touch(session_id)
        with self._lock:
            self._documents[session_id][summary.document_id] = summary
            if summary.content_hash:
                self._hashes[session_id][summary.content_hash] = summary.document_id
            self._recount(session_id)
            return summary

    def update(self, summary: DocumentSummary) -> DocumentSummary:
        return self.add(summary)

    def get(self, session_id: str, document_id: str) -> Optional[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            return self._documents.get(session_id, {}).get(document_id)

    def list(self, session_id: str) -> List[DocumentSummary]:
        """All documents of a session, newest last (upload order)."""
        session_id = require_session_id(session_id)
        with self._lock:
            docs = list(self._documents.get(session_id, {}).values())
        return sorted(docs, key=lambda d: d.created_at)

    def ready_documents(self, session_id: str) -> List[DocumentSummary]:
        return [d for d in self.list(session_id) if d.status == IngestionStatus.READY]

    def find_by_hash(self, session_id: str, content_hash: str) -> Optional[DocumentSummary]:
        """Deduplication lookup — same bytes in the same namespace."""
        session_id = require_session_id(session_id)
        with self._lock:
            document_id = self._hashes.get(session_id, {}).get(content_hash)
            if not document_id:
                return None
            return self._documents.get(session_id, {}).get(document_id)

    def remove(self, session_id: str, document_id: str) -> Optional[DocumentSummary]:
        session_id = require_session_id(session_id)
        with self._lock:
            summary = self._documents.get(session_id, {}).pop(document_id, None)
            if summary and summary.content_hash:
                self._hashes.get(session_id, {}).pop(summary.content_hash, None)
            self._recount(session_id)
            return summary

    def clear(self, session_id: str) -> List[str]:
        """Drop every document of a session; returns the removed document ids."""
        session_id = require_session_id(session_id)
        with self._lock:
            document_ids = list(self._documents.get(session_id, {}).keys())
            self._documents[session_id] = {}
            self._hashes[session_id] = {}
            self._recount(session_id)
            return document_ids

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._documents.pop(session_id, None)
            self._hashes.pop(session_id, None)

    # -- maintenance ------------------------------------------------------- #
    def expired_sessions(self, ttl_minutes: int) -> List[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, ttl_minutes))
        with self._lock:
            return [
                sid for sid, info in self._sessions.items() if info.last_active < cutoff
            ]

    def _recount(self, session_id: str) -> None:
        info = self._sessions.get(session_id)
        if info is None:
            return
        docs = self._documents.get(session_id, {}).values()
        info.document_count = len(docs)
        info.chunk_count = sum(d.chunk_count for d in docs)
        info.last_active = datetime.now(timezone.utc)


_registry: Optional[DocumentRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> DocumentRegistry:
    """Process-wide registry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = DocumentRegistry()
        return _registry


def reset_registry() -> None:
    """Test helper — start from a clean registry."""
    global _registry
    with _registry_lock:
        _registry = DocumentRegistry()


def document_ids(summaries: Iterable[DocumentSummary]) -> List[str]:
    return [s.document_id for s in summaries]
