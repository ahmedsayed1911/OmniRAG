"""Binary asset storage.

Streamlit Community Cloud gives you an *ephemeral* filesystem: anything written
is lost on restart and is not shared between replicas. The engine therefore
never assumes a durable path — it talks to a :class:`FileStore` interface and
the default implementation keeps assets in a per-process temporary workspace
with an in-memory index. Swapping in S3/GCS later means implementing three
methods, nothing else changes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from omnirag.utils.hashing import short_hash, stable_id
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StoredAsset:
    asset_id: str
    session_id: str
    media_type: str
    size_bytes: int
    path: Optional[str] = None


class FileStore(ABC):
    """Content-addressed blob store scoped by session."""

    @abstractmethod
    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        """Store bytes and return a handle. Identical bytes reuse the same id."""

    @abstractmethod
    def get(self, asset_id: str) -> Optional[bytes]:
        """Return the bytes for ``asset_id`` or ``None`` when absent/expired."""

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Remove every asset of a session; returns how many were deleted."""

    def exists(self, asset_id: str) -> bool:
        return self.get(asset_id) is not None


class LocalFileStore(FileStore):
    """Temp-directory backed store, safe for Streamlit's ephemeral disk.

    Thread-safe: Streamlit runs one thread per session and background ingestion
    may touch the store concurrently.
    """

    def __init__(self, root: Optional[str] = None, *, max_bytes: int = 900 * 1024 * 1024):
        self._root = root or os.path.join(tempfile.gettempdir(), "omnirag_workspace")
        os.makedirs(self._root, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, StoredAsset] = {}
        # Fallback for read-only/full disks: keep the bytes in RAM instead.
        self._memory: Dict[str, bytes] = {}
        self._max_bytes = max_bytes
        self._total_bytes = 0
        logger.info("Local file store initialised at %s", self._root)

    # -- paths ------------------------------------------------------------- #
    @property
    def root(self) -> str:
        return self._root

    def _session_dir(self, session_id: str) -> str:
        path = os.path.join(self._root, short_hash(session_id, 16))
        os.makedirs(path, exist_ok=True)
        return path

    # -- API --------------------------------------------------------------- #
    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        asset_id = stable_id(session_id, short_hash(data, 32), media_type)
        with self._lock:
            existing = self._index.get(asset_id)
            if existing is not None and existing.path and os.path.exists(existing.path):
                return existing

            if self._total_bytes + len(data) > self._max_bytes:
                self._evict_unsafe(len(data))

            filename = f"{asset_id}{suffix or _suffix_for(media_type)}"
            path = os.path.join(self._session_dir(session_id), filename)
            try:
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as exc:
                logger.warning("Falling back to memory for asset %s: %s", asset_id, exc)
                path = None  # type: ignore[assignment]
                self._memory[asset_id] = data

            asset = StoredAsset(
                asset_id=asset_id,
                session_id=session_id,
                media_type=media_type,
                size_bytes=len(data),
                path=path,
            )
            self._index[asset_id] = asset
            self._total_bytes += len(data)
            return asset

    def get(self, asset_id: str) -> Optional[bytes]:
        with self._lock:
            asset = self._index.get(asset_id)
            if asset is None:
                return self._memory.get(asset_id)
            if asset.path and os.path.exists(asset.path):
                try:
                    with open(asset.path, "rb") as handle:
                        return handle.read()
                except OSError as exc:
                    logger.warning("Could not read asset %s: %s", asset_id, exc)
                    return None
            return self._memory.get(asset_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            removed = 0
            for asset_id, asset in list(self._index.items()):
                if asset.session_id != session_id:
                    continue
                self._index.pop(asset_id, None)
                self._memory.pop(asset_id, None)
                self._total_bytes = max(0, self._total_bytes - asset.size_bytes)
                removed += 1
            directory = os.path.join(self._root, short_hash(session_id, 16))
            shutil.rmtree(directory, ignore_errors=True)
            logger.info("Cleared %d assets for session %s", removed, short_hash(session_id, 8))
            return removed

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"assets": len(self._index), "bytes": self._total_bytes}

    # -- internals --------------------------------------------------------- #
    def _evict_unsafe(self, needed: int) -> None:
        """Drop oldest assets until ``needed`` bytes fit. Caller holds the lock."""
        freed = 0
        for asset_id, asset in list(self._index.items()):
            if freed >= needed:
                break
            self._index.pop(asset_id, None)
            self._memory.pop(asset_id, None)
            if asset.path and os.path.exists(asset.path):
                try:
                    os.remove(asset.path)
                except OSError:
                    pass
            freed += asset.size_bytes
            self._total_bytes = max(0, self._total_bytes - asset.size_bytes)
        if freed:
            logger.info("Evicted %d bytes from the file store", freed)


class MemoryFileStore(FileStore):
    """Pure in-memory store — used by tests and read-only filesystems."""

    def __init__(self) -> None:
        self._data: Dict[str, bytes] = {}
        self._meta: Dict[str, StoredAsset] = {}
        self._lock = threading.RLock()

    def put(
        self,
        session_id: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = "",
    ) -> StoredAsset:
        asset_id = stable_id(session_id, short_hash(data, 32), media_type)
        asset = StoredAsset(
            asset_id=asset_id,
            session_id=session_id,
            media_type=media_type,
            size_bytes=len(data),
        )
        with self._lock:
            self._data[asset_id] = data
            self._meta[asset_id] = asset
        return asset

    def get(self, asset_id: str) -> Optional[bytes]:
        with self._lock:
            return self._data.get(asset_id)

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            targets = [a for a, m in self._meta.items() if m.session_id == session_id]
            for asset_id in targets:
                self._data.pop(asset_id, None)
                self._meta.pop(asset_id, None)
            return len(targets)


def _suffix_for(media_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }.get(media_type, ".bin")


_default_store: Optional[FileStore] = None
_default_lock = threading.Lock()


def get_file_store(root: Optional[str] = None) -> FileStore:
    """Process-wide default store (lazily created)."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            try:
                _default_store = LocalFileStore(root)
            except OSError as exc:  # read-only filesystem
                logger.warning("Local file store unavailable (%s); using memory", exc)
                _default_store = MemoryFileStore()
        return _default_store


def set_file_store(store: FileStore) -> None:
    """Override the default store (tests, alternative backends)."""
    global _default_store
    with _default_lock:
        _default_store = store
