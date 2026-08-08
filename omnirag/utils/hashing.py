"""Content hashing, stable IDs, and filename sanitisation."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import uuid

_UNSAFE = re.compile(r"[^\w\s.\-()؀-ۿ]", re.UNICODE)
_WS = re.compile(r"\s+")
_NAMESPACE = uuid.UUID("6f5c1c8e-0d9e-4f0a-9d3d-2f3f6f9a1b77")


def content_hash(data: bytes) -> str:
    """SHA-256 of raw bytes — the deduplication key."""
    return hashlib.sha256(data).hexdigest()


def short_hash(data: bytes | str, length: int = 12) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()[:length]


def text_hash(text: str) -> str:
    """Hash of normalised text — used to dedupe identical chunks/captions."""
    normalized = _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_id(*parts: str) -> str:
    """Deterministic UUID5 hex from the given parts.

    Used for document/chunk ids so that re-ingesting identical content in the
    same session yields identical identifiers (idempotent indexing).
    """
    name = "|".join(str(p) for p in parts)
    return uuid.uuid5(_NAMESPACE, name).hex


def stable_uuid(*parts: str) -> str:
    """Same as :func:`stable_id` but in canonical UUID form (Qdrant point ids)."""
    return str(uuid.uuid5(_NAMESPACE, "|".join(str(p) for p in parts)))


def sanitize_filename(filename: str, *, max_length: int = 120) -> str:
    """Make an uploaded filename safe to use as a display name and path part.

    Strips directory components (defeats ``../`` traversal), control characters
    and shell-hostile symbols, while preserving Arabic letters and spaces.
    """
    if not filename:
        return "untitled"

    name = unicodedata.normalize("NFKC", filename)
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = _UNSAFE.sub("_", name)
    name = _WS.sub(" ", name).strip(" .")

    if not name:
        return "untitled"

    root, ext = os.path.splitext(name)
    ext = ext[:12]
    root = root[: max(1, max_length - len(ext))] or "untitled"

    # Reserved device names on Windows.
    if root.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        root = f"_{root}"
    return f"{root}{ext}"


def file_extension(filename: str) -> str:
    """Lower-case extension without the dot (``"report.PDF"`` -> ``"pdf"``)."""
    return os.path.splitext(filename)[1].lower().lstrip(".")
