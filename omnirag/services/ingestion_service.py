"""Ingestion orchestration: upload -> parse -> understand -> chunk -> embed -> index.

Stage-by-stage progress is reported through a callback so any UI (Streamlit
today, a WebSocket tomorrow) can render it without this module knowing anything
about the presentation layer.

Failure policy: a document either lands fully indexed or is reported as failed
with a user-readable reason — never half-indexed silently. Within a document,
per-page and per-block failures are contained and surfaced as warnings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from omnirag.config.settings import AppSettings
from omnirag.core.enums import BlockType, FileType, IngestionStatus, PipelineStage
from omnirag.core.exceptions import (
    EmbeddingError,
    IngestionError,
    OmniRAGError,
    VectorStoreError,
)
from omnirag.core.models import Document, DocumentSummary, IngestionResult
from omnirag.ingestion.base import ProcessingContext
from omnirag.ingestion.router import DocumentRouter, get_router
from omnirag.rag.hybrid import get_bm25_cache
from omnirag.services.engine import OmniRAGEngine
from omnirag.storage.sessions import require_session_id
from omnirag.utils.hashing import content_hash, sanitize_filename, stable_id
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

ProgressFn = Callable[[PipelineStage, float, str], None]


def _noop(stage: PipelineStage, progress: float, message: str = "") -> None:
    return None


@dataclass
class ReindexReport:
    """Outcome of a re-index pass."""

    reindexed: List[str] = None  # type: ignore[assignment]
    missing_source: List[str] = None  # type: ignore[assignment]
    failed: List[tuple] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reindexed = self.reindexed or []
        self.missing_source = self.missing_source or []
        self.failed = self.failed or []

    @property
    def total(self) -> int:
        return len(self.reindexed) + len(self.missing_source) + len(self.failed)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing_source


@dataclass
class UploadedFile:
    """Transport-neutral upload (Streamlit's UploadedFile is adapted into this)."""

    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


class IngestionService:
    """Runs the full ingestion pipeline for one session."""

    def __init__(
        self,
        engine: OmniRAGEngine,
        *,
        router: Optional[DocumentRouter] = None,
    ):
        self.engine = engine
        self.router = router or get_router()

    @property
    def settings(self) -> AppSettings:
        return self.engine.settings

    # ------------------------------------------------------------------ #
    def ingest(
        self,
        session_id: str,
        upload: UploadedFile,
        *,
        progress: ProgressFn = _noop,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest one file. Never raises — failures come back in the result."""
        session_id = require_session_id(session_id)
        started = time.perf_counter()
        safe_name = sanitize_filename(upload.name)
        result = IngestionResult(filename=safe_name, status=IngestionStatus.PENDING)

        try:
            progress(PipelineStage.UPLOADING, 0.02, "Validating file")
            validation = self.router.validate(upload.name, upload.data, settings=self.settings)
            processor = self.router.route(validation.safe_filename)

            # -- deduplication ------------------------------------------ #
            digest = content_hash(upload.data)
            existing = self.engine.registry.find_by_hash(session_id, digest)
            if existing is not None and not force:
                logger.info("Skipping duplicate upload: %s", safe_name)
                result.status = IngestionStatus.DUPLICATE
                result.document_id = existing.document_id
                result.duplicate_of = existing.filename
                result.summary = existing
                result.chunk_count = existing.chunk_count
                result.page_count = existing.page_count
                result.duration_s = time.perf_counter() - started
                progress(PipelineStage.READY, 1.0, "Already indexed")
                return result

            # Same content in the same session always yields the same ids, so
            # re-indexing is idempotent rather than duplicating vectors.
            document_id = stable_id(session_id, digest)
            summary = DocumentSummary(
                document_id=document_id,
                session_id=session_id,
                filename=validation.safe_filename,
                file_type=validation.file_type,
                size_bytes=validation.size_bytes,
                content_hash=digest,
                status=IngestionStatus.PARSING,
            )
            self.engine.registry.add(summary)
            result.document_id = document_id

            # -- parse ---------------------------------------------------- #
            ctx = ProcessingContext(
                session_id=session_id,
                document_id=document_id,
                filename=validation.safe_filename,
                settings=self.settings,
                file_store=self.engine.file_store,
                ocr=self.engine.ocr,
                vision=self.engine.vision,
                handwriting=self.engine.handwriting,
                progress=progress,
                visual_budget=self.settings.vision.max_images_per_document,
            )

            # Keep the original bytes so the UI can offer a source preview.
            try:
                asset = self.engine.file_store.put(
                    session_id, upload.data, media_type=_media_type(validation.file_type)
                )
                source_asset_id = asset.asset_id
            except Exception:
                source_asset_id = None

            progress(PipelineStage.PARSING, 0.05, f"Reading {processor.display_name}")
            document = processor.parse(upload.data, ctx)
            document.content_hash = digest
            document.source_asset_id = source_asset_id
            document.size_bytes = validation.size_bytes

            summary = _summarize(document, summary)
            summary.source_asset_id = source_asset_id
            summary.status = IngestionStatus.CHUNKING
            self.engine.registry.update(summary)

            # -- chunk ---------------------------------------------------- #
            progress(PipelineStage.CHUNKING, 0.72, "Generating chunks")
            chunks = self.engine.chunker.chunk_document(document)
            if not chunks:
                raise IngestionError(
                    f"{safe_name} produced no chunks",
                    user_message=(
                        f"**{safe_name}** contained no indexable content after "
                        "processing."
                    ),
                )

            # -- embed ---------------------------------------------------- #
            progress(PipelineStage.EMBEDDING, 0.80, f"Embedding {len(chunks)} chunks")
            summary.status = IngestionStatus.EMBEDDING
            self.engine.registry.update(summary)

            embedded = self.engine.embedding_pipeline.embed_chunks(chunks)
            embeddings = self.engine.embeddings
            if getattr(embeddings, "fallback_active", False):
                ctx.warn(
                    "The configured embedding service was unavailable, so this "
                    "document uses offline hash embeddings. Search remains "
                    "available, but semantic and cross-lingual quality is reduced."
                )
            if not embedded.chunks:
                raise EmbeddingError(
                    "No chunk could be embedded",
                    user_message=(
                        "Embeddings could not be created for this document. "
                        "Check the embedding provider configuration."
                    ),
                )
            if embedded.failed:
                ctx.warn(
                    f"{len(embedded.failed)} chunk(s) could not be embedded and "
                    "are not searchable."
                )

            # -- index ---------------------------------------------------- #
            progress(PipelineStage.INDEXING, 0.92, "Indexing")
            summary.status = IngestionStatus.INDEXING
            self.engine.registry.update(summary)

            store = self.engine.vector_store
            store.ensure_collection(embedded.dimensions)
            written = store.upsert(session_id, embedded.chunks, embedded.vectors)
            get_bm25_cache().invalidate(session_id)

            # -- done ----------------------------------------------------- #
            summary.chunk_count = written
            summary.warnings = list(ctx.warnings)
            summary.status = IngestionStatus.READY
            self.engine.registry.update(summary)

            result.status = IngestionStatus.READY
            result.summary = summary
            result.chunk_count = written
            result.page_count = summary.page_count
            result.warnings = list(ctx.warnings)
            progress(PipelineStage.READY, 1.0, "Ready")

            logger.info(
                "Indexed %s: %d pages, %d chunks, %d visual blocks",
                safe_name,
                summary.page_count,
                written,
                summary.visual_block_count,
            )

        except OmniRAGError as exc:
            logger.warning("Ingestion failed for %s: %s", safe_name, exc.detail or exc)
            result.status = IngestionStatus.FAILED
            result.error = exc.user_message
            self._mark_failed(session_id, result.document_id, exc.user_message)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            logger.exception("Unexpected ingestion failure for %s", safe_name)
            message = (
                f"**{safe_name}** could not be processed "
                f"({type(exc).__name__}). Please check the file and try again."
            )
            result.status = IngestionStatus.FAILED
            result.error = message
            self._mark_failed(session_id, result.document_id, message)

        result.duration_s = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------ #
    def ingest_many(
        self,
        session_id: str,
        uploads: Sequence[UploadedFile],
        *,
        progress: Optional[Callable[[int, int, str, PipelineStage, float], None]] = None,
    ) -> List[IngestionResult]:
        """Ingest several files, reporting overall position."""
        results: List[IngestionResult] = []
        total = len(uploads)

        for position, upload in enumerate(uploads):
            def file_progress(
                stage: PipelineStage,
                value: float,
                message: str = "",
                _position: int = position,
                _name: str = upload.name,
            ) -> None:
                if progress is not None:
                    progress(_position, total, _name, stage, value)

            results.append(self.ingest(session_id, upload, progress=file_progress))
        return results

    def reindex(
        self,
        session_id: str,
        *,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> "ReindexReport":
        """Rebuild the index from the original uploaded bytes.

        Works for every document whose source bytes are still in the file store.
        On Streamlit Cloud that storage is ephemeral, so documents whose bytes
        were evicted (or that were indexed before a restart) are reported as
        ``missing_source`` rather than silently skipped — the user is told to
        re-upload those specific files.
        """
        session_id = require_session_id(session_id)
        report = ReindexReport()
        summaries = self.engine.registry.list(session_id)
        total = len(summaries)

        for position, summary in enumerate(summaries):
            if progress is not None:
                progress(position, total, summary.filename)

            data = None
            if summary.source_asset_id:
                try:
                    data = self.engine.file_store.get(summary.source_asset_id)
                except Exception as exc:
                    logger.warning("Could not read stored source bytes: %s", exc)

            if not data:
                report.missing_source.append(summary.filename)
                continue

            try:
                self.engine.vector_store.delete_document(session_id, summary.document_id)
            except Exception as exc:
                logger.warning("Could not clear old vectors for re-index: %s", exc)

            self.engine.registry.remove(session_id, summary.document_id)
            result = self.ingest(
                session_id, UploadedFile(name=summary.filename, data=data), force=True
            )
            if result.status == IngestionStatus.READY:
                report.reindexed.append(summary.filename)
            else:
                report.failed.append((summary.filename, result.error or "failed"))

        get_bm25_cache().invalidate(session_id)
        if progress is not None:
            progress(total, total, "")
        logger.info(
            "Re-index complete: %d rebuilt, %d missing source, %d failed",
            len(report.reindexed),
            len(report.missing_source),
            len(report.failed),
        )
        return report

    def remove_document(self, session_id: str, document_id: str) -> bool:
        """Delete one document's vectors and registry entry."""
        session_id = require_session_id(session_id)
        try:
            self.engine.vector_store.delete_document(session_id, document_id)
        except VectorStoreError as exc:
            logger.warning("Could not delete vectors: %s", exc.detail)
            return False
        self.engine.registry.remove(session_id, document_id)
        get_bm25_cache().invalidate(session_id)
        return True

    def clear_session(self, session_id: str) -> dict:
        return self.engine.clear_session(require_session_id(session_id))

    def _mark_failed(self, session_id: str, document_id: Optional[str], message: str) -> None:
        if not document_id:
            return
        summary = self.engine.registry.get(session_id, document_id)
        if summary is None:
            return
        summary.status = IngestionStatus.FAILED
        summary.error = message
        # Drop the dedup key so the user can retry the same file after a fix.
        summary.content_hash = ""
        self.engine.registry.update(summary)


def _summarize(document: Document, summary: DocumentSummary) -> DocumentSummary:
    blocks = document.blocks
    summary.page_count = document.page_count
    summary.block_count = len(blocks)
    summary.visual_block_count = sum(1 for b in blocks if b.has_visual)
    summary.table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
    summary.language = document.language
    summary.file_type = document.file_type
    summary.warnings = list(document.warnings)
    return summary


def _media_type(file_type: FileType) -> str:
    return {
        FileType.PDF: "application/pdf",
        FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        FileType.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        FileType.IMAGE: "image/png",
        FileType.TXT: "text/plain",
        FileType.MARKDOWN: "text/markdown",
    }.get(file_type, "application/octet-stream")


__all__ = ["IngestionService", "ReindexReport", "UploadedFile", "ProgressFn"]
