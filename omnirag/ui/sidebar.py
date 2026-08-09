"""Sidebar: upload, document library, selection, settings, session controls."""

from __future__ import annotations

import html
from typing import List, Optional

import streamlit as st

from omnirag.core.enums import IngestionStatus, PipelineStage
from omnirag.core.models import DocumentSummary, IngestionResult
from omnirag.ingestion.router import get_router
from omnirag.services.ingestion_service import UploadedFile
from omnirag.ui import state
from omnirag.ui.components import (
    brand,
    caption,
    divider,
    document_meta_line,
    empty_state,
    file_icon,
    pill,
    status_pill,
)
from omnirag.utils.hashing import short_hash
from omnirag.utils.logging import get_logger
from omnirag.utils.user_messages import (
    SERVICE_NOT_CONFIGURED,
    public_error_text,
    public_processing_note,
)

logger = get_logger(__name__)

STAGE_ORDER = [
    PipelineStage.UPLOADING,
    PipelineStage.PARSING,
    PipelineStage.EXTRACTING_TEXT,
    PipelineStage.ANALYZING_VISUALS,
    PipelineStage.CHUNKING,
    PipelineStage.EMBEDDING,
    PipelineStage.INDEXING,
    PipelineStage.READY,
]


def render() -> None:
    """Draw the whole sidebar."""
    with st.sidebar:
        brand()
        divider()
        _render_uploader()
        _render_library()
        divider()
        _render_actions()
        _render_settings()
        _render_footer()


# --------------------------------------------------------------------------- #
def _render_uploader() -> None:
    router = get_router()
    settings = state.settings()

    st.markdown("##### Upload documents")
    uploads = st.file_uploader(
        "Drop files here",
        type=router.supported_extensions(),
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="omnirag_uploader",
        help=(
            f"Supported: {router.supported_label()} · "
            f"up to {settings.upload.max_upload_mb:.0f} MB per file"
        ),
    )

    if not uploads:
        return

    pending = []
    for upload in uploads:
        try:
            data = upload.getvalue()
        except Exception as exc:
            logger.warning("Could not read an upload: %s", exc)
            continue
        key = f"{upload.name}:{len(data)}:{short_hash(data, 12)}"
        if state.already_processed(key):
            continue
        pending.append((key, UploadedFile(name=upload.name, data=data)))

    if not pending:
        return

    existing = len(state.documents())
    allowed = max(0, settings.upload.max_files - existing)
    if allowed <= 0:
        st.warning(
            f"Document limit reached ({settings.upload.max_files}). "
            "Remove a document before uploading more."
        )
        return
    if len(pending) > allowed:
        st.warning(f"Only the first {allowed} file(s) will be processed.")
        pending = pending[:allowed]

    _process_uploads(pending)


def _process_uploads(pending: List[tuple[str, UploadedFile]]) -> None:
    """Run ingestion with a live pipeline-progress display."""
    service = state.ingestion_service()
    session_id = state.session_id()
    results: List[IngestionResult] = []

    with st.status(f"Processing {len(pending)} file(s)…", expanded=True) as status_box:
        progress_bar = st.progress(0.0)
        for position, (key, upload) in enumerate(pending):
            label = st.empty()
            label.markdown(f"**{html.escape(upload.name)}**")
            stage_text = st.empty()

            def on_progress(
                stage: PipelineStage,
                value: float,
                message: str = "",
                _pos: int = position,
            ) -> None:
                overall = (_pos + max(0.0, min(1.0, value))) / len(pending)
                progress_bar.progress(min(1.0, overall))
                suffix = f" — {message}" if message else ""
                stage_text.caption(f"{stage.value}{suffix}")

            result = service.ingest(session_id, upload, progress=on_progress)
            results.append(result)
            state.mark_processed(key)

            if result.status == IngestionStatus.READY:
                stage_text.caption(
                    f"✅ Ready — {result.page_count} page(s), {result.chunk_count} chunks"
                )
            elif result.status == IngestionStatus.DUPLICATE:
                stage_text.caption("↩︎ Already indexed in this session")
            else:
                stage_text.caption(
                    "❌ " + public_error_text(
                        result.error or "Processing failed",
                        debug=state.settings().debug_generation,
                    )
                )

        progress_bar.progress(1.0)
        failed = [r for r in results if r.status == IngestionStatus.FAILED]
        ready = [r for r in results if r.status == IngestionStatus.READY]

        if failed and not ready:
            status_box.update(label="Processing failed", state="error")
        elif failed:
            status_box.update(
                label=f"{len(ready)} indexed, {len(failed)} failed", state="error"
            )
        else:
            status_box.update(label=f"{len(results)} file(s) ready", state="complete")

    for result in results:
        if result.status == IngestionStatus.FAILED and result.error:
            st.error(public_error_text(
                result.error, debug=state.settings().debug_generation
            ))
        for warning in result.warnings[:3]:
            st.info(public_processing_note(
                warning, debug=state.settings().debug_generation
            ), icon="ℹ️")

    st.rerun()


# --------------------------------------------------------------------------- #
def _render_library() -> None:
    documents = state.documents()
    st.markdown("##### Document library")

    if not documents:
        empty_state("📚", "No documents yet", "Upload a file to start asking questions.")
        return

    ready = [d for d in documents if d.status == IngestionStatus.READY]
    selected = state.selected_document_ids()
    active_ids = set(selected) if selected is not None else {d.document_id for d in ready}

    for summary in documents:
        _render_document_row(summary, is_active=summary.document_id in active_ids)

    if len(ready) > 1:
        divider()
        _render_selector(ready, selected)


def _render_document_row(summary: DocumentSummary, *, is_active: bool) -> None:
    columns = st.columns([0.84, 0.16], gap="small")
    with columns[0]:
        marker = "" if is_active or summary.status != IngestionStatus.READY else "  ·  muted"
        st.markdown(
            f"""
            <div class="omni-doc">
              <div class="omni-doc-name">{file_icon(summary)} {html.escape(summary.filename)}</div>
              <div class="omni-doc-meta">{html.escape(document_meta_line(summary))}{marker}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        badges = [status_pill(summary.status)]
        if summary.status == IngestionStatus.READY and not is_active:
            badges.append(pill("not in chat", "muted"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

    with columns[1]:
        if st.button(
            "✕",
            key=f"remove_{summary.document_id}",
            help="Remove this document",
            use_container_width=True,
        ):
            _remove_document(summary)

    if summary.error:
        st.caption("⚠️ " + public_error_text(
            summary.error, debug=state.settings().debug_generation
        ))
    elif summary.warnings:
        with st.expander(f"{len(summary.warnings)} processing note(s)", expanded=False):
            for warning in summary.warnings:
                st.caption("• " + public_processing_note(
                    warning, debug=state.settings().debug_generation
                ))
    st.write("")


def _remove_document(summary: DocumentSummary) -> None:
    service = state.ingestion_service()
    if service.remove_document(state.session_id(), summary.document_id):
        st.toast(f"Removed {summary.filename}")
    else:
        st.toast("Could not remove the document", icon="⚠️")
    st.rerun()


def _render_selector(ready: List[DocumentSummary], selected: Optional[List[str]]) -> None:
    """Choose which documents participate in the conversation."""
    by_name = {d.filename: d.document_id for d in ready}
    default = (
        list(by_name)
        if selected is None
        else [name for name, doc_id in by_name.items() if doc_id in set(selected)]
    )

    chosen = st.multiselect(
        "Documents in this chat",
        options=list(by_name),
        default=default,
        key="omnirag_doc_selector",
        help="Retrieval is restricted to the selected documents.",
    )

    chosen_ids = [by_name[name] for name in chosen]
    if not chosen_ids or len(chosen_ids) == len(by_name):
        state.set_selected_documents(None)
    else:
        state.set_selected_documents(chosen_ids)


# --------------------------------------------------------------------------- #
def _render_actions() -> None:
    columns = st.columns(2, gap="small")
    with columns[0]:
        if st.button("💬 New chat", use_container_width=True, help="Clear the conversation, keep documents"):
            state.new_chat()
            st.rerun()
    with columns[1]:
        if st.button("🗑️ Clear all", use_container_width=True, help="Remove every document and message"):
            state.reset_session()
            st.rerun()

    if state.documents():
        if st.button(
            "🔄 Re-index documents",
            use_container_width=True,
            help="Rebuild the search index from the uploaded files",
        ):
            _reindex()


def _reindex() -> None:
    """Rebuild the index from the stored original bytes."""
    service = state.ingestion_service()

    with st.status("Re-indexing…", expanded=True) as box:
        progress_bar = st.progress(0.0)
        label = st.empty()

        def on_progress(position: int, total: int, filename: str) -> None:
            progress_bar.progress(min(1.0, position / max(1, total)))
            if filename:
                label.caption(f"Rebuilding {filename} ({position + 1}/{total})")

        report = service.reindex(state.session_id(), progress=on_progress)
        progress_bar.progress(1.0)
        label.empty()

        if report.ok and report.reindexed:
            box.update(label=f"Re-indexed {len(report.reindexed)} document(s)", state="complete")
        elif report.reindexed:
            box.update(
                label=f"Re-indexed {len(report.reindexed)}, {len(report.missing_source) + len(report.failed)} need attention",
                state="error",
            )
        else:
            box.update(label="Nothing could be re-indexed", state="error")

    if report.missing_source:
        st.warning(
            "The original file is no longer in this session's temporary storage "
            "for: " + ", ".join(report.missing_source) + ". Please upload them again.",
            icon="⚠️",
        )
        state.forget_processed()
    for filename, error in report.failed:
        safe_error = public_error_text(
            str(error), debug=state.settings().debug_generation
        )
        st.error(f"{filename}: {safe_error}")

    st.rerun()


# --------------------------------------------------------------------------- #
def _render_settings() -> None:
    engine = state.engine()
    status = engine.status()

    diagnostic_mode = state.settings().debug_generation
    label = "⚙️ Diagnostics" if diagnostic_mode else "⚙️ System status"
    with st.expander(label, expanded=not status.ready):
        if not status.ready:
            if diagnostic_mode:
                for issue in status.issues:
                    st.error(issue, icon="🔑")
            else:
                st.error(SERVICE_NOT_CONFIGURED, icon="🔑")
        else:
            st.success("Ready")

        if diagnostic_mode:
            st.markdown("**Internal services**")
            st.markdown(
                "\n".join(
                    [
                        f"- **LLM chain:** `{status.llm_chain}`",
                        f"- **Embeddings:** `{status.embedding_provider}` / `{status.embedding_model}`",
                        f"- **Vector store:** `{status.vector_store}`",
                        f"- **Reranker:** `{status.reranker}`",
                        f"- **OCR:** `{status.ocr_provider}`",
                        f"- **Vision:** {'available' if status.vision_available else 'unavailable'}",
                    ]
                )
            )
            stats = engine.provider_stats()
            if stats.get("calls"):
                st.markdown("**Provider usage**")
                usage = ", ".join(
                    f"`{key}`: {value}" for key, value in stats["by_provider"].items()
                )
                st.caption(f"{stats['calls']} call(s) — {usage}")
                if stats.get("failovers"):
                    st.caption(f"↪︎ {stats['failovers']} failover(s)")
            for warning in status.warnings:
                st.warning(warning, icon="⚠️")
        else:
            st.caption(
                "Document search is available."
                if status.ready
                else "Document search is waiting for administrator configuration."
            )
            st.caption(
                "Visual analysis is available."
                if status.vision_available
                else "Visual analysis is currently unavailable."
            )
            settings = state.settings()
            if settings.embedding.provider == "hash":
                st.warning(
                    "Document search is running in reduced-quality mode.", icon="⚠️"
                )
            if not settings.vector_store.use_qdrant:
                st.caption("The document index resets when the app restarts.")


def _render_footer() -> None:
    divider()
    caption(
        "Documents stay in this browser session only. The AI service receives "
        "only the content needed to answer your questions."
    )


__all__ = ["render"]
