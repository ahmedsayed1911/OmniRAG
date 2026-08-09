"""Reusable presentational widgets.

Pure rendering: these functions never touch the engine, never mutate state, and
never raise. Anything that needs data receives it as an argument.
"""

from __future__ import annotations

import html
from typing import Iterable, List, Optional

import streamlit as st

from omnirag.core.enums import BlockType, IngestionStatus, Language, SourceKind
from omnirag.core.models import DocumentSummary

# --------------------------------------------------------------------------- #
STATUS_STYLE = {
    IngestionStatus.READY: ("ok", "Ready"),
    IngestionStatus.FAILED: ("err", "Failed"),
    IngestionStatus.DUPLICATE: ("muted", "Duplicate"),
    IngestionStatus.PENDING: ("info", "Queued"),
    IngestionStatus.PARSING: ("info", "Parsing"),
    IngestionStatus.ANALYZING: ("info", "Analysing"),
    IngestionStatus.CHUNKING: ("info", "Chunking"),
    IngestionStatus.EMBEDDING: ("info", "Embedding"),
    IngestionStatus.INDEXING: ("info", "Indexing"),
}

FILE_ICON = {
    "pdf": "📄",
    "docx": "📝",
    "pptx": "📊",
    "txt": "📃",
    "md": "📑",
    "image": "🖼️",
    "unknown": "📁",
}

BLOCK_LABEL = {
    BlockType.TEXT: "Text",
    BlockType.HEADING: "Heading",
    BlockType.OCR_TEXT: "Scanned text",
    BlockType.HANDWRITING: "Handwriting",
    BlockType.IMAGE: "Image",
    BlockType.TABLE: "Table",
    BlockType.CHART: "Chart",
    BlockType.DIAGRAM: "Diagram",
    BlockType.CAPTION: "Caption",
    BlockType.SPEAKER_NOTES: "Speaker notes",
    BlockType.PAGE_SNAPSHOT: "Page scan",
}

SOURCE_LABEL = {
    SourceKind.DIGITAL: "digital text",
    SourceKind.OCR: "OCR",
    SourceKind.VISION: "vision model",
    SourceKind.STRUCTURED: "parsed structure",
    SourceKind.DERIVED: "derived",
}

LANGUAGE_LABEL = {
    Language.ARABIC: "العربية",
    Language.ENGLISH: "English",
    Language.MIXED: "AR/EN",
    Language.UNKNOWN: "",
}


# --------------------------------------------------------------------------- #
def brand(title: str = "OmniRAG", subtitle: str = "Multimodal document intelligence") -> None:
    st.markdown(
        f"""
        <div class="omni-brand">
          <div class="omni-brand-mark">OR</div>
          <div class="omni-brand-text">
            <span class="omni-brand-title">{html.escape(title)}</span>
            <span class="omni-brand-sub">{html.escape(subtitle)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "info") -> str:
    """Return the markup for an inline status pill."""
    tone = tone if tone in ("ok", "warn", "err", "info", "muted") else "info"
    return f'<span class="omni-pill omni-pill-{tone}">{html.escape(str(text))}</span>'


def pills(items: Iterable[tuple[str, str]]) -> None:
    markup = " ".join(pill(text, tone) for text, tone in items)
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def render_error(message: str, *, title: str = "") -> None:
    """User-facing error. Never shows a traceback — those go to the logs."""
    st.error(f"**{title}**\n\n{message}" if title else message)


def render_warnings(warnings: List[str], *, expanded: bool = False) -> None:
    if not warnings:
        return
    label = f"⚠️ {len(warnings)} note{'s' if len(warnings) > 1 else ''} about processing"
    with st.expander(label, expanded=expanded):
        for warning in warnings:
            st.markdown(f"- {warning}")


def status_pill(status: IngestionStatus) -> str:
    tone, label = STATUS_STYLE.get(status, ("info", str(status).title()))
    return pill(label, tone)


def file_icon(summary: DocumentSummary) -> str:
    return FILE_ICON.get(str(summary.file_type), FILE_ICON["unknown"])


def document_meta_line(summary: DocumentSummary) -> str:
    """`PDF · 2.4 MB · 18 pages · 42 chunks · English`"""
    parts: List[str] = [str(summary.file_type).upper(), summary.size_label]
    if summary.page_count:
        unit = "slide" if str(summary.file_type) == "pptx" else "page"
        parts.append(f"{summary.page_count} {unit}{'s' if summary.page_count != 1 else ''}")
    if summary.chunk_count:
        parts.append(f"{summary.chunk_count} chunks")
    if summary.visual_block_count:
        parts.append(f"{summary.visual_block_count} visuals")
    if summary.table_count:
        parts.append(f"{summary.table_count} tables")
    language = LANGUAGE_LABEL.get(summary.language, "")
    if language:
        parts.append(language)
    return " · ".join(parts)


def rtl_markdown(text: str) -> None:
    """Render the exact model text without passing it through unsafe HTML."""
    st.markdown(text)


def caption(text: str) -> None:
    st.markdown(f'<div class="omni-caption">{html.escape(text)}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="omni-divider"></div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.6rem 0;opacity:.72;">
          <div style="font-size:1.7rem;margin-bottom:.4rem;">{icon}</div>
          <div style="font-weight:620;font-size:.92rem;margin-bottom:.25rem;">{html.escape(title)}</div>
          <div style="font-size:.79rem;opacity:.72;">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def provider_badge(provider: str, model: str, *, fallback_used: bool = False) -> None:
    """Small indicator showing which provider/model produced the answer."""
    if not provider and not model:
        return
    tone = "warn" if fallback_used else "muted"
    label = f"{provider} · {model}" if provider and model else (provider or model)
    suffix = " (fallback)" if fallback_used else ""
    st.markdown(pill(f"{label}{suffix}", tone), unsafe_allow_html=True)


__all__ = [
    "BLOCK_LABEL",
    "SOURCE_LABEL",
    "brand",
    "caption",
    "divider",
    "document_meta_line",
    "empty_state",
    "file_icon",
    "pill",
    "pills",
    "provider_badge",
    "render_error",
    "render_warnings",
    "rtl_markdown",
    "status_pill",
]
