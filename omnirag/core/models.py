"""Canonical internal representation used by every stage of the pipeline.

Traceability contract (enforced by tests):

    Answer -> Citation -> SearchResult -> Chunk -> ContentBlock -> Page -> Document

Every object in that chain carries stable identifiers, so a citation rendered in
the UI can always be resolved back to the exact source region of the original
upload.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnirag.core.enums import (
    BlockType,
    FileType,
    IngestionStatus,
    Language,
    Role,
    SourceKind,
)


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class OmniModel(BaseModel):
    """Base model with shared configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False)


# --------------------------------------------------------------------------- #
# Geometry & visuals
# --------------------------------------------------------------------------- #
class BoundingBox(OmniModel):
    """Axis-aligned box in PDF/page point coordinates (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class VisualRef(OmniModel):
    """Pointer to the original visual backing a block.

    The bytes themselves live in the :class:`~omnirag.storage.files.FileStore`
    so that models stay lightweight and JSON-serialisable. ``asset_id`` is the
    key used to fetch them again at answer time — this is what makes the
    "send the original image to the multimodal LLM" rule possible.
    """

    asset_id: str
    media_type: str = "image/png"
    width: Optional[int] = None
    height: Optional[int] = None
    origin: str = "embedded"  # embedded | page_render | crop | upload
    page_number: Optional[int] = None


class TableData(OmniModel):
    """A table kept in several representations at once (structure is preserved)."""

    rows: List[List[str]] = Field(default_factory=list)
    header: Optional[List[str]] = None
    markdown: str = ""
    summary: str = ""
    n_rows: int = 0
    n_cols: int = 0

    @classmethod
    def from_rows(
        cls, rows: Sequence[Sequence[Any]], *, has_header: bool = True
    ) -> "TableData":
        clean: List[List[str]] = [
            ["" if c is None else str(c).strip() for c in row] for row in rows
        ]
        clean = [r for r in clean if any(cell for cell in r)]
        header = clean[0] if (has_header and clean) else None
        body = clean[1:] if header is not None else clean
        n_cols = max((len(r) for r in clean), default=0)
        return cls(
            rows=body,
            header=header,
            markdown=_rows_to_markdown(header, body, n_cols),
            n_rows=len(body),
            n_cols=n_cols,
        )


def _rows_to_markdown(
    header: Optional[List[str]], body: List[List[str]], n_cols: int
) -> str:
    if n_cols == 0:
        return ""

    def pad(row: Sequence[str]) -> List[str]:
        cells = [str(c).replace("|", "\\|").replace("\n", " ").strip() for c in row]
        return cells + [""] * (n_cols - len(cells))

    head = pad(header) if header else [f"col_{i + 1}" for i in range(n_cols)]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
    lines.extend("| " + " | ".join(pad(r)) + " |" for r in body)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
class ContentBlock(OmniModel):
    """The atomic unit produced by ingestion.

    A block always has *some* retrievable text (``search_text``) and optionally
    keeps a reference to the original visual it was derived from.
    """

    block_id: str = Field(default_factory=_uuid)
    document_id: str
    session_id: str
    page_number: int = 1
    block_type: BlockType = BlockType.TEXT
    source_kind: SourceKind = SourceKind.DIGITAL

    text: str = ""
    visual_description: str = ""
    table: Optional[TableData] = None
    visual: Optional[VisualRef] = None
    bbox: Optional[BoundingBox] = None

    language: Language = Language.UNKNOWN
    confidence: Optional[float] = None
    uncertain: bool = False
    parent_section: Optional[str] = None
    order: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))

    @property
    def search_text(self) -> str:
        """Text used for embedding / keyword search.

        Visual blocks contribute their generated description; tables contribute
        their markdown serialisation plus summary. Nothing is thrown away.
        """
        parts: List[str] = []
        if self.parent_section:
            parts.append(self.parent_section)
        if self.text.strip():
            parts.append(self.text.strip())
        if self.visual_description.strip():
            parts.append(self.visual_description.strip())
        if self.table is not None:
            if self.table.summary:
                parts.append(self.table.summary)
            if self.table.markdown:
                parts.append(self.table.markdown)
        return "\n\n".join(parts).strip()

    @property
    def has_visual(self) -> bool:
        return self.visual is not None

    @property
    def is_empty(self) -> bool:
        return not self.search_text


class Page(OmniModel):
    """A page (PDF), slide (PPTX), or logical section (DOCX/TXT)."""

    page_id: str = Field(default_factory=_uuid)
    document_id: str
    session_id: str
    page_number: int
    label: Optional[str] = None          # e.g. "Slide 4", "Page 17"
    width: Optional[float] = None
    height: Optional[float] = None
    is_scanned: bool = False
    blocks: List[ContentBlock] = Field(default_factory=list)
    page_image: Optional[VisualRef] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.label or f"Page {self.page_number}"

    @property
    def text(self) -> str:
        return "\n\n".join(b.search_text for b in self.blocks if b.search_text)


class Document(OmniModel):
    """A single uploaded file after parsing."""

    document_id: str = Field(default_factory=_uuid)
    session_id: str
    filename: str
    file_type: FileType = FileType.UNKNOWN
    content_hash: str = ""
    size_bytes: int = 0
    page_count: int = 0
    language: Language = Language.UNKNOWN
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: datetime = Field(default_factory=_now)
    pages: List[Page] = Field(default_factory=list)
    source_asset_id: Optional[str] = None   # original bytes in the FileStore
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def blocks(self) -> List[ContentBlock]:
        return [b for p in self.pages for b in p.blocks]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def page(self, number: int) -> Optional[Page]:
        return next((p for p in self.pages if p.page_number == number), None)

    def block(self, block_id: str) -> Optional[ContentBlock]:
        return next((b for b in self.blocks if b.block_id == block_id), None)


class DocumentSummary(OmniModel):
    """Lightweight view of a document for the sidebar / library.

    Kept separate from :class:`Document` so the UI never has to hold full page
    content (and full images) in session state.
    """

    document_id: str
    session_id: str
    filename: str
    file_type: FileType
    size_bytes: int = 0
    page_count: int = 0
    chunk_count: int = 0
    block_count: int = 0
    visual_block_count: int = 0
    table_count: int = 0
    language: Language = Language.UNKNOWN
    status: IngestionStatus = IngestionStatus.PENDING
    content_hash: str = ""
    #: Key of the original uploaded bytes in the FileStore. Present while the
    #: (ephemeral) workspace still holds them; this is what makes re-indexing
    #: possible without asking the user to upload the file again.
    source_asset_id: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    @property
    def size_label(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# Chunks & retrieval
# --------------------------------------------------------------------------- #
class Chunk(OmniModel):
    """A retrievable unit. Always traceable to its source blocks."""

    chunk_id: str = Field(default_factory=_uuid)
    document_id: str
    session_id: str
    filename: str
    file_type: FileType = FileType.UNKNOWN
    page_number: int = 1
    page_label: str = ""
    block_ids: List[str] = Field(default_factory=list)
    block_type: BlockType = BlockType.TEXT
    source_kind: SourceKind = SourceKind.DIGITAL
    text: str = ""
    section: Optional[str] = None
    language: Language = Language.UNKNOWN
    visual: Optional[VisualRef] = None
    confidence: Optional[float] = None
    uncertain: bool = False
    token_estimate: int = 0
    order: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("block_ids")
    @classmethod
    def _require_provenance(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("chunk must reference at least one source block_id")
        return v

    @property
    def citation_label(self) -> str:
        return f"[{self.filename} — {self.page_label or f'Page {self.page_number}'}]"

    def to_payload(self) -> Dict[str, Any]:
        """Flat payload stored next to the vector. Used for metadata filtering."""
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": str(self.file_type),
            "page_number": self.page_number,
            "page_label": self.page_label,
            "block_ids": list(self.block_ids),
            "block_type": str(self.block_type),
            "source_kind": str(self.source_kind),
            "language": str(self.language),
            "text": self.text,
            "section": self.section,
            "confidence": self.confidence,
            "uncertain": self.uncertain,
            "order": self.order,
            "token_estimate": self.token_estimate,
            "visual_asset_id": self.visual.asset_id if self.visual else None,
            "visual_media_type": self.visual.media_type if self.visual else None,
            "visual_origin": self.visual.origin if self.visual else None,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Chunk":
        visual = None
        if payload.get("visual_asset_id"):
            visual = VisualRef(
                asset_id=payload["visual_asset_id"],
                media_type=payload.get("visual_media_type") or "image/png",
                origin=payload.get("visual_origin") or "embedded",
                page_number=payload.get("page_number"),
            )
        return cls(
            chunk_id=payload["chunk_id"],
            session_id=payload["session_id"],
            document_id=payload["document_id"],
            filename=payload.get("filename", ""),
            file_type=FileType(payload.get("file_type", "unknown")),
            page_number=int(payload.get("page_number", 1)),
            page_label=payload.get("page_label", ""),
            block_ids=list(payload.get("block_ids") or ["unknown"]),
            block_type=BlockType(payload.get("block_type", "text")),
            source_kind=SourceKind(payload.get("source_kind", "digital")),
            language=Language(payload.get("language", "unknown")),
            text=payload.get("text", ""),
            section=payload.get("section"),
            confidence=payload.get("confidence"),
            uncertain=bool(payload.get("uncertain", False)),
            order=int(payload.get("order", 0)),
            token_estimate=int(payload.get("token_estimate", 0)),
            visual=visual,
        )


class SearchResult(OmniModel):
    """A chunk with the scores that got it there."""

    chunk: Chunk
    score: float = 0.0
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None
    strategy: str = "vector"
    rank: int = 0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class RetrievalResult(OmniModel):
    """Everything the retrieval pipeline produced for one query."""

    query: str
    normalized_query: str = ""
    expanded_queries: List[str] = Field(default_factory=list)
    language: Language = Language.UNKNOWN
    results: List[SearchResult] = Field(default_factory=list)
    strategy: str = "hybrid"
    reranked: bool = False
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    query_scope: str = "FOCUSED"
    candidate_count: int = 0
    unique_pages: int = 0
    total_pages: int = 0
    structured_matches: int = 0
    completeness_pass: bool = False

    @property
    def chunks(self) -> List[Chunk]:
        return [r.chunk for r in self.results]

    @property
    def is_empty(self) -> bool:
        return not self.results


class Citation(OmniModel):
    """A source shown under an answer. Index is 1-based, matching ``[1]`` markers."""

    index: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    page_label: str = ""
    block_type: BlockType = BlockType.TEXT
    source_kind: SourceKind = SourceKind.DIGITAL
    snippet: str = ""
    score: float = 0.0
    visual_asset_id: Optional[str] = None
    visual_media_type: Optional[str] = None
    uncertain: bool = False
    confidence: Optional[float] = None

    @property
    def label(self) -> str:
        return f"[{self.filename} — {self.page_label or f'Page {self.page_number}'}]"


class ChatMessage(OmniModel):
    """One turn of the conversation."""

    message_id: str = Field(default_factory=_uuid)
    role: Role
    content: str
    created_at: datetime = Field(default_factory=_now)
    citations: List[Citation] = Field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    used_documents: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    debug: Dict[str, Any] = Field(default_factory=dict)
    reply_to_message_id: Optional[str] = None


class AnswerResult(OmniModel):
    """Output of the generation stage."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    insufficient_evidence: bool = False
    model: str = ""
    used_images: int = 0
    usage: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    finish_reason: str = ""
    continued: bool = False


class IngestionResult(OmniModel):
    """Outcome of ingesting a single uploaded file."""

    filename: str
    document_id: Optional[str] = None
    status: IngestionStatus = IngestionStatus.PENDING
    summary: Optional[DocumentSummary] = None
    chunk_count: int = 0
    page_count: int = 0
    duplicate_of: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in (IngestionStatus.READY, IngestionStatus.DUPLICATE)


class SessionInfo(OmniModel):
    """Namespace metadata for one anonymous Streamlit session."""

    session_id: str
    created_at: datetime = Field(default_factory=_now)
    last_active: datetime = Field(default_factory=_now)
    document_count: int = 0
    chunk_count: int = 0


def blocks_of(pages: Iterable[Page]) -> List[ContentBlock]:
    return [b for p in pages for b in p.blocks]
