"""Structure-aware chunking.

Not fixed-size character slicing. The chunker respects the structure ingestion
already recovered:

* **atomic blocks** — tables, charts, diagrams, images and handwriting are never
  split. Cutting a table in half destroys the numeric relationships that make it
  answerable, and half a chart description is worse than none.
* **section grouping** — consecutive text blocks are accumulated under their
  heading, and the heading is prepended to every chunk it covers so a chunk
  retrieved in isolation still knows what it is about.
* **page boundaries** — a chunk never spans two pages, because a citation must
  point at exactly one page.
* **soft boundaries** — when a section is longer than the budget, the split
  falls on a paragraph break, else a sentence break, and only as a last resort
  mid-text.
* **overlap** — carried over on sentence boundaries so a fact split across the
  seam is still retrievable from both sides.

Citation invariant (covered by tests): every chunk carries at least one
``block_id``, plus filename, page number and page label. A chunk that cannot be
traced back to its source is never produced — :class:`Chunk` itself rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from omnirag.config.settings import ChunkingSettings
from omnirag.core.enums import BlockType, Language, SourceKind
from omnirag.core.models import Chunk, ContentBlock, Document, Page
from omnirag.utils.hashing import stable_id
from omnirag.utils.language import detect_language
from omnirag.utils.logging import get_logger
from omnirag.utils.text import estimate_tokens, split_paragraphs, split_sentences

logger = get_logger(__name__)

#: Block types that must never be split or merged with neighbours.
ATOMIC_BLOCK_TYPES = frozenset(
    {
        BlockType.TABLE,
        BlockType.CHART,
        BlockType.DIAGRAM,
        BlockType.IMAGE,
        BlockType.HANDWRITING,
        BlockType.PAGE_SNAPSHOT,
    }
)


@dataclass
class _Accumulator:
    """Text being gathered for the next chunk within one page + section."""

    parts: List[str]
    block_ids: List[str]
    section: Optional[str]
    page: Page
    source_kind: SourceKind
    confidence: Optional[float] = None
    uncertain: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if p.strip()).strip()

    @property
    def length(self) -> int:
        return len(self.text)


class Chunker:
    """Turns a parsed :class:`Document` into retrievable :class:`Chunk` objects."""

    def __init__(self, settings: Optional[ChunkingSettings] = None):
        self.settings = settings or ChunkingSettings()

    # ------------------------------------------------------------------ #
    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in document.pages:
            chunks.extend(self.chunk_page(document, page))

        for order, chunk in enumerate(chunks):
            chunk.order = order
        logger.info(
            "Chunked %s: %d pages -> %d chunks", document.filename, len(document.pages), len(chunks)
        )
        return chunks

    def chunk_page(self, document: Document, page: Page) -> List[Chunk]:
        chunks: List[Chunk] = []
        accumulator: Optional[_Accumulator] = None
        section: Optional[str] = None

        for block in sorted(page.blocks, key=lambda b: b.order):
            if block.is_empty:
                continue

            # --- atomic blocks --------------------------------------- #
            if block.block_type in ATOMIC_BLOCK_TYPES:
                if accumulator is not None:
                    chunks.extend(self._flush(document, accumulator))
                    accumulator = None
                chunks.append(self._atomic_chunk(document, page, block, section))
                continue

            # --- headings start a new section ------------------------ #
            if block.block_type == BlockType.HEADING:
                if accumulator is not None:
                    chunks.extend(self._flush(document, accumulator))
                    accumulator = None
                section = block.text.strip() or section
                # The heading text itself joins the next chunk rather than
                # becoming a near-empty chunk of its own.
                accumulator = _Accumulator(
                    parts=[block.text],
                    block_ids=[block.block_id],
                    section=section,
                    page=page,
                    source_kind=block.source_kind,
                )
                continue

            # --- ordinary text --------------------------------------- #
            if accumulator is None:
                accumulator = _Accumulator(
                    parts=[],
                    block_ids=[],
                    section=section or block.parent_section,
                    page=page,
                    source_kind=block.source_kind,
                )

            text = block.search_text
            if accumulator.length + len(text) > self.settings.max_chunk_size and accumulator.parts:
                chunks.extend(self._flush(document, accumulator))
                accumulator = _Accumulator(
                    parts=[],
                    block_ids=[],
                    section=section or block.parent_section,
                    page=page,
                    source_kind=block.source_kind,
                )

            accumulator.parts.append(text)
            accumulator.block_ids.append(block.block_id)
            if block.uncertain:
                accumulator.uncertain = True
            if block.confidence is not None:
                accumulator.confidence = (
                    block.confidence
                    if accumulator.confidence is None
                    else min(accumulator.confidence, block.confidence)
                )
            if block.source_kind == SourceKind.OCR:
                accumulator.source_kind = SourceKind.OCR

            if accumulator.length >= self.settings.chunk_size:
                chunks.extend(self._flush(document, accumulator))
                accumulator = None

        if accumulator is not None:
            chunks.extend(self._flush(document, accumulator))
        return chunks

    # ------------------------------------------------------------------ #
    def _atomic_chunk(
        self, document: Document, page: Page, block: ContentBlock, section: Optional[str]
    ) -> Chunk:
        """One block -> one chunk, keeping its visual reference intact."""
        text = block.search_text
        prefix = section or block.parent_section
        if prefix and prefix not in text:
            text = f"{prefix}\n\n{text}"

        return Chunk(
            chunk_id=stable_id(document.document_id, block.block_id, "chunk"),
            document_id=document.document_id,
            session_id=document.session_id,
            filename=document.filename,
            file_type=document.file_type,
            page_number=page.page_number,
            page_label=page.display_label,
            block_ids=[block.block_id],
            block_type=block.block_type,
            source_kind=block.source_kind,
            text=text,
            section=prefix,
            language=block.language if block.language != Language.UNKNOWN else detect_language(text),
            # This is what lets retrieval send the original image to the model.
            visual=block.visual or page.page_image,
            confidence=block.confidence,
            uncertain=block.uncertain,
            token_estimate=estimate_tokens(text),
        )

    def _flush(self, document: Document, acc: _Accumulator) -> List[Chunk]:
        """Emit chunks for accumulated text, splitting on soft boundaries."""
        text = acc.text
        if not text or not acc.block_ids:
            return []

        pieces = self._split_text(text)
        chunks: List[Chunk] = []
        for index, piece in enumerate(pieces):
            body = piece
            # Repeat the section heading on continuation pieces so each chunk
            # remains self-describing when retrieved alone.
            if acc.section and index > 0 and not body.startswith(acc.section):
                body = f"{acc.section}\n\n{body}"

            chunks.append(
                Chunk(
                    chunk_id=stable_id(
                        document.document_id, acc.block_ids[0], "chunk", str(index)
                    ),
                    document_id=document.document_id,
                    session_id=document.session_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    page_number=acc.page.page_number,
                    page_label=acc.page.display_label,
                    block_ids=list(acc.block_ids),
                    block_type=BlockType.OCR_TEXT
                    if acc.source_kind == SourceKind.OCR
                    else BlockType.TEXT,
                    source_kind=acc.source_kind,
                    text=body,
                    section=acc.section,
                    language=detect_language(body),
                    # Scanned pages keep their rendering so the model can look
                    # at the actual scan when the OCR text is ambiguous.
                    visual=acc.page.page_image if acc.page.is_scanned else None,
                    confidence=acc.confidence,
                    uncertain=acc.uncertain,
                    token_estimate=estimate_tokens(body),
                )
            )

        acc.parts.clear()
        acc.block_ids.clear()
        return chunks

    # ------------------------------------------------------------------ #
    def _split_text(self, text: str) -> List[str]:
        """Split oversized text on paragraph, then sentence, boundaries."""
        limit = self.settings.chunk_size
        if len(text) <= self.settings.max_chunk_size:
            return [text]

        units = split_paragraphs(text) or [text]
        # A single paragraph longer than the budget is split by sentence.
        expanded: List[str] = []
        for unit in units:
            if len(unit) <= self.settings.max_chunk_size:
                expanded.append(unit)
            else:
                expanded.extend(self._split_long_unit(unit))

        pieces: List[str] = []
        current: List[str] = []
        current_len = 0

        for unit in expanded:
            if current and current_len + len(unit) > limit:
                pieces.append("\n\n".join(current))
                carry = self._overlap_tail("\n\n".join(current))
                current = [carry] if carry else []
                current_len = len(carry)
            current.append(unit)
            current_len += len(unit) + 2

        if current:
            tail = "\n\n".join(current)
            if pieces and len(tail) < self.settings.min_chunk_size:
                pieces[-1] = f"{pieces[-1]}\n\n{tail}"
            else:
                pieces.append(tail)

        return [p for p in pieces if p.strip()]

    def _split_long_unit(self, unit: str) -> List[str]:
        sentences = split_sentences(unit) or [unit]
        out: List[str] = []
        current: List[str] = []
        length = 0
        for sentence in sentences:
            if length + len(sentence) > self.settings.max_chunk_size and current:
                out.append(" ".join(current))
                current, length = [], 0
            if len(sentence) > self.settings.max_chunk_size:
                # Pathological input (no sentence breaks at all): hard-slice it,
                # which is the only remaining option.
                for start in range(0, len(sentence), self.settings.max_chunk_size):
                    out.append(sentence[start : start + self.settings.max_chunk_size])
                continue
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            out.append(" ".join(current))
        return out

    def _overlap_tail(self, text: str) -> str:
        """Trailing context carried into the next chunk, on a sentence boundary."""
        overlap = self.settings.chunk_overlap
        if overlap <= 0 or len(text) <= overlap:
            return ""
        tail = text[-overlap * 2 :]
        sentences = split_sentences(tail)
        carried: List[str] = []
        length = 0
        for sentence in reversed(sentences):
            if length + len(sentence) > overlap:
                break
            carried.insert(0, sentence)
            length += len(sentence) + 1
        return " ".join(carried) if carried else text[-overlap:].strip()


def chunk_documents(
    documents: Iterable[Document], settings: Optional[ChunkingSettings] = None
) -> List[Chunk]:
    chunker = Chunker(settings)
    out: List[Chunk] = []
    for document in documents:
        out.extend(chunker.chunk_document(document))
    return out


__all__ = ["ATOMIC_BLOCK_TYPES", "Chunker", "chunk_documents"]
