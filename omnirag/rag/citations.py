"""Citation construction and verification.

Citations are first-class: the context block handed to the model is numbered,
the model is told to cite those numbers, and every marker it emits is then
**verified against the numbers that were actually supplied**. A marker pointing
at a source that was never in context is not rendered as a citation — it is
stripped and reported. That is the guard against fake citations.

Traceability chain preserved end to end:

    Citation.chunk_id -> Chunk.block_ids -> ContentBlock -> Page -> Document
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from omnirag.core.models import Citation, SearchResult
from omnirag.utils.logging import get_logger
from omnirag.utils.text import snippet

logger = get_logger(__name__)

#: Matches [1], [2], [1,3], [1, 2, 5] and [1-3].
_MARKER = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
SNIPPET_CHARS = 420


@dataclass
class CitationBundle:
    """Citations plus the verification outcome for one answer."""

    citations: List[Citation]
    used_indices: Set[int]
    invalid_indices: Set[int]
    answer: str

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def coverage(self) -> float:
        """Fraction of supplied sources the answer actually cited."""
        if not self.citations:
            return 0.0
        return len(self.used_indices) / max(1, len(self.citations))


def build_citations(results: Sequence[SearchResult]) -> List[Citation]:
    """Number the retrieved contexts 1..N — these are the only legal markers."""
    citations: List[Citation] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                page_label=chunk.page_label or f"Page {chunk.page_number}",
                block_type=chunk.block_type,
                source_kind=chunk.source_kind,
                snippet=snippet(chunk.text, SNIPPET_CHARS),
                score=result.rerank_score if result.rerank_score is not None else result.score,
                visual_asset_id=chunk.visual.asset_id if chunk.visual else None,
                visual_media_type=chunk.visual.media_type if chunk.visual else None,
                uncertain=chunk.uncertain,
                confidence=chunk.confidence,
            )
        )
    return citations


def parse_markers(answer: str) -> List[int]:
    """Extract every citation index referenced in the answer text."""
    found: List[int] = []
    for match in _MARKER.finditer(answer):
        body = match.group(1)
        for part in re.split(r"\s*,\s*", body):
            range_match = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", part.strip())
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if 0 < end - start <= 30:
                    found.extend(range(start, end + 1))
                continue
            if part.strip().isdigit():
                found.append(int(part.strip()))
    return found


def verify_and_clean(
    answer: str, citations: Sequence[Citation], *, strip_invalid: bool = True
) -> CitationBundle:
    """Validate the answer's markers against the supplied sources.

    Markers outside ``1..len(citations)`` are hallucinated references. They are
    removed from the rendered answer (never silently kept), and reported so the
    UI can warn.
    """
    valid_indices = {c.index for c in citations}
    referenced = parse_markers(answer)
    used = {i for i in referenced if i in valid_indices}
    invalid = {i for i in referenced if i not in valid_indices}

    cleaned = answer
    if invalid and strip_invalid:
        cleaned = _strip_invalid_markers(answer, valid_indices)
        logger.warning(
            "Removed %d citation marker(s) referring to sources that were not "
            "provided: %s",
            len(invalid),
            sorted(invalid),
        )

    # Keep every retrieved source in the panel — the user should be able to see
    # what was considered, not only what was quoted. `used` marks the cited ones.
    return CitationBundle(
        citations=list(citations),
        used_indices=used,
        invalid_indices=invalid,
        answer=cleaned,
    )


def _strip_invalid_markers(answer: str, valid: Set[int]) -> str:
    def replace(match: re.Match) -> str:
        kept: List[str] = []
        for part in re.split(r"\s*,\s*", match.group(1)):
            range_match = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", part.strip())
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                kept.extend(str(i) for i in range(start, end + 1) if i in valid)
                continue
            if part.strip().isdigit() and int(part.strip()) in valid:
                kept.append(part.strip())
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _MARKER.sub(replace, answer)
    # Tidy up the spacing left behind by a removed marker.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()


def cited_only(bundle: CitationBundle) -> List[Citation]:
    """The subset of sources the answer actually referenced."""
    return [c for c in bundle.citations if c.index in bundle.used_indices]


def group_by_document(citations: Sequence[Citation]) -> Dict[str, List[Citation]]:
    grouped: Dict[str, List[Citation]] = {}
    for citation in citations:
        grouped.setdefault(citation.filename, []).append(citation)
    return grouped


def format_reference(citation: Citation) -> str:
    """``[annual_report.pdf — Page 18]`` — the canonical display form."""
    return f"[{citation.filename} — {citation.page_label}]"


def trace(citation: Citation, chunks_by_id: Dict[str, object]) -> Optional[Tuple[str, ...]]:
    """Resolve a citation to its source block ids (traceability check)."""
    chunk = chunks_by_id.get(citation.chunk_id)
    if chunk is None:
        return None
    return tuple(getattr(chunk, "block_ids", ()) or ())


__all__ = [
    "Citation",
    "CitationBundle",
    "build_citations",
    "cited_only",
    "format_reference",
    "group_by_document",
    "parse_markers",
    "trace",
    "verify_and_clean",
]
