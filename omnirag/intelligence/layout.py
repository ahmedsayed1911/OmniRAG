"""Layout analysis: scanned-page detection, headings, and reading order.

All deterministic — no model calls. Deciding whether a PDF page has a usable
text layer is a measurement, not a judgement call, and using an LLM for it would
be slower, costlier and less reliable.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from omnirag.core.models import BoundingBox
from omnirag.utils.language import is_rtl
from omnirag.utils.text import is_meaningful

#: A page with fewer than this many characters of extractable text is a
#: candidate for OCR (a title page can legitimately be sparse, so the image
#: coverage check below is what actually decides).
MIN_CHARS_FOR_DIGITAL_PAGE = 90
#: Fraction of the page covered by images above which we treat it as a scan.
SCAN_IMAGE_COVERAGE = 0.55
#: Heading font size must exceed the body size by this factor.
HEADING_SIZE_RATIO = 1.15


@dataclass
class TextSpan:
    """A run of text with its typographic attributes (from PyMuPDF)."""

    text: str
    size: float = 0.0
    bold: bool = False
    bbox: Optional[BoundingBox] = None
    font: str = ""


@dataclass
class PageLayoutInfo:
    is_scanned: bool = False
    char_count: int = 0
    image_coverage: float = 0.0
    reason: str = ""


def analyze_page_layout(
    *,
    text: str,
    page_area: float,
    image_area: float,
    has_text_layer: bool,
) -> PageLayoutInfo:
    """Decide whether a PDF page needs OCR.

    Rule: OCR only when there is no usable text layer. Digitally extractable
    text is never re-OCR'd — that would be slower, lossier and, with a vision
    provider, needlessly expensive.
    """
    char_count = len(text.strip())
    coverage = (image_area / page_area) if page_area > 0 else 0.0

    if not has_text_layer or char_count == 0:
        return PageLayoutInfo(
            is_scanned=True,
            char_count=char_count,
            image_coverage=coverage,
            reason="no text layer",
        )

    if char_count < MIN_CHARS_FOR_DIGITAL_PAGE and coverage >= SCAN_IMAGE_COVERAGE:
        return PageLayoutInfo(
            is_scanned=True,
            char_count=char_count,
            image_coverage=coverage,
            reason="sparse text over a full-page image",
        )

    if not is_meaningful(text, min_chars=MIN_CHARS_FOR_DIGITAL_PAGE // 3, min_tokens=3):
        return PageLayoutInfo(
            is_scanned=True,
            char_count=char_count,
            image_coverage=coverage,
            reason="text layer contains no meaningful words",
        )

    return PageLayoutInfo(
        is_scanned=False,
        char_count=char_count,
        image_coverage=coverage,
        reason="digital text layer",
    )


# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #
_NUMBERED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S")
_ALL_CAPS = re.compile(r"^[^a-z]{4,}$")


def body_font_size(spans: Sequence[TextSpan]) -> float:
    """Median span size weighted by text length — a robust body-text baseline."""
    sizes: List[float] = []
    for span in spans:
        if span.size > 0 and span.text.strip():
            sizes.extend([span.size] * max(1, len(span.text) // 20))
    return statistics.median(sizes) if sizes else 0.0


def is_heading_span(span: TextSpan, baseline: float) -> bool:
    """Typography-based heading test (size, weight, brevity)."""
    text = span.text.strip()
    if not text or len(text) > 160:
        return False
    if text.endswith((".", "،", "؛", ";")) and len(text) > 60:
        return False

    if baseline > 0 and span.size >= baseline * HEADING_SIZE_RATIO:
        return True
    if span.bold and len(text) <= 100 and baseline > 0 and span.size >= baseline * 0.98:
        return True
    return False


def looks_like_heading(line: str) -> bool:
    """Heading test for plain text / Markdown / DOCX without style info."""
    text = line.strip()
    if not text or len(text) > 150:
        return False
    if text.startswith("#"):
        return True
    if _NUMBERED_HEADING.match(text) and len(text) < 100:
        return True
    if _ALL_CAPS.match(text) and 4 <= len(text) <= 90:
        return True
    if text.endswith(":") and len(text) <= 80:
        return True
    return False


def heading_level(line: str) -> int:
    text = line.strip()
    if text.startswith("#"):
        return min(6, len(text) - len(text.lstrip("#")))
    match = _NUMBERED_HEADING.match(text)
    if match:
        return min(6, 1 + match.group(0).strip().count("."))
    return 2


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #
def sort_reading_order(
    items: Sequence[Tuple[BoundingBox, object]],
    *,
    page_width: float = 0.0,
    rtl: bool = False,
    column_tolerance: float = 0.12,
) -> List[object]:
    """Order page items top-to-bottom, respecting columns and text direction.

    Two-column layouts are common in reports and papers; naive top-to-bottom
    sorting would interleave the columns and destroy sentence continuity.
    """
    if not items:
        return []

    entries = [(bbox, payload) for bbox, payload in items if bbox is not None]
    if not entries:
        return [payload for _, payload in items]

    columns = _detect_columns(entries, page_width, column_tolerance)
    if columns <= 1:
        return [
            payload
            for _, payload in sorted(
                entries,
                key=lambda e: (round(e[0].y0, 1), -e[0].x0 if rtl else e[0].x0),
            )
        ]

    width = page_width or max(bbox.x1 for bbox, _ in entries)
    band = width / columns

    def key(entry):
        bbox = entry[0]
        column = min(columns - 1, int(bbox.x0 / band)) if band > 0 else 0
        if rtl:
            column = columns - 1 - column
        return (column, round(bbox.y0, 1), bbox.x0)

    return [payload for _, payload in sorted(entries, key=key)]


def _detect_columns(
    entries: Sequence[Tuple[BoundingBox, object]], page_width: float, tolerance: float
) -> int:
    """Detect a simple 2-column layout via a gap in the horizontal midpoints."""
    if len(entries) < 6:
        return 1
    width = page_width or max(bbox.x1 for bbox, _ in entries)
    if width <= 0:
        return 1

    midpoints = sorted((bbox.x0 + bbox.x1) / 2 / width for bbox, _ in entries)
    left = [m for m in midpoints if m < 0.5]
    right = [m for m in midpoints if m >= 0.5]
    if len(left) < 3 or len(right) < 3:
        return 1

    gap = min(right) - max(left)
    return 2 if gap > tolerance else 1


def detect_rtl_page(text: str) -> bool:
    return is_rtl(text)


__all__ = [
    "PageLayoutInfo",
    "TextSpan",
    "analyze_page_layout",
    "body_font_size",
    "detect_rtl_page",
    "heading_level",
    "is_heading_span",
    "looks_like_heading",
    "sort_reading_order",
]
