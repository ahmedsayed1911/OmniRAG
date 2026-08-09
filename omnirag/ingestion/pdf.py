"""PDF processing with PyMuPDF.

Per-page strategy (this is the crux of doing PDFs well):

1. Read the text layer and measure it.
2. **Digital page** → extract text directly, keep typography for heading
   detection, keep table structure, and analyse only the *embedded images* that
   look informative. The page is never OCR'd — that would be slower, lossier and
   more expensive for no gain.
3. **Scanned page** → render the page to an image once and send it through the
   OCR/vision pipeline. The rendered page is stored as the visual source, so the
   answering model can look at the actual scan.
4. Running headers/footers repeated across most pages are detected and removed
   from the indexed text (they otherwise pollute every chunk), while page
   numbers are preserved as metadata.

A failure on one page is contained: the page is skipped with a warning and the
rest of the document still indexes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from omnirag.core.enums import BlockType, FileType, PipelineStage, SourceKind
from omnirag.core.exceptions import CorruptedDocumentError, EmptyDocumentError
from omnirag.core.models import BoundingBox, ContentBlock, Document, Page
from omnirag.ingestion.base import BaseDocumentProcessor, ProcessingContext
from omnirag.intelligence.layout import (
    TextSpan,
    analyze_page_layout,
    body_font_size,
    detect_rtl_page,
    is_heading_span,
)
from omnirag.intelligence.tables import build_table, table_to_text
from omnirag.utils.images import is_probably_decorative
from omnirag.utils.logging import get_logger
from omnirag.utils.text import (
    clean_text,
    detect_repeated_lines,
    is_meaningful,
    remove_lines,
    split_paragraphs,
)

logger = get_logger(__name__)

try:  # PyMuPDF renamed its module; support both.
    import pymupdf  # type: ignore

    PYMUPDF_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for older releases
    try:
        import fitz as pymupdf  # type: ignore

        PYMUPDF_AVAILABLE = True
    except Exception:  # pragma: no cover
        pymupdf = None  # type: ignore[assignment]
        PYMUPDF_AVAILABLE = False

#: Embedded images smaller than this are almost always icons or rules.
MIN_EMBEDDED_IMAGE_BYTES = 2048
#: Cap on embedded images considered per page, before the document-wide budget.
MAX_IMAGES_PER_PAGE = 6


class PDFProcessor(BaseDocumentProcessor):
    extensions = ("pdf",)
    file_type = FileType.PDF
    display_name = "PDF"

    def parse(self, data: bytes, ctx: ProcessingContext) -> Document:
        if not PYMUPDF_AVAILABLE:
            raise CorruptedDocumentError(
                ctx.filename, "PyMuPDF is not installed"
            )

        document = self.new_document(data, ctx)
        ctx.progress(PipelineStage.PARSING, 0.05, "Opening PDF")

        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise CorruptedDocumentError(ctx.filename, str(exc)) from exc

        try:
            if getattr(pdf, "needs_pass", False):
                raise CorruptedDocumentError(
                    ctx.filename, "the PDF is password protected"
                )

            max_pages = ctx.settings.upload.max_pages_per_document
            total = pdf.page_count
            if total > max_pages:
                ctx.warn(
                    f"Only the first {max_pages} of {total} pages were processed "
                    "(page limit reached)."
                )
                total = max_pages

            document.metadata.update(_pdf_metadata(pdf))

            # Pass 1: text layers, to find running headers/footers.
            raw_texts: List[str] = []
            for index in range(total):
                try:
                    raw_texts.append(pdf[index].get_text("text") or "")
                except Exception:
                    raw_texts.append("")
            repeated = detect_repeated_lines(raw_texts)
            if repeated:
                logger.debug("Removing %d repeated header/footer lines", len(repeated))

            # Pass 2: full extraction.
            for index in range(total):
                ctx.progress(
                    PipelineStage.EXTRACTING_TEXT,
                    0.1 + 0.6 * (index / max(1, total)),
                    f"Page {index + 1} of {total}",
                )
                try:
                    page = self._process_page(
                        pdf[index], index + 1, ctx, raw_texts[index], repeated
                    )
                except Exception as exc:  # noqa: BLE001 - contain per-page failures
                    logger.exception("Failed to process page %d of %s", index + 1, ctx.filename)
                    ctx.warn(f"Page {index + 1} could not be processed ({type(exc).__name__}).")
                    continue
                if page is not None:
                    document.pages.append(page)
        finally:
            try:
                pdf.close()
            except Exception:  # pragma: no cover
                pass

        document = self.finalize(document, ctx)
        if not document.blocks:
            raise EmptyDocumentError(ctx.filename)
        return document

    # ------------------------------------------------------------------ #
    def _process_page(
        self,
        pdf_page: Any,
        page_number: int,
        ctx: ProcessingContext,
        raw_text: str,
        repeated: set,
    ) -> Optional[Page]:
        rect = pdf_page.rect
        page = Page(
            document_id=ctx.document_id,
            session_id=ctx.session_id,
            page_number=page_number,
            label=f"Page {page_number}",
            width=float(rect.width),
            height=float(rect.height),
        )

        image_area = _image_area(pdf_page)
        layout = analyze_page_layout(
            text=raw_text,
            page_area=float(rect.width * rect.height),
            image_area=image_area,
            has_text_layer=bool(raw_text.strip()),
        )
        page.is_scanned = layout.is_scanned
        page.metadata["layout_reason"] = layout.reason

        if layout.is_scanned:
            self._process_scanned_page(pdf_page, page, ctx)
        else:
            self._process_digital_page(pdf_page, page, ctx, raw_text, repeated)
            self._process_embedded_images(pdf_page, page, ctx)

        return page if page.blocks else None

    # -- digital --------------------------------------------------------- #
    def _process_digital_page(
        self,
        pdf_page: Any,
        page: Page,
        ctx: ProcessingContext,
        raw_text: str,
        repeated: set,
    ) -> None:
        blocks_data = self._structured_blocks(pdf_page)
        rtl = detect_rtl_page(raw_text)

        if blocks_data:
            spans = [span for _, spans in blocks_data for span in spans]
            baseline = body_font_size(spans)
            current_section: Optional[str] = None
            index = 0

            for bbox, spans in blocks_data:
                text = clean_text(" ".join(s.text for s in spans))
                text = remove_lines(text, repeated)
                if not text:
                    continue

                heading = bool(spans) and is_heading_span(
                    TextSpan(
                        text=text,
                        size=max((s.size for s in spans), default=0.0),
                        bold=any(s.bold for s in spans),
                    ),
                    baseline,
                )

                if heading:
                    current_section = text
                    block = self.make_text_block(
                        ctx,
                        page_number=page.page_number,
                        text=text,
                        index=index,
                        block_type=BlockType.HEADING,
                        bbox=bbox,
                        section=None,
                    )
                else:
                    block = self.make_text_block(
                        ctx,
                        page_number=page.page_number,
                        text=text,
                        index=index,
                        bbox=bbox,
                        section=current_section,
                    )
                index += 1
                if block is not None:
                    page.blocks.append(block)
        else:
            # No structured spans (rare): fall back to plain paragraph text.
            cleaned = remove_lines(clean_text(raw_text), repeated)
            for index, paragraph in enumerate(split_paragraphs(cleaned)):
                block = self.make_text_block(
                    ctx, page_number=page.page_number, text=paragraph, index=index
                )
                if block is not None:
                    page.blocks.append(block)

        page.metadata["rtl"] = rtl
        self._process_tables(pdf_page, page, ctx)

    def _structured_blocks(self, pdf_page: Any) -> List[Tuple[BoundingBox, List[TextSpan]]]:
        """Extract text blocks with typography, in PyMuPDF's reading order."""
        try:
            data = pdf_page.get_text("dict")
        except Exception:
            return []

        out: List[Tuple[BoundingBox, List[TextSpan]]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            spans: List[TextSpan] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    flags = int(span.get("flags", 0))
                    spans.append(
                        TextSpan(
                            text=text,
                            size=float(span.get("size", 0.0)),
                            bold=bool(flags & 2 ** 4),  # PyMuPDF bold flag
                            font=str(span.get("font", "")),
                        )
                    )
            if not spans:
                continue
            bbox_values = block.get("bbox") or (0, 0, 0, 0)
            out.append(
                (
                    BoundingBox(
                        x0=float(bbox_values[0]),
                        y0=float(bbox_values[1]),
                        x1=float(bbox_values[2]),
                        y1=float(bbox_values[3]),
                    ),
                    spans,
                )
            )
        return out

    def _process_tables(self, pdf_page: Any, page: Page, ctx: ProcessingContext) -> None:
        """Extract tables using PyMuPDF's built-in finder when available."""
        finder = getattr(pdf_page, "find_tables", None)
        if finder is None:
            return
        try:
            found = finder()
            tables = list(getattr(found, "tables", []) or [])
        except Exception as exc:
            logger.debug("Table detection failed on page %d: %s", page.page_number, exc)
            return

        for index, table in enumerate(tables[:8]):
            try:
                rows = table.extract()
            except Exception:
                continue
            data = build_table(rows, has_header=True)
            if data is None:
                continue

            bbox = None
            raw_bbox = getattr(table, "bbox", None)
            if raw_bbox and len(raw_bbox) == 4:
                bbox = BoundingBox(
                    x0=float(raw_bbox[0]),
                    y0=float(raw_bbox[1]),
                    x1=float(raw_bbox[2]),
                    y1=float(raw_bbox[3]),
                )

            # Keep a crop of the rendered table so the model can inspect the
            # real layout when the serialisation is ambiguous.
            visual = None
            if bbox is not None:
                visual = self._render_region(pdf_page, bbox, ctx, page.page_number)

            block = ContentBlock(
                block_id=self.block_id(ctx, page.page_number, index, "table"),
                document_id=ctx.document_id,
                session_id=ctx.session_id,
                page_number=page.page_number,
                block_type=BlockType.TABLE,
                source_kind=SourceKind.STRUCTURED,
                text=table_to_text(data),
                table=data,
                bbox=bbox,
                visual=visual,
                order=ctx.next_order(),
            )
            page.blocks.append(block)

    def _render_region(
        self, pdf_page: Any, bbox: BoundingBox, ctx: ProcessingContext, page_number: int
    ):
        try:
            zoom = ctx.settings.vision.page_render_dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            clip = pymupdf.Rect(*bbox.as_tuple())
            pixmap = pdf_page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            return self.store_visual(
                ctx, pixmap.tobytes("png"), origin="crop", page_number=page_number
            )
        except Exception as exc:
            logger.debug("Could not render table crop: %s", exc)
            return None

    # -- scanned --------------------------------------------------------- #
    def _process_scanned_page(self, pdf_page: Any, page: Page, ctx: ProcessingContext) -> None:
        """Render once, then OCR and visually analyse the same rendering."""
        image = self._render_page(pdf_page, ctx)
        if image is None:
            ctx.warn(f"Page {page.page_number} could not be rendered for OCR.")
            return

        page.page_image = self.store_visual(
            ctx, image, origin="page_render", page_number=page.page_number
        )

        ocr_text = ""
        confidence = None
        uncertain = False
        lazy_vision_ocr = (
            ctx.settings.vision.lazy_analysis
            and ctx.ocr is not None
            and ctx.ocr.name == "vision"
        )
        if ctx.ocr_available and not lazy_vision_ocr:
            result = ctx.ocr.recognize(image)  # type: ignore[union-attr]
            if result.ok:
                ocr_text = result.text
                confidence = result.confidence
                uncertain = result.uncertain
            elif result.error:
                ctx.warn(f"Page {page.page_number}: {result.error}")
        elif not lazy_vision_ocr:
            ctx.warn(
                f"Page {page.page_number} appears to be scanned but no OCR "
                "provider is available."
            )

        if ocr_text and is_meaningful(ocr_text, min_chars=8, min_tokens=2):
            block = ContentBlock(
                block_id=self.block_id(ctx, page.page_number, 0, "ocr_text"),
                document_id=ctx.document_id,
                session_id=ctx.session_id,
                page_number=page.page_number,
                block_type=BlockType.OCR_TEXT,
                source_kind=SourceKind.OCR,
                text=ocr_text,
                visual=page.page_image,
                confidence=confidence,
                uncertain=uncertain,
                order=ctx.next_order(),
            )
            from omnirag.utils.language import detect_language

            block.language = detect_language(ocr_text)
            page.blocks.append(block)

        # Also describe the page visually: a scan often *is* a chart, a form or
        # a handwritten note, and the description is what makes it retrievable.
        if ctx.settings.vision.describe_page_snapshots:
            snapshot = self.process_visual(
                ctx,
                image,
                page_number=page.page_number,
                index=1,
                origin="page_render",
                skip_decorative_check=True,
            )
            if snapshot is not None:
                # Avoid duplicating the OCR text we already indexed.
                if ocr_text and snapshot.block_type == BlockType.IMAGE and not snapshot.visual_description.strip():
                    return
                snapshot.block_type = (
                    snapshot.block_type
                    if snapshot.block_type in (BlockType.CHART, BlockType.DIAGRAM, BlockType.HANDWRITING, BlockType.TABLE)
                    else BlockType.PAGE_SNAPSHOT
                )
                page.blocks.append(snapshot)

    def _render_page(self, pdf_page: Any, ctx: ProcessingContext) -> Optional[bytes]:
        try:
            zoom = ctx.settings.vision.page_render_dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
            return pixmap.tobytes("png")
        except Exception as exc:
            logger.warning("Page rendering failed: %s", exc)
            return None

    # -- embedded images ------------------------------------------------- #
    def _process_embedded_images(self, pdf_page: Any, page: Page, ctx: ProcessingContext) -> None:
        """Analyse pictures embedded in an otherwise digital page."""
        if not ctx.settings.vision.enabled:
            return
        try:
            images = pdf_page.get_images(full=True)
        except Exception:
            return

        context_text = " ".join(b.text for b in page.blocks if b.text)[:500]
        analysed = 0

        for index, info in enumerate(images[:MAX_IMAGES_PER_PAGE]):
            if analysed >= MAX_IMAGES_PER_PAGE:
                break
            xref = info[0]
            try:
                extracted = pdf_page.parent.extract_image(xref)
                image_bytes = extracted.get("image")
            except Exception:
                continue
            if not image_bytes or len(image_bytes) < MIN_EMBEDDED_IMAGE_BYTES:
                continue
            if is_probably_decorative(
                image_bytes, min_pixels=ctx.settings.vision.min_image_pixels
            ):
                continue

            block = self.process_visual(
                ctx,
                image_bytes,
                page_number=page.page_number,
                index=100 + index,
                context_text=context_text,
                origin="embedded",
            )
            if block is not None:
                page.blocks.append(block)
                analysed += 1


def _image_area(pdf_page: Any) -> float:
    """Total page area covered by images, used for scan detection."""
    try:
        info = pdf_page.get_image_info()
    except Exception:
        return 0.0
    area = 0.0
    for item in info or []:
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return area


def _pdf_metadata(pdf: Any) -> Dict[str, Any]:
    try:
        raw = pdf.metadata or {}
    except Exception:
        return {}
    keep = ("title", "author", "subject", "creator", "producer", "creationDate")
    return {k: str(raw[k])[:300] for k in keep if raw.get(k)}
