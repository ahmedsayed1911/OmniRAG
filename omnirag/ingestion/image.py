"""Image ingestion (JPG / JPEG / PNG / WebP).

An uploaded image can be almost anything: a scanned invoice, a whiteboard photo,
a chart screenshot, a handwritten note, or a plain photograph. The processor
does not assume — it runs one visual analysis that *classifies* the content and
describes it, then adds OCR when the image actually contains text, and routes to
the handwriting path when handwriting is detected.

The original always stays in the file store, so a question like "what does this
diagram show?" sends the picture itself to the multimodal model.
"""

from __future__ import annotations

from typing import Optional

from omnirag.core.enums import BlockType, FileType, PipelineStage, SourceKind
from omnirag.core.exceptions import CorruptedDocumentError, EmptyDocumentError
from omnirag.core.models import ContentBlock, Document, Page
from omnirag.ingestion.base import BaseDocumentProcessor, ProcessingContext
from omnirag.utils.images import PIL_AVAILABLE, image_size, open_image
from omnirag.utils.language import detect_language
from omnirag.utils.logging import get_logger
from omnirag.utils.text import is_meaningful

logger = get_logger(__name__)

#: Block types that mean "this image contains readable text worth OCR'ing".
_TEXT_BEARING = {
    BlockType.CHART,
    BlockType.DIAGRAM,
    BlockType.TABLE,
    BlockType.IMAGE,
    BlockType.HANDWRITING,
}


class ImageProcessor(BaseDocumentProcessor):
    extensions = ("jpg", "jpeg", "png", "webp")
    file_type = FileType.IMAGE
    display_name = "Image"

    def parse(self, data: bytes, ctx: ProcessingContext) -> Document:
        if not data:
            raise EmptyDocumentError(ctx.filename)
        if PIL_AVAILABLE and open_image(data) is None:
            raise CorruptedDocumentError(ctx.filename, "the image could not be decoded")

        document = self.new_document(data, ctx)
        width, height = image_size(data)
        page = Page(
            document_id=ctx.document_id,
            session_id=ctx.session_id,
            page_number=1,
            label="Image",
            width=float(width) or None,
            height=float(height) or None,
        )

        # The uploaded image *is* the page visual.
        visual = self.store_visual(ctx, data, origin="upload", page_number=1)
        page.page_image = visual

        # -- 1. Visual understanding + classification -------------------- #
        ctx.progress(PipelineStage.ANALYZING_VISUALS, 0.3, "Analysing image")
        visual_block = self.process_visual(
            ctx,
            data,
            page_number=1,
            index=0,
            origin="upload",
            skip_decorative_check=True,
        )
        detected_type: Optional[BlockType] = None
        if visual_block is not None:
            visual_block.visual = visual_block.visual or visual
            page.blocks.append(visual_block)
            detected_type = visual_block.block_type

        # -- 2. Handwriting --------------------------------------------- #
        if detected_type == BlockType.HANDWRITING and ctx.handwriting is not None:
            ctx.progress(PipelineStage.ANALYZING_VISUALS, 0.55, "Reading handwriting")
            result = ctx.handwriting.extract(data)
            if result.ok:
                page.blocks.append(
                    ContentBlock(
                        block_id=self.block_id(ctx, 1, 1, "handwriting"),
                        document_id=ctx.document_id,
                        session_id=ctx.session_id,
                        page_number=1,
                        block_type=BlockType.HANDWRITING,
                        source_kind=SourceKind.OCR,
                        text=result.text,
                        visual=visual,
                        language=result.language,
                        confidence=result.confidence,
                        uncertain=True,
                        order=ctx.next_order(),
                    )
                )
            elif result.error:
                ctx.warn(f"Handwriting could not be transcribed: {result.error}")

        # -- 3. OCR ------------------------------------------------------ #
        # Skipped when the visual pass already transcribed the text, so a
        # screenshot does not cost two API calls.
        already_transcribed = bool(visual_block is not None and visual_block.text.strip())
        if not already_transcribed and ctx.ocr_available:
            ctx.progress(PipelineStage.EXTRACTING_TEXT, 0.7, "Reading text")
            result = ctx.ocr.recognize(data)  # type: ignore[union-attr]
            if result.ok and is_meaningful(result.text, min_chars=6, min_tokens=1):
                page.blocks.append(
                    ContentBlock(
                        block_id=self.block_id(ctx, 1, 2, "ocr_text"),
                        document_id=ctx.document_id,
                        session_id=ctx.session_id,
                        page_number=1,
                        block_type=BlockType.OCR_TEXT,
                        source_kind=SourceKind.OCR,
                        text=result.text,
                        visual=visual,
                        language=result.language or detect_language(result.text),
                        confidence=result.confidence,
                        uncertain=result.uncertain,
                        order=ctx.next_order(),
                    )
                )
            elif result.error:
                ctx.warn(f"Text recognition: {result.error}")

        if page.blocks:
            document.pages.append(page)

        document = self.finalize(document, ctx)
        document.metadata["image_size"] = f"{width}x{height}" if width else "unknown"

        if not document.blocks:
            raise EmptyDocumentError(ctx.filename)
        return document
