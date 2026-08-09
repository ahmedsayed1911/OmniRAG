"""Chunking: structure awareness, atomicity, and citation-metadata preservation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnirag.config.settings import ChunkingSettings
from omnirag.core.enums import BlockType, FileType, SourceKind
from omnirag.core.models import Chunk, ContentBlock, Document, Page, TableData, VisualRef
from omnirag.rag.chunking import ATOMIC_BLOCK_TYPES, Chunker


def build_document(blocks_per_page, *, filename="report.pdf", session="s1"):
    document = Document(
        session_id=session, filename=filename, file_type=FileType.PDF
    )
    order = 0
    for page_number, blocks in enumerate(blocks_per_page, start=1):
        page = Page(
            document_id=document.document_id,
            session_id=session,
            page_number=page_number,
            label=f"Page {page_number}",
        )
        for block in blocks:
            order += 1
            block.document_id = document.document_id
            block.session_id = session
            block.page_number = page_number
            block.order = order
            page.blocks.append(block)
        document.pages.append(page)
    return document


def text_block(text, **kwargs):
    return ContentBlock(document_id="", session_id="", text=text, **kwargs)


class TestCitationMetadata:
    def test_every_chunk_keeps_full_citation_metadata(self):
        document = build_document([[text_block("Alpha " * 30)], [text_block("Beta " * 30)]])
        chunks = Chunker().chunk_document(document)

        assert chunks
        for chunk in chunks:
            assert chunk.block_ids, "a chunk must always reference its source blocks"
            assert chunk.document_id == document.document_id
            assert chunk.session_id == "s1"
            assert chunk.filename == "report.pdf"
            assert chunk.page_number >= 1
            assert chunk.page_label

    def test_block_ids_resolve_back_to_real_blocks(self):
        document = build_document([[text_block("Alpha " * 40), text_block("Beta " * 40)]])
        chunks = Chunker().chunk_document(document)

        known = {b.block_id for b in document.blocks}
        for chunk in chunks:
            assert set(chunk.block_ids) <= known

    def test_a_chunk_never_spans_two_pages(self):
        document = build_document([
            [text_block("Page one content. " * 5)],
            [text_block("Page two content. " * 5)],
        ])
        chunks = Chunker().chunk_document(document)

        pages = {c.page_number for c in chunks}
        assert pages == {1, 2}
        for chunk in chunks:
            page_texts = {
                p.page_number: p.text for p in document.pages
            }
            assert chunk.page_number in page_texts

    def test_chunk_ids_are_stable_across_runs(self):
        document = build_document([[text_block("Stable content. " * 20)]])

        first = [c.chunk_id for c in Chunker().chunk_document(document)]
        second = [c.chunk_id for c in Chunker().chunk_document(document)]
        assert first == second


class TestAtomicBlocks:
    def test_tables_are_never_split_or_merged(self):
        table = TableData.from_rows(
            [["Region", "Q4"], ["EMEA", "3400"], ["APAC", "2600"]]
        )
        table.summary = "Regional revenue."
        document = build_document([[
            text_block("Intro paragraph. " * 10),
            text_block(
                "", block_type=BlockType.TABLE, table=table,
                source_kind=SourceKind.STRUCTURED,
            ),
            text_block("Following paragraph. " * 10),
        ]])

        chunks = Chunker().chunk_document(document)
        table_chunks = [c for c in chunks if c.block_type == BlockType.TABLE]

        assert len(table_chunks) == 1
        assert len(table_chunks[0].block_ids) == 1
        assert "| EMEA | 3400 |" in table_chunks[0].text

    @pytest.mark.parametrize("block_type", sorted(ATOMIC_BLOCK_TYPES, key=str))
    def test_atomic_types_get_their_own_chunk(self, block_type):
        document = build_document([[
            text_block("Some text. " * 10),
            text_block(
                "",
                block_type=block_type,
                visual_description=f"A {block_type.value} showing revenue growth to 8400.",
                visual=VisualRef(asset_id="asset-1"),
            ),
        ]])

        chunks = Chunker().chunk_document(document)
        matching = [c for c in chunks if c.block_type == block_type]

        assert len(matching) == 1
        assert len(matching[0].block_ids) == 1

    def test_visual_reference_survives_chunking(self):
        # Without this the "send the original image to the model" rule breaks.
        document = build_document([[
            text_block(
                "",
                block_type=BlockType.CHART,
                visual_description="Bar chart, Q4 revenue 8400.",
                visual=VisualRef(asset_id="asset-42", media_type="image/jpeg"),
            )
        ]])

        chunk = Chunker().chunk_document(document)[0]
        assert chunk.visual is not None
        assert chunk.visual.asset_id == "asset-42"
        assert chunk.visual.media_type == "image/jpeg"

    def test_visual_from_pre_reload_model_is_revalidated(self):
        """A Streamlit hot reload must not strand the old VisualRef identity."""
        models_path = Path(__file__).parents[2] / "omnirag" / "core" / "models.py"
        module_name = "omnirag.core.models_pre_reload"
        spec = importlib.util.spec_from_file_location(module_name, models_path)
        assert spec is not None and spec.loader is not None
        old_models = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = old_models
        try:
            spec.loader.exec_module(old_models)
            for model_name in ("VisualRef", "ContentBlock", "Page", "Document"):
                getattr(old_models, model_name).model_rebuild(
                    _types_namespace=vars(old_models)
                )
        finally:
            sys.modules.pop(module_name, None)

        old_visual = old_models.VisualRef(
            asset_id="asset-before-reload",
            media_type="image/jpeg",
            width=640,
            height=480,
            origin="crop",
            page_number=2,
        )
        assert type(old_visual) is not VisualRef
        assert not isinstance(old_visual, VisualRef)
        assert Chunk.model_fields["visual"].annotation != type(old_visual)

        with pytest.raises(ValidationError) as captured:
            Chunk(
                document_id="old-document",
                session_id="s1",
                filename="reload.pdf",
                block_ids=["old-block"],
                visual=old_visual,
            )
        error = captured.value.errors(include_url=False)[0]
        assert error["loc"] == ("visual",)
        assert error["type"] == "model_type"

        old_block = old_models.ContentBlock(
            document_id="old-document",
            session_id="s1",
            page_number=2,
            block_type=BlockType.TABLE,
            source_kind=SourceKind.STRUCTURED,
            text="A retained table visual.",
            visual=old_visual,
        )
        old_page = old_models.Page(
            document_id="old-document",
            session_id="s1",
            page_number=2,
            blocks=[old_block],
        )
        old_document = old_models.Document(
            document_id="old-document",
            session_id="s1",
            filename="reload.pdf",
            file_type=FileType.PDF,
            pages=[old_page],
        )

        chunk = Chunker().chunk_document(old_document)[0]

        assert isinstance(chunk.visual, VisualRef)
        assert chunk.visual.model_dump(mode="python") == old_visual.model_dump(mode="python")


class TestStructureAwareness:
    def test_headings_define_sections_carried_into_chunks(self):
        document = build_document([[
            text_block("Revenue Overview", block_type=BlockType.HEADING),
            text_block("Total revenue reached 8.4 million USD. " * 8),
        ]])

        chunks = Chunker().chunk_document(document)
        assert any(c.section == "Revenue Overview" or "Revenue Overview" in c.text for c in chunks)

    def test_long_sections_split_and_repeat_the_heading(self):
        settings = ChunkingSettings(chunk_size=300, chunk_overlap=50, max_chunk_size=400)
        document = build_document([[
            text_block("Risk Factors", block_type=BlockType.HEADING),
            text_block("\n\n".join(f"Paragraph {i} about currency volatility." for i in range(30))),
        ]])

        chunks = Chunker(settings).chunk_document(document)
        assert len(chunks) > 1
        for chunk in chunks[1:]:
            assert chunk.section == "Risk Factors"

    def test_overlap_carries_context_between_chunks(self):
        settings = ChunkingSettings(chunk_size=250, chunk_overlap=80, max_chunk_size=320)
        sentences = " ".join(f"Sentence number {i} with content." for i in range(60))
        document = build_document([[text_block(sentences)]])

        chunks = Chunker(settings).chunk_document(document)
        assert len(chunks) > 1
        # Consecutive chunks should share some text at the seam.
        overlaps = sum(
            1
            for a, b in zip(chunks, chunks[1:])
            if set(a.text.split()[-8:]) & set(b.text.split()[:20])
        )
        assert overlaps >= 1

    def test_ocr_provenance_is_propagated(self):
        document = build_document([[
            text_block(
                "Scanned page content that was read by OCR. " * 5,
                block_type=BlockType.OCR_TEXT,
                source_kind=SourceKind.OCR,
                confidence=0.42,
                uncertain=True,
            )
        ]])

        chunk = Chunker().chunk_document(document)[0]
        assert chunk.source_kind == SourceKind.OCR
        assert chunk.uncertain is True
        assert chunk.confidence == pytest.approx(0.42)

    def test_empty_blocks_are_dropped(self):
        document = build_document([[text_block(""), text_block("   ")]])
        assert Chunker().chunk_document(document) == []

    def test_chunks_are_globally_ordered(self):
        document = build_document([
            [text_block("First page. " * 20)],
            [text_block("Second page. " * 20)],
        ])
        chunks = Chunker().chunk_document(document)
        assert [c.order for c in chunks] == list(range(len(chunks)))


class TestArabicChunking:
    def test_arabic_text_is_chunked_and_detected(self):
        arabic = "بلغت الإيرادات الإجمالية ثمانية ملايين دولار في الربع الرابع. " * 12
        document = build_document([[text_block(arabic)]])

        chunks = Chunker().chunk_document(document)
        assert chunks
        assert all(str(c.language) in ("ar", "mixed") for c in chunks)
        assert "الإيرادات" in chunks[0].text

    def test_mixed_script_content_is_preserved_verbatim(self):
        mixed = "Revenue بلغت 8,400,000 USD في Q4 2024. " * 10
        document = build_document([[text_block(mixed)]])

        chunks = Chunker().chunk_document(document)
        assert "8,400,000" in chunks[0].text
        assert "بلغت" in chunks[0].text
