"""Canonical models: traceability, payload round-trips, table representations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnirag.core.enums import BlockType, FileType, Language, SourceKind
from omnirag.core.models import (
    BoundingBox,
    Chunk,
    ContentBlock,
    Document,
    DocumentSummary,
    Page,
    TableData,
    VisualRef,
)


class TestContentBlock:
    def test_search_text_combines_every_representation(self):
        block = ContentBlock(
            document_id="d",
            session_id="s",
            block_type=BlockType.CHART,
            text="Revenue by quarter",
            visual_description="Bar chart showing Q4 at 8400",
            parent_section="Financials",
        )
        text = block.search_text

        assert "Financials" in text
        assert "Revenue by quarter" in text
        assert "Bar chart showing Q4 at 8400" in text

    def test_table_contributes_summary_and_markdown(self):
        table = TableData.from_rows([["Region", "Q4"], ["EMEA", "3400"]])
        table.summary = "Columns: Region, Q4."
        block = ContentBlock(
            document_id="d", session_id="s", block_type=BlockType.TABLE, table=table
        )

        assert "Columns: Region, Q4." in block.search_text
        assert "| EMEA | 3400 |" in block.search_text

    def test_empty_block_is_detected(self):
        assert ContentBlock(document_id="d", session_id="s").is_empty is True

    def test_confidence_is_clamped(self):
        assert ContentBlock(document_id="d", session_id="s", confidence=1.7).confidence == 1.0
        assert ContentBlock(document_id="d", session_id="s", confidence=-3).confidence == 0.0

    def test_visual_blocks_are_recognised(self):
        assert BlockType.CHART.is_visual is True
        assert BlockType.DIAGRAM.is_visual is True
        assert BlockType.HANDWRITING.is_visual is True
        assert BlockType.TEXT.is_visual is False


class TestChunkTraceability:
    def test_chunk_requires_at_least_one_source_block(self):
        # This is the invariant that keeps every citation traceable.
        with pytest.raises(ValidationError):
            Chunk(
                document_id="d",
                session_id="s",
                filename="a.pdf",
                block_ids=[],
                text="orphan",
            )

    def test_citation_label_is_human_readable(self):
        chunk = Chunk(
            document_id="d",
            session_id="s",
            filename="annual_report.pdf",
            page_number=18,
            page_label="Page 18",
            block_ids=["b1"],
            text="…",
        )
        assert chunk.citation_label == "[annual_report.pdf — Page 18]"

    def test_payload_round_trip_preserves_provenance(self):
        original = Chunk(
            document_id="doc-1",
            session_id="sess-1",
            filename="deck.pptx",
            file_type=FileType.PPTX,
            page_number=7,
            page_label="Slide 7",
            block_ids=["b1", "b2"],
            block_type=BlockType.CHART,
            source_kind=SourceKind.VISION,
            text="Chart showing growth",
            section="Market",
            language=Language.ENGLISH,
            visual=VisualRef(asset_id="asset-9", media_type="image/jpeg", origin="embedded"),
            confidence=0.82,
            uncertain=True,
        )

        restored = Chunk.from_payload(original.to_payload())

        assert restored.chunk_id == original.chunk_id
        assert restored.block_ids == ["b1", "b2"]
        assert restored.page_label == "Slide 7"
        assert restored.block_type == BlockType.CHART
        assert restored.source_kind == SourceKind.VISION
        assert restored.visual is not None
        assert restored.visual.asset_id == "asset-9"
        assert restored.visual.media_type == "image/jpeg"
        assert restored.uncertain is True

    def test_payload_always_carries_the_session_id(self):
        chunk = Chunk(
            document_id="d", session_id="s-42", filename="f", block_ids=["b"], text="t"
        )
        assert chunk.to_payload()["session_id"] == "s-42"


class TestTableData:
    def test_from_rows_detects_a_header_and_builds_markdown(self):
        table = TableData.from_rows(
            [["Region", "Q3", "Q4"], ["EMEA", "2100", "3400"], ["APAC", "1800", "2600"]]
        )

        assert table.header == ["Region", "Q3", "Q4"]
        assert table.n_rows == 2
        assert table.n_cols == 3
        assert "| Region | Q3 | Q4 |" in table.markdown
        assert "| EMEA | 2100 | 3400 |" in table.markdown

    def test_pipes_in_cells_are_escaped(self):
        table = TableData.from_rows([["a|b", "c"], ["1", "2"]])
        assert r"a\|b" in table.markdown

    def test_ragged_rows_are_padded(self):
        table = TableData.from_rows([["a", "b", "c"], ["1"]])
        assert table.n_cols == 3
        assert table.markdown.count("|") > 0


class TestDocument:
    def test_blocks_are_flattened_across_pages(self):
        document = Document(session_id="s", filename="f.pdf")
        for number in (1, 2):
            page = Page(document_id=document.document_id, session_id="s", page_number=number)
            page.blocks.append(
                ContentBlock(
                    document_id=document.document_id,
                    session_id="s",
                    page_number=number,
                    text=f"page {number}",
                )
            )
            document.pages.append(page)

        assert len(document.blocks) == 2
        assert document.page(2) is not None
        assert document.page(9) is None

    def test_page_display_label_falls_back_to_page_number(self):
        page = Page(document_id="d", session_id="s", page_number=4)
        assert page.display_label == "Page 4"
        page.label = "Slide 4"
        assert page.display_label == "Slide 4"


class TestDocumentSummary:
    @pytest.mark.parametrize("size,expected", [
        (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
    ])
    def test_size_label_is_human_readable(self, size, expected):
        summary = DocumentSummary(
            document_id="d", session_id="s", filename="f", file_type=FileType.PDF,
            size_bytes=size,
        )
        assert summary.size_label == expected


class TestBoundingBox:
    def test_geometry(self):
        box = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert box.width == 100
        assert box.height == 50
        assert box.area == 5000
        assert box.as_tuple() == (10, 20, 110, 70)
