"""Document processors: each format produces the canonical representation."""

from __future__ import annotations

import pytest

from omnirag.core.enums import BlockType, FileType, SourceKind
from omnirag.core.exceptions import CorruptedDocumentError, EmptyDocumentError
from omnirag.ingestion.base import ProcessingContext
from omnirag.ingestion.docx import WordProcessor
from omnirag.ingestion.image import ImageProcessor
from omnirag.ingestion.pdf import PDFProcessor
from omnirag.ingestion.pptx import PowerPointProcessor
from omnirag.ingestion.text import TextProcessor, decode_text
from omnirag.intelligence.tables import build_table, parse_number, summarize_table


@pytest.fixture
def context(settings, session_id, file_store):
    """Processing context with visual/OCR engines disabled (offline tests)."""
    return ProcessingContext(
        session_id=session_id,
        document_id="doc-under-test",
        filename="sample",
        settings=settings,
        file_store=file_store,
        ocr=None,
        vision=None,
        handwriting=None,
    )


class TestTextProcessor:
    def test_markdown_headings_become_sections(self, context, sample_markdown):
        context.filename = "report.md"
        document = TextProcessor().parse(sample_markdown, context)

        headings = [b for b in document.blocks if b.block_type == BlockType.HEADING]
        assert any("Annual Report" in b.text for b in headings)
        assert any("Revenue Overview" in b.text for b in headings)

    def test_markdown_tables_are_parsed_structurally(self, context, sample_markdown):
        context.filename = "report.md"
        document = TextProcessor().parse(sample_markdown, context)

        tables = [b for b in document.blocks if b.block_type == BlockType.TABLE]
        assert len(tables) == 1
        table = tables[0].table
        assert table is not None
        assert table.header == ["Region", "Q3 2024", "Q4 2024"]
        assert table.n_rows == 3
        assert "| EMEA | 2100 | 3400 |" in table.markdown

    def test_arabic_content_is_preserved(self, context, sample_markdown):
        context.filename = "report.md"
        document = TextProcessor().parse(sample_markdown, context)

        joined = "\n".join(b.text for b in document.blocks)
        assert "الإيرادات" in joined
        assert "8.4" in joined

    def test_plain_text_is_split_into_paragraphs(self, context, sample_text):
        context.filename = "notes.txt"
        document = TextProcessor().parse(sample_text, context)

        assert document.blocks
        assert any("145,000 EUR" in b.text for b in document.blocks)

    def test_every_block_carries_its_page_number(self, context, sample_markdown):
        context.filename = "report.md"
        document = TextProcessor().parse(sample_markdown, context)

        for block in document.blocks:
            assert block.page_number >= 1
            assert block.document_id == "doc-under-test"
            assert block.session_id == context.session_id

    def test_empty_file_is_rejected(self, context):
        context.filename = "empty.txt"
        with pytest.raises(EmptyDocumentError):
            TextProcessor().parse(b"   \n\n  ", context)

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "cp1256"])
    def test_arabic_encodings_are_detected(self, encoding):
        text = "بلغت الإيرادات الإجمالية ثمانية ملايين دولار أمريكي"
        decoded, detected = decode_text(text.encode(encoding))

        assert "الإيرادات" in decoded
        assert detected

    def test_code_fences_are_kept_intact(self, context):
        context.filename = "doc.md"
        content = "# Title\n\n```python\ndef f():\n    return 1\n```\n\nAfter the code."
        document = TextProcessor().parse(content.encode("utf-8"), context)

        joined = "\n".join(b.text for b in document.blocks)
        assert "def f():" in joined


class TestPDFProcessor:
    def test_digital_pdf_text_is_extracted_without_ocr(self, context, sample_pdf):
        context.filename = "report.pdf"
        document = PDFProcessor().parse(sample_pdf, context)

        assert document.page_count == 2
        assert all(not p.is_scanned for p in document.pages)
        # No OCR engine is configured, yet text was still extracted.
        assert all(b.source_kind != SourceKind.OCR for b in document.blocks)
        assert any("8,400,000" in b.text for b in document.blocks)

    def test_page_numbers_are_accurate(self, context, sample_pdf):
        context.filename = "report.pdf"
        document = PDFProcessor().parse(sample_pdf, context)

        page_two = document.page(2)
        assert page_two is not None
        assert "Currency volatility" in page_two.text

    def test_headings_are_detected_from_typography(self, context, sample_pdf):
        context.filename = "report.pdf"
        document = PDFProcessor().parse(sample_pdf, context)

        headings = [b for b in document.blocks if b.block_type == BlockType.HEADING]
        assert any("Quarterly Report" in b.text for b in headings)

    def test_bounding_boxes_are_recorded(self, context, sample_pdf):
        context.filename = "report.pdf"
        document = PDFProcessor().parse(sample_pdf, context)

        boxed = [b for b in document.blocks if b.bbox is not None]
        assert boxed
        assert boxed[0].bbox.width > 0

    def test_corrupted_pdf_raises_a_friendly_error(self, context):
        context.filename = "broken.pdf"
        with pytest.raises(CorruptedDocumentError) as excinfo:
            PDFProcessor().parse(b"%PDF-1.4 this is not really a pdf", context)
        assert "broken.pdf" in excinfo.value.user_message

    def test_scanned_page_without_ocr_produces_a_warning_not_a_crash(self, context):
        pymupdf = pytest.importorskip("pymupdf")

        # A page with no text layer at all.
        doc = pymupdf.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()

        context.filename = "scan.pdf"
        with pytest.raises(EmptyDocumentError):
            PDFProcessor().parse(data, context)
        assert any("OCR" in w or "render" in w for w in context.warnings)


class TestWordProcessor:
    def test_headings_paragraphs_and_tables(self, context, sample_docx):
        context.filename = "review.docx"
        document = WordProcessor().parse(sample_docx, context)

        headings = [b for b in document.blocks if b.block_type == BlockType.HEADING]
        tables = [b for b in document.blocks if b.block_type == BlockType.TABLE]

        assert any("Operations Review" in b.text for b in headings)
        assert any("18%" in b.text for b in document.blocks)
        assert len(tables) == 1
        assert "Throughput" in tables[0].table.markdown

    def test_sections_are_labelled_for_citations(self, context, sample_docx):
        context.filename = "review.docx"
        document = WordProcessor().parse(sample_docx, context)

        assert document.pages
        assert all(p.display_label for p in document.pages)


class TestPowerPointProcessor:
    def test_slides_titles_bullets_tables_and_notes(self, context, sample_pptx):
        context.filename = "deck.pptx"
        document = PowerPointProcessor().parse(sample_pptx, context)

        assert document.page_count == 2
        assert document.pages[0].display_label == "Slide 1"

        joined = "\n".join(b.text for b in document.blocks)
        assert "Market Expansion" in joined
        assert "EMEA grew 62%" in joined

        notes = [b for b in document.blocks if b.block_type == BlockType.SPEAKER_NOTES]
        assert notes and "currency headwind" in notes[0].text

        tables = [b for b in document.blocks if b.block_type == BlockType.TABLE]
        assert tables and "62%" in tables[0].table.markdown

    def test_slide_numbers_map_to_page_numbers(self, context, sample_pptx):
        context.filename = "deck.pptx"
        document = PowerPointProcessor().parse(sample_pptx, context)

        assert [p.page_number for p in document.pages] == [1, 2]
        assert [p.display_label for p in document.pages] == ["Slide 1", "Slide 2"]


class TestImageProcessor:
    def test_image_without_vision_still_stores_the_original(self, context, sample_png):
        # No vision or OCR configured: ingestion must not crash, and the
        # original must stay available so the image is still previewable and
        # can be analysed once a provider is configured.
        context.filename = "chart.png"
        document = ImageProcessor().parse(sample_png, context)

        assert document.blocks
        block = document.blocks[0]
        assert block.block_type == BlockType.IMAGE
        assert block.visual is not None
        assert context.file_store.get(block.visual.asset_id) is not None
        assert block.uncertain is True

    def test_blank_image_is_not_indexed(self, context):
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (600, 400), "white").save(buffer, format="PNG")

        context.filename = "blank.png"
        with pytest.raises(EmptyDocumentError):
            ImageProcessor().parse(buffer.getvalue(), context)

    def test_corrupted_image_is_rejected_cleanly(self, context):
        context.filename = "broken.png"
        with pytest.raises(CorruptedDocumentError):
            ImageProcessor().parse(b"\x89PNG\r\n\x1a\n garbage bytes", context)

    def test_image_with_vision_produces_a_described_block(self, context, sample_png):
        from omnirag.core.enums import BlockType as BT
        from omnirag.intelligence.vision import VisionAnalyzer, VisualAnalysis

        class FakeVision(VisionAnalyzer):
            def __init__(self):
                super().__init__(None, enabled=True)

            @property
            def available(self):
                return True

            def analyze(self, image, *, context="", expect=None, skip_decorative_check=False):
                return VisualAnalysis(
                    block_type=BT.CHART,
                    title="Quarterly revenue",
                    description="Bar chart with Q4 at 8400 and Q3 at 6200.",
                    text="Revenue 8400",
                    entities=["Q3", "Q4"],
                    data_points=["Q4: 8400"],
                    confidence=0.9,
                )

        context.vision = FakeVision()
        context.filename = "chart.png"
        document = ImageProcessor().parse(sample_png, context)

        charts = [b for b in document.blocks if b.block_type == BT.CHART]
        assert charts
        block = charts[0]
        assert "8400" in block.visual_description
        # The original image is retained for multimodal answering.
        assert block.visual is not None
        assert context.file_store.get(block.visual.asset_id)


class TestTableIntelligence:
    def test_summary_is_deterministic_and_factual(self):
        table = build_table(
            [["Region", "Q3", "Q4"], ["EMEA", "2100", "3400"], ["APAC", "1800", "2600"]]
        )
        summary = summarize_table(table)

        assert "Region, Q3, Q4" in summary
        assert "2 data rows" in summary
        assert "EMEA" in summary
        assert "1800 to 2100" in summary or "Q3: 1800 to 2100" in summary

    @pytest.mark.parametrize("raw,expected", [
        ("1,234", 1234.0),
        ("(1,234)", -1234.0),
        ("12.5%", 12.5),
        ("٨٤٠٠", 8400.0),
        ("not a number", None),
        ("", None),
    ])
    def test_number_parsing(self, raw, expected):
        assert parse_number(raw) == expected

    def test_single_cell_is_not_a_table(self):
        assert build_table([["only"]]) is None

    def test_empty_rows_are_dropped(self):
        table = build_table([["A", "B"], ["", ""], ["1", "2"]])
        assert table is not None
        assert table.n_rows == 1
