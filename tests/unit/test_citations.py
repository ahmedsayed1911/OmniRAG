"""Citations: numbering, marker parsing, and rejection of fabricated references."""

from __future__ import annotations

import pytest

from omnirag.core.enums import BlockType, FileType, SourceKind
from omnirag.core.models import Chunk, SearchResult, VisualRef
from omnirag.rag.citations import (
    build_citations,
    cited_only,
    format_reference,
    group_by_document,
    parse_markers,
    verify_and_clean,
)


def result(filename="report.pdf", page=1, text="Revenue reached 8.4 million.", **kwargs):
    chunk = Chunk(
        document_id=kwargs.pop("document_id", "doc-1"),
        session_id="s1",
        filename=filename,
        file_type=FileType.PDF,
        page_number=page,
        page_label=f"Page {page}",
        block_ids=[f"block-{page}"],
        text=text,
        **kwargs,
    )
    return SearchResult(chunk=chunk, score=0.9)


class TestBuildCitations:
    def test_citations_are_numbered_from_one(self):
        citations = build_citations([result(page=1), result(page=2), result(page=3)])
        assert [c.index for c in citations] == [1, 2, 3]

    def test_citation_carries_full_provenance(self):
        citations = build_citations([
            result(
                filename="annual_report.pdf",
                page=18,
                block_type=BlockType.CHART,
                source_kind=SourceKind.VISION,
                visual=VisualRef(asset_id="asset-7", media_type="image/png"),
                uncertain=True,
                confidence=0.6,
            )
        ])
        citation = citations[0]

        assert citation.filename == "annual_report.pdf"
        assert citation.page_number == 18
        assert citation.block_type == BlockType.CHART
        assert citation.source_kind == SourceKind.VISION
        assert citation.visual_asset_id == "asset-7"
        assert citation.uncertain is True
        assert citation.label == "[annual_report.pdf — Page 18]"

    def test_snippet_is_populated(self):
        citations = build_citations([result(text="A long passage. " * 40)])
        assert citations[0].snippet
        assert len(citations[0].snippet) < 600


class TestParseMarkers:
    @pytest.mark.parametrize("text,expected", [
        ("Revenue grew [1].", [1]),
        ("Both sources agree [1, 3].", [1, 3]),
        ("See [2,4] for detail.", [2, 4]),
        ("Pages [1-3] cover this.", [1, 2, 3]),
        ("No citation here.", []),
        ("Multiple [1] claims [2] cited.", [1, 2]),
    ])
    def test_marker_forms(self, text, expected):
        assert parse_markers(text) == expected

    def test_absurd_ranges_are_ignored(self):
        assert parse_markers("[1-9999]") == []


class TestVerification:
    def test_valid_citations_are_kept_untouched(self):
        citations = build_citations([result(page=1), result(page=2)])
        bundle = verify_and_clean("Revenue grew [1] and risk rose [2].", citations)

        assert bundle.used_indices == {1, 2}
        assert bundle.invalid_indices == set()
        assert bundle.answer == "Revenue grew [1] and risk rose [2]."

    def test_fabricated_citations_are_stripped_and_reported(self):
        # The model cited source [7] but only two sources were supplied.
        citations = build_citations([result(page=1), result(page=2)])
        bundle = verify_and_clean("Revenue grew [1]. Margins improved [7].", citations)

        assert bundle.invalid_indices == {7}
        assert "[7]" not in bundle.answer
        assert "[1]" in bundle.answer

    def test_partially_invalid_group_keeps_the_valid_members(self):
        citations = build_citations([result(page=1), result(page=2)])
        bundle = verify_and_clean("Both agree [1, 9].", citations)

        assert bundle.invalid_indices == {9}
        assert "[1]" in bundle.answer
        assert "9" not in bundle.answer

    def test_all_retrieved_sources_stay_visible_in_the_panel(self):
        # The user should see everything considered, not only what was quoted.
        citations = build_citations([result(page=1), result(page=2), result(page=3)])
        bundle = verify_and_clean("Only the first matters [1].", citations)

        assert len(bundle.citations) == 3
        assert bundle.used_indices == {1}
        assert [c.index for c in cited_only(bundle)] == [1]

    def test_coverage_reflects_how_much_evidence_was_used(self):
        citations = build_citations([result(page=1), result(page=2), result(page=3), result(page=4)])
        bundle = verify_and_clean("Claim [1] and claim [2].", citations)
        assert bundle.coverage == 0.5

    def test_uncited_answer_is_detected(self):
        citations = build_citations([result()])
        bundle = verify_and_clean("A confident answer with no citation.", citations)

        assert bundle.used_indices == set()
        assert bundle.coverage == 0.0

    def test_arabic_answer_citations_are_handled(self):
        citations = build_citations([result(page=5)])
        bundle = verify_and_clean("بلغت الإيرادات 8.4 مليون دولار [1].", citations)

        assert bundle.used_indices == {1}
        assert "[1]" in bundle.answer


class TestFormatting:
    def test_reference_format(self):
        citation = build_citations([result(filename="deck.pptx", page=7)])[0]
        citation.page_label = "Slide 7"
        assert format_reference(citation) == "[deck.pptx — Slide 7]"

    def test_grouping_by_document(self):
        citations = build_citations([
            result(filename="a.pdf", page=1, document_id="d1"),
            result(filename="a.pdf", page=2, document_id="d1"),
            result(filename="b.pdf", page=1, document_id="d2"),
        ])
        grouped = group_by_document(citations)

        assert set(grouped) == {"a.pdf", "b.pdf"}
        assert len(grouped["a.pdf"]) == 2
