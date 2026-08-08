"""Evaluation utilities: retrieval metrics, answer metrics, OCR CER/WER."""

from __future__ import annotations

import pytest

from omnirag.core.models import AnswerResult, Citation
from omnirag.evaluation.generation_metrics import (
    citation_coverage,
    claim_citation_rate,
    evaluate_answer,
    invalid_citations,
    numeric_fidelity,
)
from omnirag.evaluation.ocr_metrics import (
    character_error_rate,
    evaluate_ocr,
    levenshtein,
    word_error_rate,
)
from omnirag.evaluation.retrieval_metrics import (
    average_precision,
    evaluate_retrieval,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def citation(index: int) -> Citation:
    return Citation(
        index=index,
        chunk_id=f"c{index}",
        document_id="d",
        filename="report.pdf",
        page_number=index,
        page_label=f"Page {index}",
    )


class TestRetrievalMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        assert recall_at_k(["x"], ["a"], 5) == 0.0

    def test_recall_respects_the_cutoff(self):
        assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 3) == 1.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4) == 0.5
        assert precision_at_k([], ["a"], 5) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_mrr_averages_across_queries(self):
        assert mean_reciprocal_rank([(["a"], ["a"]), (["x", "b"], ["b"])]) == 0.75

    def test_average_precision(self):
        assert average_precision(["a", "x", "b"], ["a", "b"]) == pytest.approx(
            (1.0 + 2 / 3) / 2
        )

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)

    def test_aggregate_report(self):
        metrics = evaluate_retrieval(
            [(["a", "b", "c"], ["a"]), (["x", "y", "b"], ["b"])], ks=(1, 3)
        )

        assert metrics.queries == 2
        assert metrics.recall[3] == 1.0
        assert metrics.recall[1] == 0.5
        assert 0 < metrics.mrr <= 1
        assert "recall" in metrics.as_dict()

    def test_empty_input(self):
        assert evaluate_retrieval([]).queries == 0


class TestGenerationMetrics:
    def test_citation_coverage(self):
        citations = [citation(1), citation(2), citation(3), citation(4)]
        assert citation_coverage("Claim [1] and [2].", citations) == 0.5
        assert citation_coverage("No citations here.", citations) == 0.0

    def test_invalid_citations_are_detected(self):
        assert invalid_citations("Claim [1] and [9].", [citation(1)]) == {9}

    def test_claim_citation_rate(self):
        text = (
            "Revenue reached 8,400,000 USD in the fourth quarter of 2024 [1]. "
            "Margins improved considerably across every operating region worldwide."
        )
        assert 0.0 < claim_citation_rate(text) < 1.0

    def test_numeric_fidelity_catches_invented_figures(self):
        context = "Revenue reached 8,400,000 USD and headcount was 1,240."
        good, missing = numeric_fidelity("Revenue was 8,400,000 USD [1].", context)
        assert good == 1.0
        assert missing == []

        bad, invented = numeric_fidelity("Revenue was 9,900,000 USD [1].", context)
        assert bad < 1.0
        assert invented

    def test_number_formatting_differences_are_tolerated(self):
        score, _ = numeric_fidelity("The figure was 8400000.", "Revenue was 8,400,000 USD.")
        assert score == 1.0

    def test_evaluate_answer_end_to_end(self):
        result = AnswerResult(
            answer="Revenue reached 8,400,000 USD [1]. It grew from 6,200,000 [2].",
            citations=[citation(1), citation(2)],
        )
        metrics = evaluate_answer(
            result, context="Revenue reached 8,400,000 USD, up from 6,200,000 USD."
        )

        assert metrics.citation_coverage == 1.0
        assert metrics.has_invalid_citations is False
        assert metrics.numeric_fidelity == 1.0
        assert "citation_coverage" in metrics.as_dict()

    def test_evaluate_answer_flags_a_fabricated_citation(self):
        result = AnswerResult(answer="Revenue grew [4].", citations=[citation(1)])
        metrics = evaluate_answer(result, context="Revenue grew.")

        assert metrics.has_invalid_citations is True
        assert metrics.invalid_citation_count == 1


class TestOCRMetrics:
    def test_levenshtein(self):
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("same", "same") == 0
        assert levenshtein("", "abc") == 3

    def test_perfect_transcription_scores_zero_error(self):
        assert character_error_rate("hello world", "hello world") == 0.0
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_errors_are_measured(self):
        assert character_error_rate("hello", "hallo") == pytest.approx(0.2)
        assert word_error_rate("one two three", "one two") == pytest.approx(1 / 3)

    def test_arabic_normalisation_forgives_orthographic_variants(self):
        reference = "الإيرادات الإجمالية"
        hypothesis = "الايرادات الاجمالية"

        strict = character_error_rate(reference, hypothesis)
        lenient = character_error_rate(reference, hypothesis, normalize=True)

        assert strict > 0
        assert lenient == 0.0

    def test_aggregate_report(self):
        metrics = evaluate_ocr([("hello world", "hello world"), ("abc", "abd")])

        assert metrics.samples == 2
        assert 0 < metrics.cer < 0.5
        assert "cer" in metrics.as_dict()

    def test_empty_input(self):
        assert evaluate_ocr([]).samples == 0
