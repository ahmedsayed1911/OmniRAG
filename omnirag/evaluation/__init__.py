"""Evaluation utilities for retrieval, generation and OCR quality."""

from omnirag.evaluation.generation_metrics import (
    GenerationMetrics,
    citation_coverage,
    evaluate_answer,
    numeric_fidelity,
)
from omnirag.evaluation.harness import (
    EvaluationCase,
    EvaluationReport,
    run_evaluation,
)
from omnirag.evaluation.ocr_metrics import (
    OCRMetrics,
    character_error_rate,
    evaluate_ocr,
    word_error_rate,
)
from omnirag.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "GenerationMetrics",
    "OCRMetrics",
    "RetrievalMetrics",
    "character_error_rate",
    "citation_coverage",
    "evaluate_answer",
    "evaluate_ocr",
    "evaluate_retrieval",
    "mean_reciprocal_rank",
    "numeric_fidelity",
    "precision_at_k",
    "recall_at_k",
    "run_evaluation",
    "word_error_rate",
]
