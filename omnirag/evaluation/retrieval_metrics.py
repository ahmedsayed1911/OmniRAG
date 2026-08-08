"""Retrieval quality metrics.

Standard IR measures computed over ``(retrieved_ids, relevant_ids)`` pairs, so
they work against any retriever configuration without touching the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items appearing in the top-k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k that is relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    window = retrieved[:k]
    hits = len(set(window) & set(relevant))
    return hits / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1/rank of the first relevant item, else 0."""
    relevant_set = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[str], Iterable[str]]]
) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    total = 0.0
    for position, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            hits += 1
            total += hits / position
    return total / len(relevant_set)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in relevant_set
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant_set), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregate scores over an evaluation set."""

    queries: int = 0
    recall: Dict[int, float] = None  # type: ignore[assignment]
    precision: Dict[int, float] = None  # type: ignore[assignment]
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg: Dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.recall = self.recall or {}
        self.precision = self.precision or {}
        self.ndcg = self.ndcg or {}

    def as_dict(self) -> Dict[str, object]:
        return {
            "queries": self.queries,
            "recall": {f"@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "precision": {f"@{k}": round(v, 4) for k, v in sorted(self.precision.items())},
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in sorted(self.ndcg.items())},
        }


def evaluate_retrieval(
    results: Sequence[tuple[Sequence[str], Iterable[str]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Aggregate metrics over ``(retrieved_ids, relevant_ids)`` pairs."""
    metrics = RetrievalMetrics(queries=len(results))
    if not results:
        return metrics

    materialised = [(list(retrieved), set(relevant)) for retrieved, relevant in results]

    for k in ks:
        metrics.recall[k] = sum(
            recall_at_k(r, rel, k) for r, rel in materialised
        ) / len(materialised)
        metrics.precision[k] = sum(
            precision_at_k(r, rel, k) for r, rel in materialised
        ) / len(materialised)
        metrics.ndcg[k] = sum(
            ndcg_at_k(r, rel, k) for r, rel in materialised
        ) / len(materialised)

    metrics.mrr = mean_reciprocal_rank(materialised)
    metrics.map_score = sum(
        average_precision(r, rel) for r, rel in materialised
    ) / len(materialised)
    return metrics


__all__ = [
    "RetrievalMetrics",
    "average_precision",
    "evaluate_retrieval",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
