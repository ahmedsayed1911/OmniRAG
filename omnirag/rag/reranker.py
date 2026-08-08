"""Reranking façade with guaranteed graceful degradation.

Wraps whichever :class:`~omnirag.providers.rerank.base.BaseReranker` is
configured and guarantees the pipeline never loses results because a rerank API
was unavailable: on any failure it falls back to the heuristic reranker, and if
that somehow fails too, the original retrieval order is preserved.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from omnirag.core.models import SearchResult
from omnirag.providers.rerank.base import BaseReranker, RerankCandidate, RerankScore
from omnirag.providers.rerank.heuristic import HeuristicReranker
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


class RerankingService:
    """Reranks :class:`SearchResult` objects, never raising."""

    def __init__(self, reranker: Optional[BaseReranker] = None, *, top_n: int = 8):
        self.reranker = reranker
        self.top_n = top_n
        self._fallback = HeuristicReranker(top_n=top_n)
        self.last_provider: str = ""

    def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_n: Optional[int] = None
    ) -> List[SearchResult]:
        if not results:
            return []
        limit = top_n or self.top_n
        candidates = [
            RerankCandidate(ref=r.chunk.chunk_id, text=r.chunk.text) for r in results
        ]

        scores = self._score(query, candidates, limit)
        if not scores:
            self.last_provider = "none"
            return list(results)[:limit]

        by_id = {r.chunk.chunk_id: r for r in results}
        ordered: List[SearchResult] = []
        for entry in scores:
            item = by_id.pop(entry.ref, None)
            if item is None:
                continue
            item.rerank_score = entry.score
            item.score = entry.score
            ordered.append(item)
        ordered.extend(by_id.values())

        for rank, item in enumerate(ordered):
            item.rank = rank
        return ordered[:limit]

    def _score(
        self, query: str, candidates: Sequence[RerankCandidate], limit: int
    ) -> List[RerankScore]:
        if self.reranker is not None:
            try:
                scores = self.reranker.rerank(query, candidates, top_n=limit)
                if scores:
                    self.last_provider = self.reranker.name
                    return scores
                logger.info("%s returned no scores — using the heuristic reranker", self.reranker.name)
            except Exception as exc:
                logger.warning(
                    "Reranker %s failed (%s) — falling back to the heuristic reranker",
                    getattr(self.reranker, "name", "?"),
                    exc,
                )

        try:
            scores = self._fallback.rerank(query, candidates, top_n=limit)
            self.last_provider = self._fallback.name
            return scores
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Heuristic reranking failed: %s", exc)
            return []


def build_reranking_service(settings=None) -> RerankingService:
    from omnirag.config.settings import get_settings
    from omnirag.providers.rerank.factory import get_reranker

    resolved = settings or get_settings()
    return RerankingService(
        get_reranker(resolved), top_n=resolved.retrieval.rerank_top_k
    )


__all__ = ["RerankingService", "build_reranking_service"]
