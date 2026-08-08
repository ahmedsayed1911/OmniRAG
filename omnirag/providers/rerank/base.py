"""Reranker interface.

Reranking is the highest-leverage retrieval upgrade after hybrid search: the
vector+BM25 stage optimises recall over a wide candidate set, and the reranker
optimises precision over the handful of passages actually shown to the LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RerankCandidate:
    """A passage to score. ``ref`` links back to the originating chunk."""

    ref: str
    text: str


@dataclass
class RerankScore:
    ref: str
    score: float
    rank: int = 0


class BaseReranker(ABC):
    name: str = "base"
    #: False for the heuristic fallback, which costs nothing but is weaker.
    is_model_based: bool = True

    def __init__(self, *, model: str = "", top_n: int = 8):
        self.model = model
        self.top_n = top_n

    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        """Return candidates ordered best-first with relevance scores."""

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "model_based": self.is_model_based}
