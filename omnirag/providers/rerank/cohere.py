"""Cohere Rerank adapter (``rerank-multilingual-v3.0`` handles Arabic)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from omnirag.core.exceptions import RerankError
from omnirag.providers.http import post_json
from omnirag.providers.rerank.base import BaseReranker, RerankCandidate, RerankScore
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.cohere.com/v1"
DEFAULT_MODEL = "rerank-multilingual-v3.0"
MAX_DOC_CHARS = 4000


class CohereReranker(BaseReranker):
    name = "cohere"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        top_n: int = 8,
        timeout_s: float = 30.0,
    ):
        super().__init__(model=model or DEFAULT_MODEL, top_n=top_n)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        limit = min(top_n or self.top_n, len(candidates))
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [c.text[:MAX_DOC_CHARS] for c in candidates],
            "top_n": limit,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/rerank",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="rerank",
            ),
            attempts=2,
            operation=f"cohere/rerank ({self.model})",
        )

        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError(
                f"Unexpected Cohere rerank payload: {str(body)[:200]}", provider=self.name
            )

        scores: List[RerankScore] = []
        for rank, item in enumerate(results):
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            scores.append(
                RerankScore(
                    ref=candidates[index].ref,
                    score=float(item.get("relevance_score", 0.0)),
                    rank=rank,
                )
            )
        return scores
