"""LLM-as-reranker.

Used when no dedicated rerank key is configured but an LLM is available. One
cheap text-only call scores the whole candidate set, so the cost is a single
request rather than one per passage. It rides on the same provider router, so a
Gemini rate limit transparently falls back to OpenRouter.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

from omnirag.core.enums import Role
from omnirag.core.exceptions import ProviderError, RerankError
from omnirag.providers.llm.base import BaseLLMProvider, LLMMessage
from omnirag.providers.llm.context import llm_operation
from omnirag.providers.rerank.base import BaseReranker, RerankCandidate, RerankScore
from omnirag.utils.logging import get_logger
from omnirag.utils.text import truncate

logger = get_logger(__name__)

MAX_DOC_CHARS = 900
MAX_CANDIDATES = 30

SYSTEM_PROMPT = """You score how well each numbered passage answers a user's question.

Rules:
- Judge only relevance to the question, never writing quality or length.
- A passage describing a chart, diagram or table is relevant if the question asks about that visual.
- Questions and passages may be in Arabic, English, or both. Judge across languages.
- Respond with a single JSON object: {"scores": [{"id": <int>, "score": <0.0-1.0>}, ...]}
- Include every passage id exactly once. No prose, no markdown."""


class LLMReranker(BaseReranker):
    name = "llm"

    def __init__(self, llm: BaseLLMProvider, *, top_n: int = 8, max_candidates: int = MAX_CANDIDATES):
        super().__init__(model=getattr(llm, "model", "llm"), top_n=top_n)
        self.llm = llm
        self.max_candidates = max_candidates

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        window = list(candidates[: self.max_candidates])
        prompt_parts = [f"Question: {query}", "", "Passages:"]
        for index, candidate in enumerate(window):
            prompt_parts.append(f"[{index}] {truncate(candidate.text, MAX_DOC_CHARS)}")
        prompt = "\n".join(prompt_parts)

        try:
            with llm_operation("rerank"):
                response = self.llm.complete(
                    [LLMMessage(role=Role.USER, text=prompt)],
                    system=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=900,
                    json_mode=True,
                )
        except ProviderError as exc:
            raise RerankError(
                f"LLM reranking failed: {exc}", provider=self.name, retryable=False
            ) from exc

        scores = _parse_scores(response.text, len(window))
        if not scores:
            raise RerankError("LLM reranker returned no usable scores", provider=self.name)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        limit = min(top_n or self.top_n, len(ordered))
        return [
            RerankScore(ref=window[index].ref, score=score, rank=rank)
            for rank, (index, score) in enumerate(ordered[:limit])
        ]


def _parse_scores(text: str, count: int) -> Dict[int, float]:
    """Tolerant JSON extraction — models occasionally wrap output in prose."""
    payload = text.strip()
    if not payload:
        return {}

    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        payload = match.group(0)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    items = data.get("scores") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}

    out: Dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id"))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            out[index] = max(0.0, min(1.0, score))
    return out
