"""Zero-cost heuristic reranker — the always-available fallback.

Not a cross-encoder, and it does not pretend to be. It combines signals that are
cheap and genuinely predictive of whether a passage answers a query:

* lexical overlap of query terms (Arabic-normalised), IDF-weighted so rare
  words count more than stop-words;
* coverage of the query (what fraction of query terms appear at all);
* proximity — terms appearing close together score higher than scattered ones;
* exact phrase match bonus;
* numeric-token match, which matters for financial/tabular questions;
* a small penalty for low-confidence OCR passages.

Used automatically when no rerank API key is configured, and as the safety net
whenever a model-based reranker errors out.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from omnirag.providers.rerank.base import BaseReranker, RerankCandidate, RerankScore
from omnirag.utils.language import normalize_for_search
from omnirag.utils.logging import get_logger
from omnirag.utils.text import tokenize

logger = get_logger(__name__)

_NUMERIC = set("0123456789٠١٢٣٤٥٦٧٨٩")


class HeuristicReranker(BaseReranker):
    name = "heuristic"
    is_model_based = False

    def __init__(self, *, top_n: int = 8):
        super().__init__(model="lexical-heuristic", top_n=top_n)

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_n: int | None = None
    ) -> List[RerankScore]:
        if not candidates:
            return []

        query_terms = [t for t in tokenize(query) if len(t) > 1]
        if not query_terms:
            # Nothing to match on — keep the retrieval order untouched.
            return [
                RerankScore(ref=c.ref, score=1.0 / (rank + 1), rank=rank)
                for rank, c in enumerate(candidates[: top_n or self.top_n])
            ]

        doc_tokens = [tokenize(c.text) for c in candidates]
        idf = self._idf(query_terms, doc_tokens)
        phrase = normalize_for_search(query)

        scored: List[tuple[float, str]] = []
        for candidate, tokens in zip(candidates, doc_tokens):
            scored.append(
                (self._score(query_terms, tokens, candidate, phrase, idf), candidate.ref)
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = min(top_n or self.top_n, len(scored))
        return [
            RerankScore(ref=ref, score=score, rank=rank)
            for rank, (score, ref) in enumerate(scored[:limit])
        ]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _idf(query_terms: Sequence[str], doc_tokens: Sequence[List[str]]) -> Dict[str, float]:
        total = max(1, len(doc_tokens))
        sets = [set(tokens) for tokens in doc_tokens]
        idf: Dict[str, float] = {}
        for term in set(query_terms):
            hits = sum(1 for token_set in sets if term in token_set)
            idf[term] = math.log(1.0 + (total - hits + 0.5) / (hits + 0.5))
        return idf

    def _score(
        self,
        query_terms: Sequence[str],
        tokens: Sequence[str],
        candidate: RerankCandidate,
        phrase: str,
        idf: Dict[str, float],
    ) -> float:
        if not tokens:
            return 0.0

        counts = Counter(tokens)
        length_norm = 1.0 / (1.0 + math.log(1.0 + len(tokens) / 120.0))

        overlap = 0.0
        matched = 0
        for term in set(query_terms):
            count = counts.get(term, 0)
            if count:
                matched += 1
                overlap += idf.get(term, 1.0) * (1.0 + math.log(count))

        if matched == 0:
            return 0.0

        coverage = matched / len(set(query_terms))
        score = overlap * length_norm * (0.5 + 0.5 * coverage)

        # Exact phrase occurrence is a strong signal.
        text_normalized = normalize_for_search(candidate.text)
        if len(phrase) > 8 and phrase in text_normalized:
            score *= 1.6

        # Proximity: reward query terms occurring near each other.
        score *= 1.0 + 0.3 * self._proximity(query_terms, tokens)

        # Numeric agreement matters for tables and financial questions.
        query_numbers = {t for t in query_terms if _is_numeric(t)}
        if query_numbers:
            hits = len(query_numbers & set(tokens))
            score *= 1.0 + 0.4 * (hits / len(query_numbers))

        return score

    @staticmethod
    def _proximity(query_terms: Sequence[str], tokens: Sequence[str]) -> float:
        """0..1 — how tightly the matched query terms cluster in the passage."""
        wanted = set(query_terms)
        positions = [i for i, token in enumerate(tokens) if token in wanted]
        if len(positions) < 2:
            return 0.0
        span = positions[-1] - positions[0] + 1
        return min(1.0, len(positions) / max(1, span))


def _is_numeric(token: str) -> bool:
    return any(ch in _NUMERIC for ch in token)
