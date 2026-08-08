"""Keyword (BM25) retrieval and hybrid fusion.

Implemented in-house rather than pulling in ``rank_bm25``: the algorithm is
thirty lines, and doing it here lets the tokeniser be Arabic-aware (diacritic
stripping, alef/yeh normalisation, character n-grams for morphology) — which an
off-the-shelf English-centric implementation would not give.

Why hybrid: dense vectors miss exact identifiers (invoice numbers, product
codes, rare proper nouns); BM25 misses paraphrases and cross-lingual matches.
Reciprocal Rank Fusion combines both without needing the two score scales to be
comparable, which is what makes it robust when the embedding provider changes.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from omnirag.core.models import Chunk
from omnirag.utils.hashing import short_hash
from omnirag.utils.logging import get_logger
from omnirag.utils.text import tokenize

logger = get_logger(__name__)

K1 = 1.5
B = 0.75
#: Character n-gram size used to give Arabic morphological variants a partial match.
NGRAM = 4


@dataclass
class BM25Index:
    """Sparse index over the chunks of a single session."""

    session_id: str
    chunk_ids: List[str] = field(default_factory=list)
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    term_frequencies: List[Counter] = field(default_factory=list)
    document_frequency: Counter = field(default_factory=Counter)
    lengths: List[int] = field(default_factory=list)
    average_length: float = 0.0
    signature: str = ""

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, *, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return ``(chunk_id, score)`` best-first."""
        terms = _expand(tokenize(query))
        if not terms or not self.chunk_ids:
            return []

        total = len(self.chunk_ids)
        idf: Dict[str, float] = {}
        for term in set(terms):
            frequency = self.document_frequency.get(term, 0)
            if frequency == 0:
                continue
            idf[term] = math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
        if not idf:
            return []

        scores: List[Tuple[str, float]] = []
        for index, chunk_id in enumerate(self.chunk_ids):
            frequencies = self.term_frequencies[index]
            length = self.lengths[index] or 1
            score = 0.0
            for term, weight in idf.items():
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                denominator = tf + K1 * (1 - B + B * length / (self.average_length or 1.0))
                score += weight * (tf * (K1 + 1)) / denominator
            if score > 0:
                scores.append((chunk_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def build_bm25_index(session_id: str, chunks: Sequence[Chunk]) -> BM25Index:
    """Build a BM25 index from a session's chunks."""
    index = BM25Index(session_id=session_id, signature=_signature(chunks))
    for chunk in chunks:
        terms = _expand(tokenize(chunk.text))
        if not terms:
            continue
        counter = Counter(terms)
        index.chunk_ids.append(chunk.chunk_id)
        index.chunks[chunk.chunk_id] = chunk
        index.term_frequencies.append(counter)
        index.lengths.append(sum(counter.values()))
        for term in counter:
            index.document_frequency[term] += 1

    index.average_length = (
        sum(index.lengths) / len(index.lengths) if index.lengths else 0.0
    )
    return index


def _expand(tokens: Sequence[str]) -> List[str]:
    """Words plus character n-grams — helps Arabic prefixes/suffixes match."""
    out: List[str] = list(tokens)
    for token in tokens:
        if len(token) > NGRAM + 1:
            padded = f"^{token}$"
            out.extend(
                f"#{padded[i : i + NGRAM]}" for i in range(len(padded) - NGRAM + 1)
            )
    return out


def _signature(chunks: Sequence[Chunk]) -> str:
    """Cheap fingerprint so a cached index is rebuilt only when content changes."""
    return short_hash("|".join(sorted(c.chunk_id for c in chunks)), 16)


class BM25IndexCache:
    """Per-session BM25 index cache, invalidated by content signature."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._indexes: Dict[str, BM25Index] = {}

    def get(self, session_id: str, chunks: Sequence[Chunk]) -> BM25Index:
        signature = _signature(chunks)
        with self._lock:
            cached = self._indexes.get(session_id)
            if cached is not None and cached.signature == signature:
                return cached
            index = build_bm25_index(session_id, chunks)
            self._indexes[session_id] = index
            logger.debug("Built BM25 index for session (%d chunks)", index.size)
            return index

    def invalidate(self, session_id: str) -> None:
        with self._lock:
            self._indexes.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._indexes.clear()


_cache = BM25IndexCache()


def get_bm25_cache() -> BM25IndexCache:
    return _cache


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Optional[Sequence[float]] = None
) -> List[Tuple[str, float]]:
    """Combine ranked id lists with weighted RRF.

    RRF uses ranks, not raw scores, so a cosine similarity and a BM25 score can
    be merged without normalising two incomparable scales.
    """
    if not rankings:
        return []
    resolved_weights = list(weights) if weights else [1.0] * len(rankings)
    if len(resolved_weights) < len(rankings):
        resolved_weights.extend([1.0] * (len(rankings) - len(resolved_weights)))

    scores: Dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, resolved_weights):
        for rank, item_id in enumerate(ranking):
            scores[item_id] += weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def normalize_scores(scores: Iterable[float]) -> List[float]:
    """Min-max normalise to 0..1 for display purposes."""
    values = list(scores)
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


__all__ = [
    "BM25Index",
    "BM25IndexCache",
    "build_bm25_index",
    "get_bm25_cache",
    "normalize_scores",
    "reciprocal_rank_fusion",
]
