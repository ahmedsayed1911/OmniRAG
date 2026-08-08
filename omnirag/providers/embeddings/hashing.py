"""Offline hashed bag-of-words embeddings.

**This is not a semantic embedding model.** It is a deterministic hashing
vectoriser (word tokens + character n-grams, sub-linear term weighting, L2
normalised). It exists so OmniRAG runs end-to-end with zero API keys — for
tests, local development, and CI — and so an embedding-key misconfiguration
degrades to lexical search instead of a hard crash.

Honest limitations, surfaced to the user in the UI as a warning:

* no semantic similarity — paraphrases do not match;
* **no cross-lingual retrieval** — an Arabic query will not match an English
  passage, which is a core OmniRAG feature;
* character n-grams do give some robustness to Arabic morphology and OCR typos.

Configure ``GEMINI_API_KEY`` or ``EMBEDDING_API_KEY`` for real retrieval.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import List, Sequence

from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.utils.language import normalize_for_search
from omnirag.utils.logging import get_logger
from omnirag.utils.text import tokenize

logger = get_logger(__name__)

DEFAULT_DIMENSIONS = 1024
_NGRAM_SIZE = 4


class HashingEmbeddings(BaseEmbeddingProvider):
    name = "hash"

    def __init__(
        self,
        *,
        model: str = "hash-1024",
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = 256,
        max_chars: int = 8000,
        use_char_ngrams: bool = True,
        **_ignored: object,
    ):
        super().__init__(model=model, batch_size=batch_size, max_chars=max_chars)
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.use_char_ngrams = use_char_ngrams
        logger.warning(
            "Using the offline `hash` embedding provider — retrieval is lexical "
            "only and cross-lingual search will not work. Configure a real "
            "embedding provider for production."
        )

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        return [self._embed_one(text) for text in texts]

    # ------------------------------------------------------------------ #
    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        counts: Counter[str] = Counter(tokenize(text))

        if self.use_char_ngrams:
            normalized = normalize_for_search(text)
            for word in normalized.split():
                if len(word) <= _NGRAM_SIZE:
                    continue
                padded = f"^{word}$"
                for i in range(len(padded) - _NGRAM_SIZE + 1):
                    counts[f"#{padded[i : i + _NGRAM_SIZE]}"] += 1

        if not counts:
            return vector

        for term, count in counts.items():
            index, sign = self._hash(term)
            # Sub-linear term weighting keeps repeated words from dominating.
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def _hash(self, term: str) -> tuple[int, float]:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # The top bit becomes a sign, which cancels hash collisions on average.
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0
