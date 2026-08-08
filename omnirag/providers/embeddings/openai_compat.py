"""OpenAI-compatible embeddings (``POST /embeddings``).

``text-embedding-3-large`` is the default: it is genuinely multilingual and
handles Arabic↔English cross-lingual retrieval well, which is a hard
requirement for OmniRAG. Any compatible gateway works via ``EMBEDDING_BASE_URL``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from omnirag.core.exceptions import EmbeddingError
from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.providers.http import post_json
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleEmbeddings(BaseEmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-large",
        base_url: str = "",
        dimensions: int = 0,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_chars: int = 8000,
    ):
        super().__init__(model=model, batch_size=batch_size, max_chars=max_chars)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.dimensions = dimensions

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        payload: Dict[str, Any] = {"model": self.model, "input": list(texts)}
        # `dimensions` is only supported by the v3 family; sending it to an
        # older model is a 400, so it is opt-in via configuration.
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/embeddings",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"embeddings ({self.model})",
        )

        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(
                f"Unexpected embeddings payload: {str(body)[:200]}", provider=self.name
            )

        # The API may return results out of order; `index` is authoritative.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [list(item.get("embedding") or []) for item in ordered]
        if any(not v for v in vectors):
            raise EmbeddingError("Embedding API returned an empty vector", provider=self.name)
        return vectors
