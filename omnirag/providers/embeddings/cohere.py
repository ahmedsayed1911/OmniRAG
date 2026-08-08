"""Cohere embeddings — a strong multilingual option for Arabic/English.

``embed-multilingual-v3.0`` covers 100+ languages and supports asymmetric
search types, which suits question→passage retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from omnirag.core.exceptions import EmbeddingError
from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.providers.http import post_json
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.cohere.com/v1"
DEFAULT_MODEL = "embed-multilingual-v3.0"
MAX_BATCH = 96


class CohereEmbeddings(BaseEmbeddingProvider):
    name = "cohere"
    supports_task_type = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        dimensions: int = 0,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_chars: int = 8000,
    ):
        super().__init__(
            model=model or DEFAULT_MODEL,
            batch_size=min(max(1, batch_size), MAX_BATCH),
            max_chars=max_chars,
        )
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.dimensions = dimensions

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "texts": list(texts),
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/embed",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"cohere/embed ({self.model})",
        )

        embeddings = body.get("embeddings")
        if isinstance(embeddings, dict):  # {"float": [[...]]}
            embeddings = embeddings.get("float")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingError(
                f"Unexpected Cohere embeddings payload: {str(body)[:200]}", provider=self.name
            )
        return [list(vector) for vector in embeddings]
