"""Google Gemini embeddings (``batchEmbedContents``).

Chosen automatically when a ``GEMINI_API_KEY`` is present and no dedicated
embedding key is configured, so the default Gemini-first deployment gets real
multilingual semantic search without a second vendor account.

Gemini distinguishes ``RETRIEVAL_QUERY`` from ``RETRIEVAL_DOCUMENT`` task types;
using them measurably improves asymmetric question→passage retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from omnirag.core.exceptions import EmbeddingError
from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.providers.http import post_json
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-embedding-001"
# Gemini caps batchEmbedContents at 100 requests per call.
MAX_BATCH = 100


class GeminiEmbeddings(BaseEmbeddingProvider):
    name = "gemini"
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

    @property
    def _model_path(self) -> str:
        return self.model if self.model.startswith("models/") else f"models/{self.model}"

    def embed_batch(self, texts: Sequence[str], *, is_query: bool = False) -> List[Vector]:
        if not texts:
            return []

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        requests: List[Dict[str, Any]] = []
        for text in texts:
            request: Dict[str, Any] = {
                "model": self._model_path,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            if self.dimensions:
                request["outputDimensionality"] = self.dimensions
            requests.append(request)

        url = f"{self.base_url}/{self._model_path}:batchEmbedContents"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        body = retry_call(
            lambda: post_json(
                url,
                {"requests": requests},
                headers=headers,
                timeout_s=self.timeout_s,
                provider="embeddings",
            ),
            attempts=3,
            operation=f"gemini/batchEmbedContents ({self.model})",
        )

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError(
                f"Unexpected Gemini embeddings payload: {str(body)[:200]}", provider=self.name
            )
        vectors = [list(item.get("values") or []) for item in embeddings]
        if any(not v for v in vectors):
            raise EmbeddingError("Gemini returned an empty vector", provider=self.name)
        return vectors
