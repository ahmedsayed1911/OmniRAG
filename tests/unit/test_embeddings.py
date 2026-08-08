"""Embedding provider selection and controlled fallback behavior."""

from __future__ import annotations

import pytest

from omnirag.core.exceptions import (
    EmbeddingError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.embeddings.gemini import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    GeminiEmbeddings,
)
from omnirag.providers.embeddings.hashing import DEFAULT_DIMENSIONS, HashingEmbeddings
from omnirag.providers.embeddings.resilient import ResilientEmbeddings


class FailingEmbeddings(BaseEmbeddingProvider):
    name = "remote"

    def __init__(self, exc: BaseException):
        super().__init__(model="remote-model", batch_size=2)
        self.exc = exc
        self.calls = 0

    def embed_batch(self, texts, *, is_query=False):
        self.calls += 1
        raise self.exc


def test_gemini_embedding_success_uses_supported_model_and_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, payload, **kwargs):
        captured.update(url=url, payload=payload, headers=kwargs["headers"])
        return {"embeddings": [{"values": [0.25, 0.75]}]}

    monkeypatch.setattr("omnirag.providers.embeddings.gemini.post_json", fake_post)
    provider = GeminiEmbeddings(api_key="test-key")

    assert provider.embed_documents(["hello"]) == [[0.25, 0.75]]
    assert DEFAULT_MODEL == "gemini-embedding-001"
    assert captured["url"] == (
        f"{DEFAULT_BASE_URL}/models/gemini-embedding-001:batchEmbedContents"
    )
    assert captured["payload"]["requests"][0]["model"] == "models/gemini-embedding-001"
    assert captured["payload"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


@pytest.mark.parametrize(
    "failure",
    [
        RateLimitError("quota", provider="remote", quota_exhausted=True),
        ProviderUnavailableError("server error", provider="remote"),
    ],
)
def test_provider_429_or_5xx_activates_sticky_hash_fallback(failure):
    primary = FailingEmbeddings(failure)
    provider = ResilientEmbeddings(primary)

    vectors = provider.embed_documents(["alpha", "beta"])
    query = provider.embed_query("alpha")

    assert provider.fallback_active
    assert provider.name == "hash"
    assert len(vectors) == 2
    assert all(len(vector) == DEFAULT_DIMENSIONS for vector in vectors)
    assert len(query) == DEFAULT_DIMENSIONS
    assert primary.calls == 1  # query remains in the hash vector space


def test_malformed_gemini_response_is_not_silently_hidden(monkeypatch):
    monkeypatch.setattr(
        "omnirag.providers.embeddings.gemini.post_json",
        lambda *args, **kwargs: {"unexpected": []},
    )
    provider = ResilientEmbeddings(GeminiEmbeddings(api_key="test-key"))

    with pytest.raises(EmbeddingError, match="Unexpected Gemini"):
        provider.embed_documents(["hello"])

    assert not provider.fallback_active


def test_programming_error_is_not_silently_hidden():
    provider = ResilientEmbeddings(FailingEmbeddings(ValueError("bug")))

    with pytest.raises(ValueError, match="bug"):
        provider.embed_documents(["hello"])

    assert not provider.fallback_active


def test_hash_embeddings_have_fixed_expected_dimension():
    provider = HashingEmbeddings()
    vectors = provider.embed_documents(["alpha", "", "مرحبا"])

    assert provider.dimensions == DEFAULT_DIMENSIONS
    assert len(vectors) == 3
    assert all(len(vector) == DEFAULT_DIMENSIONS for vector in vectors)
