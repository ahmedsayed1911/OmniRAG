"""Embedding provider adapters."""

from omnirag.providers.embeddings.base import BaseEmbeddingProvider, Vector
from omnirag.providers.embeddings.cohere import CohereEmbeddings
from omnirag.providers.embeddings.factory import (
    SUPPORTED_PROVIDERS,
    build_embedding_provider,
    get_embedding_provider,
    reset_embedding_cache,
)
from omnirag.providers.embeddings.gemini import GeminiEmbeddings
from omnirag.providers.embeddings.hashing import HashingEmbeddings
from omnirag.providers.embeddings.openai_compat import OpenAICompatibleEmbeddings

__all__ = [
    "BaseEmbeddingProvider",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "HashingEmbeddings",
    "OpenAICompatibleEmbeddings",
    "SUPPORTED_PROVIDERS",
    "Vector",
    "build_embedding_provider",
    "get_embedding_provider",
    "reset_embedding_cache",
]
