"""Embedding provider selection."""

from __future__ import annotations

import threading
from typing import Dict, Optional

from omnirag.config.settings import AppSettings, EmbeddingSettings, get_settings
from omnirag.core.exceptions import ConfigurationError, MissingCredentialError
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.embeddings.cohere import CohereEmbeddings
from omnirag.providers.embeddings.gemini import GeminiEmbeddings
from omnirag.providers.embeddings.hashing import HashingEmbeddings
from omnirag.providers.embeddings.openai_compat import OpenAICompatibleEmbeddings
from omnirag.providers.embeddings.resilient import ResilientEmbeddings
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

_cache: Dict[str, BaseEmbeddingProvider] = {}
_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("openai", "openai_compatible", "gemini", "cohere", "jina", "hash", "mock")


def build_embedding_provider(cfg: EmbeddingSettings) -> BaseEmbeddingProvider:
    """Instantiate the configured embedding backend (no caching)."""
    provider = (cfg.provider or "openai").lower()

    if provider in ("hash", "mock"):
        return HashingEmbeddings(
            model=cfg.model or "hash-1024",
            dimensions=cfg.dimensions or 1024,
            batch_size=cfg.batch_size,
            max_chars=cfg.max_chars_per_input,
        )

    if not cfg.api_key:
        raise MissingCredentialError("EMBEDDING_API_KEY", "document indexing")

    common = dict(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        dimensions=cfg.dimensions,
        batch_size=cfg.batch_size,
        timeout_s=cfg.timeout_s,
        max_chars=cfg.max_chars_per_input,
    )

    if provider in ("openai", "openai_compatible"):
        primary = OpenAICompatibleEmbeddings(**common)
        return ResilientEmbeddings(primary)
    if provider == "gemini":
        return ResilientEmbeddings(GeminiEmbeddings(**common))
    if provider == "cohere":
        return ResilientEmbeddings(CohereEmbeddings(**common))
    if provider == "jina":
        # Jina serves an OpenAI-compatible /embeddings endpoint.
        common["base_url"] = cfg.base_url or "https://api.jina.ai/v1"
        return ResilientEmbeddings(OpenAICompatibleEmbeddings(**common))

    raise ConfigurationError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}",
        user_message=(
            f"`EMBEDDING_PROVIDER={provider}` is not supported. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        ),
    )


def get_embedding_provider(settings: Optional[AppSettings] = None) -> BaseEmbeddingProvider:
    cfg = (settings or get_settings()).embedding
    key = f"{cfg.provider}|{cfg.model}|{cfg.dimensions}|{len(cfg.api_key)}|{cfg.base_url}"
    with _lock:
        provider = _cache.get(key)
        if provider is None:
            provider = build_embedding_provider(cfg)
            _cache[key] = provider
            logger.info("Embedding provider ready: %s (%s)", provider.name, provider.model)
        return provider


def reset_embedding_cache() -> None:
    with _lock:
        _cache.clear()
