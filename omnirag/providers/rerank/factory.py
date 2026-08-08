"""Reranker selection with graceful degradation.

``RERANK_PROVIDER=auto`` (the default) resolves to:

1. Cohere / Jina when a rerank API key is present — best quality;
2. the LLM reranker when an LLM chain is configured — good quality, one call;
3. the heuristic reranker — always available, no cost, no key.

Never raises for a missing key: reranking is an enhancement, and losing it must
degrade quality, not break the answer.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from omnirag.config.settings import AppSettings, RerankSettings, get_settings
from omnirag.providers.rerank.base import BaseReranker
from omnirag.providers.rerank.cohere import CohereReranker
from omnirag.providers.rerank.heuristic import HeuristicReranker
from omnirag.providers.rerank.jina import JinaReranker
from omnirag.providers.rerank.llm_reranker import LLMReranker
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

_cache: Dict[str, BaseReranker] = {}
_lock = threading.Lock()

SUPPORTED_PROVIDERS = ("auto", "cohere", "jina", "llm", "heuristic", "none")


def build_reranker(
    cfg: RerankSettings, *, top_n: int = 8, settings: Optional[AppSettings] = None
) -> BaseReranker:
    """Build the best reranker available for the current configuration."""
    provider = (cfg.provider or "auto").lower()

    if not cfg.enabled or provider == "none":
        return HeuristicReranker(top_n=top_n)

    if provider == "heuristic":
        return HeuristicReranker(top_n=top_n)

    if provider in ("cohere", "jina") and not cfg.api_key:
        logger.warning(
            "RERANK_PROVIDER=%s but no RERANK_API_KEY is set — using the heuristic reranker",
            provider,
        )
        return HeuristicReranker(top_n=top_n)

    if provider == "cohere":
        return CohereReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )
    if provider == "jina":
        return JinaReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )

    if provider == "llm":
        llm = _try_llm(settings)
        if llm is not None:
            return LLMReranker(llm, top_n=top_n)
        return HeuristicReranker(top_n=top_n)

    # --- auto ---------------------------------------------------------- #
    if cfg.api_key:
        model = (cfg.model or "").lower()
        if "jina" in model:
            return JinaReranker(
                api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
                top_n=top_n, timeout_s=cfg.timeout_s,
            )
        return CohereReranker(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            top_n=top_n, timeout_s=cfg.timeout_s,
        )

    llm = _try_llm(settings)
    if llm is not None:
        return LLMReranker(llm, top_n=top_n)

    return HeuristicReranker(top_n=top_n)


def _try_llm(settings: Optional[AppSettings]):
    """Return the LLM router, or ``None`` when no usable LLM is configured."""
    try:
        from omnirag.providers.llm.factory import get_llm_provider

        resolved = settings or get_settings()
        if not resolved.llm.is_configured:
            return None
        # The offline mock cannot score relevance; the heuristic reranker is
        # both better and cheaper in that configuration.
        if all(e.provider == "mock" for e in resolved.llm.configured_endpoints):
            return None
        return get_llm_provider(resolved)
    except Exception as exc:  # configuration problems must not break retrieval
        logger.info("LLM reranker unavailable (%s) — falling back to heuristic", exc)
        return None


def get_reranker(settings: Optional[AppSettings] = None) -> BaseReranker:
    resolved = settings or get_settings()
    cfg = resolved.rerank
    key = f"{cfg.provider}|{cfg.model}|{len(cfg.api_key)}|{cfg.enabled}|{resolved.retrieval.rerank_top_k}"
    with _lock:
        reranker = _cache.get(key)
        if reranker is None:
            reranker = build_reranker(
                cfg, top_n=resolved.retrieval.rerank_top_k, settings=resolved
            )
            _cache[key] = reranker
            logger.info("Reranker ready: %s (%s)", reranker.name, reranker.model)
        return reranker


def reset_rerank_cache() -> None:
    with _lock:
        _cache.clear()
