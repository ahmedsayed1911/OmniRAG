"""External provider adapters (LLM, embeddings, OCR, reranking).

Each sub-package exposes a `Base*Provider` interface plus a `factory` that picks
an implementation from configuration. The RAG engine imports only the
interfaces, never a concrete vendor.
"""

from omnirag.providers.errors import FailureClass, classify, should_failover

__all__ = ["FailureClass", "classify", "should_failover"]
