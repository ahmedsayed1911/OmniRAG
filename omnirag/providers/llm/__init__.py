"""LLM provider adapters and the failover router.

The application always talks to :func:`get_llm_provider`, which returns a
:class:`~omnirag.providers.llm.router.FallbackLLMProvider` wrapping the
configured chain (Gemini primary → OpenRouter fallback by default).
"""

from omnirag.providers.llm.anthropic import AnthropicLLM
from omnirag.providers.llm.base import (
    BaseLLMProvider,
    ImagePart,
    LLMMessage,
    LLMResponse,
)
from omnirag.providers.llm.factory import (
    SUPPORTED_PROVIDERS,
    build_endpoint_provider,
    build_llm_provider,
    get_llm_provider,
    reset_llm_cache,
)
from omnirag.providers.llm.gemini import GeminiLLM
from omnirag.providers.llm.mock import MockLLM
from omnirag.providers.llm.openai_compat import OpenAICompatibleLLM
from omnirag.providers.llm.openrouter import OpenRouterLLM, model_supports_images
from omnirag.providers.llm.router import (
    FallbackLLMProvider,
    ProviderAttempt,
    RouterStats,
)

__all__ = [
    "AnthropicLLM",
    "BaseLLMProvider",
    "FallbackLLMProvider",
    "GeminiLLM",
    "ImagePart",
    "LLMMessage",
    "LLMResponse",
    "MockLLM",
    "OpenAICompatibleLLM",
    "OpenRouterLLM",
    "ProviderAttempt",
    "RouterStats",
    "SUPPORTED_PROVIDERS",
    "build_endpoint_provider",
    "build_llm_provider",
    "get_llm_provider",
    "model_supports_images",
    "reset_llm_cache",
]
