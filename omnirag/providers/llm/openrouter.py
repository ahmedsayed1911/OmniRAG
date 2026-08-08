"""OpenRouter adapter — the fallback path of the provider chain.

OpenRouter speaks the OpenAI chat-completions dialect, so this subclasses
:class:`~omnirag.providers.llm.openai_compat.OpenAICompatibleLLM` and adds:

* the OpenRouter base URL and the ``HTTP-Referer`` / ``X-Title`` attribution
  headers the service asks integrators to send;
* **model-level image-capability detection**, because OpenRouter proxies
  hundreds of models and many are text-only. A multimodal request routed to a
  text-only model must fail loudly (:class:`ProviderCapabilityError`) rather
  than quietly discarding the page screenshot the answer depends on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.exceptions import ProviderCapabilityError
from omnirag.providers.llm.base import LLMMessage, LLMResponse
from omnirag.providers.llm.openai_compat import OpenAICompatibleLLM
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

#: Substrings of OpenRouter model slugs that are known to accept image input.
#: Matching is conservative — an unknown model is treated as text-only so we
#: fail with a clear message instead of dropping evidence.
VISION_MODEL_MARKERS: tuple[str, ...] = (
    "gemini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-5",
    "o3",
    "o4-mini",
    "claude-3",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "pixtral",
    "llama-3.2-11b",
    "llama-3.2-90b",
    "llama-4",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "internvl",
    "molmo",
    "grok-2-vision",
    "grok-4",
    "mistral-medium-3",
    "-vision",
    "vision-",
    "multimodal",
)

#: Explicitly text-only families that would otherwise match a marker above.
TEXT_ONLY_MARKERS: tuple[str, ...] = (
    "gemma-2",
    "deepseek-r1-distill",
    "text-embedding",
)


def model_supports_images(model: str) -> bool:
    """Best-effort capability check from an OpenRouter model slug."""
    slug = (model or "").lower()
    if not slug:
        return False
    if any(marker in slug for marker in TEXT_ONLY_MARKERS):
        return False
    return any(marker in slug for marker in VISION_MODEL_MARKERS)


class OpenRouterLLM(OpenAICompatibleLLM):
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = "",
        retry_attempts: int = 2,
        supports_images_override: Optional[bool] = None,
        app_url: str = "https://github.com/",
        app_title: str = "OmniRAG",
    ):
        super().__init__(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            vision_model=vision_model,
            retry_attempts=retry_attempts,
        )
        self._supports_images_override = supports_images_override
        self.app_url = app_url
        self.app_title = app_title
        self.supports_vision = True  # the API transport always can

    # -- capability -------------------------------------------------------- #
    def supports_images(self, model: Optional[str] = None) -> bool:
        if self._supports_images_override is not None:
            return self._supports_images_override
        return model_supports_images(model or self.vision_model or self.model)

    # -- transport --------------------------------------------------------- #
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Attribution headers requested by OpenRouter; they carry no secrets.
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_title,
        }

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        has_images = any(m.has_images for m in messages)
        target_model = model or (self.vision_model if has_images else self.model)

        if has_images and not self.supports_images(target_model):
            raise ProviderCapabilityError(
                f"OpenRouter model '{target_model}' does not accept image input",
                provider="openrouter",
                capability="images",
                user_message=(
                    f"The configured OpenRouter model `{target_model}` cannot read images, "
                    "so the visual evidence for this request could not be analysed. "
                    "Set `OPENROUTER_MODEL` to a vision-capable model "
                    "(for example `google/gemini-2.0-flash-001` or `openai/gpt-4o-mini`)."
                ),
            )

        return super().complete(
            messages,
            system=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model=target_model,
            json_mode=json_mode,
        )
