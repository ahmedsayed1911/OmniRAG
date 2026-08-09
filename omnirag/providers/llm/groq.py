"""Groq adapter using the official OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import Optional

from omnirag.providers.llm.openai_compat import OpenAICompatibleLLM

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"

# Unknown models are text-only until an operator explicitly opts in with
# GROQ_MODEL_SUPPORTS_IMAGES=true or configures a known vision model.
VISION_MODELS = frozenset({DEFAULT_VISION_MODEL})


def model_supports_images(model: str) -> bool:
    return (model or "").strip().lower() in VISION_MODELS


class GroqLLM(OpenAICompatibleLLM):
    """Text and vision generation through GroqCloud."""

    name = "groq"
    max_tokens_field = "max_completion_tokens"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = DEFAULT_VISION_MODEL,
        retry_attempts: int = 2,
        supports_images_override: Optional[bool] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            vision_model=vision_model or DEFAULT_VISION_MODEL,
            retry_attempts=retry_attempts,
        )
        self._supports_images_override = supports_images_override
        self.supports_vision = self.supports_images()

    def supports_images(self, model: Optional[str] = None) -> bool:
        if self._supports_images_override is not None:
            return self._supports_images_override
        return model_supports_images(model or self.vision_model)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_VISION_MODEL",
    "GroqLLM",
    "model_supports_images",
]
