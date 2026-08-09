"""Groq adapter using the official OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import Optional, Sequence

from omnirag.core.exceptions import ProviderTokenBudgetExceededError
from omnirag.providers.llm.base import LLMMessage, LLMRequestRequirements, LLMResponse
from omnirag.providers.llm.openai_compat import OpenAICompatibleLLM
from omnirag.utils.logging import get_logger
from omnirag.utils.text import estimate_tokens

logger = get_logger(__name__)

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
        max_rate_limit_wait_seconds: float = 20.0,
        tpm_limit: int = 8000,
        estimated_image_tokens: int = 2048,
        focused_vision_max_output_tokens: int = 1024,
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
        self.retry_max_delay = max(0.0, max_rate_limit_wait_seconds)
        self.skip_if_retry_after_exceeds_max = True
        self.tpm_limit = max(0, tpm_limit)
        self.estimated_image_tokens = max(0, estimated_image_tokens)
        self.focused_vision_max_output_tokens = max(
            128, focused_vision_max_output_tokens
        )
        self.supports_vision = self.supports_images()

    def supports_images(self, model: Optional[str] = None) -> bool:
        if self._supports_images_override is not None:
            return self._supports_images_override
        return model_supports_images(model or self.vision_model)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        requirements: Optional[LLMRequestRequirements] = None,
    ) -> LLMResponse:
        requested = max_output_tokens or self.max_output_tokens
        operation = requirements.operation if requirements else "unspecified"
        image_count = sum(len(message.images) for message in messages)
        input_estimate = estimate_tokens(system or "") + sum(
            estimate_tokens(message.text)
            + len(message.images) * self.estimated_image_tokens
            for message in messages
        )
        effective = requested
        if image_count and operation in {
            "final_answer",
            "final_answer_continuation",
        }:
            effective = min(effective, self.focused_vision_max_output_tokens)
        if self.tpm_limit:
            safe_limit = max(256, int(self.tpm_limit * 0.90))
            available = safe_limit - input_estimate - 128
            if available < 128:
                raise ProviderTokenBudgetExceededError(
                    "Groq request skipped before HTTP: estimated input exceeds TPM window",
                    provider=self.name,
                )
            effective = min(effective, max(128, available))
            if effective < requested:
                if requested > self.max_output_tokens:
                    raise ProviderTokenBudgetExceededError(
                        "Groq request skipped before HTTP: exhaustive output budget "
                        "does not fit the estimated TPM window",
                        provider=self.name,
                    )
                logger.info(
                    "LLM operation=%s provider=groq token_budget requested=%d "
                    "input_estimate=%d effective=%d estimated_total=%d "
                    "selected_visuals=%d safe_tpm_limit=%d tpm_limit=%d",
                    operation,
                    requested,
                    input_estimate,
                    effective,
                    input_estimate + effective,
                    image_count,
                    safe_limit,
                    self.tpm_limit,
                )
        try:
            response = super().complete(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=effective,
                model=model,
                json_mode=json_mode,
                requirements=requirements,
            )
        except ProviderTokenBudgetExceededError:
            # One bounded retry can salvage a focused visual answer without
            # dropping its exact-page evidence or image. Identical retries are
            # disabled for this exception; a second 413 reaches the router.
            if not image_count or operation not in {
                "final_answer",
                "final_answer_continuation",
            } or effective <= 512:
                raise
            reduced = max(256, min(512, effective // 2))
            logger.warning(
                "LLM operation=%s provider=groq retry=token_budget_reduction "
                "previous_output=%d reduced_output=%d selected_visuals=%d",
                operation,
                effective,
                reduced,
                image_count,
            )
            response = super().complete(
                messages,
                system=system,
                temperature=temperature,
                max_output_tokens=reduced,
                model=model,
                json_mode=json_mode,
                requirements=requirements,
            )
            effective = reduced

        response.diagnostics.setdefault("estimated_input_tokens", input_estimate)
        response.diagnostics.setdefault("requested_output_tokens", requested)
        response.diagnostics.setdefault("effective_output_tokens", effective)
        response.diagnostics.setdefault(
            "estimated_total_tokens", input_estimate + effective
        )
        response.diagnostics.setdefault("selected_visuals", image_count)
        return response


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_VISION_MODEL",
    "GroqLLM",
    "model_supports_images",
]
