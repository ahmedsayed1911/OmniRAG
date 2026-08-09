"""OpenAI / OpenAI-compatible chat-completions adapter.

Works unchanged against OpenAI, Azure-style gateways, OpenRouter, Groq,
Together, DeepSeek, Fireworks, vLLM and Ollama's OpenAI shim — anything that
speaks ``POST /chat/completions``. Vision is sent as ``image_url`` parts using
base64 data URLs, which every one of those backends accepts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.enums import Role
from omnirag.core.exceptions import (
    LLMError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPolicyError,
    ProviderPaymentRequiredError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.providers.http import attach_http_context, is_quota_exhausted, post_json
from omnirag.providers.llm.base import (
    BaseLLMProvider, LLMMessage, LLMRequestRequirements, LLMResponse,
)
from omnirag.utils.images import to_data_url
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Model families known to accept images. Unknown models are assumed capable —
# the alternative (silently dropping images) would break the multimodal rule.
_TEXT_ONLY_HINTS = ("embedding", "whisper", "tts", "moderation", "instruct")


class OpenAICompatibleLLM(BaseLLMProvider):
    name = "openai_compatible"
    supports_vision = True
    max_tokens_field = "max_tokens"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.1,
        max_output_tokens: int = 1400,
        timeout_s: float = 90.0,
        vision_model: str = "",
        retry_attempts: int = 2,
    ):
        super().__init__(model=model, temperature=temperature, max_output_tokens=max_output_tokens)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.vision_model = vision_model or model
        self.retry_attempts = max(1, retry_attempts)
        self.supports_vision = not any(h in model.lower() for h in _TEXT_ONLY_HINTS)

    def supports_images(self, model: Optional[str] = None) -> bool:
        target = (model or self.vision_model or self.model).lower()
        return not any(hint in target for hint in _TEXT_ONLY_HINTS)

    # ------------------------------------------------------------------ #
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
        has_images = any(m.has_images for m in messages)
        target_model = model or (self.vision_model if has_images else self.model)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": self._build_messages(messages, system),
            "temperature": self.temperature if temperature is None else temperature,
            self.max_tokens_field: max_output_tokens or self.max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            if self.name == "openrouter":
                # Ask OpenRouter to choose only routes that support the
                # structured-output parameter instead of silently ignoring it.
                payload["provider"] = {"require_parameters": True}

        headers = self._headers()

        diagnostics: Dict[str, Any] = {}

        def _call() -> Dict[str, Any]:
            return post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider=self.name,
                diagnostics=diagnostics,
            )

        body = retry_call(
            _call,
            attempts=self.retry_attempts,
            operation=f"{self.name}/chat-completions ({target_model})",
            max_delay=2.0,
        )
        response = self._parse(body, target_model)
        response.diagnostics.update(diagnostics)
        response.diagnostics.setdefault("provider_raw_chars", len(response.text))
        response.diagnostics.setdefault("parsed_chars", len(response.text))
        return response

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    def _build_messages(
        self, messages: Sequence[LLMMessage], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for message in messages:
            role = "assistant" if message.role == Role.ASSISTANT else (
                "system" if message.role == Role.SYSTEM else "user"
            )
            if not message.images:
                out.append({"role": role, "content": message.text})
                continue

            parts: List[Dict[str, Any]] = []
            if message.text:
                parts.append({"type": "text", "text": message.text})
            for image in message.images:
                if image.label:
                    parts.append({"type": "text", "text": image.label})
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(image.data, image.media_type)},
                    }
                )
            out.append({"role": role, "content": parts})
        return out

    def _parse(self, body: Dict[str, Any], model: str) -> LLMResponse:
        # Some gateways report upstream failures inside a 200 body.
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            raise_gateway_error(error, provider=self.name)

        try:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):  # some gateways return parts
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            text = (content or "").strip()
            finish_reason = choice.get("finish_reason") or ""
        except Exception as exc:
            raise LLMError(
                f"Unexpected chat/completions payload: {str(body)[:300]}",
                provider=self.name,
            ) from exc

        if finish_reason == "content_filter":
            raise ProviderPolicyError(
                f"{self.name} blocked the request (content_filter)",
                provider=self.name,
                reason="content_filter",
            )

        if not text:
            raise LLMError(
                "The model returned an empty completion",
                provider=self.name,
                user_message="The model returned an empty answer. Try rephrasing your question.",
            )

        return LLMResponse(
            text=text,
            model=body.get("model", model),
            finish_reason=finish_reason,
            usage=body.get("usage") or {},
            provider=self.name,
        )


def raise_gateway_error(error: Dict[str, Any], *, provider: str) -> None:
    """Translate an error object returned inside a 200 response.

    OpenRouter in particular surfaces upstream 429/5xx this way; classifying it
    here is what lets the router fail over instead of treating it as a bug.
    """
    message = str(error.get("message", ""))[:300]
    code = error.get("code")
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = 0

    if status == 429 or "rate limit" in message.lower():
        raise attach_http_context(RateLimitError(
            f"{provider} rate limited: {message}",
            provider=provider,
            quota_exhausted=is_quota_exhausted(message),
        ), status or 429, message)
    if status in (408, 504) or "timeout" in message.lower():
        raise attach_http_context(
            ProviderTimeoutError(f"{provider} timeout: {message}", provider=provider),
            status,
            message,
        )
    if status >= 500:
        raise attach_http_context(ProviderUnavailableError(
            f"{provider} upstream error {status}: {message}", provider=provider
        ), status, message)
    if status in (401, 403):
        raise attach_http_context(ProviderAuthError(
            f"{provider} auth error: {message}",
            provider=provider,
            user_message=f"The {provider} API rejected your credentials.",
        ), status, message)
    if status == 402:
        raise attach_http_context(ProviderPaymentRequiredError(
            f"{provider} payment required: {message}",
            provider=provider,
            user_message="The configured OpenRouter route requires credits.",
        ), status, message)
    raise attach_http_context(ProviderBadRequestError(
        f"{provider} returned an error: {message}", provider=provider
    ), status, message)
