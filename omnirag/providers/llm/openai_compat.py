"""OpenAI / OpenAI-compatible chat-completions adapter.

Works unchanged against OpenAI, Azure-style gateways, OpenRouter, Groq,
Together, DeepSeek, Fireworks, vLLM and Ollama's OpenAI shim — anything that
speaks ``POST /chat/completions``. Vision is sent as ``image_url`` parts using
base64 data URLs, which every one of those backends accepts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.enums import Role
from omnirag.core.exceptions import (
    LLMError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPolicyError,
    ProviderPaymentRequiredError,
    ProviderTimeoutError,
    ProviderTokenBudgetExceededError,
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
_REASONING_PART_TYPES = frozenset({"analysis", "reasoning", "thinking"})
_VISIBLE_TEXT_PART_TYPES = frozenset({"", "text", "output_text"})
_LEADING_THINK_RE = re.compile(
    r"^\s*<think(?:\s[^>]*)?>(.*?)</think>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


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

    def model_for_request(
        self, messages: Sequence[LLMMessage], model: Optional[str] = None
    ) -> str:
        return model or (
            self.vision_model if any(message.has_images for message in messages) else self.model
        )

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
        target_model = self.model_for_request(messages, model)

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
        payload.update(
            self._provider_payload(
                target_model,
                json_mode=json_mode,
                requirements=requirements,
            )
        )

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
            max_delay=getattr(self, "retry_max_delay", 2.0),
            skip_if_retry_after_exceeds_max=getattr(
                self, "skip_if_retry_after_exceeds_max", False
            ),
        )
        response = self._parse(body, target_model)
        response.diagnostics.update(diagnostics)
        response.diagnostics.setdefault("provider_raw_chars", len(response.text))
        response.diagnostics.setdefault("parsed_chars", len(response.text))
        return response

    def _provider_payload(
        self,
        model: str,
        *,
        json_mode: bool,
        requirements: Optional[LLMRequestRequirements],
    ) -> Dict[str, Any]:
        """Provider-specific request options without leaking into callers."""
        return {}

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
            raw_text, structured_reasoning_chars = _visible_message_text(message)
            text, tagged_reasoning_chars = _without_leading_thinking(raw_text)
            reasoning_chars = structured_reasoning_chars + tagged_reasoning_chars
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
            diagnostics={
                "provider_raw_chars": len(raw_text),
                "parsed_chars": len(text),
                "reasoning_chars": reasoning_chars,
                "reasoning_suppressed": bool(reasoning_chars),
            },
        )


def _visible_message_text(message: Dict[str, Any]) -> tuple[str, int]:
    """Return assistant answer text while retaining only safe reasoning metrics.

    OpenAI-compatible gateways may expose reasoning as ``message.reasoning``
    or as explicitly typed content blocks. Neither belongs in answer content.
    Untyped and ordinary text blocks remain compatible with older gateways.
    """
    reasoning_chars = _nested_text_length(message.get("reasoning"))
    content = message.get("content")
    if isinstance(content, str):
        return content, reasoning_chars
    if not isinstance(content, list):
        return "", reasoning_chars

    visible: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").strip().lower()
        part_text = part.get("text")
        if part_type in _REASONING_PART_TYPES:
            reasoning_chars += _nested_text_length(part_text)
        elif part_type in _VISIBLE_TEXT_PART_TYPES and isinstance(part_text, str):
            visible.append(part_text)
    return "".join(visible), reasoning_chars


def _without_leading_thinking(text: str) -> tuple[str, int]:
    """Remove only the provider's documented leading ``<think>`` envelope."""
    remaining = text or ""
    suppressed = 0
    while True:
        match = _LEADING_THINK_RE.match(remaining)
        if not match:
            break
        suppressed += len(match.group(1))
        remaining = remaining[match.end():]
    if re.match(r"^\s*<think(?:\s[^>]*)?>", remaining, flags=re.IGNORECASE):
        # An unterminated thinking-only response has no trustworthy answer
        # boundary. Treat it as empty so the normal provider failure path runs.
        return "", suppressed + len(remaining)
    return remaining.strip(), suppressed


def _nested_text_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_nested_text_length(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _nested_text_length(value.get(key))
            for key in ("text", "content", "reasoning")
            if key in value
        )
    return 0


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

    if status == 413 and any(
        marker in message.lower()
        for marker in ("token", "tpm", "request too large", "rate_limit_exceeded")
    ):
        raise attach_http_context(
            ProviderTokenBudgetExceededError(
                f"{provider} token budget exceeded: {message}",
                provider=provider,
            ),
            status,
            message,
        )
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
