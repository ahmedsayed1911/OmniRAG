"""Anthropic Messages API adapter (vision-capable)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.enums import Role
from omnirag.core.exceptions import LLMError
from omnirag.providers.http import post_json
from omnirag.providers.llm.base import (
    BaseLLMProvider, LLMMessage, LLMRequestRequirements, LLMResponse,
)
from omnirag.utils.images import to_base64
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
# Anthropic accepts these image media types only.
_SUPPORTED_MEDIA = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class AnthropicLLM(BaseLLMProvider):
    name = "anthropic"
    supports_vision = True

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
    ):
        super().__init__(model=model, temperature=temperature, max_output_tokens=max_output_tokens)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.vision_model = vision_model or model

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

        system_prompt = system or ""
        if json_mode:
            # Anthropic has no response_format flag; steer via the system prompt.
            system_prompt = (system_prompt + "\n\nRespond with a single valid JSON object and nothing else.").strip()

        payload: Dict[str, Any] = {
            "model": target_model,
            "max_tokens": max_output_tokens or self.max_output_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "messages": self._build_messages(messages),
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        body = retry_call(
            lambda: post_json(
                f"{self.base_url}/messages",
                payload,
                headers=headers,
                timeout_s=self.timeout_s,
                provider="LLM",
            ),
            attempts=3,
            operation=f"anthropic/messages ({target_model})",
        )
        return self._parse(body, target_model)

    def _build_messages(self, messages: Sequence[LLMMessage]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                continue  # carried in the top-level `system` field
            role = "assistant" if message.role == Role.ASSISTANT else "user"
            content: List[Dict[str, Any]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for image in message.images:
                media_type = (
                    image.media_type if image.media_type in _SUPPORTED_MEDIA else "image/png"
                )
                if image.label:
                    content.append({"type": "text", "text": image.label})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": to_base64(image.data),
                        },
                    }
                )
            out.append({"role": role, "content": content or [{"type": "text", "text": ""}]})

        # The Messages API requires the conversation to start with a user turn.
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out or [{"role": "user", "content": [{"type": "text", "text": ""}]}]

    def _parse(self, body: Dict[str, Any], model: str) -> LLMResponse:
        blocks = body.get("content") or []
        text = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        ).strip()
        if not text:
            raise LLMError(
                f"Empty Anthropic response: {str(body)[:300]}",
                provider="LLM",
                user_message="The model returned an empty answer. Try rephrasing your question.",
            )
        return LLMResponse(
            text=text,
            model=body.get("model", model),
            finish_reason=body.get("stop_reason", ""),
            usage=body.get("usage") or {},
        )
