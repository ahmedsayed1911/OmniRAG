"""Google Gemini ``generateContent`` adapter (vision-capable)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.enums import Role
from omnirag.core.exceptions import LLMError, ProviderPolicyError
from omnirag.providers.http import post_json
from omnirag.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from omnirag.utils.images import to_base64
from omnirag.utils.logging import get_logger
from omnirag.utils.retry import retry_call

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.6-flash"

#: Gemini finish reasons that are deliberate refusals, not transient faults.
POLICY_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}

#: Text-only Gemini/Gemma models exposed on the same endpoint.
_TEXT_ONLY_HINTS = ("embedding", "gemma", "aqa")


class GeminiLLM(BaseLLMProvider):
    name = "gemini"
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
        retry_attempts: int = 2,
    ):
        super().__init__(
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.vision_model = vision_model or self.model
        self.retry_attempts = max(1, retry_attempts)

    def supports_images(self, model: Optional[str] = None) -> bool:
        target = (model or self.vision_model or self.model).lower()
        return not any(hint in target for hint in _TEXT_ONLY_HINTS)

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

        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_output_tokens or self.max_output_tokens,
        }
        # Gemini 3.5/3.6 removed the legacy sampling knobs from the current API
        # contract. Sending temperature makes an otherwise valid model request
        # fail immediately with HTTP 400.
        if not target_model.startswith(("gemini-3.5-", "gemini-3.6-")):
            generation_config["temperature"] = (
                self.temperature if temperature is None else temperature
            )
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": self._build_contents(messages),
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        if has_images and not self.supports_images(target_model):
            from omnirag.core.exceptions import ProviderCapabilityError

            raise ProviderCapabilityError(
                f"Gemini model '{target_model}' does not accept image input",
                provider=self.name,
                capability="images",
                user_message=(
                    f"The configured Gemini model `{target_model}` cannot read images. "
                    "Set `GEMINI_MODEL` to a multimodal model such as `gemini-3.6-flash`."
                ),
            )

        url = f"{self.base_url}/models/{target_model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        body = retry_call(
            lambda: post_json(
                url, payload, headers=headers, timeout_s=self.timeout_s, provider=self.name
            ),
            attempts=self.retry_attempts,
            operation=f"gemini/generateContent ({target_model})",
        )
        return self._parse(body, target_model)

    def _build_contents(self, messages: Sequence[LLMMessage]) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                continue
            role = "model" if message.role == Role.ASSISTANT else "user"
            parts: List[Dict[str, Any]] = []
            if message.text:
                parts.append({"text": message.text})
            for image in message.images:
                if image.label:
                    parts.append({"text": image.label})
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": image.media_type or "image/png",
                            "data": to_base64(image.data),
                        }
                    }
                )
            contents.append({"role": role, "parts": parts or [{"text": ""}]})
        return contents or [{"role": "user", "parts": [{"text": ""}]}]

    def _parse(self, body: Dict[str, Any], model: str) -> LLMResponse:
        candidates = body.get("candidates") or []
        if not candidates:
            blocked = (body.get("promptFeedback") or {}).get("blockReason")
            if blocked:
                # A deliberate refusal: another vendor would not "fix" this, so
                # it must never trigger the fallback chain.
                raise ProviderPolicyError(
                    f"Gemini blocked the prompt (blockReason={blocked})",
                    provider=self.name,
                    reason=str(blocked),
                    user_message=(
                        "The model declined to answer this request on safety grounds. "
                        "Try rephrasing your question."
                    ),
                )
            raise LLMError(
                "Gemini returned no candidates",
                provider=self.name,
                user_message="The model returned an empty answer. Try rephrasing your question.",
            )

        candidate = candidates[0]
        finish_reason = str(candidate.get("finishReason") or "")
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

        if not text and finish_reason in POLICY_FINISH_REASONS:
            raise ProviderPolicyError(
                f"Gemini refused to answer (finishReason={finish_reason})",
                provider=self.name,
                reason=finish_reason,
                user_message=(
                    "The model declined to answer this request on safety grounds."
                ),
            )
        if not text:
            raise LLMError(
                f"Empty Gemini completion (finishReason={finish_reason})",
                provider=self.name,
                user_message="The model returned an empty answer. Try rephrasing your question.",
            )

        return LLMResponse(
            text=text,
            model=model,
            finish_reason=finish_reason,
            usage=body.get("usageMetadata") or {},
            provider=self.name,
        )
