"""LLM provider interface.

Nothing outside ``omnirag/providers/llm`` knows which vendor is in use. The
engine builds vendor-neutral :class:`LLMMessage` objects (text + optional
images) and receives an :class:`LLMResponse`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.enums import Role


@dataclass
class ImagePart:
    """An image attached to a message.

    ``data`` holds the raw bytes; providers encode them as base64 in whatever
    shape their API expects. ``label`` is rendered as adjacent text so the model
    can tie a picture to its citation (e.g. ``[report.pdf — Page 12]``).
    """

    data: bytes
    media_type: str = "image/png"
    label: str = ""


@dataclass
class LLMMessage:
    role: Role = Role.USER
    text: str = ""
    images: List[ImagePart] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None
    #: Which adapter actually produced this text (``gemini``, ``openrouter``…).
    provider: str = ""
    #: True when the primary provider failed and a fallback answered instead.
    fallback_used: bool = False
    #: Human-readable trail of provider attempts, e.g.
    #: ``["gemini: RateLimitError[recoverable]", "openrouter: ok"]``.
    attempts: List[str] = field(default_factory=list)

    @property
    def provider_label(self) -> str:
        """``provider · model`` string for the UI badge."""
        if self.provider and self.model:
            return f"{self.provider} · {self.model}"
        return self.provider or self.model or "unknown"


class BaseLLMProvider(ABC):
    """Contract every LLM adapter implements."""

    name: str = "base"
    #: Whether the *adapter's API* can carry images at all.
    supports_vision: bool = False

    def __init__(self, *, model: str, temperature: float = 0.1, max_output_tokens: int = 1400):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------ #
    def supports_images(self, model: Optional[str] = None) -> bool:
        """Whether the *selected model* accepts image inputs.

        Distinct from :attr:`supports_vision`: an OpenAI-compatible endpoint can
        carry images, but the specific model configured on it may be text-only.
        Callers use this to avoid silently dropping visual evidence.
        """
        return self.supports_vision

    @abstractmethod
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
        """Generate a completion. Raises :class:`~omnirag.core.exceptions.LLMError`."""

    def complete_text(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """Convenience wrapper for single-turn text prompts."""
        response = self.complete([LLMMessage(role=Role.USER, text=prompt)], system=system, **kwargs)
        return response.text

    def health(self) -> bool:
        """Cheap availability probe; adapters may override with a real call."""
        return True

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "vision": self.supports_vision,
            "images": self.supports_images(),
        }
