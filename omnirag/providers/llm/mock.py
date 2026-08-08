"""Deterministic offline LLM used by the test suite.

This is **not** a production answering path. It is only selected when
``LLM_PROVIDER=mock`` is set explicitly, and it exists so the full pipeline
(ingest → retrieve → rerank → generate → cite) can be exercised in CI without
any paid API key. It never invents facts: it extracts sentences from the
context it is given and cites them.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

from omnirag.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

_CONTEXT_RE = re.compile(r"\[(\d+)\]\s*\[(?P<label>[^\]]+)\]")


class MockLLM(BaseLLMProvider):
    """Extractive stand-in for a real model."""

    name = "mock"
    supports_vision = True

    def __init__(
        self,
        *,
        model: str = "mock-llm",
        temperature: float = 0.0,
        max_output_tokens: int = 1400,
        responder: Optional[Callable[[Sequence[LLMMessage], Optional[str]], str]] = None,
    ):
        super().__init__(model=model, temperature=temperature, max_output_tokens=max_output_tokens)
        self.responder = responder
        self.calls: List[Dict[str, object]] = []

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
        self.calls.append(
            {
                "system": system,
                "text": "\n".join(m.text for m in messages),
                "images": sum(len(m.images) for m in messages),
                "json_mode": json_mode,
            }
        )

        if self.responder is not None:
            return LLMResponse(text=self.responder(messages, system), model=self.model)

        user_text = "\n".join(m.text for m in messages if m.text)

        if json_mode:
            return LLMResponse(text="{}", model=self.model)

        return LLMResponse(text=self._extractive_answer(user_text), model=self.model)

    @staticmethod
    def _extractive_answer(prompt: str) -> str:
        """Echo the first lines of each numbered context block with citations."""
        indices = [int(m.group(1)) for m in _CONTEXT_RE.finditer(prompt)]
        if not indices:
            return "The provided documents do not contain enough information to answer this question."

        lines: List[str] = []
        blocks = re.split(r"\n(?=\[\d+\]\s*\[)", prompt)
        for block in blocks:
            match = _CONTEXT_RE.match(block.strip())
            if not match:
                continue
            body = block[match.end():].strip().split("\n")
            content = " ".join(part for part in body if part.strip())[:220]
            if content:
                lines.append(f"{content} [{match.group(1)}]")

        if not lines:
            return "The provided documents do not contain enough information to answer this question."
        return "\n\n".join(lines[:4])
