"""Provider router with conservative, classified failover.

The rest of OmniRAG depends only on :class:`~omnirag.providers.llm.base.BaseLLMProvider`.
This adapter *is* one, and internally holds an ordered chain of real providers
(Gemini primary, OpenRouter fallback by default).

Failover policy
---------------
A provider is abandoned for the next one **only** when the failure is classified
as :attr:`~omnirag.providers.errors.FailureClass.RECOVERABLE` — 429/quota, 5xx,
timeout, transient network fault, temporary model unavailability. Auth failures,
malformed requests, safety refusals and ordinary Python bugs propagate
immediately, because a second vendor would fail identically (or, worse, hide a
defect).

Multimodal policy
-----------------
When a request carries images, providers whose selected model cannot read images
are skipped rather than being sent a silently text-only request. If no provider
in the chain can see the images, a :class:`ProviderCapabilityError` explains
exactly what to change — the visual evidence is never dropped without saying so.

Latency policy
--------------
Each provider does at most a couple of quick internal retries; the router adds
no sleeps of its own. Worst case is bounded so the Streamlit UI never hangs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from omnirag.core.exceptions import (
    AllProvidersFailedError,
    ProviderCapabilityError,
    RateLimitError,
)
from omnirag.providers.errors import FailureClass, classify, describe
from omnirag.providers.llm.base import (
    BaseLLMProvider, LLMMessage, LLMRequestRequirements, LLMResponse,
)
from omnirag.providers.llm.context import current_llm_operation, current_llm_session_id
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderAttempt:
    """One entry of the failover trail (safe to log and to show in the UI)."""

    provider: str
    model: str
    outcome: str                      # ok | skipped | failed
    operation: str = "unspecified"
    failure_class: Optional[str] = None
    error_type: Optional[str] = None
    duration_ms: float = 0.0

    def __str__(self) -> str:
        if self.outcome == "ok":
            return f"{self.operation}/{self.provider}: ok ({self.duration_ms:.0f} ms)"
        if self.outcome == "skipped":
            return f"{self.operation}/{self.provider}: skipped ({self.failure_class})"
        return f"{self.operation}/{self.provider}: {self.error_type} [{self.failure_class}]"


@dataclass
class RouterStats:
    """Lightweight counters powering the UI's provider indicator."""

    calls: int = 0
    failovers: int = 0
    by_provider: Dict[str, int] = field(default_factory=dict)
    last_provider: str = ""
    last_model: str = ""
    last_attempts: List[str] = field(default_factory=list)


class FallbackLLMProvider(BaseLLMProvider):
    """Ordered chain of providers presented as a single provider."""

    name = "router"

    def __init__(
        self,
        providers: Sequence[BaseLLMProvider],
        *,
        enable_fallback: bool = True,
        rate_limit_cooldown_seconds: float = 60.0,
        clock=time.monotonic,
    ):
        active = [p for p in providers if p is not None]
        if not active:
            raise ValueError("FallbackLLMProvider requires at least one provider")

        primary = active[0]
        super().__init__(
            model=primary.model,
            temperature=primary.temperature,
            max_output_tokens=primary.max_output_tokens,
        )
        self.providers: List[BaseLLMProvider] = list(active)
        self.enable_fallback = enable_fallback and len(self.providers) > 1
        self.supports_vision = any(p.supports_vision for p in self.providers)
        self.stats = RouterStats()
        self._lock = threading.Lock()
        self.rate_limit_cooldown_seconds = max(0.0, rate_limit_cooldown_seconds)
        self._clock = clock
        self._rate_limit_cooldowns: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    @property
    def primary(self) -> BaseLLMProvider:
        return self.providers[0]

    @property
    def chain(self) -> List[BaseLLMProvider]:
        """Providers actually eligible for a call (respects the fallback flag)."""
        return self.providers if self.enable_fallback else self.providers[:1]

    def supports_images(self, model: Optional[str] = None) -> bool:
        return any(p.supports_images() for p in self.chain)

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.primary.model,
            "vision": self.supports_vision,
            "images": self.supports_images(),
            "chain": [
                {"provider": p.name, "model": p.model, "images": p.supports_images()}
                for p in self.chain
            ],
            "fallback_enabled": self.enable_fallback,
        }

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
        needs_images = any(m.has_images for m in messages)
        operation = current_llm_operation()
        requirements = requirements or LLMRequestRequirements(
            requires_text=True,
            requires_images=needs_images,
            requires_structured_output=json_mode,
            operation=operation,
        )
        needs_images = requirements.requires_images
        session_id = current_llm_session_id()
        attempts: List[ProviderAttempt] = []
        failures: List[tuple[str, BaseException]] = []
        capability_error: Optional[ProviderCapabilityError] = None
        cooldown_error: Optional[RateLimitError] = None

        for index, provider in enumerate(self.chain):
            if provider.name == "gemini" and self._cooldown_active(session_id):
                attempts.append(ProviderAttempt(
                    provider=provider.name,
                    model=model or provider.model,
                    operation=operation,
                    outcome="skipped",
                    failure_class="rate_limit_cooldown",
                ))
                logger.info(
                    "LLM operation=%s provider=gemini skipped=session_rate_limit_cooldown",
                    operation,
                )
                cooldown_error = RateLimitError(
                    "Gemini is temporarily skipped after a session rate limit",
                    provider="gemini",
                    quota_exhausted=True,
                )
                continue
            # -- capability gate: never silently drop visual evidence -------- #
            if needs_images and not provider.supports_images(model):
                logger.info(
                    "Skipping %s for a multimodal request: model %s has no image support",
                    provider.name,
                    model or provider.model,
                )
                attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        model=model or provider.model,
                        operation=operation,
                        outcome="skipped",
                        failure_class=FailureClass.CAPABILITY.value,
                    )
                )
                capability_error = ProviderCapabilityError(
                    f"{provider.name} model '{model or provider.model}' cannot read images",
                    provider=provider.name,
                    capability="images",
                    user_message=(
                        f"The configured {provider.name} model "
                        f"`{model or provider.model}` cannot read images, so the visual "
                        "evidence for this question could not be analysed. Configure a "
                        "vision-capable model to use OmniRAG's multimodal features."
                    ),
                )
                continue

            started = time.perf_counter()
            role = "primary" if index == 0 else f"fallback #{index}"
            logger.info(
                "LLM operation=%s provider=%s role=%s model=%s base_url=%s%s",
                operation,
                provider.name,
                role,
                model or provider.model,
                getattr(provider, "base_url", "local"),
                ", multimodal" if needs_images else "",
            )
            try:
                response = provider.complete(
                    messages,
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    model=model,
                    json_mode=json_mode,
                    requirements=requirements,
                )
            except BaseException as exc:  # noqa: BLE001 - classified below
                elapsed = (time.perf_counter() - started) * 1000
                failure = classify(exc)
                attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        model=model or provider.model,
                        operation=operation,
                        outcome="failed",
                        failure_class=failure.value,
                        error_type=type(exc).__name__,
                        duration_ms=elapsed,
                    )
                )
                failures.append((provider.name, exc))

                if provider.name == "gemini" and isinstance(exc, RateLimitError):
                    self._start_cooldown(session_id)

                logger.warning(
                    "LLM operation=%s provider=%s model=%s status=%s "
                    "classified_error=%s failure_class=%s body=%s",
                    operation,
                    provider.name,
                    model or provider.model,
                    getattr(exc, "status_code", "n/a"),
                    type(exc).__name__,
                    failure.value,
                    getattr(exc, "safe_body", "<unavailable>"),
                )

                if failure is not FailureClass.RECOVERABLE:
                    # Bugs, bad keys, malformed requests and safety refusals are
                    # surfaced as-is: another vendor cannot fix them.
                    if len(failures) == 1:
                        logger.warning(
                            "%s failed with a non-recoverable error (%s) — not failing over",
                            provider.name,
                            describe(exc),
                        )
                        raise
                    # A fallback failure must not replace the primary failure.
                    # Break so AllProvidersFailedError retains the complete trail.
                    logger.error(
                        "Fallback provider %s failed non-recoverably; preserving %d failures",
                        provider.name,
                        len(failures),
                    )
                    break

                remaining = len(self.chain) - index - 1
                logger.warning("%s failed: %s", provider.name, describe(exc))
                if remaining <= 0:
                    logger.error("No fallback provider left after %s", provider.name)
                    break
                logger.info(
                    "Switching to %s fallback", self.chain[index + 1].name
                )
                continue

            elapsed = (time.perf_counter() - started) * 1000
            attempts.append(
                ProviderAttempt(
                    provider=provider.name,
                    model=response.model or provider.model,
                    operation=operation,
                    outcome="ok",
                    duration_ms=elapsed,
                )
            )
            logger.info("%s request succeeded in %.0f ms", provider.name, elapsed)

            response.provider = response.provider or provider.name
            response.fallback_used = response.fallback_used or index > 0
            response.attempts = [str(a) for a in attempts]
            self._record(response, failover=index > 0)
            return response

        # Every provider was skipped or failed.
        if capability_error is not None and not failures:
            raise capability_error
        if cooldown_error is not None and failures:
            failures.insert(0, ("gemini", cooldown_error))
        if capability_error is not None and failures:
            failures.append((capability_error.provider or "provider", capability_error))
        if not failures:
            if cooldown_error is not None:
                raise cooldown_error
            raise ProviderCapabilityError(
                "No configured provider could serve this request",
                provider=self.name,
                user_message="No configured AI provider could handle this request.",
            )
        raise AllProvidersFailedError(failures)

    def _cooldown_active(self, session_id: str) -> bool:
        if not session_id or self.rate_limit_cooldown_seconds <= 0:
            return False
        with self._lock:
            expiry = self._rate_limit_cooldowns.get(session_id, 0.0)
            if expiry <= self._clock():
                self._rate_limit_cooldowns.pop(session_id, None)
                return False
            return True

    def _start_cooldown(self, session_id: str) -> None:
        if not session_id or self.rate_limit_cooldown_seconds <= 0:
            return
        with self._lock:
            self._rate_limit_cooldowns[session_id] = (
                self._clock() + self.rate_limit_cooldown_seconds
            )

    # ------------------------------------------------------------------ #
    def _record(self, response: LLMResponse, *, failover: bool) -> None:
        with self._lock:
            self.stats.calls += 1
            if failover:
                self.stats.failovers += 1
            provider = response.provider or "unknown"
            self.stats.by_provider[provider] = self.stats.by_provider.get(provider, 0) + 1
            self.stats.last_provider = provider
            self.stats.last_model = response.model
            self.stats.last_attempts = list(response.attempts)
