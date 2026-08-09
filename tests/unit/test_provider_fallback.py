"""Provider failover: Gemini primary -> OpenRouter fallback.

Covers the full behaviour contract:

* recoverable failures (429, quota, timeout, 5xx, network) DO fail over;
* non-recoverable failures (bad key, malformed request, safety refusal,
  programming bugs) do NOT fail over;
* every configuration combination works (both / Gemini only / OpenRouter only /
  neither);
* multimodal capability is respected instead of silently dropping images.

No API keys, no network: the HTTP layer is monkeypatched.
"""

from __future__ import annotations

import httpx
import pytest

from omnirag.config.settings import build_settings
from omnirag.core.exceptions import (
    AllProvidersFailedError,
    MissingCredentialError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderCapabilityError,
    ProviderPolicyError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.providers.errors import FailureClass, classify, should_failover
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage, LLMResponse
from omnirag.providers.llm.factory import build_llm_provider
from omnirag.providers.llm.gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL, GeminiLLM
from omnirag.providers.llm.openrouter import (
    DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL,
    OpenRouterLLM,
    model_supports_images,
)
from omnirag.providers.llm.router import FallbackLLMProvider


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ScriptedProvider(BaseLLMProvider):
    """Provider that raises a queued exception or returns a canned answer."""

    def __init__(self, name: str, *, outcomes, images: bool = True, model: str = "m"):
        super().__init__(model=model)
        self.name = name
        self.supports_vision = images
        self._images = images
        self.outcomes = list(outcomes)
        self.call_count = 0

    def supports_images(self, model=None) -> bool:
        return self._images

    def complete(self, messages, *, system=None, temperature=None,
                 max_output_tokens=None, model=None, json_mode=False):
        self.call_count += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(text=str(outcome), model=self.model, provider=self.name)


def message(text: str = "hi", *, with_image: bool = False) -> list[LLMMessage]:
    images = [ImagePart(data=b"\x89PNG-fake", media_type="image/png")] if with_image else []
    return [LLMMessage(text=text, images=images)]


def configure(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return build_settings()


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
class TestErrorClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            RateLimitError("429", provider="gemini"),
            RateLimitError("quota", provider="gemini", quota_exhausted=True),
            ProviderTimeoutError("timeout", provider="gemini"),
            ProviderUnavailableError("503", provider="gemini"),
        ],
    )
    def test_recoverable_errors_fail_over(self, exc):
        assert classify(exc) is FailureClass.RECOVERABLE
        assert should_failover(exc) is True

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ProviderAuthError("401", provider="gemini"), FailureClass.AUTH),
            (ProviderBadRequestError("400", provider="gemini"), FailureClass.BAD_REQUEST),
            (ProviderPolicyError("safety", provider="gemini"), FailureClass.POLICY),
            (ProviderCapabilityError("no images", provider="openrouter"), FailureClass.CAPABILITY),
            (ValueError("bug in our code"), FailureClass.BUG),
            (TypeError("bug"), FailureClass.BUG),
        ],
    )
    def test_non_recoverable_errors_do_not_fail_over(self, exc, expected):
        assert classify(exc) is expected
        assert should_failover(exc) is False

    def test_quota_exhaustion_is_not_retried_on_the_same_provider(self):
        from omnirag.providers.errors import should_retry_same_provider

        transient = RateLimitError("slow down", provider="gemini")
        exhausted = RateLimitError("quota", provider="gemini", quota_exhausted=True)

        assert should_retry_same_provider(transient) is True
        # Waiting will not refill a quota; switch provider instead of hanging.
        assert should_retry_same_provider(exhausted) is False


# --------------------------------------------------------------------------- #
# Router failover
# --------------------------------------------------------------------------- #
class TestRouterFailover:
    def test_gemini_success_never_touches_the_fallback(self):
        gemini = ScriptedProvider("gemini", outcomes=["primary answer"])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        response = router.complete(message())

        assert response.text == "primary answer"
        assert response.provider == "gemini"
        assert response.fallback_used is False
        assert openrouter.call_count == 0
        assert router.stats.failovers == 0

    @pytest.mark.parametrize(
        "failure",
        [
            RateLimitError("429 rate limited", provider="gemini"),
            RateLimitError("RESOURCE_EXHAUSTED", provider="gemini", quota_exhausted=True),
            ProviderTimeoutError("deadline exceeded", provider="gemini"),
            ProviderUnavailableError("503 model overloaded", provider="gemini"),
            ProviderUnavailableError("connection reset", provider="gemini"),
        ],
        ids=["rate_limit", "quota", "timeout", "server_error", "network"],
    )
    def test_recoverable_gemini_failure_falls_back_to_openrouter(self, failure):
        gemini = ScriptedProvider("gemini", outcomes=[failure])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        response = router.complete(message())

        assert response.text == "fallback answer"
        assert response.provider == "openrouter"
        assert response.fallback_used is True
        assert gemini.call_count == 1
        assert openrouter.call_count == 1
        assert router.stats.failovers == 1
        assert len(response.attempts) == 2

    def test_invalid_api_key_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderAuthError("401 invalid key", provider="gemini")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderAuthError):
            router.complete(message())

        # A second vendor cannot fix a bad key — it must not be charged for it.
        assert openrouter.call_count == 0

    def test_policy_refusal_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderPolicyError("blocked", provider="gemini", reason="SAFETY")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback answer"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderPolicyError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_malformed_request_does_not_fall_back(self):
        gemini = ScriptedProvider(
            "gemini", outcomes=[ProviderBadRequestError("400 bad model", provider="gemini")]
        )
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ProviderBadRequestError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_programming_bug_propagates_untouched(self):
        gemini = ScriptedProvider("gemini", outcomes=[ValueError("our bug")])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(ValueError):
            router.complete(message())
        assert openrouter.call_count == 0

    def test_fallback_failure_reports_every_provider(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        openrouter = ScriptedProvider(
            "openrouter", outcomes=[ProviderUnavailableError("503", provider="openrouter")]
        )
        router = FallbackLLMProvider([gemini, openrouter])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(message())

        failures = dict(excinfo.value.failures)
        assert set(failures) == {"gemini", "openrouter"}
        assert excinfo.value.user_message

    def test_nonrecoverable_fallback_failure_preserves_primary_failure(self):
        primary_error = ProviderUnavailableError("503", provider="gemini")
        fallback_error = ProviderBadRequestError("404 model", provider="openrouter")
        router = FallbackLLMProvider([
            ScriptedProvider("gemini", outcomes=[primary_error]),
            ScriptedProvider("openrouter", outcomes=[fallback_error]),
        ])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(message())

        assert excinfo.value.failures == [
            ("gemini", primary_error),
            ("openrouter", fallback_error),
        ]
        assert "gemini:" in excinfo.value.user_message
        assert "openrouter:" in excinfo.value.user_message

    def test_fallback_can_be_disabled(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        openrouter = ScriptedProvider("openrouter", outcomes=["fallback"])
        router = FallbackLLMProvider([gemini, openrouter], enable_fallback=False)

        with pytest.raises(AllProvidersFailedError):
            router.complete(message())
        assert openrouter.call_count == 0


# --------------------------------------------------------------------------- #
# Multimodal capability
# --------------------------------------------------------------------------- #
class TestMultimodalCapability:
    def test_text_only_fallback_is_skipped_for_image_requests(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        text_only = ScriptedProvider("openrouter", outcomes=["fallback"], images=False)
        router = FallbackLLMProvider([gemini, text_only])

        with pytest.raises(AllProvidersFailedError):
            router.complete(message(with_image=True))

        # Never sent: dropping the image silently would produce a confident
        # answer about a picture the model never saw.
        assert text_only.call_count == 0

    def test_text_only_fallback_still_serves_text_requests(self):
        gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
        text_only = ScriptedProvider("openrouter", outcomes=["fallback"], images=False)
        router = FallbackLLMProvider([gemini, text_only])

        response = router.complete(message(with_image=False))
        assert response.text == "fallback"
        assert text_only.call_count == 1

    def test_capability_error_when_no_provider_can_see_images(self):
        text_only = ScriptedProvider("openrouter", outcomes=["never"], images=False)
        router = FallbackLLMProvider([text_only])

        with pytest.raises(ProviderCapabilityError) as excinfo:
            router.complete(message(with_image=True))

        assert "cannot read images" in excinfo.value.user_message
        assert text_only.call_count == 0

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("google/gemini-3.6-flash", True),
            ("openai/gpt-4o-mini", True),
            ("anthropic/claude-sonnet-4.5", True),
            ("mistralai/pixtral-12b", True),
            ("qwen/qwen2.5-vl-72b-instruct", True),
            ("meta-llama/llama-3.1-8b-instruct", False),
            ("deepseek/deepseek-r1-distill-llama-70b", False),
            ("google/gemma-2-27b-it", False),
            ("", False),
        ],
    )
    def test_openrouter_model_capability_detection(self, model, expected):
        assert model_supports_images(model) is expected

    def test_openrouter_raises_before_calling_a_text_only_model(self):
        provider = OpenRouterLLM(
            api_key="test-key", model="meta-llama/llama-3.1-8b-instruct"
        )
        with pytest.raises(ProviderCapabilityError) as excinfo:
            provider.complete(message(with_image=True))
        assert "OPENROUTER_MODEL" in excinfo.value.user_message

    def test_capability_override_is_honoured(self):
        provider = OpenRouterLLM(
            api_key="k", model="some/unknown-model", supports_images_override=True
        )
        assert provider.supports_images() is True


# --------------------------------------------------------------------------- #
# Configuration combinations
# --------------------------------------------------------------------------- #
class TestConfigurationCombinations:
    def test_both_configured_gives_gemini_primary_openrouter_fallback(self, monkeypatch):
        settings = configure(
            monkeypatch, GEMINI_API_KEY="g-key", OPENROUTER_API_KEY="or-key"
        )
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["gemini", "openrouter"]
        assert router.enable_fallback is True
        assert settings.llm.fallback_active is True

    def test_only_gemini_configured_works_without_fallback(self, monkeypatch):
        settings = configure(monkeypatch, GEMINI_API_KEY="g-key")
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["gemini"]
        assert router.enable_fallback is False
        assert settings.llm.is_configured is True

    def test_only_openrouter_configured_becomes_the_active_provider(self, monkeypatch):
        settings = configure(monkeypatch, OPENROUTER_API_KEY="or-key")
        router = build_llm_provider(settings.llm)

        assert [p.name for p in router.chain] == ["openrouter"]
        assert settings.llm.provider == "openrouter"
        assert settings.llm.is_configured is True

    def test_neither_configured_reports_a_clear_error(self, monkeypatch):
        settings = configure(monkeypatch)

        assert settings.llm.is_configured is False
        issues = settings.validation_issues()
        assert issues and "GEMINI_API_KEY" in issues[0]

        with pytest.raises(MissingCredentialError):
            build_llm_provider(settings.llm)

    def test_fallback_can_be_disabled_by_configuration(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g-key",
            OPENROUTER_API_KEY="or-key",
            ENABLE_PROVIDER_FALLBACK="false",
        )
        router = build_llm_provider(settings.llm)

        assert router.enable_fallback is False
        assert len(router.chain) == 1

    def test_models_are_configurable(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g",
            GEMINI_MODEL="gemini-2.5-pro",
            OPENROUTER_API_KEY="o",
            OPENROUTER_MODEL="openai/gpt-4o",
        )
        chain = {e.provider: e.model for e in settings.llm.configured_endpoints}
        assert chain["gemini"] == "gemini-2.5-pro"
        assert chain["openrouter"] == "openai/gpt-4o"

    def test_current_default_model_identifiers(self, monkeypatch):
        settings = configure(
            monkeypatch,
            GEMINI_API_KEY="g",
            OPENROUTER_API_KEY="o",
        )
        chain = {e.provider: e.model for e in settings.llm.configured_endpoints}

        assert GEMINI_DEFAULT_MODEL == "gemini-3.6-flash"
        assert OPENROUTER_DEFAULT_MODEL == "google/gemini-3.6-flash"
        assert chain == {
            "gemini": "gemini-3.6-flash",
            "openrouter": "google/gemini-3.6-flash",
        }

    def test_no_api_key_is_ever_hardcoded(self, monkeypatch):
        settings = configure(monkeypatch)
        for endpoint in settings.llm.endpoints:
            assert endpoint.api_key == ""


# --------------------------------------------------------------------------- #
# HTTP-level classification (adapters, still no network)
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class TestHTTPClassification:
    @pytest.fixture(autouse=True)
    def _patch_client(self, monkeypatch):
        self.responses = []
        self.requests = []

        class FakeClient:
            is_closed = False

            def post(_self, url, json=None, headers=None):
                self.requests.append({"url": url, "json": json})
                if not self.responses:
                    raise AssertionError("unexpected extra HTTP call")
                item = self.responses.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

        monkeypatch.setattr(
            "omnirag.providers.http.get_client", lambda timeout_s=60.0: FakeClient()
        )

    def test_429_becomes_a_rate_limit_error(self):
        self.responses = [FakeResponse(429, text="Too many requests")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.quota_exhausted is False

    def test_429_with_resource_exhausted_marks_quota(self):
        self.responses = [FakeResponse(429, text='{"error":{"status":"RESOURCE_EXHAUSTED"}}')]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.quota_exhausted is True

    def test_401_becomes_an_auth_error(self):
        self.responses = [FakeResponse(401, text="API key not valid")]
        provider = GeminiLLM(api_key="bad", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderAuthError):
            provider.complete(message())

    def test_503_becomes_a_provider_unavailable_error(self):
        self.responses = [FakeResponse(503, text="model overloaded")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.complete(message())
        assert excinfo.value.status_code == 503
        assert excinfo.value.safe_body == "model overloaded"

    def test_timeout_becomes_a_provider_timeout_error(self):
        self.responses = [httpx.ReadTimeout("timed out")]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderTimeoutError):
            provider.complete(message())

    def test_gemini_safety_block_becomes_a_policy_error(self):
        self.responses = [
            FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})
        ]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderPolicyError):
            provider.complete(message())

    def test_gemini_success_is_parsed(self):
        self.responses = [
            FakeResponse(
                200,
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello"}, {"text": " world"}]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 12,
                    },
                },
            )
        ]
        provider = GeminiLLM(api_key="k", model="gemini-3.6-flash", retry_attempts=1)
        response = provider.complete(message())

        assert response.text == "Hello world"
        assert response.provider == "gemini"
        assert response.finish_reason == "STOP"
        assert response.usage["candidatesTokenCount"] == 7
        assert self.requests[0]["url"].endswith(
            "/models/gemini-3.6-flash:generateContent"
        )
        assert "temperature" not in self.requests[0]["json"]["generationConfig"]

    def test_openrouter_content_filter_becomes_a_policy_error(self):
        self.responses = [
            FakeResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": ""}, "finish_reason": "content_filter"}
                    ]
                },
            )
        ]
        provider = OpenRouterLLM(api_key="k", model="google/gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(ProviderPolicyError):
            provider.complete(message())

    def test_openrouter_gateway_error_inside_200_is_classified(self):
        self.responses = [
            FakeResponse(200, {"error": {"code": 429, "message": "rate limit exceeded"}})
        ]
        provider = OpenRouterLLM(api_key="k", model="google/gemini-3.6-flash", retry_attempts=1)
        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(message())
        assert excinfo.value.status_code == 429
        assert excinfo.value.safe_body == "rate limit exceeded"


# --------------------------------------------------------------------------- #
# Isolation from the rest of the system
# --------------------------------------------------------------------------- #
def test_llm_failure_does_not_touch_embeddings_or_vector_store(monkeypatch, session_id):
    """A dead LLM chain must not destroy an existing index."""
    from omnirag.core.enums import FileType
    from omnirag.core.models import Chunk
    from omnirag.rag.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    chunk = Chunk(
        document_id="d1",
        session_id=session_id,
        filename="a.pdf",
        file_type=FileType.PDF,
        block_ids=["b1"],
        text="Revenue was 8,400,000 USD.",
    )
    store.upsert(session_id, [chunk], [[0.1] * 8])
    assert store.count(session_id) == 1

    gemini = ScriptedProvider("gemini", outcomes=[RateLimitError("429", provider="gemini")])
    openrouter = ScriptedProvider(
        "openrouter", outcomes=[ProviderUnavailableError("503", provider="openrouter")]
    )
    router = FallbackLLMProvider([gemini, openrouter])

    with pytest.raises(AllProvidersFailedError):
        router.complete(message())

    # The index is untouched by the LLM outage.
    assert store.count(session_id) == 1
    assert store.list_chunks(session_id)[0].text == "Revenue was 8,400,000 USD."
