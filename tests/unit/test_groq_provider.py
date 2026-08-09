"""Groq adapter, ordered routing, cooldown, and configuration regressions."""

from __future__ import annotations

import httpx
import pytest

from omnirag.config.settings import build_settings, reset_settings_cache
from omnirag.core.exceptions import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderPaymentRequiredError,
    ProviderPolicyError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage, LLMResponse
from omnirag.providers.llm.context import llm_session
from omnirag.providers.llm.factory import build_llm_provider, get_llm_provider
from omnirag.providers.llm.groq import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    GroqLLM,
    model_supports_images,
)
from omnirag.providers.llm.router import FallbackLLMProvider


class ScriptedProvider(BaseLLMProvider):
    def __init__(self, name: str, outcomes, *, images: bool = True):
        super().__init__(model=f"{name}-model")
        self.name = name
        self.outcomes = list(outcomes)
        self._images = images
        self.call_count = 0
        self.last_messages = None

    def supports_images(self, model=None) -> bool:
        return self._images

    def complete(
        self,
        messages,
        *,
        system=None,
        temperature=None,
        max_output_tokens=None,
        model=None,
        json_mode=False,
        requirements=None,
    ):
        self.call_count += 1
        self.last_messages = messages
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(text=str(outcome), model=self.model, provider=self.name)


def _messages(*, image: bool = False):
    images = [ImagePart(data=b"\x89PNG-fake", media_type="image/png")] if image else []
    return [LLMMessage(text="question", images=images)]


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture
def groq_http(monkeypatch):
    calls = []
    responses = []

    class FakeClient:
        is_closed = False

        def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            item = responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    monkeypatch.setattr(
        "omnirag.providers.http.get_client", lambda timeout_s=60.0: FakeClient()
    )
    return calls, responses


class TestGroqConfiguration:
    def test_default_chain_and_current_models(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

        settings = build_settings()

        assert [e.provider for e in settings.llm.configured_endpoints] == [
            "gemini",
            "groq",
            "openrouter",
        ]
        groq = next(e for e in settings.llm.endpoints if e.provider == "groq")
        assert groq.model == DEFAULT_MODEL == "openai/gpt-oss-20b"
        assert groq.vision_model == DEFAULT_VISION_MODEL == "qwen/qwen3.6-27b"

    def test_explicit_chain_skips_missing_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "gemini,groq,openrouter")
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")

        settings = build_settings()

        assert [e.provider for e in settings.llm.configured_endpoints] == [
            "gemini",
            "openrouter",
        ]

    def test_groq_and_openrouter_work_without_gemini(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "q")
        monkeypatch.setenv("OPENROUTER_API_KEY", "o")

        assert [e.provider for e in build_settings().llm.configured_endpoints] == [
            "groq",
            "openrouter",
        ]

    def test_grok_alias_is_accepted(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROK_API_KEY", "legacy-key")

        endpoint = build_settings().llm.configured_endpoints[0]

        assert endpoint.provider == "groq"
        assert endpoint.api_key == "legacy-key"

    def test_canonical_groq_key_wins_over_alias(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "canonical-key")
        monkeypatch.setenv("GROK_API_KEY", "legacy-key")

        assert build_settings().llm.configured_endpoints[0].api_key == "canonical-key"

    def test_same_length_key_change_rebuilds_cached_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "first-key-1")
        first = get_llm_provider(build_settings())

        monkeypatch.setenv("GROQ_API_KEY", "other-key-2")
        reset_settings_cache()
        second = get_llm_provider(build_settings())

        assert second is not first
        assert second.primary.api_key == "other-key-2"

    def test_model_and_chain_changes_rebuild_streamlit_engine(self, monkeypatch):
        from omnirag.ui import state

        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "key")
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        reset_settings_cache()
        first = state.engine()

        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
        reset_settings_cache()
        second = state.engine()

        assert second is not first
        assert second.settings.llm.primary_endpoint.model == "openai/gpt-oss-120b"


class TestGroqTransport:
    def test_text_response_metadata_and_official_endpoint(self, groq_http):
        calls, responses = groq_http
        responses.append(FakeResponse(
            200,
            {
                "id": "chatcmpl-safe",
                "model": DEFAULT_MODEL,
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            headers={"content-length": "180"},
        ))
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        response = provider.complete(_messages(), max_output_tokens=321)

        assert calls[0]["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
        assert calls[0]["json"]["model"] == DEFAULT_MODEL
        assert calls[0]["json"]["max_completion_tokens"] == 321
        assert "max_tokens" not in calls[0]["json"]
        assert response.text == "answer"
        assert response.model == DEFAULT_MODEL
        assert response.finish_reason == "stop"
        assert response.usage["completion_tokens"] == 2
        assert response.provider == "groq"
        assert response.diagnostics["http_status"] == 200

    def test_multimodal_request_uses_vision_model_and_preserves_mime(self, groq_http):
        calls, responses = groq_http
        responses.append(FakeResponse(200, {
            "model": DEFAULT_VISION_MODEL,
            "choices": [{"message": {"content": "visual answer"}, "finish_reason": "stop"}],
        }))
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        response = provider.complete(_messages(image=True))

        payload = calls[0]["json"]
        assert payload["model"] == DEFAULT_VISION_MODEL
        parts = payload["messages"][0]["content"]
        image = next(part for part in parts if part["type"] == "image_url")
        assert image["image_url"]["url"].startswith("data:image/png;base64,")
        assert response.text == "visual answer"

    @pytest.mark.parametrize(
        "response,error_type",
        [
            (FakeResponse(401, text="invalid key"), ProviderAuthError),
            (FakeResponse(404, text="model not found"), ProviderBadRequestError),
            (FakeResponse(429, text="rate limit"), RateLimitError),
            (FakeResponse(503, text="temporarily unavailable"), ProviderUnavailableError),
            (httpx.ReadTimeout("timed out"), ProviderTimeoutError),
        ],
    )
    def test_typed_errors(self, groq_http, response, error_type):
        _, responses = groq_http
        responses.append(response)
        provider = GroqLLM(api_key="test-key", retry_attempts=1)

        with pytest.raises(error_type):
            provider.complete(_messages())

    def test_capability_detection_is_conservative(self):
        assert model_supports_images(DEFAULT_VISION_MODEL) is True
        assert model_supports_images(DEFAULT_MODEL) is False
        assert model_supports_images("unknown/future-model") is False

    def test_retry_after_ten_seconds_waits_once_then_succeeds(
        self, groq_http, monkeypatch
    ):
        calls, responses = groq_http
        sleeps = []
        monkeypatch.setattr("omnirag.utils.retry.time.sleep", sleeps.append)
        responses.extend(
            [
                FakeResponse(
                    429,
                    text="TPM limit reached; try again in 10s",
                    headers={"retry-after": "10"},
                ),
                FakeResponse(
                    200,
                    {
                        "model": DEFAULT_MODEL,
                        "choices": [
                            {"message": {"content": "ok"}, "finish_reason": "stop"}
                        ],
                    },
                ),
            ]
        )
        provider = GroqLLM(
            api_key="test-key",
            retry_attempts=2,
            max_rate_limit_wait_seconds=20,
        )

        assert provider.complete(_messages()).text == "ok"
        assert len(calls) == 2
        assert sleeps == [10]

    def test_retry_after_over_bound_fails_over_without_sleep(
        self, groq_http, monkeypatch
    ):
        calls, responses = groq_http
        sleeps = []
        monkeypatch.setattr("omnirag.utils.retry.time.sleep", sleeps.append)
        responses.append(
            FakeResponse(
                429,
                text="TPM limit reached; try again in 30s",
                headers={"retry-after": "30"},
            )
        )
        groq = GroqLLM(
            api_key="test-key",
            retry_attempts=2,
            max_rate_limit_wait_seconds=20,
        )
        fallback = ScriptedProvider("openrouter", ["fallback"])

        response = FallbackLLMProvider([groq, fallback]).complete(_messages())

        assert response.provider == "openrouter"
        assert len(calls) == 1
        assert sleeps == []

    def test_router_logs_actual_groq_vision_model(self, groq_http, caplog):
        _, responses = groq_http
        responses.append(
            FakeResponse(
                200,
                {
                    "model": DEFAULT_VISION_MODEL,
                    "choices": [
                        {
                            "message": {"content": "visual"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        router = FallbackLLMProvider(
            [GroqLLM(api_key="test-key", retry_attempts=1)]
        )

        with caplog.at_level("INFO"):
            router.complete(_messages(image=True))

        assert any(
            f"provider=groq role=primary model={DEFAULT_VISION_MODEL}" in record.message
            for record in caplog.records
        )
        assert any(
            f"requested_model={DEFAULT_VISION_MODEL} "
            f"response_model={DEFAULT_VISION_MODEL}" in record.message
            for record in caplog.records
        )


class TestThreeProviderRouting:
    def test_gemini_success_does_not_call_fallbacks(self):
        gemini = ScriptedProvider("gemini", ["primary"])
        groq = ScriptedProvider("groq", ["middle"])
        openrouter = ScriptedProvider("openrouter", ["last"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert response.provider == "gemini"
        assert (gemini.call_count, groq.call_count, openrouter.call_count) == (1, 0, 0)

    def test_gemini_429_uses_groq_without_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["fast fallback"])
        openrouter = ScriptedProvider("openrouter", ["last"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert response.provider == "groq"
        assert response.text == "fast fallback"
        assert openrouter.call_count == 0
        assert response.diagnostics["fallback_position"] == 1

    def test_gemini_and_groq_429_use_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [RateLimitError("429", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["final fallback"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert response.provider == "openrouter"
        assert response.text == "final fallback"
        assert response.diagnostics["fallback_position"] == 2

    def test_payment_required_uses_next_provider_without_same_route_retry(self):
        gemini = ScriptedProvider(
            "gemini", [ProviderPaymentRequiredError("402", provider="gemini")]
        )
        groq = ScriptedProvider("groq", ["answer"])
        openrouter = ScriptedProvider("openrouter", ["never"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert response.provider == "groq"
        assert (gemini.call_count, groq.call_count, openrouter.call_count) == (1, 1, 0)

    def test_gemini_cooldown_routes_directly_to_groq(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["first", "second"])
        openrouter = ScriptedProvider("openrouter", ["never"])
        router = FallbackLLMProvider([gemini, groq, openrouter])

        with llm_session("session"):
            assert router.complete(_messages()).provider == "groq"
            assert router.complete(_messages()).provider == "groq"

        assert gemini.call_count == 1
        assert groq.call_count == 2
        assert openrouter.call_count == 0

    def test_gemini_and_groq_cooldowns_route_directly_to_openrouter(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [RateLimitError("429", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["first", "second"])
        router = FallbackLLMProvider([gemini, groq, openrouter])

        with llm_session("session"):
            assert router.complete(_messages()).provider == "openrouter"
            assert router.complete(_messages()).provider == "openrouter"

        assert gemini.call_count == 1
        assert groq.call_count == 1
        assert openrouter.call_count == 2

    @pytest.mark.parametrize(
        "failure",
        [
            ProviderUnavailableError("503", provider="groq"),
            ProviderTimeoutError("timeout", provider="groq"),
        ],
    )
    def test_recoverable_groq_failure_uses_openrouter(self, failure):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [failure])
        openrouter = ScriptedProvider("openrouter", ["answer"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert response.provider == "openrouter"
        assert openrouter.call_count == 1

    def test_groq_auth_error_is_nonrecoverable_configuration_failure(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", [ProviderAuthError("401", provider="groq")])
        openrouter = ScriptedProvider("openrouter", ["must not run"])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert [name for name, _ in excinfo.value.failures] == ["gemini", "groq"]
        assert openrouter.call_count == 0

    def test_visual_request_skips_text_only_groq_and_preserves_image(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["must not run"], images=False)
        openrouter = ScriptedProvider("openrouter", ["visual answer"], images=True)

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(
            _messages(image=True)
        )

        assert response.provider == "openrouter"
        assert groq.call_count == 0
        assert openrouter.last_messages[0].images[0].media_type == "image/png"

    def test_visual_request_uses_capable_groq(self):
        gemini = ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")])
        groq = ScriptedProvider("groq", ["visual answer"], images=True)
        openrouter = ScriptedProvider("openrouter", ["never"])

        response = FallbackLLMProvider([gemini, groq, openrouter]).complete(
            _messages(image=True)
        )

        assert response.provider == "groq"
        assert groq.last_messages[0].images
        assert openrouter.call_count == 0

    def test_safety_refusal_never_falls_back(self):
        gemini = ScriptedProvider(
            "gemini", [ProviderPolicyError("safety", provider="gemini")]
        )
        groq = ScriptedProvider("groq", ["must not run"])
        openrouter = ScriptedProvider("openrouter", ["must not run"])

        with pytest.raises(ProviderPolicyError):
            FallbackLLMProvider([gemini, groq, openrouter]).complete(_messages())

        assert groq.call_count == openrouter.call_count == 0

    def test_all_provider_failures_keep_complete_attempt_trail(self):
        router = FallbackLLMProvider([
            ScriptedProvider("gemini", [RateLimitError("429", provider="gemini")]),
            ScriptedProvider("groq", [ProviderUnavailableError("503", provider="groq")]),
            ScriptedProvider(
                "openrouter", [ProviderUnavailableError("503", provider="openrouter")]
            ),
        ])

        with pytest.raises(AllProvidersFailedError) as excinfo:
            router.complete(_messages())

        assert [name for name, _ in excinfo.value.failures] == [
            "gemini",
            "groq",
            "openrouter",
        ]
