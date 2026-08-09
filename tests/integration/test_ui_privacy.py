"""Normal Streamlit rendering must not expose provider implementation details."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from omnirag.core.enums import Role
from omnirag.core.models import ChatMessage
from omnirag.ui.state import MESSAGES_KEY, SESSION_KEY

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = "openrouter"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


def _configure(monkeypatch, *, debug: bool) -> None:
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "mock")
    monkeypatch.setenv("FALLBACK_LLM_PROVIDER", "none")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    monkeypatch.setenv("VISION_ENABLED", "false")
    monkeypatch.setenv("OMNIRAG_DEBUG_GENERATION", "true" if debug else "false")
    monkeypatch.setenv("DEBUG_PANELS", "true")  # must not bypass the privacy gate


def _fallback_message() -> ChatMessage:
    return ChatMessage(
        role=Role.ASSISTANT,
        content="Grounded answer.",
        debug={
            "provider": PROVIDER,
            "model": MODEL,
            "provider_attempts": [
                "final_answer/gemini: RateLimitError [recoverable]",
                "final_answer/openrouter: ok",
            ],
            "openrouter_free_fallback": True,
            "contexts": 8,
            "images_sent": 1,
            "elapsed_s": 12.4,
            "query_scope": "MULTI_PART",
            "requested_max_output_tokens": 8192,
            "usage": {"completion_tokens": 700},
            "warnings": [],
        },
    )


def _visible_text(app: AppTest) -> str:
    values = []
    for name in ("markdown", "caption", "error", "warning", "info", "success", "json"):
        for element in getattr(app, name, []):
            values.append(str(getattr(element, "value", "")))
    values.extend(str(item.label) for item in app.expander)
    return "\n".join(values)


def test_normal_ui_hides_provider_model_fallback_and_internal_diagnostics(
    monkeypatch, session_id
):
    _configure(monkeypatch, debug=False)
    message = _fallback_message()
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state[SESSION_KEY] = session_id
    app.session_state[MESSAGES_KEY] = [message]

    app.run(timeout=30)

    assert not app.exception
    visible = _visible_text(app).lower()
    for secret_detail in (
        PROVIDER,
        MODEL,
        "gemini",
        "fallback",
        "failover",
        "rate limit",
        "8192",
        "completion_tokens",
        "multi_part",
        "retrieval mode",
        "reranked",
        "hash-1024",
        "vector store",
        "provider usage",
    ):
        assert secret_detail.lower() not in visible
    assert "8 sources · 1 image analysed · 12.4s" in visible
    assert not any("fallback" in item.value.lower() for item in app.warning)
    # Observability data remains intact in internal state.
    stored = app.session_state[MESSAGES_KEY][0]
    assert stored.debug["provider"] == PROVIDER
    assert stored.debug["model"] == MODEL
    assert len(stored.debug["provider_attempts"]) == 2


def test_provider_and_model_are_visible_only_in_generation_diagnostic_mode(
    monkeypatch, session_id
):
    _configure(monkeypatch, debug=True)
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state[SESSION_KEY] = session_id
    app.session_state[MESSAGES_KEY] = [_fallback_message()]

    app.run(timeout=30)

    assert not app.exception
    visible = _visible_text(app).lower()
    assert PROVIDER in visible
    assert MODEL in visible
    assert "fallback" in visible
    assert "generation diagnostics" in visible


def test_raw_provider_error_is_sanitized_in_normal_ui(monkeypatch, session_id):
    _configure(monkeypatch, debug=False)
    raw = (
        'openrouter HTTP 402 {"error":{"message":"insufficient credits"}} '
        "after gemini 429 fallback"
    )
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state[SESSION_KEY] = session_id
    app.session_state[MESSAGES_KEY] = [
        ChatMessage(role=Role.ASSISTANT, content=raw, error=raw)
    ]

    app.run(timeout=30)

    assert not app.exception
    visible = _visible_text(app).lower()
    assert "temporarily unavailable" in visible
    for detail in ("openrouter", "gemini", "402", "429", "credits", "fallback"):
        assert detail not in visible
