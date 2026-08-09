"""Configuration: env parsing, defaults, validation, secret handling."""

from __future__ import annotations

import pytest

from omnirag.config.bootstrap import apply_secrets, load_dotenv
from omnirag.config.settings import build_settings, get_settings, reset_settings_cache
from omnirag.core.exceptions import ConfigurationError, MissingCredentialError


class TestDefaults:
    def test_defaults_are_sensible_without_any_environment(self):
        settings = build_settings()

        assert settings.llm.provider == "gemini"
        assert settings.retrieval.top_k > settings.retrieval.rerank_top_k
        assert settings.chunking.chunk_overlap < settings.chunking.chunk_size
        assert settings.upload.max_upload_mb > 0
        assert "pdf" in settings.upload.allowed_extensions

    def test_no_provider_means_not_ready_with_a_helpful_message(self):
        settings = build_settings()

        assert settings.is_ready is False
        issues = settings.validation_issues()
        assert any("GEMINI_API_KEY" in issue for issue in issues)

    def test_hash_embeddings_are_selected_offline_and_warned_about(self):
        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert any("hash" in w for w in settings.warnings())


class TestParsing:
    def test_numeric_settings_are_parsed(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "40")
        monkeypatch.setenv("RERANK_TOP_K", "6")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("MAX_UPLOAD_MB", "12.5")

        settings = build_settings()

        assert settings.retrieval.top_k == 40
        assert settings.retrieval.rerank_top_k == 6
        assert settings.llm.temperature == 0.7
        assert settings.upload.max_upload_mb == 12.5

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("off", False),
    ])
    def test_boolean_settings_are_parsed(self, monkeypatch, value, expected):
        monkeypatch.setenv("VISION_ENABLED", value)
        assert build_settings().vision.enabled is expected

    def test_invalid_number_raises_a_helpful_configuration_error(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "not-a-number")

        with pytest.raises(ConfigurationError) as excinfo:
            build_settings()
        assert "TOP_K" in excinfo.value.user_message

    def test_overlap_larger_than_chunk_size_is_rejected(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "200")
        monkeypatch.setenv("CHUNK_OVERLAP", "400")

        with pytest.raises(ConfigurationError) as excinfo:
            build_settings()
        assert "CHUNK_OVERLAP" in excinfo.value.user_message


class TestSecretHandling:
    def test_redacted_view_masks_every_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSuperSecretValue123")
        monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret-value")

        redacted = build_settings().redacted()
        serialized = str(redacted)

        assert "AIzaSuperSecretValue123" not in serialized
        assert "qdrant-secret-value" not in serialized

    def test_require_key_raises_a_missing_credential_error(self):
        settings = build_settings()
        with pytest.raises(MissingCredentialError):
            settings.llm.require_key()

    def test_apply_secrets_maps_flat_keys(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        applied = apply_secrets({"GEMINI_API_KEY": "from-secrets"})

        assert applied == 1
        assert build_settings().llm.endpoints[0].api_key == "from-secrets"

    def test_apply_secrets_maps_sections(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        apply_secrets({"qdrant": {"url": "https://example.qdrant.io", "collection": "c"}})

        settings = build_settings()
        assert settings.vector_store.url == "https://example.qdrant.io"
        assert settings.vector_store.collection == "c"

    def test_real_environment_wins_over_secrets(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        apply_secrets({"GEMINI_API_KEY": "from-secrets"})

        assert build_settings().llm.endpoints[0].api_key == "from-env"

    def test_streamlit_secret_value_refreshes_without_overriding_host_env(
        self, monkeypatch
    ):
        monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "1400"})
        assert build_settings().llm.max_output_tokens == 1400

        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "4096"})
        assert build_settings().llm.max_output_tokens == 4096

        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "7777")
        apply_secrets({"LLM_MAX_OUTPUT_TOKENS": "8192"})
        assert build_settings().llm.max_output_tokens == 7777

    def test_missing_dotenv_file_is_not_an_error(self):
        assert load_dotenv("definitely-not-a-real-file.env") == 0


class TestCaching:
    def test_get_settings_is_cached_and_resettable(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "11")
        first = get_settings()
        assert first.retrieval.top_k == 11

        monkeypatch.setenv("TOP_K", "22")
        assert get_settings().retrieval.top_k == 11  # still cached

        reset_settings_cache()
        assert get_settings().retrieval.top_k == 22

    def test_streamlit_engine_rebuilds_when_output_budget_changes(self, monkeypatch):
        from omnirag.services.engine import reset_engine
        from omnirag.ui import state

        reset_engine()
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1400")
        monkeypatch.setenv("LLM_EXHAUSTIVE_MAX_OUTPUT_TOKENS", "1400")
        reset_settings_cache()
        first = state.engine()

        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("LLM_EXHAUSTIVE_MAX_OUTPUT_TOKENS", "8192")
        reset_settings_cache()
        second = state.engine()

        assert second is not first
        assert second.settings.llm.max_output_tokens == 4096
        assert second.settings.llm.exhaustive_max_output_tokens == 8192

    def test_generation_debug_mode_is_disabled_by_default_and_configurable(
        self, monkeypatch
    ):
        assert build_settings().debug_generation is False
        monkeypatch.setenv("OMNIRAG_DEBUG_GENERATION", "true")
        assert build_settings().debug_generation is True


class TestEmbeddingIndependence:
    def test_embeddings_use_gemini_key_when_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        settings = build_settings()

        assert settings.embedding.provider == "gemini"
        assert settings.embedding.api_key == "g-key"

    def test_dedicated_embedding_key_is_preferred(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setenv("EMBEDDING_API_KEY", "e-key")

        settings = build_settings()
        assert settings.embedding.api_key == "e-key"

    def test_openrouter_only_falls_back_to_hash_embeddings_with_a_warning(self, monkeypatch):
        # OpenRouter serves no embedding endpoint; the app must still run and
        # say clearly that retrieval quality is degraded.
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert any("hash" in w for w in settings.warnings())

    def test_explicit_remote_embedding_provider_without_any_key_uses_hash(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

        settings = build_settings()

        assert settings.embedding.provider == "hash"
        assert settings.embedding.model == "hash-1024"
        assert any("hash" in warning for warning in settings.warnings())
