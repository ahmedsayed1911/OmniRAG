"""Startup and architecture guarantees.

These tests protect two properties that are easy to break silently:

1. the app imports and starts cleanly with **no configuration at all**;
2. the RAG engine never imports Streamlit — the precondition for the planned
   FastAPI migration.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UI_ONLY_MODULES = {"omnirag.ui", "omnirag.config.bootstrap"}


def iter_modules():
    import omnirag

    for info in pkgutil.walk_packages(omnirag.__path__, prefix="omnirag."):
        yield info.name


class TestImports:
    def test_every_module_imports(self):
        failures = []
        for name in iter_modules():
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 - reported below
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert not failures, "modules failed to import:\n" + "\n".join(failures)

    def test_app_entry_point_compiles(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError on failure

    def test_public_packages_expose_their_api(self):
        import omnirag.core.models as models
        import omnirag.evaluation as evaluation
        import omnirag.ingestion as ingestion
        import omnirag.providers.llm as llm
        import omnirag.rag as rag
        import omnirag.services as services

        assert hasattr(models, "Chunk")
        assert hasattr(rag, "Retriever")
        assert hasattr(ingestion, "get_router")
        assert hasattr(services, "OmniRAGEngine")
        assert hasattr(llm, "FallbackLLMProvider")
        assert hasattr(evaluation, "evaluate_retrieval")


class TestUIIndependence:
    """The core engine must stay framework-agnostic."""

    def _imports_streamlit(self, path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "streamlit" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "streamlit":
                    return True
        return False

    def test_engine_modules_never_import_streamlit(self):
        offenders = []
        for path in (ROOT / "omnirag").rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            module = relative.replace("/", ".").removesuffix(".py").removesuffix(".__init__")
            if any(module.startswith(allowed) for allowed in UI_ONLY_MODULES):
                continue
            if self._imports_streamlit(path):
                offenders.append(relative)

        assert not offenders, (
            "these engine modules import Streamlit, which breaks the "
            f"FastAPI migration guarantee: {offenders}"
        )

    def test_rag_and_ingestion_work_without_streamlit_state(
        self, engine, session_id, sample_text, fake_embeddings
    ):
        """The whole vertical slice runs with no Streamlit runtime present."""
        from omnirag.services.ingestion_service import IngestionService, UploadedFile

        engine._embeddings = fake_embeddings
        result = IngestionService(engine).ingest(
            session_id, UploadedFile(name="notes.txt", data=sample_text)
        )
        assert result.status.value == "ready"


class TestZeroConfigurationStartup:
    def test_settings_build_without_any_environment(self):
        from omnirag.config.settings import build_settings

        settings = build_settings()
        assert settings.is_ready is False
        assert settings.validation_issues()

    def test_engine_builds_and_reports_status_without_keys(self):
        from omnirag.services.engine import OmniRAGEngine

        status = OmniRAGEngine().status()

        assert status.ready is False
        assert status.issues
        assert status.vector_store in ("memory", "qdrant")

    def test_router_is_usable_without_keys(self):
        from omnirag.ingestion.router import get_router

        assert "pdf" in get_router().supported_extensions()

    def test_chat_without_configuration_returns_a_helpful_message(self, session_id):
        from omnirag.services.chat_service import ChatRequest, ChatService
        from omnirag.services.engine import OmniRAGEngine

        message = ChatService(OmniRAGEngine()).answer(
            ChatRequest(question="hello", session_id=session_id)
        )

        assert message.error
        assert "GEMINI_API_KEY" in message.content


class TestRepositoryHygiene:
    def test_no_env_file_is_committed(self):
        assert not (ROOT / ".env").exists(), "a real .env must never be committed"

    def test_no_real_secrets_file_is_committed(self):
        assert not (ROOT / ".streamlit" / "secrets.toml").exists()

    def test_gitignore_covers_the_sensitive_paths(self):
        content = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in (".env", ".streamlit/secrets.toml", "__pycache__/", "*.log"):
            assert pattern in content

    def test_example_files_contain_no_real_keys(self):
        import re

        patterns = [
            re.compile(r"sk-[A-Za-z0-9]{20,}"),
            re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
            re.compile(r"sk-or-v1-[a-f0-9]{40,}"),
        ]
        for name in (".env.example", ".streamlit/secrets.toml.example"):
            content = (ROOT / name).read_text(encoding="utf-8")
            for pattern in patterns:
                assert not pattern.search(content), f"{name} looks like it contains a real key"

    def test_requirements_lists_every_runtime_import(self):
        content = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("streamlit", "pydantic", "pymupdf", "python-docx",
                        "python-pptx", "pillow", "qdrant-client", "httpx", "numpy"):
            assert package in content
