"""Environment bootstrap for the Streamlit entry point.

This is the *only* place where Streamlit secrets meet the core engine: secrets
are copied into ``os.environ`` before ``omnirag.config.settings`` is read, so
nothing under ``omnirag/`` (except the ``ui`` package) needs to import
Streamlit. A future FastAPI process simply skips this module and relies on real
environment variables.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

_SECTION_PREFIXES = {
    "llm": "LLM_",
    "embedding": "EMBEDDING_",
    "qdrant": "QDRANT_",
    "rerank": "RERANK_",
    "ocr": "OCR_",
    "vision": "VISION_",
}


def load_dotenv(path: str = ".env", *, override: bool = False) -> int:
    """Minimal ``.env`` loader (no extra dependency).

    Returns the number of variables applied. Silently does nothing when the file
    is absent, which is the normal case in cloud deployments.
    """
    if not os.path.isfile(path):
        return 0
    applied = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                applied += 1
    return applied


def apply_secrets(secrets: Mapping[str, Any], *, override: bool = False) -> int:
    """Copy a (possibly nested) secrets mapping into ``os.environ``.

    Supports both flat keys::

        LLM_API_KEY = "sk-..."

    and sectioned TOML::

        [llm]
        api_key = "sk-..."
        model = "gpt-4o"
    """
    applied = 0
    for key, value in _iter_items(secrets):
        if isinstance(value, Mapping):
            prefix = _SECTION_PREFIXES.get(str(key).lower())
            if prefix is None:
                continue
            for sub_key, sub_value in _iter_items(value):
                if isinstance(sub_value, (Mapping, list, tuple)):
                    continue
                env_name = f"{prefix}{str(sub_key).upper()}"
                applied += _set_env(env_name, sub_value, override)
            continue
        if isinstance(value, (list, tuple)):
            continue
        applied += _set_env(str(key).upper(), value, override)
    return applied


def _iter_items(mapping: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    try:
        return list(mapping.items())
    except Exception:  # pragma: no cover - defensive for exotic mappings
        return []


def _set_env(name: str, value: Any, override: bool) -> int:
    if value is None:
        return 0
    if not override and name in os.environ and os.environ[name]:
        return 0
    os.environ[name] = str(value)
    return 1


def bootstrap_environment() -> None:
    """Load ``.env`` then Streamlit secrets, then reset the settings cache.

    Order matters: real environment variables win, then ``.env``, then secrets
    fill the remaining gaps. Import of ``streamlit`` is local and optional so
    this function is safe to call from non-Streamlit processes.
    """
    load_dotenv()
    try:
        import streamlit as st  # local import: keeps core UI-free

        secrets = st.secrets
    except Exception:
        secrets = None

    if secrets is not None:
        try:
            apply_secrets(secrets)
        except Exception:
            # A malformed/absent secrets.toml must never block startup.
            pass

    from omnirag.config.settings import reset_settings_cache

    reset_settings_cache()
