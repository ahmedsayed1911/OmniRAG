"""Streamlit presentation layer.

The ONLY package in OmniRAG that imports Streamlit. Everything below it
(services, rag, ingestion, providers, core) is UI-agnostic, which is what makes
the FastAPI migration a matter of adding a new entry point rather than a
rewrite.
"""

__all__ = ["chat", "components", "sidebar", "sources", "state", "styles"]
