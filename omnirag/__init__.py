"""OmniRAG — Universal Multimodal Document Intelligence & RAG Platform.

Layering (imports only ever point downwards):

    ui/            Streamlit presentation      -> services
    services/      application orchestration   -> ingestion, rag, storage
    ingestion/     file -> canonical Document  -> intelligence, core
    intelligence/  OCR / vision / tables       -> providers, core
    rag/           chunk, embed, index, answer -> providers, storage, core
    providers/     external API adapters       -> core, utils
    core/          models, enums, exceptions   -> (nothing)

Nothing outside ``omnirag.ui`` imports Streamlit, which is what makes the
FastAPI migration a matter of adding a new entry point.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
