"""Application services — the layer the UI talks to.

Streamlit (and later FastAPI) calls only these classes. They own orchestration
and error translation; they never own presentation.
"""

from omnirag.services.chat_service import ChatRequest, ChatService
from omnirag.services.engine import (
    EngineStatus,
    OmniRAGEngine,
    get_engine,
    reset_engine,
)
from omnirag.services.ingestion_service import IngestionService, UploadedFile

__all__ = [
    "ChatRequest",
    "ChatService",
    "EngineStatus",
    "IngestionService",
    "OmniRAGEngine",
    "UploadedFile",
    "get_engine",
    "reset_engine",
]
