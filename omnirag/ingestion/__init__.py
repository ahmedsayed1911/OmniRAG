"""Document ingestion: raw upload -> canonical Document."""

from omnirag.ingestion.base import BaseDocumentProcessor, ProcessingContext
from omnirag.ingestion.docx import WordProcessor
from omnirag.ingestion.image import ImageProcessor
from omnirag.ingestion.pdf import PDFProcessor
from omnirag.ingestion.pptx import PowerPointProcessor
from omnirag.ingestion.router import (
    DocumentRouter,
    FileValidation,
    get_router,
    reset_router,
)
from omnirag.ingestion.text import TextProcessor

__all__ = [
    "BaseDocumentProcessor",
    "DocumentRouter",
    "FileValidation",
    "ImageProcessor",
    "PDFProcessor",
    "PowerPointProcessor",
    "ProcessingContext",
    "TextProcessor",
    "WordProcessor",
    "get_router",
    "reset_router",
]
