"""File-type routing and upload validation.

Adding a format (XLSX, HTML, …) means writing one processor and registering it
here — nothing else in the pipeline changes, because every processor emits the
same canonical :class:`~omnirag.core.models.Document`.

Validation happens *before* any parsing: extension allow-list, size limit,
emptiness, and content-sniffing to catch a file whose extension lies about its
type. Uploaded files are only ever read as data — nothing is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

from omnirag.config.settings import AppSettings, get_settings
from omnirag.core.enums import FileType
from omnirag.core.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from omnirag.ingestion.base import BaseDocumentProcessor
from omnirag.ingestion.docx import WordProcessor
from omnirag.ingestion.image import ImageProcessor
from omnirag.ingestion.pdf import PDFProcessor
from omnirag.ingestion.pptx import PowerPointProcessor
from omnirag.ingestion.text import TextProcessor
from omnirag.utils.hashing import file_extension, sanitize_filename
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

#: Registered processors, in resolution order.
PROCESSOR_CLASSES: Tuple[Type[BaseDocumentProcessor], ...] = (
    PDFProcessor,
    WordProcessor,
    PowerPointProcessor,
    ImageProcessor,
    TextProcessor,
)

#: Magic-number signatures used to sanity-check the declared extension.
_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF8", "gif"),
    (b"PK\x03\x04", "zip"),  # docx/pptx/xlsx are ZIP containers
)
_ZIP_FORMATS = {"docx", "pptx", "xlsx"}


@dataclass(frozen=True)
class FileValidation:
    filename: str
    safe_filename: str
    extension: str
    file_type: FileType
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class DocumentRouter:
    """Maps a filename to the processor that can parse it."""

    def __init__(self, processors: Optional[Sequence[BaseDocumentProcessor]] = None):
        self._processors: List[BaseDocumentProcessor] = list(
            processors if processors is not None else [cls() for cls in PROCESSOR_CLASSES]
        )
        self._by_extension: Dict[str, BaseDocumentProcessor] = {}
        for processor in self._processors:
            for extension in processor.extensions:
                self._by_extension[extension.lower()] = processor

    # ------------------------------------------------------------------ #
    @property
    def processors(self) -> List[BaseDocumentProcessor]:
        return list(self._processors)

    def supported_extensions(self) -> List[str]:
        return sorted(self._by_extension)

    def supported_label(self) -> str:
        return ", ".join(f".{e}" for e in self.supported_extensions())

    def is_supported(self, filename: str) -> bool:
        return file_extension(filename) in self._by_extension

    def file_type_for(self, filename: str) -> FileType:
        processor = self._by_extension.get(file_extension(filename))
        if processor is None:
            return FileType.UNKNOWN
        if processor.file_type == FileType.TXT and filename.lower().endswith(
            (".md", ".markdown")
        ):
            return FileType.MARKDOWN
        return processor.file_type

    def route(self, filename: str) -> BaseDocumentProcessor:
        """Return the processor for ``filename`` or raise a user-facing error."""
        extension = file_extension(filename)
        processor = self._by_extension.get(extension)
        if processor is None:
            raise UnsupportedFileTypeError(
                filename, extension or "(none)", self.supported_label()
            )
        return processor

    def register(self, processor: BaseDocumentProcessor) -> None:
        """Register an additional processor (e.g. a future XLSX handler)."""
        self._processors.append(processor)
        for extension in processor.extensions:
            self._by_extension[extension.lower()] = processor
        logger.info(
            "Registered processor %s for %s",
            type(processor).__name__,
            ", ".join(processor.extensions),
        )

    # ------------------------------------------------------------------ #
    def validate(
        self, filename: str, data: bytes, *, settings: Optional[AppSettings] = None
    ) -> FileValidation:
        """Check an upload before any parsing happens."""
        resolved = settings or get_settings()
        safe_name = sanitize_filename(filename)
        extension = file_extension(safe_name)

        if extension not in self._by_extension:
            raise UnsupportedFileTypeError(
                safe_name, extension or "(none)", self.supported_label()
            )

        if not data:
            raise EmptyDocumentError(safe_name)

        limit_mb = resolved.upload.max_upload_mb
        size_mb = len(data) / (1024 * 1024)
        if size_mb > limit_mb:
            raise FileTooLargeError(safe_name, size_mb, limit_mb)

        self._check_signature(safe_name, extension, data)

        return FileValidation(
            filename=filename,
            safe_filename=safe_name,
            extension=extension,
            file_type=self.file_type_for(safe_name),
            size_bytes=len(data),
        )

    @staticmethod
    def _check_signature(filename: str, extension: str, data: bytes) -> None:
        """Reject files whose bytes clearly contradict their extension.

        Text formats are exempt (they have no magic number), and an unknown
        signature is allowed through — the goal is catching obvious mismatches,
        not enforcing a strict format registry.
        """
        if extension in ("txt", "md", "markdown", "text", "webp"):
            return

        header = data[:16]
        detected: Optional[str] = None
        for signature, name in _SIGNATURES:
            if header.startswith(signature):
                detected = name
                break
        if detected is None:
            return

        expected_zip = extension in _ZIP_FORMATS
        if detected == "zip" and expected_zip:
            return
        if detected == extension:
            return
        if detected in ("png", "jpg", "gif") and extension in ("png", "jpg", "jpeg", "webp"):
            return  # image containers are interchangeable enough for Pillow

        raise UnsupportedFileTypeError(
            filename,
            extension,
            f"the file contents look like {detected}, not {extension}",
        )


_router: Optional[DocumentRouter] = None


def get_router() -> DocumentRouter:
    """Process-wide router singleton."""
    global _router
    if _router is None:
        _router = DocumentRouter()
    return _router


def reset_router() -> None:
    global _router
    _router = None


__all__ = [
    "DocumentRouter",
    "FileValidation",
    "PROCESSOR_CLASSES",
    "get_router",
    "reset_router",
]
