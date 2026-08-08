"""File routing, extension support, and upload validation."""

from __future__ import annotations

import pytest

from omnirag.core.enums import FileType
from omnirag.core.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from omnirag.ingestion.docx import WordProcessor
from omnirag.ingestion.image import ImageProcessor
from omnirag.ingestion.pdf import PDFProcessor
from omnirag.ingestion.pptx import PowerPointProcessor
from omnirag.ingestion.router import DocumentRouter, get_router
from omnirag.ingestion.text import TextProcessor


@pytest.fixture
def router() -> DocumentRouter:
    return DocumentRouter()


class TestSupportedExtensions:
    def test_every_required_format_is_supported(self, router):
        required = {"pdf", "docx", "pptx", "txt", "md", "jpg", "jpeg", "png", "webp"}
        assert required <= set(router.supported_extensions())

    @pytest.mark.parametrize("filename,processor", [
        ("report.pdf", PDFProcessor),
        ("REPORT.PDF", PDFProcessor),
        ("notes.docx", WordProcessor),
        ("deck.pptx", PowerPointProcessor),
        ("photo.jpg", ImageProcessor),
        ("photo.JPEG", ImageProcessor),
        ("scan.png", ImageProcessor),
        ("shot.webp", ImageProcessor),
        ("notes.txt", TextProcessor),
        ("readme.md", TextProcessor),
    ])
    def test_routing(self, router, filename, processor):
        assert isinstance(router.route(filename), processor)

    @pytest.mark.parametrize("filename", ["a.exe", "a.zip", "a.csv", "noextension", "a.xlsx"])
    def test_unsupported_types_raise_a_friendly_error(self, router, filename):
        with pytest.raises(UnsupportedFileTypeError) as excinfo:
            router.route(filename)
        assert filename.split(".")[-1] in excinfo.value.user_message or "supported" in excinfo.value.user_message.lower()

    def test_markdown_gets_its_own_file_type(self, router):
        assert router.file_type_for("readme.md") == FileType.MARKDOWN
        assert router.file_type_for("notes.txt") == FileType.TXT

    def test_is_supported(self, router):
        assert router.is_supported("a.pdf") is True
        assert router.is_supported("a.exe") is False

    def test_a_new_processor_can_be_registered(self, router):
        # Demonstrates the extension point for XLSX/HTML.
        class FakeExcelProcessor(TextProcessor):
            extensions = ("xlsx",)
            file_type = FileType.XLSX

        router.register(FakeExcelProcessor())
        assert "xlsx" in router.supported_extensions()
        assert isinstance(router.route("book.xlsx"), FakeExcelProcessor)


class TestValidation:
    def test_valid_upload_passes(self, router, settings):
        validation = router.validate("report.md", b"# Title\n\nSome content.", settings=settings)

        assert validation.safe_filename == "report.md"
        assert validation.extension == "md"
        assert validation.size_bytes > 0

    def test_empty_file_is_rejected(self, router, settings):
        with pytest.raises(EmptyDocumentError):
            router.validate("empty.txt", b"", settings=settings)

    def test_oversized_file_is_rejected(self, router, settings, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
        from omnirag.config.settings import build_settings

        with pytest.raises(FileTooLargeError) as excinfo:
            router.validate("big.txt", b"x" * 50_000, settings=build_settings())
        assert "MB" in excinfo.value.user_message

    def test_filename_is_sanitised(self, router, settings):
        validation = router.validate(
            "../../etc/passwd.txt", b"content here", settings=settings
        )
        assert "/" not in validation.safe_filename
        assert ".." not in validation.safe_filename

    def test_arabic_filenames_are_preserved(self, router, settings):
        validation = router.validate(
            "تقرير سنوي.txt", "محتوى المستند".encode("utf-8"), settings=settings
        )
        assert "تقرير" in validation.safe_filename

    def test_content_signature_mismatch_is_caught(self, router, settings):
        # A ZIP renamed to .pdf must not reach the PDF parser.
        with pytest.raises(UnsupportedFileTypeError):
            router.validate("fake.pdf", b"PK\x03\x04rest of a zip", settings=settings)

    def test_real_pdf_signature_passes(self, router, settings):
        assert router.validate("real.pdf", b"%PDF-1.7\nbody", settings=settings)

    def test_office_zip_signature_passes(self, router, settings):
        assert router.validate("doc.docx", b"PK\x03\x04word/", settings=settings)


def test_router_singleton_is_shared():
    assert get_router() is get_router()
