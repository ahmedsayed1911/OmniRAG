"""End-to-end pipeline integration.

Exercises the full vertical slice with mocked providers:

    upload -> validate -> parse -> chunk -> embed -> index
           -> retrieve -> rerank -> generate -> cite

No API keys, no network.
"""

from __future__ import annotations

import pytest

from omnirag.core.enums import BlockType, IngestionStatus, Role
from omnirag.core.exceptions import ProviderUnavailableError
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.embeddings.resilient import ResilientEmbeddings
from omnirag.rag.generation import AnswerGenerator
from omnirag.services.chat_service import ChatRequest, ChatService
from omnirag.services.ingestion_service import UploadedFile
from omnirag.storage.sessions import new_session_id


@pytest.fixture
def wired(engine, fake_embeddings, recording_llm):
    """Engine with deterministic offline embeddings and a scripted LLM."""
    engine._embeddings = fake_embeddings
    engine._llm = recording_llm
    return engine


@pytest.fixture
def service(wired):
    from omnirag.services.ingestion_service import IngestionService

    return IngestionService(wired)


class TestFullPipeline:
    def test_markdown_upload_to_cited_answer(
        self, service, wired, session_id, sample_markdown, recording_llm
    ):
        result = service.ingest(
            session_id, UploadedFile(name="annual_report.md", data=sample_markdown)
        )

        assert result.status == IngestionStatus.READY
        assert result.chunk_count > 0
        assert wired.vector_store.count(session_id) == result.chunk_count

        recording_llm.responses = ["Total revenue reached 8,400,000 USD in Q4 2024 [1]."]
        chat = ChatService(wired)
        message = chat.answer(
            ChatRequest(question="What was total revenue in Q4?", session_id=session_id)
        )

        assert message.role == Role.ASSISTANT
        assert message.error is None
        assert "8,400,000" in message.content
        assert message.citations
        assert message.citations[0].filename == "annual_report.md"
        assert message.debug["contexts"] > 0

    def test_pdf_upload_to_answer_with_page_citation(
        self, service, wired, session_id, sample_pdf, recording_llm
    ):
        result = service.ingest(session_id, UploadedFile(name="report.pdf", data=sample_pdf))
        assert result.status == IngestionStatus.READY
        assert result.page_count == 2

        recording_llm.responses = ["Currency volatility is the principal risk [1]."]
        message = ChatService(wired).answer(
            ChatRequest(question="What is the principal risk?", session_id=session_id)
        )

        assert message.citations
        cited = message.citations[0]
        assert cited.filename == "report.pdf"
        assert cited.page_number in (1, 2)
        assert cited.page_label.startswith("Page")

    def test_document_ingestion_completes_with_runtime_hash_fallback(
        self, engine, ingestion_service, session_id, sample_pdf
    ):
        class UnavailableEmbeddings(BaseEmbeddingProvider):
            name = "gemini"

            def __init__(self):
                super().__init__(model="gemini-embedding-001")

            def embed_batch(self, texts, *, is_query=False):
                raise ProviderUnavailableError("mock 503", provider="gemini")

        engine._embeddings = ResilientEmbeddings(UnavailableEmbeddings())

        result = ingestion_service.ingest(
            session_id, UploadedFile(name="fallback.pdf", data=sample_pdf)
        )

        assert result.status == IngestionStatus.READY
        assert result.chunk_count > 0
        assert engine.embeddings.fallback_active
        assert any("offline hash embeddings" in warning for warning in result.warnings)

    def test_pptx_answer_cites_slide_numbers(
        self, service, wired, session_id, sample_pptx, recording_llm
    ):
        service.ingest(session_id, UploadedFile(name="deck.pptx", data=sample_pptx))

        recording_llm.responses = ["EMEA grew 62 percent [1]."]
        message = ChatService(wired).answer(
            ChatRequest(question="How much did EMEA grow?", session_id=session_id)
        )

        assert message.citations
        assert any("Slide" in c.page_label for c in message.citations)

    def test_traceability_chain_is_complete(
        self, service, wired, session_id, sample_markdown, recording_llm
    ):
        """Answer -> citation -> chunk -> block ids -> page -> document."""
        service.ingest(session_id, UploadedFile(name="report.md", data=sample_markdown))

        recording_llm.responses = ["Revenue grew [1]."]
        message = ChatService(wired).answer(
            ChatRequest(question="revenue", session_id=session_id)
        )

        chunks = {c.chunk_id: c for c in wired.vector_store.list_chunks(session_id)}
        summaries = {d.document_id for d in wired.registry.list(session_id)}

        for citation in message.citations:
            chunk = chunks[citation.chunk_id]
            assert chunk.block_ids
            assert chunk.document_id in summaries
            assert chunk.page_number >= 1
            assert chunk.filename == citation.filename


class TestDeduplication:
    def test_same_file_is_not_reprocessed(self, service, session_id, sample_markdown):
        first = service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        second = service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))

        assert first.status == IngestionStatus.READY
        assert second.status == IngestionStatus.DUPLICATE
        assert second.document_id == first.document_id

    def test_same_content_under_a_different_name_is_still_a_duplicate(
        self, service, session_id, sample_markdown
    ):
        service.ingest(session_id, UploadedFile(name="original.md", data=sample_markdown))
        renamed = service.ingest(session_id, UploadedFile(name="copy.md", data=sample_markdown))

        assert renamed.status == IngestionStatus.DUPLICATE

    def test_different_content_is_processed(self, service, session_id, sample_markdown, sample_text):
        first = service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        second = service.ingest(session_id, UploadedFile(name="b.txt", data=sample_text))

        assert second.status == IngestionStatus.READY
        assert second.document_id != first.document_id

    def test_the_same_file_in_another_session_is_processed(
        self, service, sample_markdown
    ):
        alice, bob = new_session_id(), new_session_id()
        service.ingest(alice, UploadedFile(name="shared.md", data=sample_markdown))
        result = service.ingest(bob, UploadedFile(name="shared.md", data=sample_markdown))

        assert result.status == IngestionStatus.READY


class TestMultiDocument:
    def test_retrieval_spans_multiple_documents(
        self, service, wired, session_id, sample_markdown, sample_text, recording_llm
    ):
        service.ingest(session_id, UploadedFile(name="finance.md", data=sample_markdown))
        service.ingest(session_id, UploadedFile(name="project.txt", data=sample_text))

        recording_llm.responses = ["Comparison of both documents [1][2]."]
        message = ChatService(wired).answer(
            ChatRequest(
                question="Compare the finance report with the project report",
                session_id=session_id,
            )
        )

        filenames = {c.filename for c in message.citations}
        assert len(filenames) >= 2

    def test_document_selection_restricts_retrieval(
        self, service, wired, session_id, sample_markdown, sample_text, recording_llm
    ):
        first = service.ingest(session_id, UploadedFile(name="finance.md", data=sample_markdown))
        service.ingest(session_id, UploadedFile(name="project.txt", data=sample_text))

        recording_llm.responses = ["Answer [1]."]
        message = ChatService(wired).answer(
            ChatRequest(
                question="What was the total cost?",
                session_id=session_id,
                document_ids=[first.document_id],
            )
        )

        assert {c.filename for c in message.citations} <= {"finance.md"}


class TestSessionLifecycle:
    def test_documents_never_leak_between_sessions(
        self, service, wired, sample_markdown, recording_llm
    ):
        alice, bob = new_session_id(), new_session_id()
        service.ingest(alice, UploadedFile(name="confidential.md", data=sample_markdown))

        recording_llm.responses = ["Should not happen"]
        message = ChatService(wired).answer(
            ChatRequest(question="What was total revenue?", session_id=bob)
        )

        assert message.citations == []
        assert message.debug.get("contexts", 0) == 0 or not message.debug

    def test_removing_a_document_removes_its_vectors(
        self, service, wired, session_id, sample_markdown
    ):
        result = service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        assert wired.vector_store.count(session_id) > 0

        service.remove_document(session_id, result.document_id)

        assert wired.vector_store.count(session_id) == 0
        assert wired.registry.list(session_id) == []

    def test_clearing_a_session_removes_everything(
        self, service, wired, session_id, sample_markdown
    ):
        service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        stats = service.clear_session(session_id)

        assert stats["documents"] >= 1
        assert wired.vector_store.count(session_id) == 0
        assert wired.registry.list(session_id) == []

    def test_reindex_rebuilds_from_stored_bytes(
        self, service, wired, session_id, sample_markdown
    ):
        first = service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        original_count = wired.vector_store.count(session_id)

        report = service.reindex(session_id)

        assert report.reindexed == ["a.md"]
        assert report.missing_source == []
        assert wired.vector_store.count(session_id) == original_count
        assert wired.registry.list(session_id)[0].filename == "a.md"


class TestFailureHandling:
    def test_unsupported_file_fails_gracefully(self, service, session_id):
        result = service.ingest(session_id, UploadedFile(name="virus.exe", data=b"MZ\x90\x00"))

        assert result.status == IngestionStatus.FAILED
        assert result.error
        assert "supported" in result.error.lower() or "not a supported" in result.error.lower()

    def test_empty_file_fails_gracefully(self, service, session_id):
        result = service.ingest(session_id, UploadedFile(name="empty.txt", data=b""))

        assert result.status == IngestionStatus.FAILED
        assert result.error

    def test_corrupted_pdf_fails_gracefully(self, service, session_id):
        result = service.ingest(
            session_id, UploadedFile(name="broken.pdf", data=b"%PDF-1.4 not a real pdf")
        )

        assert result.status == IngestionStatus.FAILED
        assert "broken.pdf" in result.error

    def test_one_bad_file_does_not_block_the_others(
        self, service, session_id, sample_markdown
    ):
        results = service.ingest_many(
            session_id,
            [
                UploadedFile(name="bad.exe", data=b"MZ"),
                UploadedFile(name="good.md", data=sample_markdown),
            ],
        )

        assert results[0].status == IngestionStatus.FAILED
        assert results[1].status == IngestionStatus.READY

    def test_llm_outage_leaves_the_index_intact(
        self, service, wired, session_id, sample_markdown
    ):
        from omnirag.core.exceptions import AllProvidersFailedError, RateLimitError

        service.ingest(session_id, UploadedFile(name="a.md", data=sample_markdown))
        indexed = wired.vector_store.count(session_id)

        class DeadLLM:
            name = "dead"
            model = "dead"
            supports_vision = True

            def supports_images(self, model=None):
                return True

            def complete(self, *args, **kwargs):
                raise AllProvidersFailedError(
                    [("gemini", RateLimitError("429")), ("openrouter", RateLimitError("429"))]
                )

        wired._llm = DeadLLM()
        message = ChatService(wired).answer(
            ChatRequest(question="What was revenue?", session_id=session_id)
        )

        assert message.error is not None
        assert "failed" in message.content.lower() or "provider" in message.content.lower()
        # The index survived the outage untouched.
        assert wired.vector_store.count(session_id) == indexed

    def test_question_without_documents_is_answered_honestly(self, wired, session_id):
        message = ChatService(wired).answer(
            ChatRequest(question="What was revenue?", session_id=session_id)
        )
        assert message.citations == []

    def test_empty_question_is_rejected(self, wired, session_id):
        message = ChatService(wired).answer(ChatRequest(question="   ", session_id=session_id))
        assert message.error


class TestArabicEndToEnd:
    def test_arabic_question_retrieves_from_an_arabic_section(
        self, service, wired, session_id, sample_markdown, recording_llm
    ):
        service.ingest(session_id, UploadedFile(name="report.md", data=sample_markdown))

        recording_llm.responses = ["بلغت الإيرادات 8.4 مليون دولار [1]."]
        message = ChatService(wired).answer(
            ChatRequest(question="ما هي الإيرادات الإجمالية؟", session_id=session_id)
        )

        assert message.citations
        assert "الإيرادات" in message.content

    def test_answer_language_instruction_reaches_the_prompt(
        self, service, wired, session_id, sample_markdown, recording_llm
    ):
        service.ingest(session_id, UploadedFile(name="report.md", data=sample_markdown))

        recording_llm.responses = ["بلغت الإيرادات 8.4 مليون [1]."]
        ChatService(wired).answer(
            ChatRequest(
                question="Answer in Arabic: what was the total revenue?",
                session_id=session_id,
            )
        )

        system_prompt = recording_llm.calls[-1]["system"]
        assert "Arabic" in system_prompt


class TestSuggestedPrompts:
    def test_prompts_adapt_to_session_content(
        self, service, wired, session_id, sample_markdown
    ):
        service.ingest(session_id, UploadedFile(name="report.md", data=sample_markdown))
        prompts = ChatService(wired).suggested_prompts(session_id)

        assert prompts
        assert any("Summarize" in p for p in prompts)

    def test_no_documents_means_no_prompts(self, wired, session_id):
        assert ChatService(wired).suggested_prompts(session_id) == []
