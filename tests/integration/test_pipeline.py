"""End-to-end pipeline integration.

Exercises the full vertical slice with mocked providers:

    upload -> validate -> parse -> chunk -> embed -> index
           -> retrieve -> rerank -> generate -> cite

No API keys, no network.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from omnirag.core.enums import BlockType, IngestionStatus, Role
from omnirag.core.exceptions import ProviderUnavailableError
from omnirag.core.models import ChatMessage
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.embeddings.resilient import ResilientEmbeddings
from omnirag.providers.llm.base import BaseLLMProvider, LLMResponse
from omnirag.providers.llm.context import current_llm_operation
from omnirag.providers.llm.router import FallbackLLMProvider
from omnirag.rag.generation import AnswerGenerator
from omnirag.services.chat_service import ChatRequest, ChatService
from omnirag.services.chat_history import apply_regeneration, plan_regeneration
from omnirag.services.ingestion_service import IngestionService, UploadedFile
from omnirag.storage.sessions import new_session_id
from omnirag.ui.message_actions import copy_component_html


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

    def test_regeneration_reruns_retrieval_without_reingestion(
        self,
        service,
        wired,
        session_id,
        sample_markdown,
        recording_llm,
        monkeypatch,
    ):
        ingested = service.ingest(
            session_id, UploadedFile(name="report.md", data=sample_markdown)
        )
        indexed_before = wired.vector_store.count(session_id)
        documents_before = [
            item.model_dump() for item in wired.registry.list(session_id)
        ]
        user = ChatMessage(role=Role.USER, content="What was total revenue?")
        recording_llm.responses = ["Revenue was 8,400,000 USD [1]."]
        first = ChatService(wired).answer(
            ChatRequest(
                question=user.content,
                session_id=session_id,
                user_message_id=user.message_id,
            )
        )
        history = [user, first]

        searches = 0
        original_search = wired.vector_store.search

        def counted_search(*args, **kwargs):
            nonlocal searches
            searches += 1
            return original_search(*args, **kwargs)

        monkeypatch.setattr(wired.vector_store, "search", counted_search)
        recording_llm.model = "mock-regenerated"
        recording_llm.responses = ["Regenerated revenue answer [1]."]
        plan = plan_regeneration(history, first.message_id)
        regenerated = ChatService(wired).answer(
            ChatRequest(
                question=plan.prompt,
                session_id=session_id,
                history=plan.history,
                user_message_id=plan.user_message_id,
            )
        )
        updated = apply_regeneration(history, plan, regenerated)

        assert ingested.status == IngestionStatus.READY
        assert searches > 0, "regeneration must execute retrieval again"
        assert len(updated) == 2 and updated[1].content.startswith("Regenerated")
        assert updated[1].retrieval is not first.retrieval
        assert updated[1].citations is not first.citations
        assert updated[1].debug["model"] == "mock-regenerated"
        assert wired.vector_store.count(session_id) == indexed_before
        assert [item.model_dump() for item in wired.registry.list(session_id)] == documents_before

    def test_long_continued_answer_is_stored_copied_and_regenerated_completely(
        self, service, wired, session_id, sample_markdown
    ):
        service.ingest(session_id, UploadedFile(name="report.md", data=sample_markdown))
        indexed = wired.vector_store.count(session_id)

        class ContinuedLLM(BaseLLMProvider):
            name = "gemini"

            def __init__(self):
                super().__init__(model="gemini-3.6-flash")
                self.responses = [
                    ("All failures begin with TC", "MAX_TOKENS"),
                    ("03 and finish naturally [1].", "STOP"),
                    ("Regenerated list begins with TC", "MAX_TOKENS"),
                    ("03 and also finishes naturally [1].", "STOP"),
                ]

            def complete(self, messages, **kwargs):
                text, reason = self.responses.pop(0)
                return LLMResponse(
                    text=text,
                    model=self.model,
                    provider=self.name,
                    finish_reason=reason,
                    usage={"candidatesTokenCount": 10},
                )

        wired._llm = ContinuedLLM()
        user = ChatMessage(role=Role.USER, content="List all failed test cases")
        first = ChatService(wired).answer(
            ChatRequest(
                question=user.content,
                session_id=session_id,
                user_message_id=user.message_id,
            )
        )

        assert first.content == "All failures begin with TC03 and finish naturally [1]."
        assert first.debug["finish_reason"] == "STOP"
        assert first.debug["continued"] is True
        assert first.debug["returned_chars"] == len(first.content)
        assert first.content in copy_component_html(first.content)

        history = [user, first]
        plan = plan_regeneration(history, first.message_id)
        regenerated = ChatService(wired).answer(
            ChatRequest(
                question=plan.prompt,
                session_id=session_id,
                history=plan.history,
                user_message_id=plan.user_message_id,
            )
        )
        updated = apply_regeneration(history, plan, regenerated)

        assert updated[-1].content == (
            "Regenerated list begins with TC03 and also finishes naturally [1]."
        )
        assert updated[-1].debug["continued"] is True
        assert wired.vector_store.count(session_id) == indexed

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

    def test_arabic_chat_uses_mocked_provider_chain_for_rewrite_and_final_answer(
        self, engine, fake_embeddings, session_id, sample_markdown
    ):
        class UnavailablePrimary(BaseLLMProvider):
            name = "gemini"
            supports_vision = True

            def __init__(self):
                super().__init__(model="gemini-3.6-flash")

            def complete(self, messages, **kwargs):
                raise ProviderUnavailableError("mock 503", provider=self.name)

        class OperationAwareFallback(BaseLLMProvider):
            name = "openrouter"
            supports_vision = True

            def __init__(self):
                super().__init__(model="google/gemini-3.6-flash")
                self.operations = []

            def complete(self, messages, **kwargs):
                operation = current_llm_operation()
                self.operations.append(operation)
                if operation == "query_rewrite":
                    text = '{"queries":["total revenue Q4 2024"]}'
                else:
                    text = "بلغت الإيرادات 8.4 مليون دولار في الربع الرابع [1]."
                return LLMResponse(text=text, model=self.model, provider=self.name)

        engine.settings = replace(
            engine.settings,
            retrieval=replace(engine.settings.retrieval, query_rewrite=True),
        )
        engine._embeddings = fake_embeddings
        ingestion = IngestionService(engine)
        uploaded = ingestion.ingest(
            session_id, UploadedFile(name="annual.md", data=sample_markdown)
        )
        assert uploaded.status == IngestionStatus.READY

        fallback = OperationAwareFallback()
        engine._llm = FallbackLLMProvider([UnavailablePrimary(), fallback])
        answer = ChatService(engine).answer(
            ChatRequest(question="ما إجمالي الإيرادات في الربع الرابع؟", session_id=session_id)
        )

        assert answer.error is None
        assert "8.4" in answer.content
        assert answer.citations
        assert fallback.operations == ["query_rewrite", "final_answer"]
        attempts = answer.debug["provider_attempts"]
        assert any("final_answer/gemini" in attempt for attempt in attempts)
        assert any("final_answer/openrouter" in attempt for attempt in attempts)

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
