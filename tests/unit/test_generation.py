"""Grounded generation: prompt contract, multimodal attachment, honesty rules."""

from __future__ import annotations

import pytest

from omnirag.config.settings import build_settings
from omnirag.core.enums import BlockType, FileType, Language, Role, SourceKind
from omnirag.core.models import (
    ChatMessage,
    Chunk,
    RetrievalResult,
    SearchResult,
    VisualRef,
)
from omnirag.rag.generation import (
    INSUFFICIENT_MARKER,
    SYSTEM_PROMPT,
    AnswerGenerator,
    GenerationRequest,
)


def chunk(text, *, page=1, filename="report.pdf", **kwargs):
    return Chunk(
        document_id=kwargs.pop("document_id", "doc-1"),
        session_id="s1",
        filename=filename,
        file_type=FileType.PDF,
        page_number=page,
        page_label=f"Page {page}",
        block_ids=[f"b{page}"],
        text=text,
        **kwargs,
    )


def retrieval(*chunks) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        results=[SearchResult(chunk=c, score=0.9 - i * 0.1) for i, c in enumerate(chunks)],
    )


@pytest.fixture
def generator(recording_llm, file_store, settings):
    return AnswerGenerator(recording_llm, file_store=file_store, settings=settings)


class TestPromptContract:
    def test_system_prompt_states_every_grounding_rule(self):
        lowered = SYSTEM_PROMPT.lower()

        assert "only the numbered sources" in lowered
        assert INSUFFICIENT_MARKER in SYSTEM_PROMPT
        assert "never invent" in lowered
        assert "exactly" in lowered                 # numeric fidelity
        assert "not evidence" in lowered            # history is not evidence
        assert "language of the user's question" in lowered

    def test_context_is_numbered_and_labelled(self, generator, recording_llm):
        recording_llm.responses = ["Revenue was 8.4M [1]."]
        generator.generate(
            GenerationRequest(
                question="What was revenue?",
                retrieval=retrieval(
                    chunk("Revenue reached 8,400,000 USD.", page=4),
                    chunk("Risks include currency volatility.", page=9),
                ),
                session_id="s1",
            )
        )

        prompt = recording_llm.calls[0]["text"]
        assert "[1] [report.pdf — Page 4]" in prompt
        assert "[2] [report.pdf — Page 9]" in prompt
        assert "SOURCES" in prompt

    def test_provenance_is_described_to_the_model(self, generator, recording_llm):
        recording_llm.responses = ["ok [1]"]
        generator.generate(
            GenerationRequest(
                question="q",
                retrieval=retrieval(
                    chunk(
                        "Handwritten note about the deadline.",
                        block_type=BlockType.HANDWRITING,
                        source_kind=SourceKind.OCR,
                        uncertain=True,
                        confidence=0.4,
                    )
                ),
                session_id="s1",
            )
        )

        prompt = recording_llm.calls[0]["text"]
        assert "handwritten note" in prompt
        assert "LOW CONFIDENCE" in prompt

    def test_whole_documents_are_never_sent(self, generator, recording_llm):
        recording_llm.responses = ["ok [1]"]
        long_chunk = chunk("x" * 50_000)
        generator.generate(
            GenerationRequest(question="q", retrieval=retrieval(long_chunk), session_id="s1")
        )

        # Context is truncated per chunk rather than dumping the document.
        assert len(recording_llm.calls[0]["text"]) < 12_000

    def test_history_is_passed_as_separate_turns(self, generator, recording_llm):
        recording_llm.responses = ["ok [1]"]
        history = [
            ChatMessage(role=Role.USER, content="What was revenue?"),
            ChatMessage(role=Role.ASSISTANT, content="It was 8.4M [1]."),
        ]
        generator.generate(
            GenerationRequest(
                question="And the risks?",
                retrieval=retrieval(chunk("Risks include currency volatility.")),
                session_id="s1",
                history=history,
            )
        )

        roles = recording_llm.calls[0]["roles"]
        assert roles.count("user") >= 2
        assert "assistant" in roles


class TestCitationEnforcement:
    def test_valid_citations_pass_through(self, generator, recording_llm):
        recording_llm.responses = ["Revenue reached 8,400,000 USD [1]."]
        result = generator.generate(
            GenerationRequest(
                question="revenue?",
                retrieval=retrieval(chunk("Revenue reached 8,400,000 USD.", page=4)),
                session_id="s1",
            )
        )

        assert "[1]" in result.answer
        assert result.citations[0].page_number == 4
        assert not result.warnings

    def test_fabricated_citations_are_stripped_and_flagged(self, generator, recording_llm):
        recording_llm.responses = ["Revenue grew [1]. Margins improved [5]."]
        result = generator.generate(
            GenerationRequest(
                question="q",
                retrieval=retrieval(chunk("Revenue reached 8.4M.")),
                session_id="s1",
            )
        )

        assert "[5]" not in result.answer
        assert any("not provided" in w for w in result.warnings)

    def test_uncited_answer_is_flagged(self, generator, recording_llm):
        recording_llm.responses = ["Revenue was strong and margins improved substantially."]
        result = generator.generate(
            GenerationRequest(
                question="q", retrieval=retrieval(chunk("Revenue reached 8.4M.")), session_id="s1"
            )
        )

        assert any("did not cite" in w for w in result.warnings)

    def test_insufficient_evidence_marker_is_handled(self, generator, recording_llm):
        recording_llm.responses = [
            f"{INSUFFICIENT_MARKER}\nThe documents do not mention the 2026 forecast."
        ]
        result = generator.generate(
            GenerationRequest(
                question="What is the 2026 forecast?",
                retrieval=retrieval(chunk("2024 revenue was 8.4M.")),
                session_id="s1",
            )
        )

        assert result.insufficient_evidence is True
        assert INSUFFICIENT_MARKER not in result.answer

    def test_no_retrieval_means_no_llm_call_at_all(self, generator, recording_llm):
        result = generator.generate(
            GenerationRequest(question="q", retrieval=RetrievalResult(query="q"), session_id="s1")
        )

        assert result.insufficient_evidence is True
        assert result.citations == []
        assert recording_llm.calls == [], "must not pay for a call with no evidence"

    def test_no_evidence_message_matches_the_question_language(self, generator):
        result = generator.generate(
            GenerationRequest(
                question="ما هي الإيرادات؟",
                retrieval=RetrievalResult(query="q", language=Language.ARABIC),
                session_id="s1",
            )
        )
        assert "المستندات" in result.answer


class TestMultimodalAttachment:
    def test_original_image_is_attached_for_visual_sources(
        self, recording_llm, file_store, settings
    ):
        asset = file_store.put("s1", b"fake-png-bytes", media_type="image/png")
        generator = AnswerGenerator(recording_llm, file_store=file_store, settings=settings)
        recording_llm.responses = ["The chart shows Q4 at 8400 [1]."]

        result = generator.generate(
            GenerationRequest(
                question="What does the chart show?",
                retrieval=retrieval(
                    chunk(
                        "Bar chart of quarterly revenue.",
                        block_type=BlockType.CHART,
                        source_kind=SourceKind.VISION,
                        visual=VisualRef(asset_id=asset.asset_id, media_type="image/png"),
                    )
                ),
                session_id="s1",
            )
        )

        assert recording_llm.calls[0]["images"] == 1
        assert result.used_images == 1

    def test_text_sources_attach_no_images(self, generator, recording_llm):
        recording_llm.responses = ["ok [1]"]
        generator.generate(
            GenerationRequest(
                question="q", retrieval=retrieval(chunk("Plain text passage.")), session_id="s1"
            )
        )
        assert recording_llm.calls[0]["images"] == 0

    def test_image_budget_is_respected(self, recording_llm, file_store, monkeypatch):
        monkeypatch.setenv("MAX_IMAGES_PER_ANSWER", "2")
        monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "mock")
        settings = build_settings()

        chunks = []
        for index in range(5):
            asset = file_store.put("s1", f"image-{index}".encode(), media_type="image/png")
            chunks.append(
                chunk(
                    f"Chart {index}",
                    page=index + 1,
                    block_type=BlockType.CHART,
                    visual=VisualRef(asset_id=asset.asset_id),
                )
            )

        generator = AnswerGenerator(recording_llm, file_store=file_store, settings=settings)
        recording_llm.responses = ["ok [1]"]
        generator.generate(
            GenerationRequest(question="q", retrieval=retrieval(*chunks), session_id="s1")
        )

        assert recording_llm.calls[0]["images"] == 2

    def test_missing_asset_does_not_break_the_answer(self, generator, recording_llm):
        recording_llm.responses = ["Answer [1]"]
        result = generator.generate(
            GenerationRequest(
                question="q",
                retrieval=retrieval(
                    chunk(
                        "Chart description",
                        block_type=BlockType.CHART,
                        visual=VisualRef(asset_id="asset-that-was-evicted"),
                    )
                ),
                session_id="s1",
            )
        )

        assert result.answer
        assert result.used_images == 0

    def test_capability_error_retries_text_only_and_warns(self, file_store, settings):
        from omnirag.core.exceptions import ProviderCapabilityError
        from omnirag.providers.llm.base import LLMResponse

        class CapabilityLimitedLLM:
            name = "openrouter"
            model = "text-only"
            supports_vision = True

            def __init__(self):
                self.calls = []

            def supports_images(self, model=None):
                return True  # the router let it through; the API rejects it

            def complete(self, messages, **kwargs):
                has_images = any(m.images for m in messages)
                self.calls.append(has_images)
                if has_images:
                    raise ProviderCapabilityError(
                        "model cannot read images",
                        provider="openrouter",
                        user_message="The configured model cannot read images.",
                    )
                return LLMResponse(text="Text-only answer [1]", model="text-only")

        asset = file_store.put("s1", b"png", media_type="image/png")
        llm = CapabilityLimitedLLM()
        generator = AnswerGenerator(llm, file_store=file_store, settings=settings)

        result = generator.generate(
            GenerationRequest(
                question="What does the chart show?",
                retrieval=retrieval(
                    chunk(
                        "Chart of revenue",
                        block_type=BlockType.CHART,
                        visual=VisualRef(asset_id=asset.asset_id),
                    )
                ),
                session_id="s1",
            )
        )

        assert llm.calls == [True, False]
        assert result.used_images == 0
        # The user is told the visual evidence could not be read.
        assert any("cannot read images" in w for w in result.warnings)
