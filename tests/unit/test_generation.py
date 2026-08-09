"""Grounded generation: prompt contract, multimodal attachment, honesty rules."""

from __future__ import annotations

import pytest

from omnirag.config.settings import build_settings
from omnirag.core.enums import BlockType, FileType, Language, QueryScope, Role, SourceKind
from omnirag.core.models import (
    ChatMessage,
    Chunk,
    RetrievalResult,
    SearchResult,
    VisualRef,
)
from omnirag.core.exceptions import ProviderUnavailableError
from omnirag.providers.llm.base import BaseLLMProvider, LLMResponse
from omnirag.rag.generation import (
    INSUFFICIENT_MARKER,
    SYSTEM_PROMPT,
    AnswerGenerator,
    GenerationRequest,
)
from omnirag.rag.query_rewrite import parse_query


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


class FinishReasonLLM(BaseLLMProvider):
    name = "gemini"
    supports_vision = True

    def __init__(self, outcomes):
        super().__init__(model="gemini-3.6-flash")
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


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

    def test_exhaustive_prompt_requires_every_supported_item_and_its_citation(
        self, generator, recording_llm
    ):
        recording_llm.responses = ["TC03 failed [1]."]
        plan = parse_query("List all failed test cases")
        assert plan.scope == QueryScope.EXHAUSTIVE
        generator.generate(
            GenerationRequest(
                question=plan.original,
                retrieval=retrieval(chunk("TC03 | Status | Fail", page=2)),
                session_id="s1",
                plan=plan,
            )
        )

        system = recording_llm.calls[0]["system"]
        assert "comprehensive coverage" in system.lower()
        assert "each reported item" in system.lower()
        assert "never invent a relationship" in system.lower()


class TestOutputCompletion:
    def _generate(self, settings, file_store, outcomes, *, question="List all failures"):
        llm = FinishReasonLLM(outcomes)
        generator = AnswerGenerator(llm, file_store=file_store, settings=settings)
        plan = parse_query(question)
        result = generator.generate(
            GenerationRequest(
                question=question,
                retrieval=retrieval(chunk("TC03 failed because validation was absent.", page=2)),
                session_id="s1",
                plan=plan,
            )
        )
        return result, llm

    def test_normal_short_answer_is_unchanged(self, settings, file_store):
        response = LLMResponse(
            text="TC03 failed [1].",
            model="gemini-3.6-flash",
            provider="gemini",
            finish_reason="STOP",
            usage={"candidatesTokenCount": 8},
        )
        result, llm = self._generate(settings, file_store, [response])
        assert result.answer == response.text
        assert result.finish_reason == "STOP"
        assert result.continued is False
        assert len(llm.calls) == 1

    def test_exhaustive_answer_uses_larger_configured_budget(
        self, settings, file_store
    ):
        object.__setattr__(settings.llm, "max_output_tokens", 2048)
        object.__setattr__(settings.llm, "exhaustive_max_output_tokens", 9000)
        response = LLMResponse(text="Complete [1].", finish_reason="STOP")
        _, llm = self._generate(settings, file_store, [response])
        assert llm.calls[0]["max_output_tokens"] == 9000

    @pytest.mark.parametrize(
        "first,second,combined",
        [
            ("The failure rea", "son is timeout [1].", "The failure reason is timeout [1]."),
            ("سبب الف", "شل هو انتهاء المهلة [1].", "سبب الفشل هو انتهاء المهلة [1]."),
            ("TC03 — سبب الف", "شل: timeout [1].", "TC03 — سبب الفشل: timeout [1]."),
        ],
    )
    def test_max_tokens_performs_one_exact_multilingual_continuation(
        self, settings, file_store, first, second, combined
    ):
        outcomes = [
            LLMResponse(
                text=first,
                finish_reason="MAX_TOKENS",
                usage={"candidatesTokenCount": 100},
            ),
            LLMResponse(
                text=second,
                finish_reason="STOP",
                usage={"candidatesTokenCount": 40},
            ),
        ]
        result, llm = self._generate(settings, file_store, outcomes)
        assert result.answer == combined
        assert result.continued is True
        assert result.finish_reason == "STOP"
        assert len(llm.calls) == 2
        continuation = llm.calls[1]["messages"][-1].text.lower()
        assert "continue exactly" in continuation
        assert "do not repeat" in continuation
        assert "citation numbering" in continuation

    def test_continuation_removes_repeated_overlap_and_keeps_citations(
        self, settings, file_store
    ):
        first = "### TC03\nFailure detail repeated phrase"
        second = "detail repeated phrase and completed [1]."
        result, _ = self._generate(
            settings,
            file_store,
            [
                LLMResponse(text=first, finish_reason="MAX_TOKENS"),
                LLMResponse(text=second, finish_reason="STOP"),
            ],
        )
        assert result.answer.count("detail repeated phrase") == 1
        assert result.citations and "[1]" in result.answer

    def test_no_more_than_one_continuation(self, settings, file_store):
        result, llm = self._generate(
            settings,
            file_store,
            [
                LLMResponse(text="Part one ", finish_reason="MAX_TOKENS"),
                LLMResponse(text="part two [1]", finish_reason="MAX_TOKENS"),
                LLMResponse(text="must not be called", finish_reason="STOP"),
            ],
        )
        assert len(llm.calls) == 2
        assert result.continued is True
        assert any("no further" in warning.lower() for warning in result.warnings)

    def test_continuation_failure_preserves_first_response(self, settings, file_store):
        first = "Complete generated portion [1]."
        result, llm = self._generate(
            settings,
            file_store,
            [
                LLMResponse(text=first, finish_reason="MAX_TOKENS"),
                ProviderUnavailableError("mock outage", provider="gemini"),
            ],
        )
        assert result.answer == first
        assert result.continued is False
        assert len(llm.calls) == 2
        assert any("continuation failed" in warning.lower() for warning in result.warnings)

    def test_stop_response_has_no_arbitrary_answer_slicing(self, settings, file_store):
        long_answer = ("Complete sentence with evidence [1]. " * 500).strip()
        result, _ = self._generate(
            settings,
            file_store,
            [LLMResponse(text=long_answer, finish_reason="STOP")],
        )
        assert result.answer == long_answer

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

    def test_page_visuals_are_scoped_and_deduplicated(
        self, recording_llm, file_store, monkeypatch
    ):
        monkeypatch.setenv("MAX_VISUALS_PER_QUERY", "3")
        monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "mock")
        settings = build_settings()
        page = file_store.put("s1", b"page-3", media_type="image/png")
        duplicate = file_store.put("s1", b"page-3", media_type="image/png")
        crop = file_store.put("s1", b"page-3-crop", media_type="image/png")
        other = file_store.put("s1", b"page-4", media_type="image/png")
        chunks = [
            chunk("full page", page=3, block_type=BlockType.PAGE_SNAPSHOT,
                  visual=VisualRef(asset_id=page.asset_id, origin="page_render")),
            chunk("duplicate page", page=3, block_type=BlockType.PAGE_SNAPSHOT,
                  visual=VisualRef(asset_id=duplicate.asset_id, origin="page_render")),
            chunk("diagram crop", page=3, block_type=BlockType.DIAGRAM,
                  visual=VisualRef(asset_id=crop.asset_id, origin="crop")),
            chunk("unrelated", page=4, block_type=BlockType.DIAGRAM,
                  visual=VisualRef(asset_id=other.asset_id, origin="page_render")),
        ]
        recording_llm.responses = ["Grounded answer [1]."]
        result = AnswerGenerator(
            recording_llm, file_store=file_store, settings=settings
        ).generate(GenerationRequest(
            question="Explain the diagram on Page 3",
            retrieval=retrieval(*chunks),
            session_id="s1",
            plan=parse_query("Explain the diagram on Page 3"),
        ))
        # Exact-page focused requests send the preferred full-page visual only;
        # duplicate/crop representations remain textual context.
        assert recording_llm.calls[0]["images"] == 1
        assert result.used_images == 1

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

    def test_required_visual_is_never_silently_dropped(self, file_store, settings):
        from omnirag.core.exceptions import ProviderCapabilityError

        class RejectingVisionLLM:
            name = "openrouter"
            model = "openrouter/free"

            def supports_images(self, model=None):
                return True

            def complete(self, messages, **kwargs):
                raise ProviderCapabilityError(
                    "no compatible image route",
                    provider="openrouter",
                    capability="images",
                    user_message="No free multimodal OpenRouter model is currently available.",
                )

        asset = file_store.put("s1", b"visual", media_type="image/png")
        plan = parse_query("Explain the diagram on Page 3")
        generator = AnswerGenerator(
            RejectingVisionLLM(), file_store=file_store, settings=settings
        )
        with pytest.raises(ProviderCapabilityError):
            generator.generate(GenerationRequest(
                question=plan.original,
                retrieval=retrieval(chunk(
                    "diagram", page=3, block_type=BlockType.DIAGRAM,
                    visual=VisualRef(asset_id=asset.asset_id, origin="page_render"),
                )),
                session_id="s1",
                plan=plan,
            ))
