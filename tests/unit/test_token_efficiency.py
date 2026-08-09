"""Production token budgets, lazy visual understanding, and bounded Groq waits."""

from __future__ import annotations

from dataclasses import replace

import pytest

from omnirag.config.settings import build_settings
from omnirag.core.enums import BlockType, FileType, Role, SourceKind
from omnirag.core.exceptions import AllProvidersFailedError, RateLimitError
from omnirag.core.models import Chunk, RetrievalResult, SearchResult, VisualRef
from omnirag.ingestion.base import ProcessingContext
from omnirag.ingestion.pdf import PDFProcessor
from omnirag.intelligence.vision import VisionAnalyzer
from omnirag.intelligence.ocr import OCREngine
from omnirag.providers.ocr.vision_llm import VisionOCRProvider
from omnirag.providers.llm.base import (
    BaseLLMProvider,
    ImagePart,
    LLMMessage,
    LLMRequestRequirements,
    LLMResponse,
)
from omnirag.providers.llm.groq import DEFAULT_VISION_MODEL, GroqLLM
from omnirag.providers.llm.context import llm_session
from omnirag.providers.llm.router import FallbackLLMProvider
from omnirag.rag.generation import AnswerGenerator, GenerationRequest
from omnirag.rag.query_rewrite import parse_query


class CountingVisionLLM(BaseLLMProvider):
    name = "counting"
    supports_vision = True

    def __init__(self):
        super().__init__(model="counting-vision")
        self.calls = []

    def supports_images(self, model=None):
        return True

    def complete(self, messages, **kwargs):
        self.calls.append(
            {
                "images": sum(len(message.images) for message in messages),
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "json_mode": kwargs.get("json_mode", False),
            }
        )
        if kwargs.get("json_mode"):
            text = (
                '{"type":"diagram","title":"Page 3","description":'
                '"A labelled process diagram.","text":"مرحلة ١ ثم مرحلة ٢",'
                '"entities":[],"data_points":[],"confidence":0.9,'
                '"unreadable":false}'
            )
        else:
            text = "يوضح الرسم مرحلتين مترابطتين. [1]"
        return LLMResponse(text=text, model=self.model, provider=self.name)


class OutcomeProvider(BaseLLMProvider):
    def __init__(self, name, outcomes):
        super().__init__(model=f"{name}-model")
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return LLMResponse(text=outcome, model=self.model, provider=self.name)


def test_production_default_operation_budgets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_CHAIN", "mock")
    settings = build_settings()

    assert settings.vision.analysis_max_output_tokens == 800
    assert settings.ocr.max_output_tokens == 1200
    assert settings.retrieval.query_rewrite_max_output_tokens == 256
    assert settings.retrieval.rerank_max_output_tokens == 256
    assert settings.llm.max_output_tokens == 2048
    assert settings.llm.exhaustive_max_output_tokens == 4096
    assert settings.llm.groq_max_rate_limit_wait_seconds == 20
    assert settings.llm.groq_estimated_image_tokens == 2048
    assert settings.llm.groq_focused_vision_max_output_tokens == 1024


def test_scanned_pdf_ingestion_makes_zero_vision_calls(
    settings, file_store, sample_png
):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for _ in range(8):
        page = pdf.new_page(width=500, height=500)
        page.insert_image(page.rect, stream=sample_png)
    data = pdf.tobytes()
    pdf.close()

    llm = CountingVisionLLM()
    lazy_settings = replace(
        settings,
        vision=replace(settings.vision, enabled=True, lazy_analysis=True),
    )
    ctx = ProcessingContext(
        session_id="lazy-session",
        document_id="lazy-document",
        filename="scan.pdf",
        settings=lazy_settings,
        file_store=file_store,
        vision=VisionAnalyzer(llm, min_image_pixels=1),
        ocr=OCREngine(VisionOCRProvider(llm, max_output_tokens=1200)),
    )

    document = PDFProcessor().parse(data, ctx)

    assert document.page_count == 8
    assert len(document.blocks) == 8
    assert all(block.visual is not None for block in document.blocks)
    assert all(
        block.metadata["visual_analysis_pending"] is True
        for block in document.blocks
    )
    assert llm.calls == []


def test_page_three_visual_analysis_is_cached_across_generator_rebuilds(
    settings, file_store, sample_png
):
    llm = CountingVisionLLM()
    visual = file_store.put("session", sample_png, media_type="image/png")
    chunk = Chunk(
        document_id="page-three-document",
        session_id="session",
        filename="scan.pdf",
        file_type=FileType.PDF,
        page_number=3,
        page_label="Page 3",
        block_ids=["page-three-block"],
        block_type=BlockType.PAGE_SNAPSHOT,
        source_kind=SourceKind.VISION,
        text="Image on page 3 of scan.pdf.",
        visual=VisualRef(
            asset_id=visual.asset_id,
            media_type="image/png",
            origin="page_render",
            page_number=3,
        ),
        metadata={"visual_analysis_pending": True},
    )
    retrieval = RetrievalResult(
        query="page 3 اشرحلي الرسومات الموجودة في",
        results=[
            SearchResult(chunk=chunk, rank=0),
            SearchResult(
                chunk=chunk.model_copy(
                    update={
                        "chunk_id": "page-four-chunk",
                        "page_number": 4,
                        "page_label": "Page 4",
                        "block_ids": ["page-four-block"],
                    }
                ),
                rank=1,
            ),
        ],
    )
    plan = parse_query(retrieval.query)
    vision = VisionAnalyzer(llm, min_image_pixels=1, max_output_tokens=800)
    vision.clear_cache()

    for _ in range(2):
        generator = AnswerGenerator(
            llm, file_store=file_store, vision=vision, settings=replace(
                settings,
                vision=replace(settings.vision, enabled=True, lazy_analysis=True),
            )
        )
        result = generator.generate(
            GenerationRequest(
                question=retrieval.query,
                retrieval=retrieval,
                session_id="session",
                plan=plan,
            )
        )
        assert "[1]" in result.answer

    vision_calls = [call for call in llm.calls if call["json_mode"]]
    final_calls = [call for call in llm.calls if not call["json_mode"]]
    assert len(vision_calls) == 1
    assert vision_calls[0]["max_output_tokens"] == 800
    assert len(final_calls) == 2
    assert all(call["max_output_tokens"] == 2048 for call in final_calls)
    assert all(call["images"] == 1 for call in [*vision_calls, *final_calls])
    assert vision.cache_hits == 1


def test_focused_page_compaction_keeps_evidence_and_drops_unrelated_pages(
    settings, file_store, sample_png
):
    llm = CountingVisionLLM()
    asset = file_store.put("session", sample_png, media_type="image/png")
    visual = VisualRef(
        asset_id=asset.asset_id,
        media_type="image/png",
        origin="page_render",
        page_number=3,
    )
    first = Chunk(
        document_id="doc",
        session_id="session",
        filename="scan.pdf",
        page_number=3,
        block_ids=["vision"],
        block_type=BlockType.PAGE_SNAPSHOT,
        source_kind=SourceKind.VISION,
        text="Visual description of the coordination diagram.",
        visual=visual,
    )
    duplicate = first.model_copy(
        update={
            "chunk_id": "ocr-copy",
            "block_ids": ["ocr"],
            "text": "OCR labels: salary processing and bank information.",
        }
    )
    unrelated = first.model_copy(
        update={
            "chunk_id": "page-four",
            "page_number": 4,
            "page_label": "Page 4",
            "block_ids": ["page-four"],
            "text": "Unrelated Page 4 evidence.",
        }
    )
    generator = AnswerGenerator(llm, file_store=file_store, settings=settings)
    plan = parse_query("اشرح بيدج 3")

    compacted = generator._focused_page_results(
        [
            SearchResult(chunk=first, rank=0),
            SearchResult(chunk=duplicate, rank=1),
            SearchResult(chunk=unrelated, rank=2),
        ],
        plan,
    )

    assert len(compacted) == 1
    assert compacted[0].chunk.page_number == 3
    assert "coordination diagram" in compacted[0].chunk.text
    assert "salary processing" in compacted[0].chunk.text
    assert compacted[0].chunk.block_ids == ["vision", "ocr"]


def test_standalone_vision_ocr_obeys_1200_token_ceiling(sample_png):
    llm = CountingVisionLLM()
    provider = VisionOCRProvider(llm, max_output_tokens=1200)

    provider.recognize(sample_png)

    assert llm.calls == [
        {"images": 1, "max_output_tokens": 1200, "json_mode": True}
    ]


def test_groq_multimodal_diagnostics_select_actual_vision_model():
    provider = GroqLLM(api_key="not-sent", retry_attempts=1)
    messages = [
        LLMMessage(
            role=Role.USER,
            text="page 3",
            images=[ImagePart(data=b"image", media_type="image/png")],
        )
    ]

    assert provider.model_for_request(messages) == DEFAULT_VISION_MODEL


def test_groq_tpm_preflight_caps_output_without_dropping_evidence(monkeypatch):
    captured = {}

    def fake_complete(self, messages, **kwargs):
        captured.update(kwargs)
        captured["messages"] = messages
        return LLMResponse(text="ok", model=self.model, provider="groq")

    monkeypatch.setattr(
        "omnirag.providers.llm.openai_compat.OpenAICompatibleLLM.complete",
        fake_complete,
    )
    provider = GroqLLM(
        api_key="not-sent",
        tpm_limit=1000,
        estimated_image_tokens=100,
    )
    messages = [LLMMessage(text="evidence " * 120)]

    provider.complete(messages, max_output_tokens=900)

    assert captured["messages"] is messages
    assert 128 <= captured["max_output_tokens"] < 900


def test_focused_groq_page_vision_stays_below_tpm_and_preserves_image(monkeypatch):
    captured = {}

    def fake_complete(self, messages, **kwargs):
        captured.update(kwargs)
        captured["messages"] = messages
        return LLMResponse(text="ok", model=self.vision_model, provider="groq")

    monkeypatch.setattr(
        "omnirag.providers.llm.openai_compat.OpenAICompatibleLLM.complete",
        fake_complete,
    )
    provider = GroqLLM(
        api_key="not-sent",
        max_output_tokens=2048,
        tpm_limit=8000,
        estimated_image_tokens=2048,
        focused_vision_max_output_tokens=1024,
    )
    image = ImagePart(data=b"page-three-image", media_type="image/png")
    messages = [
        LLMMessage(
            text="Only the compact Page 3 evidence and citation identity.",
            images=[image],
        )
    ]

    response = provider.complete(
        messages,
        max_output_tokens=2048,
        requirements=LLMRequestRequirements(
            requires_images=True,
            operation="final_answer",
        ),
    )

    assert captured["messages"] is messages
    assert captured["messages"][0].images == [image]
    assert captured["max_output_tokens"] == 1024
    assert response.diagnostics["selected_visuals"] == 1
    assert response.diagnostics["estimated_total_tokens"] < 8000


def test_groq_retry_after_waits_once_when_within_bound(monkeypatch):
    from omnirag.utils.retry import retry_call

    sleeps = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("TPM", provider="groq", retry_after=10)
        return "ok"

    result = retry_call(
        operation,
        attempts=2,
        max_delay=20,
        skip_if_retry_after_exceeds_max=True,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [10]


def test_groq_retry_after_over_bound_falls_back_without_wait():
    from omnirag.utils.retry import retry_call

    sleeps = []
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise RateLimitError("TPM", provider="groq", retry_after=30)

    with pytest.raises(RateLimitError):
        retry_call(
            operation,
            attempts=2,
            max_delay=20,
            skip_if_retry_after_exceeds_max=True,
            sleep=sleeps.append,
        )

    assert attempts == 1
    assert sleeps == []


def test_hard_gemini_quota_is_dead_for_only_the_affected_session():
    gemini = OutcomeProvider(
        "gemini",
        [
            RateLimitError(
                "daily free quota exhausted",
                provider="gemini",
                quota_exhausted=True,
                quota_scope="hard_quota",
            ),
            "new-session-primary",
        ],
    )
    groq = OutcomeProvider("groq", ["fallback-one", "fallback-two"])
    router = FallbackLLMProvider([gemini, groq])

    with llm_session("quota-session"):
        assert router.complete([LLMMessage(text="one")]).provider == "groq"
        assert router.complete([LLMMessage(text="two")]).provider == "groq"
    with llm_session("different-session"):
        assert router.complete([LLMMessage(text="three")]).provider == "gemini"

    assert gemini.calls == 2
    assert groq.calls == 2


def test_openrouter_zero_remaining_is_skipped_until_reported_reset():
    now = [0.0]
    openrouter = OutcomeProvider(
        "openrouter",
        [
            RateLimitError(
                "daily quota",
                provider="openrouter",
                quota_exhausted=True,
                quota_scope="daily_or_account",
                reset_at="100",
            ),
            "after-reset",
        ],
    )
    router = FallbackLLMProvider(
        [openrouter],
        clock=lambda: now[0],
        wall_clock=lambda: 0.0,
    )

    with llm_session("openrouter-session"):
        with pytest.raises(AllProvidersFailedError):
            router.complete([LLMMessage(text="one")])
        with pytest.raises(AllProvidersFailedError):
            router.complete([LLMMessage(text="two")])
        assert openrouter.calls == 1
        now[0] = 101
        assert router.complete([LLMMessage(text="three")]).text == "after-reset"

    assert openrouter.calls == 2
