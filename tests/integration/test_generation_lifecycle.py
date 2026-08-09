"""Exact text preservation through provider, RAG, state and Streamlit reruns."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from omnirag.core.enums import FileType, Role
from omnirag.core.models import ChatMessage, Chunk, RetrievalResult, SearchResult
from omnirag.providers.llm.base import BaseLLMProvider, LLMResponse
from omnirag.services.chat_service import ChatRequest, ChatService
from omnirag.services.ingestion_service import IngestionService, UploadedFile
from omnirag.ui.message_actions import copy_component_html
from omnirag.ui.state import MESSAGES_KEY, SESSION_KEY

ROOT = Path(__file__).resolve().parents[2]


def long_arabic_markdown() -> str:
    unit = (
        "### ملخص متطلبات الاختبار\n\n"
        "- **Requirement:** English detail with exact citations [1], [2], [1-16].\n"
        "- نتيجة عربية كاملة مع شرح السبب والنتيجة المتوقعة والنتيجة الفعلية [3].\n\n"
    )
    text = unit
    while len(text) < 8_200:
        text += unit
    return text + "FINAL_SENTENCE: اكتملت الإجابة العربية دون اقتطاع [16]."


def fixed_retrieval() -> RetrievalResult:
    results = []
    for index in range(1, 17):
        chunk = Chunk(
            document_id="doc-1",
            session_id="generation-lifecycle",
            filename="sanitized.pdf",
            file_type=FileType.PDF,
            page_number=index,
            page_label=f"Page {index}",
            block_ids=[f"block-{index}"],
            text=f"Evidence item {index} supports requirement and test result {index}.",
        )
        results.append(SearchResult(chunk=chunk, score=1.0 - index / 100))
    return RetrievalResult(
        query="List all requirements, tests, reports, and errors",
        results=results,
        query_scope="MULTI_PART",
        unique_pages=16,
        total_pages=16,
    )


class LongResponseLLM(BaseLLMProvider):
    name = "gemini"

    def __init__(self, text: str):
        super().__init__(model="gemini-3.6-flash")
        self.text = text
        self.requested_output_tokens = 0

    def complete(self, messages, **kwargs):
        self.requested_output_tokens = kwargs["max_output_tokens"]
        return LLMResponse(
            text=self.text,
            model=self.model,
            provider=self.name,
            finish_reason="STOP",
            usage={"candidatesTokenCount": 3000},
            diagnostics={
                "provider_raw_chars": len(self.text),
                "parsed_chars": len(self.text),
                "candidate_count": 1,
                "content_parts_count": 1,
                "http_status": 200,
            },
        )


class FixedRetrievalChatService(ChatService):
    def _retrieve(self, request, session_id):
        return fixed_retrieval()


def test_long_arabic_answer_is_exact_at_every_boundary_and_after_rerun(
    engine, session_id, monkeypatch, sample_markdown
):
    original = long_arabic_markdown()
    assert len(original) >= 8_000
    llm = LongResponseLLM(original)
    engine._llm = llm
    ingestion = IngestionService(engine).ingest(
        session_id, UploadedFile(name="sanitized.md", data=sample_markdown)
    )
    assert ingestion.status.value == "ready"

    message = FixedRetrievalChatService(engine).answer(
        ChatRequest(
            question="List all requirements, tests, reports, and errors",
            session_id=session_id,
            generation_id="generation-exact-equality",
        )
    )

    assert llm.requested_output_tokens == 4096
    assert message.content == original
    assert message.debug["provider_raw_chars"] == len(original)
    assert message.debug["parsed_chars"] == len(original)
    assert message.debug["grounded_result_chars"] == len(original)
    assert message.debug["returned_chars"] == len(original)

    clipboard = copy_component_html(message.content, message.message_id)
    assert f"const text = {json.dumps(original, ensure_ascii=False)};" in clipboard

    monkeypatch.setenv("OMNIRAG_DEBUG_GENERATION", "true")
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state[SESSION_KEY] = session_id
    app.session_state[MESSAGES_KEY] = [
        ChatMessage(role=Role.USER, content="سؤال شامل"),
        message,
    ]
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state[MESSAGES_KEY][1].content == original
    assert any(markdown.value == original for markdown in app.markdown)
    assert any(
        "FINAL_SENTENCE: اكتملت الإجابة العربية دون اقتطاع [16]."
        in markdown.value
        for markdown in app.markdown
    )
    assert [button.label for button in app.button[:3]] == [
        "Edit",
        "Regenerate",
        "Regenerate",
    ]
    assert any(expander.label == "Generation diagnostics" for expander in app.expander)

    app.run(timeout=30)
    assert not app.exception
    assert app.session_state[MESSAGES_KEY][1].content == original
    assert any(markdown.value == original for markdown in app.markdown)
