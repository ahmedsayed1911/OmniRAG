"""Global/exhaustive retrieval remains complete but bounded."""

from __future__ import annotations

from omnirag.config.settings import build_settings
from omnirag.core.enums import BlockType, FileType, QueryScope, SourceKind
from omnirag.core.models import Chunk
from omnirag.providers.rerank.heuristic import HeuristicReranker
from omnirag.rag.query_rewrite import classify_query_scope, parse_query
from omnirag.rag.retrieval import RetrievalRequest, Retriever
from omnirag.rag.vector_store import InMemoryVectorStore


def _chunk(session_id: str, page: int, text: str, *, kind=BlockType.TEXT) -> Chunk:
    return Chunk(
        document_id="qa-report",
        session_id=session_id,
        filename="qa.pdf",
        file_type=FileType.PDF,
        page_number=page,
        page_label=f"Page {page}",
        block_ids=[f"block-{page}-{kind.value}-{len(text)}"],
        block_type=kind,
        source_kind=SourceKind.STRUCTURED if kind == BlockType.TABLE else SourceKind.DIGITAL,
        text=text,
    )


def _retriever(session_id, fake_embeddings, chunks, **settings_overrides):
    store = InMemoryVectorStore()
    store.upsert(
        session_id, chunks, fake_embeddings.embed_documents([chunk.text for chunk in chunks])
    )
    settings = build_settings()
    for key, value in settings_overrides.items():
        object.__setattr__(settings.retrieval, key, value)
    return Retriever(
        vector_store=store,
        embeddings=fake_embeddings,
        reranker=HeuristicReranker(top_n=80),
        llm=None,
        settings=settings,
    )


def test_scope_classifier_covers_english_arabic_and_mixed_queries():
    assert classify_query_scope("What happened in TC03?") == QueryScope.FOCUSED
    assert classify_query_scope("Summarize this document") == QueryScope.GLOBAL
    assert classify_query_scope("اذكر كل الحالات الفاشلة") == QueryScope.EXHAUSTIVE
    assert (
        classify_query_scope("لخص الملف كله and list all failed test cases")
        == QueryScope.MULTI_PART
    )


def test_arabic_exhaustive_decomposition_adds_english_structured_terms():
    plan = parse_query("اذكر كل الحالات الفاشلة في الملف")
    assert plan.search_queries[0] == "اذكر كل الحالات الفاشلة في الملف"
    joined = " ".join(plan.expansions).lower()
    assert "failed test cases" in joined
    assert "actual result" in joined
    assert len(plan.expansions) <= 6


def test_small_document_scan_recovers_failures_across_all_pages(
    session_id, fake_embeddings
):
    chunks = [_chunk(session_id, page, f"General section on page {page}.") for page in range(1, 12)]
    failures = {
        2: "| Test Case | Actual Result | Status |\n| TC02 | Login rejected | Fail |",
        6: "| Test Case | Actual Result | Status |\n| TC06 | Timeout | Failed |",
        10: "| Test Case | Actual Result | Status |\n| TC10 | Wrong total | Fail |",
    }
    chunks.extend(_chunk(session_id, page, text, kind=BlockType.TABLE) for page, text in failures.items())
    chunks.extend(
        [
            _chunk(session_id, 9, "Bug Report BUG-002 Severity High linked to TC02."),
            _chunk(session_id, 11, "Bug Report BUG-010 Severity High linked to TC10."),
        ]
    )
    # Duplicate representation of the same structured evidence must not occupy
    # a second context position.
    chunks.append(_chunk(session_id, 2, failures[2], kind=BlockType.TABLE))
    retriever = _retriever(
        session_id,
        fake_embeddings,
        chunks,
        exhaustive_scan_max_chunks=120,
        exhaustive_final_k=24,
        max_context_chars=50000,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="List all failed test cases and every bug report",
            session_id=session_id,
        )
    )

    text = "\n".join(item.chunk.text for item in result.results)
    assert result.query_scope == QueryScope.EXHAUSTIVE.value
    assert result.structured_matches == 3
    assert all(case in text for case in ("TC02", "TC06", "TC10"))
    assert all(bug in text for bug in ("BUG-002", "BUG-010"))
    assert {2, 6, 9, 10, 11} <= {item.chunk.page_number for item in result.results}
    assert sum(item.chunk.text == failures[2] for item in result.results) == 1


def test_large_document_exhaustive_scan_is_bounded(session_id, fake_embeddings):
    chunks = [
        _chunk(session_id, page + 1, f"Section {page + 1} ordinary content")
        for page in range(150)
    ]
    retriever = _retriever(
        session_id,
        fake_embeddings,
        chunks,
        exhaustive_scan_max_chunks=20,
        exhaustive_final_k=18,
    )

    result = retriever.retrieve(
        RetrievalRequest(query="Summarize the entire document", session_id=session_id)
    )

    assert result.query_scope == QueryScope.GLOBAL.value
    assert len(result.results) <= 18
    assert result.candidate_count < len(chunks)


def test_high_severity_structured_match_excludes_medium_bugs(
    session_id, fake_embeddings
):
    chunks = [
        _chunk(
            session_id,
            7,
            "| Bug ID | BUG-001 |\n| Severity | Medium |",
            kind=BlockType.TABLE,
        ),
        _chunk(
            session_id,
            9,
            "| Bug ID | BUG-003 |\n| Severity | High |",
            kind=BlockType.TABLE,
        ),
    ]
    result = _retriever(session_id, fake_embeddings, chunks).retrieve(
        RetrievalRequest(
            query="What are all high-severity bugs in this document?",
            session_id=session_id,
        )
    )

    assert result.structured_matches == 1
    assert any("BUG-003" in item.chunk.text for item in result.results)
