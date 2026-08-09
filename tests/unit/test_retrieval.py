"""Retrieval: hybrid fusion, filtering, reranking fallback, query parsing."""

from __future__ import annotations

import pytest

from omnirag.config.settings import build_settings
from omnirag.core.enums import BlockType, FileType, Language
from omnirag.core.models import Chunk, SearchResult
from omnirag.providers.rerank.base import RerankCandidate
from omnirag.providers.rerank.heuristic import HeuristicReranker
from omnirag.rag.hybrid import build_bm25_index, reciprocal_rank_fusion
from omnirag.rag.query_rewrite import expand_with_llm, parse_query
from omnirag.rag.retrieval import RetrievalRequest, Retriever
from omnirag.rag.vector_store import InMemoryVectorStore


def make_chunk(session_id, text, *, page=1, document_id="d1", filename="report.pdf", **kwargs):
    return Chunk(
        document_id=document_id,
        session_id=session_id,
        filename=filename,
        file_type=FileType.PDF,
        page_number=page,
        page_label=f"Page {page}",
        block_ids=[f"b-{document_id}-{page}"],
        text=text,
        **kwargs,
    )


@pytest.fixture
def indexed(session_id, fake_embeddings):
    """A small session-scoped index for retrieval tests."""
    store = InMemoryVectorStore()
    chunks = [
        make_chunk(session_id, "Total revenue reached 8,400,000 USD in Q4 2024.", page=1),
        make_chunk(session_id, "Currency volatility remains the principal risk for 2025.", page=2),
        make_chunk(session_id, "The EMEA region grew 62 percent year over year.", page=3),
        make_chunk(session_id, "Employee headcount increased to 1,240 people.", page=4),
        make_chunk(
            session_id,
            "Bar chart showing quarterly revenue with Q4 at 8400 and Q3 at 6200.",
            page=5,
            block_type=BlockType.CHART,
        ),
    ]
    vectors = fake_embeddings.embed_documents([c.text for c in chunks])
    store.upsert(session_id, chunks, vectors)
    return store, chunks


def build_retriever(store, embeddings, **overrides):
    settings = build_settings()
    for key, value in overrides.items():
        object.__setattr__(settings.retrieval, key, value)
    return Retriever(
        vector_store=store,
        embeddings=embeddings,
        reranker=HeuristicReranker(top_n=settings.retrieval.rerank_top_k),
        llm=None,
        settings=settings,
    )


class TestBM25:
    def test_keyword_search_finds_exact_terms(self, session_id):
        chunks = [
            make_chunk(session_id, "Invoice number INV-2024-8891 was paid.", page=1),
            make_chunk(session_id, "General information about billing processes.", page=2),
        ]
        index = build_bm25_index(session_id, chunks)

        results = index.search("INV-2024-8891", top_k=5)
        assert results
        assert results[0][0] == chunks[0].chunk_id

    def test_arabic_keyword_search_matches_normalised_forms(self, session_id):
        chunks = [
            make_chunk(session_id, "بلغت الإيرادات الإجمالية ثمانية ملايين دولار", page=1),
            make_chunk(session_id, "عدد الموظفين ألف ومائتان", page=2),
        ]
        index = build_bm25_index(session_id, chunks)

        # Query written without the hamza still matches.
        results = index.search("الايرادات", top_k=5)
        assert results
        assert results[0][0] == chunks[0].chunk_id

    def test_empty_index_returns_nothing(self, session_id):
        assert build_bm25_index(session_id, []).search("anything") == []


class TestFusion:
    def test_rrf_rewards_agreement_between_rankers(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "b"]], k=60)
        assert fused[0][0] == "a"

    def test_rrf_handles_disjoint_rankings(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]], k=60)
        assert {item for item, _ in fused} == {"a", "b"}

    def test_weights_shift_the_balance(self):
        heavy_first = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[5.0, 1.0])
        assert heavy_first[0][0] == "a"


class TestRetrieverPipeline:
    def test_relevant_chunk_is_retrieved(self, indexed, session_id, fake_embeddings):
        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)

        result = retriever.retrieve(
            RetrievalRequest(query="What was total revenue?", session_id=session_id)
        )

        assert result.results
        assert "8,400,000" in result.results[0].chunk.text

    def test_results_never_cross_sessions(self, indexed, session_id, fake_embeddings):
        from omnirag.storage.sessions import new_session_id

        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)

        result = retriever.retrieve(
            RetrievalRequest(query="revenue", session_id=new_session_id())
        )
        assert result.results == []

    def test_document_filter_is_applied(self, session_id, fake_embeddings):
        store = InMemoryVectorStore()
        chunks = [
            make_chunk(session_id, "Alpha report revenue was 100.", document_id="doc-a", filename="a.pdf"),
            make_chunk(session_id, "Beta report revenue was 200.", document_id="doc-b", filename="b.pdf"),
        ]
        store.upsert(session_id, chunks, fake_embeddings.embed_documents([c.text for c in chunks]))

        retriever = build_retriever(store, fake_embeddings)
        result = retriever.retrieve(
            RetrievalRequest(query="revenue", session_id=session_id, document_ids=["doc-a"])
        )

        assert result.results
        assert {r.chunk.document_id for r in result.results} == {"doc-a"}

    def test_page_intent_narrows_retrieval(self, indexed, session_id, fake_embeddings):
        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)

        result = retriever.retrieve(
            RetrievalRequest(query="Explain page 3", session_id=session_id)
        )

        assert result.results
        assert {r.chunk.page_number for r in result.results} == {3}

    def test_page_specific_query_skips_llm_rewrite(self, indexed, session_id, fake_embeddings):
        class RewriteSpy:
            calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("page-specific query must not call query rewrite")

        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings, query_rewrite=True)
        retriever.llm = RewriteSpy()
        result = retriever.retrieve(
            RetrievalRequest(query="Explain the diagram on Page 3", session_id=session_id)
        )
        assert retriever.llm.calls == 0
        assert any("Skipped LLM query rewrite" in note for note in result.notes)

    def test_production_arabic_page_query_is_strictly_page_three(
        self, session_id, fake_embeddings
    ):
        class RewriteSpy:
            calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("deterministic page query must not be rewritten")

        class ModelReranker:
            is_model_based = True
            calls = 0

            def rerank(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("deterministic page query must not use LLM reranking")

        store = InMemoryVectorStore()
        chunks = [
            make_chunk(
                session_id,
                f"Image on page {page}.",
                page=page,
                block_type=BlockType.PAGE_SNAPSHOT,
            )
            for page in range(1, 9)
        ]
        store.upsert(
            session_id,
            chunks,
            fake_embeddings.embed_documents([chunk.text for chunk in chunks]),
        )
        retriever = build_retriever(store, fake_embeddings, query_rewrite=True)
        retriever.llm = RewriteSpy()
        retriever.reranker = ModelReranker()

        result = retriever.retrieve(
            RetrievalRequest(
                query="اشرحلي الرسومات الموجودة ف بيدج 3",
                session_id=session_id,
            )
        )

        assert retriever.llm.calls == 0
        assert retriever.reranker.calls == 0
        assert result.unique_pages == 1
        assert [item.chunk.page_number for item in result.results] == [3]
        assert result.candidate_count == 1

    def test_page_filtered_candidates_avoid_model_reranker(self, session_id, fake_embeddings):
        class ModelReranker:
            is_model_based = True
            calls = 0

            def rerank(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("model reranker must not be used")

        store = InMemoryVectorStore()
        chunks = [
            make_chunk(session_id, "Page 3 full diagram", page=3),
            make_chunk(session_id, "Salary Software System crop", page=3),
            make_chunk(session_id, "Unrelated page", page=4),
        ]
        store.upsert(session_id, chunks, fake_embeddings.embed_documents([c.text for c in chunks]))
        retriever = build_retriever(store, fake_embeddings)
        retriever.reranker = ModelReranker()
        result = retriever.retrieve(
            RetrievalRequest(query="Explain Page 3", session_id=session_id)
        )
        assert retriever.reranker.calls == 0
        assert {item.chunk.page_number for item in result.results} == {3}

    def test_missing_page_reports_a_note_instead_of_wrong_content(
        self, indexed, session_id, fake_embeddings
    ):
        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)

        result = retriever.retrieve(
            RetrievalRequest(query="Explain page 97", session_id=session_id)
        )

        assert result.results == []
        assert any("97" in note for note in result.notes)

    def test_result_limit_is_respected(self, indexed, session_id, fake_embeddings):
        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)

        result = retriever.retrieve(
            RetrievalRequest(query="revenue risk region", session_id=session_id, final_k=2)
        )
        assert len(result.results) <= 2

    def test_empty_index_returns_empty_result(self, session_id, fake_embeddings):
        retriever = build_retriever(InMemoryVectorStore(), fake_embeddings)
        result = retriever.retrieve(
            RetrievalRequest(query="anything", session_id=session_id)
        )
        assert result.is_empty

    def test_a_broken_reranker_does_not_lose_results(self, indexed, session_id, fake_embeddings):
        class BrokenReranker:
            name = "broken"

            def rerank(self, query, candidates, *, top_n=None):
                raise RuntimeError("rerank API is down")

        store, _ = indexed
        retriever = build_retriever(store, fake_embeddings)
        retriever.reranker = BrokenReranker()

        result = retriever.retrieve(
            RetrievalRequest(query="What was total revenue?", session_id=session_id)
        )

        assert result.results, "retrieval must degrade gracefully, not fail"
        assert result.reranked is False


class TestHeuristicReranker:
    def test_lexical_overlap_ranks_the_right_passage_first(self):
        reranker = HeuristicReranker(top_n=3)
        candidates = [
            RerankCandidate(ref="a", text="Employee headcount grew to 1,240 people."),
            RerankCandidate(ref="b", text="Total revenue reached 8,400,000 USD in Q4."),
            RerankCandidate(ref="c", text="The office relocated to a new building."),
        ]

        scores = reranker.rerank("What was total revenue in Q4?", candidates)
        assert scores[0].ref == "b"

    def test_no_matching_terms_preserves_input_order(self):
        reranker = HeuristicReranker(top_n=2)
        candidates = [RerankCandidate(ref="a", text="xxx"), RerankCandidate(ref="b", text="yyy")]

        scores = reranker.rerank("!!!", candidates)
        assert [s.ref for s in scores] == ["a", "b"]

    def test_empty_candidates(self):
        assert HeuristicReranker().rerank("q", []) == []


class TestQueryParsing:
    @pytest.mark.parametrize("query,pages", [
        ("Explain page 17", [17]),
        ("Explain page 03", [3]),
        ("What is on p. 3?", [3]),
        ("Summarise pages 4-6", [4, 5, 6]),
        ("Compare pages 3 and 4", [3, 4]),
        ("What does slide 12 show?", [12]),
        ("اشرح الصفحة 23", [23]),
        ("اشرح الصفحة ٢٣", [23]),
        ("اشرح الصفحة الثالثة", [3]),
        ("قارن صفحات 3 و4", [3, 4]),
        ("اشرحلي الرسومات الموجودة ف بيدج 3", [3]),
        ("اشرحلي الرسومات الموجودة ف بيج 3", [3]),
        ("قارن بيدج 3 و4", [3, 4]),
        ("قارن بيج 3 و 4", [3, 4]),
        ("Tell me about revenue", []),
    ])
    def test_page_intent_extraction(self, query, pages):
        assert parse_query(query).page_filter == pages

    @pytest.mark.parametrize("query", [
        "What does the chart on page 23 show?",
        "Explain this diagram",
        "Describe the table",
        "ما الذي يوضحه الرسم البياني؟",
    ])
    def test_visual_intent_detection(self, query):
        assert parse_query(query).wants_visual is True

    @pytest.mark.parametrize("query", [
        "Compare document A with document B",
        "Where do these reports disagree?",
        "قارن بين التقريرين",
    ])
    def test_comparison_intent_detection(self, query):
        assert parse_query(query).is_comparison is True

    @pytest.mark.parametrize("query,expected", [
        ("Answer in Arabic: what is the revenue?", Language.ARABIC),
        ("Summarize in English please", Language.ENGLISH),
        ("أجب بالعربية عن سؤالي", Language.ARABIC),
        ("What is the revenue?", None),
    ])
    def test_answer_language_intent(self, query, expected):
        assert parse_query(query).answer_language == expected

    def test_query_language_detection(self):
        assert parse_query("ما هي الإيرادات؟").language == Language.ARABIC
        assert parse_query("What is the revenue?").language == Language.ENGLISH

    def test_followup_is_expanded_with_the_previous_question(self):
        from omnirag.core.enums import Role
        from omnirag.core.models import ChatMessage

        history = [
            ChatMessage(role=Role.USER, content="What was revenue in 2024?"),
            ChatMessage(role=Role.ASSISTANT, content="Revenue was 8.4M [1]."),
        ]
        plan = parse_query("And what about 2023?", history)

        assert plan.is_followup is True
        assert any("2024" in q for q in plan.search_queries)

    def test_the_original_query_is_always_searched(self):
        plan = parse_query("What is the revenue?")
        assert plan.search_queries[0] == "What is the revenue?"

    def test_intent_is_never_rewritten_away(self):
        original = "What was the exact revenue figure for Q4 2024?"
        plan = parse_query(original)
        assert plan.normalized == original

    def test_plain_text_query_rewrite_avoids_structured_output(self):
        class PlainRewriteLLM:
            json_modes = []

            def complete(self, *args, **kwargs):
                self.json_modes.append(kwargs.get("json_mode"))
                from omnirag.providers.llm.base import LLMResponse

                return LLMResponse(
                    text="quarterly revenue chart\nمخطط الإيرادات ربع السنوية"
                )

        llm = PlainRewriteLLM()
        plan = expand_with_llm(parse_query("Explain the revenue chart"), llm)

        assert llm.json_modes == [False]
        assert "quarterly revenue chart" in plan.expansions
        assert "مخطط الإيرادات ربع السنوية" in plan.expansions

    def test_query_rewrite_json_validation_failure_degrades_safely(self):
        from omnirag.core.exceptions import ProviderBadRequestError

        class BrokenRewriteLLM:
            def complete(self, *args, **kwargs):
                raise ProviderBadRequestError(
                    "json_validate_failed",
                    provider="groq",
                )

        plan = expand_with_llm(
            parse_query("Explain revenue trends"), BrokenRewriteLLM()
        )

        assert plan.search_queries == ["Explain revenue trends"]
        assert any("unavailable" in note.lower() for note in plan.notes)
