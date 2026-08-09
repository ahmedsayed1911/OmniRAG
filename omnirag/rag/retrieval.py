"""The retrieval pipeline.

    query
      -> normalise + parse intent (pages, visuals, comparison, language)
      -> optional LLM expansion (adds a cross-lingual variant)
      -> dense vector search   (per query variant, session-filtered)
      -> BM25 keyword search   (session-filtered)
      -> reciprocal rank fusion
      -> rerank (Cohere / Jina / LLM / heuristic)
      -> diversity + context-budget selection
      -> RetrievalResult

Every stage is optional and independently swappable; the orchestration lives
here so the UI and the future FastAPI layer share exactly one code path.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from omnirag.config.settings import AppSettings, get_settings
from omnirag.core.enums import BlockType, QueryScope
from omnirag.core.exceptions import RetrievalError, VectorStoreError
from omnirag.core.models import Chunk, RetrievalResult, SearchResult
from omnirag.providers.embeddings.base import BaseEmbeddingProvider
from omnirag.providers.rerank.base import BaseReranker, RerankCandidate
from omnirag.providers.rerank.heuristic import HeuristicReranker
from omnirag.rag.hybrid import get_bm25_cache, reciprocal_rank_fusion
from omnirag.rag.query_rewrite import QueryPlan, expand_with_llm, parse_query
from omnirag.rag.vector_store import BaseVectorStore, SearchFilter
from omnirag.storage.sessions import require_session_id
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

#: Weight given to dense vs keyword rankings during fusion. Dense is favoured
#: because it is the only path that works cross-lingually.
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.7
#: At most this many chunks from any single page, so one dense page cannot
#: monopolise the context window.
MAX_PER_PAGE = 3


@dataclass
class RetrievalRequest:
    """Everything needed for one retrieval, explicit rather than implicit."""

    query: str
    session_id: str
    document_ids: Optional[Sequence[str]] = None
    history: Optional[Sequence] = None
    top_k: Optional[int] = None
    final_k: Optional[int] = None
    use_rerank: bool = True
    use_expansion: bool = True


class Retriever:
    """Hybrid retriever. All dependencies injected — no globals, no Streamlit."""

    def __init__(
        self,
        *,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddingProvider,
        reranker: Optional[BaseReranker] = None,
        llm=None,
        settings: Optional[AppSettings] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.reranker = reranker
        self.llm = llm
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        session_id = require_session_id(request.session_id)
        cfg = self.settings.retrieval
        top_k = request.top_k or cfg.top_k
        final_k = request.final_k or cfg.rerank_top_k
        timings: Dict[str, float] = {}

        # -- 1. plan ---------------------------------------------------- #
        started = time.perf_counter()
        plan = parse_query(request.query, request.history)
        if (
            request.use_expansion
            and cfg.query_rewrite
            and self.llm is not None
            and not plan.page_filter
        ):
            plan = expand_with_llm(
                plan,
                self.llm,
                max_expansions=cfg.max_expansions,
                max_output_tokens=cfg.query_rewrite_max_output_tokens,
            )
        elif plan.page_filter:
            plan.notes.append("Skipped LLM query rewrite for an explicit page constraint.")
        timings["plan_ms"] = (time.perf_counter() - started) * 1000

        broad = plan.scope != QueryScope.FOCUSED
        corpus: List[Chunk] = []
        if broad:
            try:
                corpus = self.vector_store.list_chunks(
                    session_id, document_ids=request.document_ids
                )
            except Exception as exc:
                logger.warning("Broad corpus scan unavailable: %s", exc)
            if plan.page_filter:
                allowed = set(plan.page_filter)
                corpus = [chunk for chunk in corpus if chunk.page_number in allowed]
            if plan.scope in (QueryScope.EXHAUSTIVE, QueryScope.MULTI_PART):
                final_k = max(final_k, cfg.exhaustive_final_k)
            else:
                final_k = max(final_k, min(16, cfg.exhaustive_final_k))
            if corpus and len(corpus) <= cfg.exhaustive_scan_max_chunks:
                top_k = max(top_k, len(corpus))

        result = RetrievalResult(
            query=request.query,
            normalized_query=plan.normalized,
            expanded_queries=list(plan.expansions),
            language=plan.language,
            strategy=cfg.strategy,
            notes=list(plan.notes),
            query_scope=plan.scope.value,
            total_pages=len({(c.document_id, c.page_number) for c in corpus}),
        )

        filters = self._build_filters(plan, request.document_ids)

        # -- 2. dense --------------------------------------------------- #
        started = time.perf_counter()
        try:
            vector_hits = self._vector_search(session_id, plan, filters, top_k)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Vector search failed: {exc}",
                user_message="Search over your documents failed. Please try again.",
            ) from exc
        timings["vector_ms"] = (time.perf_counter() - started) * 1000

        # -- 3. keyword ------------------------------------------------- #
        keyword_hits: Dict[str, float] = {}
        keyword_order: List[str] = []
        chunk_pool: Dict[str, Chunk] = {c.chunk_id: c for c, _ in vector_hits}

        if cfg.use_keyword and cfg.strategy in ("hybrid", "keyword"):
            started = time.perf_counter()
            keyword_order, keyword_hits, keyword_chunks = self._keyword_search(
                session_id, plan, request.document_ids, top_k, filters
            )
            chunk_pool.update(keyword_chunks)
            timings["keyword_ms"] = (time.perf_counter() - started) * 1000

        structured = _structured_matches(corpus, plan)
        if plan.scope in (QueryScope.EXHAUSTIVE, QueryScope.MULTI_PART):
            final_k = max(final_k, len(structured))
        structured_order = [chunk.chunk_id for chunk in structured]
        chunk_pool.update({chunk.chunk_id: chunk for chunk in structured})

        coverage_order: List[str] = []
        if corpus and len(corpus) <= cfg.exhaustive_scan_max_chunks:
            # Small selected collections can be inspected completely. This is
            # a low-weight recall ranking; relevance rankings still dominate.
            coverage_order = [chunk.chunk_id for chunk in corpus]
            chunk_pool.update({chunk.chunk_id: chunk for chunk in corpus})

        if not chunk_pool:
            result.timings_ms = timings
            if plan.page_filter:
                result.notes.append(
                    f"No content was found on the requested page(s): "
                    f"{', '.join(str(p) for p in plan.page_filter)}."
                )
            return result

        # -- 4. fusion -------------------------------------------------- #
        vector_order = [c.chunk_id for c, _ in vector_hits]
        vector_scores = {c.chunk_id: score for c, score in vector_hits}

        if cfg.strategy == "vector" or not keyword_order:
            rankings = [vector_order]
            weights = [VECTOR_WEIGHT]
            strategy = "vector"
        elif cfg.strategy == "keyword":
            rankings = [keyword_order]
            weights = [KEYWORD_WEIGHT]
            strategy = "keyword"
        else:
            rankings = [vector_order, keyword_order]
            weights = [VECTOR_WEIGHT, KEYWORD_WEIGHT]
            strategy = "hybrid"
        if structured_order:
            rankings.append(structured_order)
            weights.append(1.8)
        if coverage_order:
            rankings.append(coverage_order)
            weights.append(0.2)
        if len(rankings) > 1:
            fused = reciprocal_rank_fusion(
                rankings,
                k=cfg.rrf_k,
                weights=weights,
            )
        else:
            only = rankings[0]
            score_map = vector_scores if strategy == "vector" else keyword_hits
            fused = [(cid, score_map.get(cid, 0.0)) for cid in only]
        result.strategy = strategy

        candidates = [
            SearchResult(
                chunk=chunk_pool[cid],
                score=score,
                vector_score=vector_scores.get(cid),
                keyword_score=keyword_hits.get(cid),
                strategy=strategy,
            )
            for cid, score in fused
            if cid in chunk_pool
        ][: max(top_k, final_k * 3)]

        # -- 5. rerank -------------------------------------------------- #
        if request.use_rerank and self.reranker is not None and len(candidates) > 1:
            started = time.perf_counter()
            active_reranker = self.reranker
            if getattr(active_reranker, "is_model_based", False) and (
                plan.page_filter or len(candidates) <= max(12, final_k)
            ):
                active_reranker = HeuristicReranker(top_n=final_k * 3)
                result.notes.append(
                    "Used deterministic reranking for the bounded candidate set."
                )
            candidates, reranked = self._rerank(
                plan, candidates, final_k * 3, active_reranker
            )
            result.reranked = reranked
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000

        # -- 6. selection ----------------------------------------------- #
        selected = self._select(candidates, plan, final_k, cfg.max_context_chars)
        completeness_pass = False
        if plan.scope in (QueryScope.EXHAUSTIVE, QueryScope.MULTI_PART) and structured:
            missing = [
                chunk for chunk in structured
                if chunk.chunk_id not in {item.chunk.chunk_id for item in selected}
            ]
            if missing:
                completeness_pass = True
                by_id = {item.chunk.chunk_id: item for item in candidates}
                selected = self._recover_structured(
                    selected,
                    [by_id[c.chunk_id] for c in structured if c.chunk_id in by_id],
                    final_k,
                    cfg.max_context_chars,
                )
        for rank, item in enumerate(selected):
            item.rank = rank
        result.results = selected
        result.candidate_count = len(candidates)
        result.unique_pages = len(
            {(item.chunk.document_id, item.chunk.page_number) for item in selected}
        )
        result.structured_matches = sum(
            chunk.block_type == BlockType.TABLE for chunk in structured
        )
        result.completeness_pass = completeness_pass
        result.timings_ms = timings

        logger.info(
            "query_scope=%s subqueries=%d candidate_count=%d unique_pages=%d "
            "structured_matches=%d final_context_chunks=%d completeness_pass=%s "
            "retrieval_ms=%.0f strategy=%s reranked=%s",
            plan.scope.value,
            len(plan.search_queries),
            len(candidates),
            result.unique_pages,
            result.structured_matches,
            len(selected),
            completeness_pass,
            sum(timings.values()),
            strategy,
            result.reranked,
        )
        return result

    # ------------------------------------------------------------------ #
    def _build_filters(
        self, plan: QueryPlan, document_ids: Optional[Sequence[str]]
    ) -> SearchFilter:
        return SearchFilter(
            document_ids=list(document_ids) if document_ids else None,
            page_numbers=plan.page_filter or None,
        )

    def _vector_search(
        self, session_id: str, plan: QueryPlan, filters: SearchFilter, top_k: int
    ) -> List[tuple[Chunk, float]]:
        queries = plan.search_queries
        if not queries:
            return []

        try:
            vectors = self.embeddings.embed_documents(queries) if len(queries) > 1 else [
                self.embeddings.embed_query(queries[0])
            ]
        except Exception as exc:
            raise RetrievalError(
                f"Could not embed the query: {exc}",
                user_message=(
                    "Your question could not be processed by the embedding "
                    "provider. Check the embedding configuration."
                ),
            ) from exc

        best: Dict[str, tuple[Chunk, float]] = {}
        per_query = max(5, top_k // max(1, len(queries)) + 5)
        for vector in vectors:
            if not vector:
                continue
            for hit in self.vector_store.search(
                session_id, vector, top_k=per_query, filters=filters
            ):
                current = best.get(hit.chunk.chunk_id)
                if current is None or hit.score > current[1]:
                    best[hit.chunk.chunk_id] = (hit.chunk, hit.score)

        ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _keyword_search(
        self,
        session_id: str,
        plan: QueryPlan,
        document_ids: Optional[Sequence[str]],
        top_k: int,
        filters: SearchFilter,
    ) -> tuple[List[str], Dict[str, float], Dict[str, Chunk]]:
        try:
            chunks = self.vector_store.list_chunks(session_id, document_ids=document_ids)
        except Exception as exc:
            logger.warning("Keyword search unavailable: %s", exc)
            return [], {}, {}
        if not chunks:
            return [], {}, {}

        if plan.page_filter:
            allowed = set(plan.page_filter)
            chunks = [c for c in chunks if c.page_number in allowed]
            if not chunks:
                return [], {}, {}

        index = get_bm25_cache().get(session_id, chunks)
        scores: Dict[str, float] = {}
        for query in plan.search_queries:
            for chunk_id, score in index.search(query, top_k=top_k):
                scores[chunk_id] = max(scores.get(chunk_id, 0.0), score)

        order = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        pool = {cid: index.chunks[cid] for cid in order if cid in index.chunks}
        return order, scores, pool

    def _rerank(
        self,
        plan: QueryPlan,
        candidates: List[SearchResult],
        limit: int,
        reranker: Optional[BaseReranker] = None,
    ) -> tuple[List[SearchResult], bool]:
        reranker = reranker or self.reranker
        if reranker is None:
            return candidates, False
        window = candidates[:limit]
        payload = [
            RerankCandidate(ref=item.chunk.chunk_id, text=item.chunk.text) for item in window
        ]
        try:
            scores = reranker.rerank(plan.normalized, payload, top_n=len(payload))
        except Exception as exc:
            logger.warning("Reranking failed (%s) — keeping fusion order", exc)
            return candidates, False

        if not scores:
            return candidates, False

        by_id = {item.chunk.chunk_id: item for item in window}
        ordered: List[SearchResult] = []
        for entry in scores:
            item = by_id.pop(entry.ref, None)
            if item is None:
                continue
            item.rerank_score = entry.score
            item.score = entry.score
            ordered.append(item)

        # Anything the reranker dropped keeps its fusion order behind the rest.
        ordered.extend(by_id.values())
        ordered.extend(candidates[limit:])
        return ordered, True

    def _select(
        self,
        candidates: Sequence[SearchResult],
        plan: QueryPlan,
        final_k: int,
        max_chars: int,
    ) -> List[SearchResult]:
        """Pick the final context set: relevance, then diversity, then budget."""
        selected: List[SearchResult] = []
        per_page: Dict[tuple[str, int], int] = {}
        per_document: Dict[str, int] = {}
        used_chars = 0

        # A comparison question must see every document, so reserve slots.
        document_cap = final_k
        if plan.is_comparison:
            distinct = len({c.chunk.document_id for c in candidates})
            if distinct > 1:
                document_cap = max(2, final_k // distinct + 1)

        prioritized = _deduplicate_candidates(candidates)
        if plan.wants_visual:
            # Explicit visual question: float visual blocks to the front.
            prioritized.sort(key=lambda r: 0 if r.chunk.block_type.is_visual else 1)

        # Broad queries first take the best item from each page, then fill the
        # remaining slots by score. This prevents adjacent duplicate pages from
        # monopolising a supposedly document-wide answer.
        if plan.scope != QueryScope.FOCUSED:
            first_by_page: Dict[tuple[str, int], SearchResult] = {}
            for item in prioritized:
                key = (item.chunk.document_id, item.chunk.page_number)
                first_by_page.setdefault(key, item)
            representatives = list(first_by_page.values())
            representative_ids = {item.chunk.chunk_id for item in representatives}
            prioritized = representatives + [
                item for item in prioritized if item.chunk.chunk_id not in representative_ids
            ]

        for item in prioritized:
            if len(selected) >= final_k:
                break
            chunk = item.chunk
            page_key = (chunk.document_id, chunk.page_number)
            if per_page.get(page_key, 0) >= MAX_PER_PAGE:
                continue
            if per_document.get(chunk.document_id, 0) >= document_cap:
                continue
            if used_chars + len(chunk.text) > max_chars and selected:
                continue

            selected.append(item)
            per_page[page_key] = per_page.get(page_key, 0) + 1
            per_document[chunk.document_id] = per_document.get(chunk.document_id, 0) + 1
            used_chars += len(chunk.text)

        return selected

    def _recover_structured(
        self,
        selected: List[SearchResult],
        missing: Sequence[SearchResult],
        final_k: int,
        max_chars: int,
    ) -> List[SearchResult]:
        """One bounded recovery pass for structured exhaustive matches."""
        required_ids = {item.chunk.chunk_id for item in missing}
        recovered = list(missing[:final_k])
        used_chars = sum(len(item.chunk.text) for item in recovered)
        covered_pages = {
            (item.chunk.document_id, item.chunk.page_number) for item in recovered
        }
        # Required rows may consume the nominal character budget. Preserve one
        # representative from every otherwise-missing page first so a combined
        # "summarize + enumerate" request still sees the whole document. The
        # hard context-count limit keeps this bounded.
        for item in selected:
            page = (item.chunk.document_id, item.chunk.page_number)
            if (
                item.chunk.chunk_id in required_ids
                or page in covered_pages
                or len(recovered) >= final_k
            ):
                continue
            recovered.append(item)
            covered_pages.add(page)
            used_chars += len(item.chunk.text)
        recovered_ids = {item.chunk.chunk_id for item in recovered}
        for item in selected:
            if item.chunk.chunk_id in required_ids or len(recovered) >= final_k:
                continue
            if item.chunk.chunk_id in recovered_ids:
                continue
            if used_chars + len(item.chunk.text) > max_chars:
                continue
            recovered.append(item)
            recovered_ids.add(item.chunk.chunk_id)
            used_chars += len(item.chunk.text)
        return recovered


def _structured_matches(chunks: Sequence[Chunk], plan: QueryPlan) -> List[Chunk]:
    """Return exact structured rows/blocks that satisfy explicit query terms."""
    failed = _FAILED_QUERY.search(plan.normalized)
    bugs = _BUG_QUERY.search(plan.normalized)
    severity = _HIGH_SEVERITY_QUERY.search(plan.normalized)
    matches: List[Chunk] = []
    for chunk in chunks:
        text = chunk.text
        if severity:
            if _HIGH_SEVERITY_ROW.search(text) and _BUG_ROW.search(text):
                matches.append(chunk)
        elif failed:
            if chunk.block_type == BlockType.TABLE and _FAILED_ROW.search(text):
                matches.append(chunk)
            elif _BUG_ROW.search(text):
                # Bug reports are retained beside failed rows so the generator
                # can make only evidence-supported test-case links.
                matches.append(chunk)
        elif bugs and _BUG_ROW.search(text):
            matches.append(chunk)
    return _dedupe_chunks(matches)


_FAILED_QUERY = re.compile(r"\b(?:fail|failed|failure)s?\b|(?:فشل|فاشل|فاشلة|الفشل)", re.I)
_BUG_QUERY = re.compile(r"\bbugs?\b|(?:الأخطاء|المشاكل)", re.I)
_HIGH_SEVERITY_QUERY = re.compile(r"\bhigh[- ]severity\b|(?:عالية\s+الخطورة)", re.I)
_FAILED_ROW = re.compile(
    r"(?:\bstatus\b|\|)\s*[:|=-]?\s*(?:fail|failed)\b|\|\s*(?:fail|failed)\s*\|",
    re.I,
)
_BUG_ROW = re.compile(r"\bBUG[-_ ]?\d+\b|\bbug\s+(?:id|report)\b", re.I)
_HIGH_SEVERITY_ROW = re.compile(
    r"\|\s*severity\s*\|\s*high\s*\||\bseverity\s*[:=-]\s*high\b",
    re.I,
)


def _evidence_key(chunk: Chunk) -> str:
    identifiers = sorted(set(_IDENTIFIER.findall(chunk.text.upper())))
    if identifiers:
        return (
            f"{chunk.document_id}:{chunk.page_number}:"
            f"{','.join(identifiers)}"
        )
    text = re.sub(r"\W+", " ", chunk.text.lower()).strip()
    return f"{chunk.document_id}:{chunk.page_number}:{text[:500]}"


_IDENTIFIER = re.compile(r"\b(?:TC|BUG)[-_ ]?\d+\b", re.I)


def _richness(chunk: Chunk) -> tuple[int, int, int]:
    return (
        int(chunk.block_type == BlockType.TABLE),
        int(chunk.visual is not None),
        len(chunk.text),
    )


def _dedupe_chunks(chunks: Sequence[Chunk]) -> List[Chunk]:
    positions: Dict[str, int] = {}
    out: List[Chunk] = []
    for chunk in chunks:
        key = _evidence_key(chunk)
        if key in positions:
            index = positions[key]
            if _richness(chunk) > _richness(out[index]):
                out[index] = chunk
            continue
        positions[key] = len(out)
        out.append(chunk)
    return out


def _deduplicate_candidates(candidates: Sequence[SearchResult]) -> List[SearchResult]:
    positions: Dict[str, int] = {}
    out: List[SearchResult] = []
    for item in candidates:
        key = _evidence_key(item.chunk)
        if key in positions:
            index = positions[key]
            if _richness(item.chunk) > _richness(out[index].chunk):
                out[index] = item
            continue
        positions[key] = len(out)
        out.append(item)
    return out


def build_retriever(settings: Optional[AppSettings] = None) -> Retriever:
    """Wire a retriever from configuration (providers resolved lazily)."""
    from omnirag.providers.embeddings.factory import get_embedding_provider
    from omnirag.providers.rerank.factory import get_reranker
    from omnirag.rag.vector_store import get_vector_store

    resolved = settings or get_settings()
    llm = None
    if resolved.retrieval.query_rewrite and resolved.llm.is_configured:
        try:
            from omnirag.providers.llm.factory import get_llm_provider

            llm = get_llm_provider(resolved)
        except Exception as exc:
            logger.info("Query rewriting disabled: %s", exc)

    return Retriever(
        vector_store=get_vector_store(),
        embeddings=get_embedding_provider(resolved),
        reranker=get_reranker(resolved),
        llm=llm,
        settings=resolved,
    )


__all__ = ["RetrievalRequest", "Retriever", "build_retriever", "BlockType"]
