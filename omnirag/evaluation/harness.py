"""Evaluation harness.

Runs a labelled question set against a live session and reports retrieval and
generation metrics together. Designed for offline use (a script or notebook),
not for the request path.

Example::

    from omnirag.evaluation.harness import EvaluationCase, run_evaluation
    from omnirag.services.engine import get_engine

    cases = [
        EvaluationCase(
            question="What was Q4 revenue?",
            relevant_pages=[("report.pdf", 4)],
        )
    ]
    report = run_evaluation(get_engine(), session_id, cases)
    print(report.as_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from omnirag.core.models import ChatMessage
from omnirag.evaluation.generation_metrics import GenerationMetrics, evaluate_answer
from omnirag.evaluation.retrieval_metrics import RetrievalMetrics, evaluate_retrieval
from omnirag.services.chat_service import ChatRequest, ChatService
from omnirag.services.engine import OmniRAGEngine
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationCase:
    """One labelled question.

    Relevance can be expressed at page level (``relevant_pages``), which is far
    easier to label by hand than chunk ids, or at chunk level when known.
    """

    question: str
    relevant_pages: Sequence[Tuple[str, int]] = field(default_factory=list)
    relevant_chunk_ids: Sequence[str] = field(default_factory=list)
    expected_substrings: Sequence[str] = field(default_factory=list)
    document_ids: Optional[Sequence[str]] = None
    note: str = ""


@dataclass
class CaseResult:
    case: EvaluationCase
    message: ChatMessage
    retrieved_keys: List[str]
    relevant_keys: List[str]
    generation: GenerationMetrics
    substring_hits: int = 0

    @property
    def substring_recall(self) -> float:
        if not self.case.expected_substrings:
            return 1.0
        return self.substring_hits / len(self.case.expected_substrings)


@dataclass
class EvaluationReport:
    retrieval: RetrievalMetrics
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def mean_citation_coverage(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.generation.citation_coverage for c in self.cases) / len(self.cases)

    @property
    def mean_numeric_fidelity(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.generation.numeric_fidelity for c in self.cases) / len(self.cases)

    @property
    def answers_with_invalid_citations(self) -> int:
        return sum(1 for c in self.cases if c.generation.has_invalid_citations)

    @property
    def mean_substring_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.substring_recall for c in self.cases) / len(self.cases)

    def as_dict(self) -> Dict[str, object]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "generation": {
                "cases": len(self.cases),
                "mean_citation_coverage": round(self.mean_citation_coverage, 4),
                "mean_numeric_fidelity": round(self.mean_numeric_fidelity, 4),
                "mean_substring_recall": round(self.mean_substring_recall, 4),
                "answers_with_invalid_citations": self.answers_with_invalid_citations,
            },
        }


def run_evaluation(
    engine: OmniRAGEngine,
    session_id: str,
    cases: Sequence[EvaluationCase],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> EvaluationReport:
    """Run every case and aggregate the metrics."""
    service = ChatService(engine)
    pairs: List[Tuple[List[str], List[str]]] = []
    results: List[CaseResult] = []

    for case in cases:
        message = service.answer(
            ChatRequest(
                question=case.question,
                session_id=session_id,
                document_ids=case.document_ids,
            )
        )

        retrieved_keys = _retrieved_keys(message, use_chunks=bool(case.relevant_chunk_ids))
        relevant_keys = (
            list(case.relevant_chunk_ids)
            if case.relevant_chunk_ids
            else [_page_key(f, p) for f, p in case.relevant_pages]
        )
        pairs.append((retrieved_keys, relevant_keys))

        hits = sum(
            1
            for needle in case.expected_substrings
            if needle.lower() in message.content.lower()
        )

        results.append(
            CaseResult(
                case=case,
                message=message,
                retrieved_keys=retrieved_keys,
                relevant_keys=relevant_keys,
                generation=_answer_metrics(message),
                substring_hits=hits,
            )
        )

    return EvaluationReport(retrieval=evaluate_retrieval(pairs, ks=ks), cases=results)


def _retrieved_keys(message: ChatMessage, *, use_chunks: bool) -> List[str]:
    if message.retrieval is None:
        return []
    if use_chunks:
        return [r.chunk.chunk_id for r in message.retrieval.results]
    return [
        _page_key(r.chunk.filename, r.chunk.page_number) for r in message.retrieval.results
    ]


def _page_key(filename: str, page: int) -> str:
    return f"{filename}#p{page}"


def _answer_metrics(message: ChatMessage) -> GenerationMetrics:
    from omnirag.core.models import AnswerResult

    result = AnswerResult(
        answer=message.content,
        citations=message.citations,
        insufficient_evidence=bool((message.debug or {}).get("insufficient_evidence")),
    )
    context = ""
    if message.retrieval is not None:
        context = "\n".join(r.chunk.text for r in message.retrieval.results)
    return evaluate_answer(result, context=context)


__all__ = [
    "CaseResult",
    "EvaluationCase",
    "EvaluationReport",
    "run_evaluation",
]
