"""Answer-quality metrics.

Deterministic checks first (citation coverage, citation validity, numeric
fidelity) because they are cheap, reproducible and catch the failures that
matter most in a RAG system. Model-graded hooks (groundedness, relevance) are
provided as an interface for teams that want them, and are never invoked
automatically — grading every answer with an LLM would double the cost of the
product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set

from omnirag.core.models import AnswerResult, Citation
from omnirag.rag.citations import parse_markers

_NUMBER = re.compile(r"-?\d[\d,،.]*\d|\d")
_CITATION_MARKER = re.compile(r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]")


def strip_citation_markers(text: str) -> str:
    """Remove ``[1]`` / ``[2, 4]`` markers so they are not read as figures."""
    return _CITATION_MARKER.sub(" ", text or "")


@dataclass
class GenerationMetrics:
    """Per-answer quality signals."""

    citation_coverage: float = 0.0       # cited sources / retrieved sources
    claim_citation_rate: float = 0.0     # sentences with a citation / sentences
    has_invalid_citations: bool = False
    invalid_citation_count: int = 0
    numeric_fidelity: float = 1.0        # answer numbers found in the context
    unsupported_numbers: List[str] = field(default_factory=list)
    answered: bool = True
    insufficient_evidence: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "citation_coverage": round(self.citation_coverage, 4),
            "claim_citation_rate": round(self.claim_citation_rate, 4),
            "has_invalid_citations": self.has_invalid_citations,
            "invalid_citation_count": self.invalid_citation_count,
            "numeric_fidelity": round(self.numeric_fidelity, 4),
            "unsupported_numbers": self.unsupported_numbers[:10],
            "answered": self.answered,
            "insufficient_evidence": self.insufficient_evidence,
        }


def citation_coverage(answer: str, citations: Sequence[Citation]) -> float:
    """Share of supplied sources the answer actually cited."""
    if not citations:
        return 0.0
    valid = {c.index for c in citations}
    used = {i for i in parse_markers(answer) if i in valid}
    return len(used) / len(valid)


def claim_citation_rate(answer: str) -> float:
    """Share of substantive sentences carrying a citation marker."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?؟])\s+|\n+", answer)
        if len(s.strip()) > 40
    ]
    if not sentences:
        return 1.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+", s))
    return cited / len(sentences)


def invalid_citations(answer: str, citations: Sequence[Citation]) -> Set[int]:
    """Markers pointing at sources that were never supplied — i.e. fabricated."""
    valid = {c.index for c in citations}
    return {i for i in parse_markers(answer) if i not in valid}


def numeric_fidelity(answer: str, context: str) -> tuple[float, List[str]]:
    """Check that numbers in the answer actually appear in the retrieved context.

    Catches the most damaging RAG failure: a plausible but invented figure.
    Numbers written differently (1,200 vs 1200) are normalised before comparison,
    and citation markers are removed first so ``[1]`` is not mistaken for a claim.
    """
    answer_numbers = _normalized_numbers(strip_citation_markers(answer))
    if not answer_numbers:
        return 1.0, []
    context_numbers = _normalized_numbers(context)
    missing = [n for n in answer_numbers if n not in context_numbers]
    return 1.0 - len(missing) / len(answer_numbers), missing


def _normalized_numbers(text: str) -> Set[str]:
    out: Set[str] = set()
    for match in _NUMBER.finditer(text or ""):
        raw = match.group(0).replace(",", "").replace("،", "").rstrip(".")
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        out.add(f"{value:g}")
    return out


def evaluate_answer(
    result: AnswerResult, *, context: Optional[str] = None
) -> GenerationMetrics:
    """Score one generated answer against the evidence it was given."""
    invalid = invalid_citations(result.answer, result.citations)
    metrics = GenerationMetrics(
        citation_coverage=citation_coverage(result.answer, result.citations),
        claim_citation_rate=claim_citation_rate(result.answer),
        has_invalid_citations=bool(invalid),
        invalid_citation_count=len(invalid),
        answered=bool(result.answer.strip()),
        insufficient_evidence=result.insufficient_evidence,
    )

    evidence = context
    if evidence is None:
        evidence = "\n".join(c.snippet for c in result.citations)
    fidelity, missing = numeric_fidelity(result.answer, evidence)
    metrics.numeric_fidelity = fidelity
    metrics.unsupported_numbers = missing
    return metrics


class Grader(Protocol):
    """Hook for model-graded evaluation (groundedness, relevance, …)."""

    def score(self, question: str, answer: str, context: str) -> Dict[str, float]:
        ...


class LLMGrader:
    """Optional LLM-based grader.

    Not used anywhere in the request path — instantiate it explicitly in an
    offline evaluation script so grading cost is a deliberate choice.
    """

    SYSTEM = """You grade an answer against the context it was given.

Return a single JSON object:
{"groundedness": <0.0-1.0>, "relevance": <0.0-1.0>, "completeness": <0.0-1.0>, "notes": "<short>"}

- groundedness: is every claim supported by the context? Unsupported claims score low.
- relevance: does the answer address the question?
- completeness: does it use the available evidence fully?"""

    def __init__(self, llm) -> None:
        self.llm = llm

    def score(self, question: str, answer: str, context: str) -> Dict[str, float]:
        import json

        from omnirag.core.enums import Role
        from omnirag.providers.llm.base import LLMMessage

        prompt = (
            f"CONTEXT:\n{context[:12000]}\n\n"
            f"QUESTION: {question}\n\nANSWER:\n{answer[:6000]}"
        )
        try:
            response = self.llm.complete(
                [LLMMessage(role=Role.USER, text=prompt)],
                system=self.SYSTEM,
                temperature=0.0,
                max_output_tokens=300,
                json_mode=True,
            )
            data = json.loads(re.search(r"\{.*\}", response.text, re.DOTALL).group(0))
        except Exception:
            return {}
        return {
            key: max(0.0, min(1.0, float(data.get(key, 0.0))))
            for key in ("groundedness", "relevance", "completeness")
            if key in data
        }


__all__ = [
    "GenerationMetrics",
    "Grader",
    "LLMGrader",
    "citation_coverage",
    "claim_citation_rate",
    "evaluate_answer",
    "invalid_citations",
    "numeric_fidelity",
    "strip_citation_markers",
]
