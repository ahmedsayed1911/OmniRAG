"""Query transformation.

Three deterministic stages plus one optional model-assisted stage:

1. **normalisation** — whitespace, Arabic diacritics, Unicode form. Never
   changes wording.
2. **structural parsing** — recognises explicit user intent that a model should
   not be asked to guess: "page 17", "الصفحة ٢٣", "slide 4", "compare A and B",
   "answer in Arabic". These become retrieval *filters*, not prompt hints, which
   is far more reliable than hoping the LLM notices.
3. **conversational resolution** — a follow-up like "and what about 2023?" is
   expanded with the previous question's subject, so retrieval has something to
   match on.
4. **LLM expansion** (optional, ``QUERY_REWRITE=true``) — produces alternative
   phrasings *and* a cross-lingual variant, so an Arabic question can retrieve
   from English documents. Intent is explicitly preserved: the original query is
   always searched too, and expansions only ever add candidates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from omnirag.core.enums import Language, Role
from omnirag.core.exceptions import ProviderError
from omnirag.core.models import ChatMessage
from omnirag.providers.llm.base import BaseLLMProvider, LLMMessage
from omnirag.providers.llm.context import llm_operation
from omnirag.utils.language import detect_language, normalize_arabic
from omnirag.utils.logging import get_logger
from omnirag.utils.text import dedupe_preserve_order

logger = get_logger(__name__)

# "page 17", "p. 17", "pages 3-5", "slide 4", "الصفحة ١٧", "صفحة 17", "شريحة 4"
_PAGE_PATTERNS = (
    re.compile(r"\b(?:pages?|pg\.?|p\.)\s*(\d{1,4})(?:\s*[-–to]+\s*(\d{1,4}))?\b", re.I),
    re.compile(r"\bslides?\s*(\d{1,4})(?:\s*[-–to]+\s*(\d{1,4}))?\b", re.I),
    re.compile(r"(?:الصفحه|الصفحة|صفحة|صفحه|ص)\s*[:\-]?\s*([\d٠-٩]{1,4})"),
    re.compile(r"(?:الشريحة|الشريحه|شريحة|شريحه)\s*[:\-]?\s*([\d٠-٩]{1,4})"),
)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_ANSWER_LANGUAGE_PATTERNS = (
    (re.compile(r"\b(?:answer|reply|respond|summari[sz]e)\s+(?:me\s+)?in\s+arabic\b", re.I), Language.ARABIC),
    (re.compile(r"\b(?:answer|reply|respond|summari[sz]e)\s+(?:me\s+)?in\s+english\b", re.I), Language.ENGLISH),
    (re.compile(r"(?:أجب|اجب|جاوب|الإجابة|الاجابة)\s*(?:لي\s*)?(?:باللغة\s*)?(?:بال)?عربي"), Language.ARABIC),
    (re.compile(r"(?:أجب|اجب|جاوب|الإجابة|الاجابة)\s*(?:لي\s*)?(?:باللغة\s*)?(?:بال)?(?:انجليزي|إنجليزي|انكليزي)"), Language.ENGLISH),
)

_VISUAL_HINTS = re.compile(
    r"\b(chart|graph|plot|figure|diagram|flowchart|drawing|image|picture|photo|"
    r"screenshot|table|handwrit|sketch)\w*\b",
    re.I,
)
_VISUAL_HINTS_AR = re.compile(
    r"(رسم|مخطط|بياني|شكل|صورة|جدول|خريطة|مكتوب بخط اليد|بخط اليد)"
)

_COMPARISON = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differ|disagree|contrast)\b", re.I
)
_COMPARISON_AR = re.compile(r"(قارن|مقارنة|الفرق|الفروق|يختلف|تختلف)")

_FOLLOWUP = re.compile(
    r"^\s*(and\s+|what\s+about\s+|how\s+about\s+|و?ماذا\s+عن\s+|وماذا\s+عن\s+|و\s*)",
    re.I,
)
_PRONOUN_ONLY = re.compile(
    r"^\s*(?:what|who|when|where|why|how)?\s*(?:does|do|did|is|are|was|were)?\s*"
    r"(?:it|this|that|they|them|he|she|these|those)\b",
    re.I,
)

SYSTEM_PROMPT = """You rewrite a user's question into alternative search queries for a document search engine.

Rules:
- Preserve the user's intent exactly. Never answer the question, never add new topics.
- Produce 2-3 alternative phrasings that use different wording for the same information need.
- ALWAYS include one translation of the query (Arabic query -> English variant, English query -> Arabic variant), because the documents may be in a different language than the question.
- Keep proper nouns, numbers, codes and units unchanged in every variant.
- Output a single JSON object: {"queries": ["...", "..."]}"""


@dataclass
class QueryPlan:
    """The parsed, expanded form of one user question."""

    original: str
    normalized: str
    language: Language = Language.UNKNOWN
    #: Language the *answer* should use, when the user asked explicitly.
    answer_language: Optional[Language] = None
    expansions: List[str] = field(default_factory=list)
    page_filter: List[int] = field(default_factory=list)
    wants_visual: bool = False
    is_comparison: bool = False
    is_followup: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def search_queries(self) -> List[str]:
        """Original first — expansions only ever add recall, never replace."""
        return dedupe_preserve_order([self.normalized, *self.expansions])


def normalize_query(query: str) -> str:
    """Whitespace/Unicode normalisation. Wording is untouched."""
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    return cleaned


def parse_query(query: str, history: Optional[Sequence[ChatMessage]] = None) -> QueryPlan:
    """Deterministic query analysis — no model call."""
    normalized = normalize_query(query)
    plan = QueryPlan(
        original=query,
        normalized=normalized,
        language=detect_language(normalized),
    )

    for pattern, language in _ANSWER_LANGUAGE_PATTERNS:
        if pattern.search(normalized):
            plan.answer_language = language
            break

    plan.page_filter = _extract_pages(normalized)
    plan.wants_visual = bool(
        _VISUAL_HINTS.search(normalized) or _VISUAL_HINTS_AR.search(normalized)
    )
    plan.is_comparison = bool(
        _COMPARISON.search(normalized) or _COMPARISON_AR.search(normalized)
    )

    resolved = _resolve_followup(normalized, history)
    if resolved != normalized:
        plan.is_followup = True
        plan.expansions.append(resolved)
        plan.notes.append("Resolved a follow-up question using the previous turn.")

    return plan


def _extract_pages(query: str) -> List[int]:
    pages: List[int] = []
    for pattern in _PAGE_PATTERNS:
        for match in pattern.finditer(query):
            start_raw = (match.group(1) or "").translate(_ARABIC_DIGITS)
            if not start_raw.isdigit():
                continue
            start = int(start_raw)
            end = start
            if match.lastindex and match.lastindex >= 2 and match.group(2):
                end_raw = match.group(2).translate(_ARABIC_DIGITS)
                if end_raw.isdigit():
                    end = int(end_raw)
            if end < start:
                start, end = end, start
            # Guard against absurd ranges from a mis-parse.
            if end - start > 40:
                end = start
            pages.extend(range(start, end + 1))
    return sorted(set(p for p in pages if 1 <= p <= 5000))


def _resolve_followup(query: str, history: Optional[Sequence[ChatMessage]]) -> str:
    """Expand an elliptical follow-up with the previous user question."""
    if not history:
        return query
    if len(query) > 90:
        return query
    if not (_FOLLOWUP.match(query) or _PRONOUN_ONLY.match(query)):
        return query

    previous = next(
        (m.content for m in reversed(history) if m.role == Role.USER and m.content.strip()),
        "",
    )
    if not previous or previous.strip() == query.strip():
        return query
    return f"{previous.strip()} — {query.strip()}"


def expand_with_llm(
    plan: QueryPlan,
    llm: Optional[BaseLLMProvider],
    *,
    max_expansions: int = 3,
) -> QueryPlan:
    """Add model-generated paraphrases and a cross-lingual variant.

    Failure is non-fatal: retrieval proceeds with the deterministic plan, which
    already contains the original query.
    """
    if llm is None or max_expansions <= 0 or not plan.normalized:
        return plan

    try:
        with llm_operation("query_rewrite"):
            response = llm.complete(
                [LLMMessage(role=Role.USER, text=f"Question: {plan.normalized}")],
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=320,
                json_mode=True,
            )
    except ProviderError as exc:
        logger.info("Query expansion skipped: %s", exc)
        plan.notes.append("Query expansion was unavailable for this question.")
        return plan
    except Exception as exc:  # noqa: BLE001 - never break retrieval
        logger.warning("Query expansion failed: %s", exc)
        return plan

    for variant in _parse_queries(response.text)[:max_expansions]:
        if variant and variant.lower() != plan.normalized.lower():
            plan.expansions.append(variant)
    plan.expansions = dedupe_preserve_order(plan.expansions)
    return plan


def _parse_queries(raw: str) -> List[str]:
    payload = (raw or "").strip()
    if not payload:
        return []
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", payload).strip()
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        payload = match.group(0)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    items = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        text = normalize_query(str(item))
        if 2 < len(text) <= 400:
            out.append(text)
    return out


def normalize_for_index(text: str) -> str:
    """Arabic-normalised form used for keyword matching."""
    return normalize_arabic(text)


__all__ = [
    "QueryPlan",
    "expand_with_llm",
    "normalize_query",
    "parse_query",
]
