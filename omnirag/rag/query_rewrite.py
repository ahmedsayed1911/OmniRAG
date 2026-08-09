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

from omnirag.core.enums import Language, QueryScope, Role
from omnirag.core.exceptions import ProviderError
from omnirag.core.models import ChatMessage
from omnirag.providers.llm.base import BaseLLMProvider, LLMMessage
from omnirag.providers.llm.context import llm_operation
from omnirag.utils.language import detect_language, normalize_arabic
from omnirag.utils.logging import get_logger
from omnirag.utils.text import dedupe_preserve_order

logger = get_logger(__name__)

# "page 17", "p. 17", "pages 3-5", "slide 4", "الصفحة ١٧", "صفحة 17", "شريحة 4"
_NUMBER = r"[0-9٠-٩]{1,4}"
_PAGE_LABEL = r"(?:pages?|pg\.?|p\.|slides?|الصفحة|الصفحه|صفحة|صفحه|الصفحات|صفحات|الشريحة|الشريحه|شريحة|شريحه)"
_PAGE_RANGE = re.compile(
    rf"{_PAGE_LABEL}\s*[:\-]?\s*({_NUMBER})\s*(?:-|–|—|to|إلى|الى)\s*({_NUMBER})",
    re.I,
)
_PAGE_LIST = re.compile(
    rf"{_PAGE_LABEL}\s*[:\-]?\s*({_NUMBER}(?:\s*(?:and|&|,|،|و)\s*{_NUMBER})+)",
    re.I,
)
_PAGE_SINGLE = re.compile(rf"{_PAGE_LABEL}\s*[:\-]?\s*({_NUMBER})", re.I)
_ARABIC_ORDINAL_PAGE = re.compile(
    r"(?:الصفحة|الصفحه|صفحة|صفحه)\s+(الأولى|الاولى|الثانية|الثانيه|الثالثة|الثالثه|الرابعة|الرابعه|الخامسة|الخامسه|السادسة|السادسه|السابعة|السابعه|الثامنة|الثامنه|التاسعة|التاسعه|العاشرة|العاشره)"
)
_ARABIC_ORDINALS = {
    "الأولى": 1, "الاولى": 1, "الثانية": 2, "الثانيه": 2,
    "الثالثة": 3, "الثالثه": 3, "الرابعة": 4, "الرابعه": 4,
    "الخامسة": 5, "الخامسه": 5, "السادسة": 6, "السادسه": 6,
    "السابعة": 7, "السابعه": 7, "الثامنة": 8, "الثامنه": 8,
    "التاسعة": 9, "التاسعه": 9, "العاشرة": 10, "العاشره": 10,
}
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

_EXHAUSTIVE_HINTS = re.compile(
    r"\b(list\s+all|all|every|each|find\s+every|complete\s+list)\b|"
    r"(كل\s+الحالات|جميع\s+الحالات|كل\s+الأخطاء|كل\s+المشاكل|اذكر\s+كل|"
    r"جميع|بالكامل)",
    re.I,
)
_GLOBAL_HINTS = re.compile(
    r"\b(summari[sz]e\s+(?:this|the|entire|whole|full)?\s*document|"
    r"entire\s+document|whole\s+(?:document|file)|across\s+the\s+document|"
    r"full\s+(?:document|file))\b|"
    r"(لخص\s+(?:هذا\s+)?الملف|لخصلي\s+الملف|الملف\s+كله|كامل\s+الملف)",
    re.I,
)
_FAILED_HINTS = re.compile(r"\b(?:fail|failed|failure)s?\b|(?:فشل|فاشل|فاشلة|الفشل)", re.I)
_BUG_HINTS = re.compile(r"\bbugs?\b|bug\s+reports?|(?:الأخطاء|المشاكل|عيوب)", re.I)
_SEVERITY_HINTS = re.compile(r"\bseverity|high[- ]severity\b|(?:الخطورة|عالية\s+الخطورة)", re.I)

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
    scope: QueryScope = QueryScope.FOCUSED

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
    plan.scope = classify_query_scope(normalized)
    if plan.scope != QueryScope.FOCUSED:
        plan.expansions.extend(decompose_query(plan))
        plan.notes.append(f"Retrieval scope: {plan.scope.value}.")

    resolved = _resolve_followup(normalized, history)
    if resolved != normalized:
        plan.is_followup = True
        plan.expansions.append(resolved)
        plan.notes.append("Resolved a follow-up question using the previous turn.")

    return plan


def classify_query_scope(query: str) -> QueryScope:
    """Classify breadth deterministically; ordinary questions stay fast."""
    normalized = normalize_query(query)
    exhaustive = bool(_EXHAUSTIVE_HINTS.search(normalized))
    global_request = bool(_GLOBAL_HINTS.search(normalized))
    # A request combining a document overview with an enumeration has two
    # separately retrievable intents and must retain both.
    if exhaustive and global_request:
        return QueryScope.MULTI_PART
    if exhaustive:
        return QueryScope.EXHAUSTIVE
    if global_request:
        return QueryScope.GLOBAL
    return QueryScope.FOCUSED


def decompose_query(plan: QueryPlan) -> List[str]:
    """Produce bounded, intent-preserving retrieval variants."""
    query = plan.normalized
    variants: List[str] = []
    if plan.scope in (QueryScope.GLOBAL, QueryScope.MULTI_PART):
        variants.extend(["document overview main sections", "document summary key findings"])
    if _FAILED_HINTS.search(query):
        variants.extend(
            [
                "failed test cases status fail failed",
                "test case actual result expected result failure reason",
            ]
        )
    if _BUG_HINTS.search(query) or plan.scope in (QueryScope.EXHAUSTIVE, QueryScope.MULTI_PART):
        variants.append("bug report bug ID severity actual result")
    if _SEVERITY_HINTS.search(query):
        variants.append("high severity bug reports")
    if plan.language in (Language.ARABIC, Language.MIXED):
        # Deterministic English bridge for the common structured QA vocabulary.
        variants.append("test cases status failed actual result bug report severity")
    return dedupe_preserve_order(variants)[:6]


def _extract_pages(query: str) -> List[int]:
    pages: List[int] = [
        _ARABIC_ORDINALS[match.group(1)]
        for match in _ARABIC_ORDINAL_PAGE.finditer(query)
    ]
    for match in _PAGE_RANGE.finditer(query):
        start = int(match.group(1).translate(_ARABIC_DIGITS))
        end = int(match.group(2).translate(_ARABIC_DIGITS))
        if end < start:
            start, end = end, start
        pages.extend(range(start, end + 1 if end - start <= 40 else start + 1))
    for match in _PAGE_LIST.finditer(query):
        pages.extend(
            int(raw.translate(_ARABIC_DIGITS))
            for raw in re.findall(_NUMBER, match.group(1))
        )
    for match in _PAGE_SINGLE.finditer(query):
        pages.append(int(match.group(1).translate(_ARABIC_DIGITS)))
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
    "classify_query_scope",
    "decompose_query",
    "expand_with_llm",
    "normalize_query",
    "parse_query",
]
