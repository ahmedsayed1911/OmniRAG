"""Grounded, multimodal answer generation.

Contract enforced by the prompt *and* by code:

* the model sees only the retrieved contexts, never whole documents;
* every context is numbered, and only those numbers may be cited — markers are
  verified afterwards in :mod:`omnirag.rag.citations`;
* when a retrieved block has a visual (chart, diagram, scan, photo, handwritten
  note), **the original image is attached to the request**, not just its
  description. This is the multimodal retrieval rule;
* conversation history is passed as *context for what the user means*, and the
  prompt states explicitly that previous answers are not evidence — only the
  numbered sources are;
* insufficient evidence must be stated, not papered over.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from omnirag.config.settings import AppSettings, get_settings
from omnirag.core.enums import BlockType, Language, QueryScope, Role, SourceKind
from omnirag.core.exceptions import ProviderCapabilityError, ProviderError
from omnirag.core.models import (
    AnswerResult,
    ChatMessage,
    Chunk,
    Citation,
    RetrievalResult,
    SearchResult,
)
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage, LLMResponse
from omnirag.providers.llm.context import generation_context, llm_operation
from omnirag.rag.citations import build_citations, verify_and_clean
from omnirag.rag.query_rewrite import QueryPlan
from omnirag.storage.files import FileStore
from omnirag.intelligence.vision import VisionAnalyzer
from omnirag.utils.language import language_name
from omnirag.utils.hashing import short_hash
from omnirag.utils.logging import get_logger
from omnirag.utils.text import dedupe_preserve_order, estimate_tokens, truncate

logger = get_logger(__name__)

INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"
FINAL_ANSWER_ONLY_INSTRUCTION = (
    "Return only the final answer. Do not output analysis, reasoning, planning, "
    "drafting notes, internal monologue, self-evaluation, preparation steps, or "
    "generation instructions."
)

SYSTEM_PROMPT = f"""You are OmniRAG, a document-analysis assistant. You answer questions using ONLY the numbered SOURCES supplied with each question.

GROUNDING
- Base every factual statement on the SOURCES. Never use outside knowledge to add facts about the user's documents.
- If the SOURCES do not contain the answer, say so plainly and state what is missing. Begin such an answer with {INSUFFICIENT_MARKER} on its own line.
- Never invent a document, page, figure, value or quotation that is not in the SOURCES.

CITATIONS
- Cite with square-bracket numbers that refer to the SOURCE numbers, e.g. [1] or [2, 4].
- Put the citation immediately after the sentence or clause it supports.
- Only cite numbers that actually appear in the SOURCES. Never cite a number you were not given.
- Every factual claim needs a citation. Uncited prose is only acceptable for connecting sentences.

ACCURACY
- Reproduce numbers, dates, units, names and identifiers EXACTLY as they appear. Never round, convert or recompute unless the user asks, and then show that you did.
- Distinguish what a source states from what you infer. Mark inference explicitly, e.g. "The report states X [1]; this suggests Y."
- Preserve uncertainty. A source marked as OCR or handwriting is a best-effort reading: say so when the answer depends on it.
- If sources disagree, say so and cite each side. Do not silently pick one.

VISUALS
- Some sources are charts, diagrams, tables, scans or photographs. Where the image itself is attached, read it directly and describe what you actually see.
- When answering about a visual, mention its concrete elements (axis labels, series, components, arrows, column headers) rather than speaking generally.

CONVERSATION
- Earlier turns tell you what the user means. They are NOT evidence. Never cite a previous answer; re-derive facts from the SOURCES.

LANGUAGE
- Answer in the language of the user's question, unless they ask for another language.
- The documents may be in a different language than the question. Translate the content into the answer language, but keep proper nouns, identifiers and numbers unchanged, and quote key terms in the original script when helpful.

STYLE
- Be direct and specific. Lead with the answer, then the supporting detail.
- Use short paragraphs or bullets. Use a Markdown table when comparing several items.
- Do not describe your own process or mention "sources provided to me".

OUTPUT PRIVACY
- {FINAL_ANSWER_ONLY_INSTRUCTION}"""

MAX_CONTEXT_CHARS_PER_CHUNK = 3200
OUTPUT_LIMIT_REASONS = frozenset({"MAX_TOKENS", "MAX_OUTPUT_TOKENS", "LENGTH"})
CONTINUATION_PROMPT = """Continue exactly where your previous response stopped.
Do not restart the answer and do not repeat any completed section or sentence.
Use only the same numbered sources already supplied; do not add outside facts.
Preserve the existing citation numbering and cite every continued factual item.
Finish the remaining answer naturally and compactly.
Return only the continued final-answer text; never output reasoning, planning, or drafting notes."""


@dataclass
class GenerationRequest:
    question: str
    retrieval: RetrievalResult
    session_id: str
    history: Sequence[ChatMessage] = field(default_factory=list)
    plan: Optional[QueryPlan] = None
    answer_language: Optional[Language] = None
    generation_id: str = ""


class AnswerGenerator:
    """Builds the grounded prompt, attaches visuals, and verifies citations."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        file_store: Optional[FileStore] = None,
        vision: Optional[VisionAnalyzer] = None,
        settings: Optional[AppSettings] = None,
    ):
        self.llm = llm
        self.file_store = file_store
        self.vision = vision
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> AnswerResult:
        results = self._focused_page_results(
            list(request.retrieval.results), request.plan
        )
        if not results:
            return self._no_evidence_answer(request)

        citations = build_citations(results)
        images = self._collect_images(results, citations, request.plan)
        self._enrich_lazy_visuals(results, citations, images, request.plan)
        citations = build_citations(results)
        context_text = self._format_context(results, citations)

        messages = self._build_messages(request, context_text, images)
        active_messages = messages
        output_budget = self._output_budget(request)
        generation_id = request.generation_id or uuid.uuid4().hex
        input_token_estimate = sum(estimate_tokens(message.text) for message in messages)
        logger.info(
            "Generation request detected_page_filter=%s retrieved_pages=%s "
            "context_chunks=%d estimated_input_tokens=%d "
            "requested_output_tokens=%d selected_visuals=%d",
            list(request.plan.page_filter) if request.plan else [],
            sorted({item.chunk.page_number for item in results}),
            len(results),
            input_token_estimate,
            output_budget,
            len(images),
        )
        started = time.perf_counter()
        generation_warnings: List[str] = []

        try:
            with generation_context(generation_id), llm_operation("final_answer"):
                response = self.llm.complete(
                    messages,
                    system=self._system_prompt(request),
                    temperature=self.settings.llm.temperature,
                    max_output_tokens=output_budget,
                )
        except ProviderCapabilityError as exc:
            # The chain could not read the attached visuals. Rather than
            # silently answering without the evidence, retry text-only and say
            # so explicitly in the result.
            logger.warning("Multimodal generation unavailable: %s", exc)
            if not images:
                raise
            if request.plan and request.plan.wants_visual:
                # The question explicitly depends on the attached visual.
                # Answering text-only would hide an evidence loss.
                raise
            active_messages = self._build_messages(request, context_text, [])
            with generation_context(generation_id), llm_operation("final_answer_text_only"):
                response = self.llm.complete(
                    active_messages,
                    system=self._system_prompt(request),
                    temperature=self.settings.llm.temperature,
                    max_output_tokens=output_budget,
                )
            images = []
            generation_warnings.append(exc.user_message)

        response, continued, continuation_warnings = self._continue_once(
            request, active_messages, response, output_budget, generation_id
        )
        generation_warnings.extend(continuation_warnings)

        elapsed = (time.perf_counter() - started) * 1000
        returned_tokens = _output_tokens(response)
        logger.info(
            "Generation query_scope=%s context_chunks=%d input_token_estimate=%d "
            "requested_max_output_tokens=%d provider=%s model=%s finish_reason=%s "
            "returned_chars=%d returned_tokens=%d continued=%s generation_ms=%.0f",
            request.plan.scope.value if request.plan else "FOCUSED",
            len(results),
            input_token_estimate,
            output_budget,
            response.provider or self.llm.name,
            response.model or self.llm.model,
            response.finish_reason or "unspecified",
            len(response.text),
            returned_tokens,
            continued,
            elapsed,
        )
        result = self._finish(
            response,
            citations,
            len(images),
            continued=continued,
            generation_id=generation_id,
            requested_output_tokens=output_budget,
        )
        result.warnings.extend(generation_warnings)
        logger.info(
            "Generation lifecycle stage=generation_result generation_id=%s "
            "finish_reason=%s generation_result_chars=%d",
            generation_id,
            result.finish_reason or "unspecified",
            len(result.answer),
        )
        return result

    def _output_budget(self, request: GenerationRequest) -> int:
        base = max(1, self.settings.llm.max_output_tokens)
        if request.plan and request.plan.scope in (
            QueryScope.GLOBAL,
            QueryScope.EXHAUSTIVE,
            QueryScope.MULTI_PART,
        ):
            return max(base, self.settings.llm.exhaustive_max_output_tokens)
        return base

    def _focused_page_results(
        self,
        results: Sequence[SearchResult],
        plan: Optional[QueryPlan],
    ) -> List[SearchResult]:
        """Defensively restrict and compact exact-page evidence.

        Retrieval already applies the page filter. This boundary prevents an
        older/stale index result from widening generation and merges duplicate
        OCR/vision representations that reference the same stored page image.
        """
        if not plan or not plan.page_filter or plan.scope != QueryScope.FOCUSED:
            return list(results)

        allowed = set(plan.page_filter)
        compacted: List[SearchResult] = []
        by_visual: Dict[tuple[str, int, str], SearchResult] = {}
        for result in results:
            chunk = result.chunk
            if chunk.page_number not in allowed:
                continue
            asset_id = chunk.visual.asset_id if chunk.visual else ""
            if not asset_id:
                compacted.append(result)
                continue
            key = (chunk.document_id, chunk.page_number, asset_id)
            existing = by_visual.get(key)
            if existing is None:
                copied = result.model_copy(deep=True)
                by_visual[key] = copied
                compacted.append(copied)
                continue
            if chunk.text and chunk.text not in existing.chunk.text:
                existing.chunk.text = (
                    f"{existing.chunk.text}\n\n{chunk.text}".strip()
                )
            existing.chunk.block_ids = dedupe_preserve_order(
                [*existing.chunk.block_ids, *chunk.block_ids]
            )
            existing.chunk.metadata["merged_page_representations"] = True
        return compacted

    def _continue_once(
        self,
        request: GenerationRequest,
        messages: Sequence[LLMMessage],
        response: LLMResponse,
        output_budget: int,
        generation_id: str,
    ) -> tuple[LLMResponse, bool, List[str]]:
        """Continue once only when the provider explicitly hit its token cap."""
        if response.finish_reason.upper() not in OUTPUT_LIMIT_REASONS:
            return response, False, []

        continuation_messages = [
            *messages,
            LLMMessage(role=Role.ASSISTANT, text=response.text),
            LLMMessage(role=Role.USER, text=CONTINUATION_PROMPT),
        ]
        try:
            with generation_context(generation_id), llm_operation(
                "final_answer_continuation"
            ):
                continuation = self.llm.complete(
                    continuation_messages,
                    system=self._system_prompt(request),
                    temperature=self.settings.llm.temperature,
                    max_output_tokens=output_budget,
                )
        except Exception as exc:  # noqa: BLE001 - retain the complete first portion
            logger.warning(
                "One-shot continuation failed safely: provider=%s error=%s",
                getattr(self.llm, "name", "unknown"),
                type(exc).__name__,
            )
            return response, False, [
                "The model reached its output limit, and automatic continuation failed. "
                "The complete generated portion is shown above."
            ]

        combined = _join_continuation(response.text, continuation.text)
        usage = dict(response.usage or {})
        usage["continuation"] = dict(continuation.usage or {})
        combined_response = LLMResponse(
            text=combined,
            model=continuation.model or response.model,
            finish_reason=continuation.finish_reason,
            usage=usage,
            provider=continuation.provider or response.provider,
            fallback_used=response.fallback_used or continuation.fallback_used,
            attempts=[*response.attempts, *continuation.attempts],
            diagnostics={
                **dict(response.diagnostics or {}),
                "continuation_provider_raw_chars": (continuation.diagnostics or {}).get(
                    "provider_raw_chars", len(continuation.text)
                ),
                "continuation_parsed_chars": (continuation.diagnostics or {}).get(
                    "parsed_chars", len(continuation.text)
                ),
                "combined_chars": len(combined),
            },
        )
        warnings = ["Response continued automatically because the model reached its output limit."]
        if continuation.finish_reason.upper() in OUTPUT_LIMIT_REASONS:
            warnings.append(
                "The continued response also reached the provider output limit; no further "
                "automatic continuation was attempted."
            )
        return combined_response, True, warnings

    # ------------------------------------------------------------------ #
    def _finish(
        self,
        response: LLMResponse,
        citations: List[Citation],
        image_count: int,
        *,
        continued: bool = False,
        generation_id: str = "",
        requested_output_tokens: int = 0,
    ) -> AnswerResult:
        bundle = verify_and_clean(response.text, citations)
        answer = bundle.answer
        insufficient = INSUFFICIENT_MARKER in answer
        if insufficient:
            answer = answer.replace(INSUFFICIENT_MARKER, "").strip()

        warnings: List[str] = []
        if bundle.invalid_indices:
            warnings.append(
                "The model referenced sources that were not provided; those "
                "citations were removed."
            )
        if not bundle.used_indices and not insufficient:
            warnings.append(
                "The answer did not cite any source. Treat it with caution and "
                "check the sources below."
            )

        return AnswerResult(
            answer=answer,
            citations=bundle.citations,
            insufficient_evidence=insufficient,
            model=response.model or self.llm.model,
            used_images=image_count,
            usage=dict(response.usage or {}),
            warnings=warnings,
            finish_reason=response.finish_reason,
            continued=continued,
            generation_id=generation_id,
            generation_debug={
                **dict(response.diagnostics or {}),
                "provider_raw_chars": (response.diagnostics or {}).get(
                    "provider_raw_chars", len(response.text)
                ),
                "parsed_chars": (response.diagnostics or {}).get(
                    "parsed_chars", len(response.text)
                ),
                "grounded_result_chars": len(answer),
                "requested_output_tokens": requested_output_tokens,
                "continuation_count": 1 if continued else 0,
            },
        )

    def _no_evidence_answer(self, request: GenerationRequest) -> AnswerResult:
        """No retrieval hits: say so honestly instead of calling the model."""
        arabic = request.retrieval.language == Language.ARABIC or (
            request.answer_language == Language.ARABIC
        )
        message = (
            "لم أعثر على أي محتوى في المستندات المحددة يجيب عن هذا السؤال. "
            "جرّب إعادة صياغة السؤال، أو تأكد من اختيار المستند المناسب."
            if arabic
            else (
                "I could not find anything in the selected documents that answers "
                "this question. Try rephrasing it, or check that the right "
                "documents are selected."
            )
        )
        return AnswerResult(
            answer=message,
            citations=[],
            insufficient_evidence=True,
            model=self.llm.model,
        )

    # ------------------------------------------------------------------ #
    def _system_prompt(self, request: GenerationRequest) -> str:
        prompt = SYSTEM_PROMPT
        if request.plan and request.plan.scope in (
            QueryScope.EXHAUSTIVE,
            QueryScope.MULTI_PART,
        ):
            prompt += """

COMPREHENSIVE MODE
- The user requested comprehensive coverage, not representative examples.
- Enumerate every matching item supported by the numbered evidence.
- Give each reported item its own nearby citation; do not cite one unrelated source for a whole list.
- Preserve links explicitly supported by evidence (for example test case -> actual result -> bug ID -> severity), but never invent a relationship.
- Do not claim the list is complete unless the retrieval evidence supports that claim.
- If evidence appears incomplete or ambiguous, state that limitation explicitly."""
            prompt += """
- Keep the overview concise, then use a compact structured list for exhaustive items.
- Do not repeat document headings or background under every item.
- Prefer: item/title; page; expected; actual; reason; linked bug/severity; citations."""
        target = request.answer_language or (
            request.plan.answer_language if request.plan else None
        )
        if target and target != Language.UNKNOWN:
            prompt += f"\n\nThe user asked for the answer in {language_name(target)}. Answer in {language_name(target)}."
        return prompt

    def _format_context(
        self, results: Sequence[SearchResult], citations: Sequence[Citation]
    ) -> str:
        """Render the numbered SOURCES block."""
        parts: List[str] = []
        for citation, result in zip(citations, results):
            chunk = result.chunk
            header = f"[{citation.index}] [{chunk.filename} — {citation.page_label}]"
            descriptors = [_describe_block(chunk)]
            if chunk.section:
                descriptors.append(f"section: {truncate(chunk.section, 90)}")
            if chunk.uncertain:
                descriptors.append("LOW CONFIDENCE — verify before relying on it")
            if chunk.confidence is not None:
                descriptors.append(f"confidence {chunk.confidence:.0%}")
            if chunk.visual is not None:
                descriptors.append("original image attached below")

            body = truncate(chunk.text, MAX_CONTEXT_CHARS_PER_CHUNK)
            parts.append(f"{header}\n({'; '.join(descriptors)})\n{body}")

        return "\n\n---\n\n".join(parts)

    def _collect_images(
        self,
        results: Sequence[SearchResult],
        citations: Sequence[Citation],
        plan: Optional[QueryPlan] = None,
    ) -> List[Tuple[int, ImagePart]]:
        """Load the original visuals for retrieved visual blocks.

        This is what makes retrieval genuinely multimodal: the description got
        the block found, the image itself is what the model reasons over.
        """
        if not self.settings.llm.enable_multimodal or self.file_store is None:
            return []
        if not self.llm.supports_images():
            return []

        budget = max(0, self.settings.llm.max_images_per_answer)
        if budget == 0:
            return []

        out: List[Tuple[int, ImagePart]] = []
        seen_assets: set[str] = set()
        seen_content: set[str] = set()
        per_page: Dict[int, int] = {}
        requested_pages = set(plan.page_filter if plan else [])
        if requested_pages and plan and plan.scope == QueryScope.FOCUSED:
            budget = min(budget, len(requested_pages))
        candidates = list(zip(citations, results))
        if requested_pages:
            candidates = [
                pair for pair in candidates
                if pair[1].chunk.page_number in requested_pages
            ]
            candidates.sort(
                key=lambda pair: (
                    pair[1].chunk.visual is None,
                    0 if (
                        pair[1].chunk.visual is not None
                        and pair[1].chunk.visual.origin == "page_render"
                    ) else 1,
                    pair[1].rank,
                )
            )

        for citation, result in candidates:
            if len(out) >= budget:
                break
            chunk = result.chunk
            if chunk.visual is None:
                continue
            # Only send images that carry real information for this answer.
            if not (chunk.block_type.is_visual or chunk.source_kind == SourceKind.OCR):
                continue
            asset_id = chunk.visual.asset_id
            if asset_id in seen_assets:
                continue
            if requested_pages and per_page.get(chunk.page_number, 0) >= 2:
                continue

            data = self.file_store.get(asset_id)
            if not data:
                logger.debug("Visual asset %s is no longer available", asset_id[:8])
                continue
            content_id = short_hash(data, 24)
            if content_id in seen_content:
                continue
            # One full-page render plus at most one distinct high-ranked crop
            # gives the model direct evidence without near-duplicate payloads.
            if (
                requested_pages
                and per_page.get(chunk.page_number, 0) >= 1
                and chunk.visual.origin == "page_render"
            ):
                continue
            seen_assets.add(asset_id)
            seen_content.add(content_id)
            per_page[chunk.page_number] = per_page.get(chunk.page_number, 0) + 1
            out.append(
                (
                    citation.index,
                    ImagePart(
                        data=data,
                        media_type=chunk.visual.media_type or "image/png",
                        label=f"Image for source [{citation.index}] — {chunk.filename}, {citation.page_label}",
                    ),
                )
            )
        return out

    def _enrich_lazy_visuals(
        self,
        results: Sequence[SearchResult],
        citations: Sequence[Citation],
        images: Sequence[Tuple[int, ImagePart]],
        plan: Optional[QueryPlan],
    ) -> None:
        """Understand at most one canonical image for each retrieved page.

        The combined vision result contains both transcription and semantic
        description. It enriches only this request's context; the content-hash
        cache makes later questions and Streamlit reruns reuse it.
        """
        if not self.settings.vision.lazy_analysis or self.vision is None:
            return
        if plan is not None and not (plan.wants_visual or plan.page_filter):
            return

        by_citation = {
            citation.index: result
            for citation, result in zip(citations, results)
        }
        seen_pages: set[tuple[str, int]] = set()
        for citation_index, image in images:
            result = by_citation.get(citation_index)
            if result is None:
                continue
            chunk = result.chunk
            page_key = (chunk.document_id, chunk.page_number)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            analysis = self.vision.analyze(
                image.data,
                expect=chunk.block_type if chunk.block_type.is_visual else None,
                skip_decorative_check=True,
                document_id=chunk.document_id,
                document_hash=str(chunk.metadata.get("document_hash") or ""),
                page_number=chunk.page_number,
            )
            if analysis.ok and analysis.searchable_text not in chunk.text:
                chunk.text = f"{chunk.text}\n\n{analysis.searchable_text}".strip()
                chunk.metadata["visual_analysis_pending"] = False
                chunk.metadata["visual_analysis_cached"] = True

    def _build_messages(
        self,
        request: GenerationRequest,
        context_text: str,
        images: Sequence[Tuple[int, ImagePart]],
    ) -> List[LLMMessage]:
        messages: List[LLMMessage] = []

        # Recent conversation, clearly framed as intent context, not evidence.
        history = _recent_history(request.history, self.settings.max_history_messages)
        for message in history:
            messages.append(
                LLMMessage(
                    role=message.role,
                    text=truncate(message.content, 1200),
                )
            )

        prompt_parts = [
            "SOURCES (the only evidence you may use):",
            "",
            context_text,
            "",
            "---",
            "",
            f"QUESTION: {request.question}",
            "",
            FINAL_ANSWER_ONLY_INSTRUCTION,
        ]
        if images:
            numbers = ", ".join(f"[{index}]" for index, _ in images)
            prompt_parts.append(
                f"\nThe original images for sources {numbers} are attached. "
                "Read them directly when they matter to the answer."
            )

        messages.append(
            LLMMessage(
                role=Role.USER,
                text="\n".join(prompt_parts),
                images=[part for _, part in images],
            )
        )
        return messages


def _describe_block(chunk: Chunk) -> str:
    """Human-readable provenance label shown to the model."""
    kind = {
        BlockType.TABLE: "table",
        BlockType.CHART: "chart",
        BlockType.DIAGRAM: "diagram",
        BlockType.IMAGE: "image",
        BlockType.HANDWRITING: "handwritten note",
        BlockType.OCR_TEXT: "scanned text (OCR)",
        BlockType.PAGE_SNAPSHOT: "page scan",
        BlockType.HEADING: "heading",
        BlockType.SPEAKER_NOTES: "speaker notes",
        BlockType.CAPTION: "caption",
    }.get(chunk.block_type, "text")

    source = {
        SourceKind.OCR: "read by OCR",
        SourceKind.VISION: "described by a vision model",
        SourceKind.STRUCTURED: "parsed from the file structure",
        SourceKind.DIGITAL: "extracted text",
        SourceKind.DERIVED: "derived",
    }.get(chunk.source_kind, "extracted text")

    return f"{kind}, {source}"


def _recent_history(
    history: Sequence[ChatMessage], limit: int
) -> List[ChatMessage]:
    """Last N turns, dropping errors and empty messages."""
    usable = [
        m
        for m in history
        if m.content.strip() and m.error is None and m.role in (Role.USER, Role.ASSISTANT)
    ]
    return usable[-max(0, limit) :] if limit else []


def _join_continuation(first: str, continuation: str) -> str:
    """Join an exact continuation while removing a bounded repeated prefix."""
    left = first
    right = continuation
    if not right:
        return left
    max_overlap = min(600, len(left), len(right))
    overlap = next(
        (size for size in range(max_overlap, 19, -1) if left[-size:] == right[:size]),
        0,
    )
    if overlap:
        right = right[overlap:]
    # MAX_TOKENS frequently cuts inside a word. The continuation instruction
    # asks for the exact remaining characters, so concatenate without inventing
    # whitespace unless the provider begins a fresh Markdown block.
    separator = (
        "\n"
        if right.startswith(("#", "- ", "* ", ">", "```"))
        and not left.endswith(("\n", " "))
        else ""
    )
    return f"{left}{separator}{right}"


def _output_tokens(response: LLMResponse) -> int:
    usage = response.usage or {}
    keys = (
        "candidatesTokenCount",
        "output_tokens",
        "completion_tokens",
        "outputTokenCount",
    )
    total = next((int(usage[key]) for key in keys if usage.get(key) is not None), 0)
    continuation = usage.get("continuation")
    if isinstance(continuation, dict):
        total += next(
            (int(continuation[key]) for key in keys if continuation.get(key) is not None),
            0,
        )
    return total or estimate_tokens(response.text)


def build_generator(
    settings: Optional[AppSettings] = None, file_store: Optional[FileStore] = None
) -> AnswerGenerator:
    from omnirag.providers.llm.factory import get_llm_provider
    from omnirag.storage.files import get_file_store

    resolved = settings or get_settings()
    return AnswerGenerator(
        get_llm_provider(resolved),
        file_store=file_store or get_file_store(),
        settings=resolved,
    )


__all__ = [
    "AnswerGenerator",
    "GenerationRequest",
    "INSUFFICIENT_MARKER",
    "SYSTEM_PROMPT",
    "FINAL_ANSWER_ONLY_INSTRUCTION",
    "build_generator",
]
