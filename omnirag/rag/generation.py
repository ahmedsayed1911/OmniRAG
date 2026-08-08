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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from omnirag.config.settings import AppSettings, get_settings
from omnirag.core.enums import BlockType, Language, Role, SourceKind
from omnirag.core.exceptions import ProviderCapabilityError, ProviderError
from omnirag.core.models import (
    AnswerResult,
    ChatMessage,
    Chunk,
    Citation,
    RetrievalResult,
    SearchResult,
)
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage
from omnirag.rag.citations import build_citations, verify_and_clean
from omnirag.rag.query_rewrite import QueryPlan
from omnirag.storage.files import FileStore
from omnirag.utils.language import language_name
from omnirag.utils.logging import get_logger
from omnirag.utils.text import truncate

logger = get_logger(__name__)

INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"

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
- Do not describe your own process or mention "sources provided to me"."""

MAX_CONTEXT_CHARS_PER_CHUNK = 3200


@dataclass
class GenerationRequest:
    question: str
    retrieval: RetrievalResult
    session_id: str
    history: Sequence[ChatMessage] = field(default_factory=list)
    plan: Optional[QueryPlan] = None
    answer_language: Optional[Language] = None


class AnswerGenerator:
    """Builds the grounded prompt, attaches visuals, and verifies citations."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        file_store: Optional[FileStore] = None,
        settings: Optional[AppSettings] = None,
    ):
        self.llm = llm
        self.file_store = file_store
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> AnswerResult:
        results = list(request.retrieval.results)
        if not results:
            return self._no_evidence_answer(request)

        citations = build_citations(results)
        context_text = self._format_context(results, citations)
        images = self._collect_images(results, citations)

        messages = self._build_messages(request, context_text, images)
        started = time.perf_counter()

        try:
            response = self.llm.complete(
                messages,
                system=self._system_prompt(request),
                temperature=self.settings.llm.temperature,
                max_output_tokens=self.settings.llm.max_output_tokens,
            )
        except ProviderCapabilityError as exc:
            # The chain could not read the attached visuals. Rather than
            # silently answering without the evidence, retry text-only and say
            # so explicitly in the result.
            logger.warning("Multimodal generation unavailable: %s", exc)
            if not images:
                raise
            response = self.llm.complete(
                self._build_messages(request, context_text, []),
                system=self._system_prompt(request),
                temperature=self.settings.llm.temperature,
                max_output_tokens=self.settings.llm.max_output_tokens,
            )
            result = self._finish(response, citations, len(images))
            result.used_images = 0
            result.warnings.append(exc.user_message)
            return result

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "Generated answer in %.0f ms (%d contexts, %d images, provider=%s)",
            elapsed,
            len(results),
            len(images),
            response.provider or self.llm.name,
        )
        return self._finish(response, citations, len(images))

    # ------------------------------------------------------------------ #
    def _finish(self, response, citations: List[Citation], image_count: int) -> AnswerResult:
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
        self, results: Sequence[SearchResult], citations: Sequence[Citation]
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
        seen: set[str] = set()
        for citation, result in zip(citations, results):
            if len(out) >= budget:
                break
            chunk = result.chunk
            if chunk.visual is None:
                continue
            # Only send images that carry real information for this answer.
            if not (chunk.block_type.is_visual or chunk.source_kind == SourceKind.OCR):
                continue
            asset_id = chunk.visual.asset_id
            if asset_id in seen:
                continue

            data = self.file_store.get(asset_id)
            if not data:
                logger.debug("Visual asset %s is no longer available", asset_id[:8])
                continue
            seen.add(asset_id)
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
    "build_generator",
]
