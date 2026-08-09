"""Conversational question answering.

One call does retrieval → generation → citation verification, and returns a
fully-formed :class:`ChatMessage` ready to render. The service owns the rule
that documents — not the conversation — are the source of truth: history is
passed to the model as *intent* context only, and every factual claim must come
back cited to a retrieved source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from omnirag.config.settings import AppSettings
from omnirag.core.enums import Role
from omnirag.core.exceptions import (
    MissingCredentialError,
    OmniRAGError,
    ProviderCapabilityError,
    ProviderError,
)
from omnirag.core.models import ChatMessage, RetrievalResult
from omnirag.rag.generation import AnswerGenerator, GenerationRequest
from omnirag.rag.query_rewrite import parse_query
from omnirag.rag.retrieval import RetrievalRequest, Retriever
from omnirag.services.engine import OmniRAGEngine
from omnirag.storage.sessions import require_session_id
from omnirag.utils.logging import get_logger
from omnirag.utils.text import estimate_tokens

logger = get_logger(__name__)


@dataclass
class ChatRequest:
    question: str
    session_id: str
    document_ids: Optional[Sequence[str]] = None
    history: Sequence[ChatMessage] = field(default_factory=list)
    user_message_id: Optional[str] = None
    generation_id: str = ""


class ChatService:
    """Answers questions over a session's indexed documents."""

    def __init__(self, engine: OmniRAGEngine):
        self.engine = engine

    @property
    def settings(self) -> AppSettings:
        return self.engine.settings

    # ------------------------------------------------------------------ #
    def answer(self, request: ChatRequest) -> ChatMessage:
        """Run the full RAG turn. Errors become a user-readable message."""
        session_id = require_session_id(request.session_id)
        question = (request.question or "").strip()
        started = time.perf_counter()

        if not question:
            return _error_message("Please enter a question.")

        if not self.settings.llm.is_configured:
            return _error_message(
                "No language-model provider is configured. Add `GEMINI_API_KEY` "
                "(primary) or `OPENROUTER_API_KEY` (fallback) to your secrets, "
                "then reload the app."
            )

        try:
            retrieval = self._retrieve(request, session_id)
        except OmniRAGError as exc:
            logger.warning("Retrieval failed: %s", exc.detail or exc)
            return _error_message(exc.user_message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected retrieval failure")
            return _error_message(
                f"Search over your documents failed ({type(exc).__name__})."
            )

        try:
            generation_started = time.perf_counter()
            generator = self._generator()
            plan = parse_query(question, request.history)
            result = generator.generate(
                GenerationRequest(
                    question=question,
                    retrieval=retrieval,
                    session_id=session_id,
                    history=request.history,
                    plan=plan,
                    answer_language=plan.answer_language,
                    generation_id=request.generation_id,
                )
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
        except MissingCredentialError as exc:
            return _error_message(exc.user_message)
        except ProviderCapabilityError as exc:
            return _error_message(exc.user_message)
        except ProviderError as exc:
            logger.warning("Generation failed: %s", exc.detail or exc)
            return _error_message(exc.user_message, retrieval=retrieval)
        except OmniRAGError as exc:
            return _error_message(exc.user_message, retrieval=retrieval)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected generation failure")
            return _error_message(
                f"The answer could not be generated ({type(exc).__name__}).",
                retrieval=retrieval,
            )

        elapsed = time.perf_counter() - started
        message = ChatMessage(
            role=Role.ASSISTANT,
            content=result.answer,
            citations=result.citations,
            retrieval=retrieval,
            used_documents=sorted({c.document_id for c in result.citations}),
            debug={
                "model": result.model,
                "provider": self._last_provider(),
                "images_sent": result.used_images,
                "contexts": len(retrieval.results),
                "strategy": retrieval.strategy,
                "reranked": retrieval.reranked,
                "insufficient_evidence": result.insufficient_evidence,
                "elapsed_s": round(elapsed, 2),
                "timings_ms": retrieval.timings_ms,
                "warnings": result.warnings,
                "usage": result.usage,
                "provider_attempts": self._last_attempts(),
                "query_scope": retrieval.query_scope,
                "pages_covered": retrieval.unique_pages,
                "total_pages": retrieval.total_pages,
                "candidate_count": retrieval.candidate_count,
                "structured_matches": retrieval.structured_matches,
                "completeness_pass": retrieval.completeness_pass,
                "generation_ms": round(generation_ms, 1),
                "requested_max_output_tokens": (
                    max(
                        self.settings.llm.max_output_tokens,
                        self.settings.llm.exhaustive_max_output_tokens,
                    )
                    if retrieval.query_scope != "FOCUSED"
                    else self.settings.llm.max_output_tokens
                ),
                "finish_reason": result.finish_reason,
                "continued": result.continued,
                "returned_chars": len(result.answer),
                "returned_token_estimate": estimate_tokens(result.answer),
                "generation_id": result.generation_id,
                **dict(result.generation_debug or {}),
            },
            reply_to_message_id=request.user_message_id,
        )
        logger.info(
            "Stored assistant message query_scope=%s provider=%s model=%s "
            "finish_reason=%s chat_service_chars=%d stored_token_estimate=%d "
            "continued=%s generation_id=%s message_id=%s",
            retrieval.query_scope,
            message.debug.get("provider", ""),
            message.debug.get("model", ""),
            result.finish_reason or "unspecified",
            len(message.content),
            estimate_tokens(message.content),
            result.continued,
            result.generation_id,
            message.message_id,
        )
        return message

    # ------------------------------------------------------------------ #
    def _retrieve(self, request: ChatRequest, session_id: str) -> RetrievalResult:
        retriever = Retriever(
            vector_store=self.engine.vector_store,
            embeddings=self.engine.embeddings,
            reranker=self._reranker(),
            llm=self.engine.llm if self.settings.retrieval.query_rewrite else None,
            settings=self.settings,
        )
        return retriever.retrieve(
            RetrievalRequest(
                query=request.question,
                session_id=session_id,
                document_ids=list(request.document_ids) if request.document_ids else None,
                history=request.history,
            )
        )

    def _reranker(self):
        try:
            from omnirag.providers.rerank.factory import get_reranker

            return get_reranker(self.settings)
        except Exception as exc:
            logger.info("Reranking unavailable: %s", exc)
            return None

    def _generator(self) -> AnswerGenerator:
        llm = self.engine.llm
        if llm is None:
            raise MissingCredentialError(
                "GEMINI_API_KEY or OPENROUTER_API_KEY", "answer generation"
            )
        return AnswerGenerator(
            llm, file_store=self.engine.file_store, settings=self.settings
        )

    def _last_provider(self) -> str:
        stats = getattr(self.engine.llm, "stats", None)
        if stats is None:
            return getattr(self.engine.llm, "name", "")
        return stats.last_provider or getattr(self.engine.llm, "name", "")

    def _last_attempts(self) -> List[str]:
        stats = getattr(self.engine.llm, "stats", None)
        return list(stats.last_attempts) if stats is not None else []

    # ------------------------------------------------------------------ #
    def suggested_prompts(self, session_id: str) -> List[str]:
        """Starter prompts, adapted to what is actually in the session."""
        documents = self.engine.registry.ready_documents(session_id)
        if not documents:
            return []

        prompts: List[str] = ["Summarize these documents."]
        if len(documents) > 1:
            prompts.append(
                f"Compare {documents[0].filename} with {documents[1].filename}."
            )
        if any(d.visual_block_count for d in documents):
            prompts.append("Explain the charts and diagrams in these documents.")
        if any(d.table_count for d in documents):
            prompts.append("What do the tables show?")
        if any(d.page_count and d.page_count > 3 for d in documents):
            prompts.append("Explain page 3.")

        from omnirag.core.enums import Language

        if any(d.language in (Language.ARABIC, Language.MIXED) for d in documents):
            prompts.append("لخّص هذه المستندات بالعربية.")
        else:
            prompts.append("Answer in Arabic: what are the key findings?")
        return prompts[:6]


def _error_message(text: str, retrieval: Optional[RetrievalResult] = None) -> ChatMessage:
    return ChatMessage(
        role=Role.ASSISTANT,
        content=text,
        error=text,
        retrieval=retrieval,
    )


__all__ = ["ChatRequest", "ChatService"]
