"""Main panel: welcome state, conversation, answers, citations."""

from __future__ import annotations

import html
from typing import List, Optional

import streamlit as st

from omnirag.core.enums import Role
from omnirag.core.models import ChatMessage
from omnirag.services.chat_service import ChatRequest
from omnirag.services.chat_history import apply_regeneration, plan_regeneration
from omnirag.ui import state
from omnirag.ui.message_actions import action_key, render_copy_button
from omnirag.ui.components import (
    caption,
    provider_badge,
    render_error,
    rtl_markdown,
)
from omnirag.ui.sources import render_sources
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

FEATURES = [
    ("📄", "Any document", "PDF, scans, Word, PowerPoint, images, Markdown and text."),
    ("👁️", "Sees the visuals", "Charts, diagrams, tables and handwriting — not just OCR."),
    ("🌍", "Arabic & English", "Ask in one language, search documents written in the other."),
    ("🔎", "Cited answers", "Every claim links back to a file, page and passage."),
]

FALLBACK_PROMPTS = [
    "Summarize these documents.",
    "What are the key findings?",
    "Explain the charts and diagrams.",
    "لخّص هذه المستندات بالعربية.",
]


def render() -> None:
    """Draw the main panel."""
    history = state.messages()
    documents = state.ready_documents()
    action_error = state.take_action_error()
    if action_error:
        st.error(action_error)

    if not history:
        _render_welcome(bool(documents))

    for index, message in enumerate(history):
        _render_message(message, index)

    _handle_input(bool(documents))


# --------------------------------------------------------------------------- #
def _render_welcome(has_documents: bool) -> None:
    st.markdown(
        """
        <div class="omni-hero">
          <h1>Chat with your documents</h1>
          <p>Upload PDFs, scans, Word files, slide decks or images — then ask
          anything. OmniRAG reads the text <em>and</em> the visuals, and cites
          every answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_documents:
        columns = st.columns(4, gap="small")
        for column, (icon, title, body) in zip(columns, FEATURES):
            with column:
                st.markdown(
                    f"""
                    <div class="omni-feature">
                      <div class="omni-feature-title">{icon} {html.escape(title)}</div>
                      <div class="omni-feature-body">{html.escape(body)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")
        st.info("Upload a document in the sidebar to get started.", icon="👈")
        return

    _render_example_prompts()


def _render_example_prompts() -> None:
    try:
        prompts = state.chat_service().suggested_prompts(state.session_id())
    except Exception as exc:
        logger.debug("Could not build suggested prompts: %s", exc)
        prompts = []
    prompts = prompts or FALLBACK_PROMPTS

    st.markdown("###### Try asking")
    columns = st.columns(2, gap="small")
    for index, prompt in enumerate(prompts[:6]):
        with columns[index % 2]:
            if st.button(prompt, key=f"example_{index}", use_container_width=True):
                state.set_pending_prompt(prompt)
                st.rerun()


# --------------------------------------------------------------------------- #
def _render_message(message: ChatMessage, index: int) -> None:
    avatar = "🧑" if message.role == Role.USER else "🔷"
    with st.chat_message(message.role.value, avatar=avatar):
        if message.role == Role.USER and state.editing_message_id() == message.message_id:
            _render_editor(message)
            return
        if message.error:
            render_error(message.content)
        else:
            rtl_markdown(message.content)

        if message.role == Role.USER:
            _render_user_actions(message)
            return

        if message.citations:
            render_sources(
                message.citations,
                used_indices=_used_indices(message),
                file_store=state.engine().file_store,
                key_prefix=f"m{index}",
                retrieval=message.retrieval,
            )

        _render_message_footer(message)
        _render_assistant_actions(message)


def _render_user_actions(message: ChatMessage) -> None:
    columns = st.columns([0.34, 0.18, 0.22, 0.26], gap="small")
    with columns[0]:
        render_copy_button(
            text=message.content, key=action_key("copy_user", message.message_id)
        )
    with columns[1]:
        if st.button(
            "Edit",
            key=action_key("edit_user", message.message_id),
            help="Edit this prompt",
        ):
            state.set_editing_message(message.message_id)
            st.rerun()
    with columns[2]:
        if st.button(
            "Regenerate",
            key=action_key("regen_user", message.message_id),
            help="Resend this prompt and replace later turns",
        ):
            _regenerate(message.message_id)


def _render_assistant_actions(message: ChatMessage) -> None:
    columns = st.columns([0.34, 0.24, 0.42], gap="small")
    with columns[0]:
        render_copy_button(
            text=message.content,
            key=action_key("copy_assistant", message.message_id),
        )
    with columns[1]:
        if st.button(
            "Regenerate",
            key=action_key("regen_assistant", message.message_id),
            help="Regenerate this answer from its preceding prompt",
        ):
            _regenerate(message.message_id)


def _render_editor(message: ChatMessage) -> None:
    edited = st.text_area(
        "Edit message",
        value=message.content,
        key=action_key("edit_text", message.message_id),
        label_visibility="collapsed",
    )
    cancel, save, _ = st.columns([0.18, 0.32, 0.5], gap="small")
    with cancel:
        if st.button(
            "Cancel",
            key=action_key("cancel_edit", message.message_id),
            help="Cancel editing",
        ):
            state.set_editing_message(None)
            st.rerun()
    with save:
        if st.button(
            "Save & Regenerate",
            key=action_key("save_edit", message.message_id),
            help="Save this prompt and regenerate from this point",
        ):
            _regenerate(message.message_id, edited_text=edited)


def _regenerate(message_id: str, *, edited_text: Optional[str] = None) -> None:
    """Run a complete new RAG turn, committing history only on success."""
    current = list(state.messages())
    try:
        plan = plan_regeneration(current, message_id, edited_text=edited_text)
        with st.spinner("Searching your documents again…"):
            answer = state.chat_service().answer(
                ChatRequest(
                    question=plan.prompt,
                    session_id=state.session_id(),
                    document_ids=state.selected_document_ids(),
                    history=plan.history,
                    user_message_id=plan.user_message_id,
                )
            )
        if answer.error:
            state.set_action_error(answer.content)
        else:
            state.replace_messages(apply_regeneration(current, plan, answer))
            state.set_editing_message(None)
    except Exception as exc:  # noqa: BLE001 - preserve the valid old turn
        logger.exception("Message regeneration failed")
        state.set_action_error(f"Could not regenerate this message ({type(exc).__name__}).")
    st.rerun()


def _render_message_footer(message: ChatMessage) -> None:
    debug = message.debug or {}
    warnings: List[str] = list(debug.get("warnings") or [])
    for warning in warnings:
        st.warning(warning, icon="⚠️")

    if not debug:
        return

    columns = st.columns([0.45, 0.55])
    with columns[0]:
        provider_badge(
            str(debug.get("provider", "")),
            str(debug.get("model", "")),
            fallback_used=_used_fallback(debug),
        )
    with columns[1]:
        bits = []
        if debug.get("contexts"):
            bits.append(f"{debug['contexts']} sources")
        if debug.get("images_sent"):
            bits.append(f"{debug['images_sent']} images analysed")
        if debug.get("elapsed_s"):
            bits.append(f"{debug['elapsed_s']}s")
        scope = str(debug.get("query_scope", "FOCUSED"))
        if scope != "FOCUSED":
            bits.append(f"Retrieval mode: {scope.title().replace('_', ' ')}")
            covered = debug.get("pages_covered")
            total = debug.get("total_pages")
            if covered and total:
                bits.append(f"Pages covered: {covered}/{total}")
        if bits:
            caption(" · ".join(bits))

    if state.settings().debug_panels:
        with st.expander("🔧 Debug", expanded=False):
            st.json(debug, expanded=False)


def _used_fallback(debug: dict) -> bool:
    attempts = debug.get("provider_attempts") or []
    return len(attempts) > 1


def _used_indices(message: ChatMessage) -> set:
    from omnirag.rag.citations import parse_markers

    valid = {c.index for c in message.citations}
    return {i for i in parse_markers(message.content) if i in valid}


# --------------------------------------------------------------------------- #
def _handle_input(has_documents: bool) -> None:
    settings = state.settings()
    placeholder = (
        "Ask anything about your documents…"
        if has_documents
        else "Upload a document first…"
    )

    typed = st.chat_input(placeholder, disabled=not has_documents)
    prompt = typed or state.take_pending_prompt()

    if not prompt:
        return
    if not has_documents:
        return
    if not settings.llm.is_configured:
        st.error(
            "No language-model provider is configured. Add `GEMINI_API_KEY` "
            "(primary) or `OPENROUTER_API_KEY` (fallback) to your Streamlit "
            "secrets, then reload."
        )
        return

    user_message = ChatMessage(role=Role.USER, content=prompt)
    state.add_message(user_message)

    with st.chat_message("user", avatar="🧑"):
        rtl_markdown(prompt)

    with st.chat_message("assistant", avatar="🔷"):
        with st.spinner("Searching your documents…"):
            answer = _answer(prompt, user_message.message_id)
    state.add_message(answer)
    st.rerun()


def _answer(prompt: str, user_message_id: Optional[str] = None) -> ChatMessage:
    service = state.chat_service()
    history = [m for m in state.messages() if m.role in (Role.USER, Role.ASSISTANT)][:-1]
    try:
        return service.answer(
            ChatRequest(
                question=prompt,
                session_id=state.session_id(),
                document_ids=state.selected_document_ids(),
                history=history,
                user_message_id=user_message_id,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the UI must never crash
        logger.exception("Unhandled error while answering")
        message = f"Something went wrong while answering ({type(exc).__name__})."
        return ChatMessage(role=Role.ASSISTANT, content=message, error=message)


__all__ = ["render"]
