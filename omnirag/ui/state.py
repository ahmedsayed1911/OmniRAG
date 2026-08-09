"""Streamlit session-state management.

Owns the browser-session lifecycle and the one thing the engine cannot do for
itself: creating and remembering the ``session_id`` namespace that every
document, vector and query is scoped to.

The engine itself is cached with ``st.cache_resource`` (shared across all
visitors, as intended — it holds only stateless clients), while everything
user-specific lives in ``st.session_state`` and is keyed by that session id.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from omnirag.config.settings import AppSettings, get_settings
from omnirag.core.models import ChatMessage, DocumentSummary
from omnirag.services.chat_service import ChatService
from omnirag.services.engine import OmniRAGEngine, get_engine
from omnirag.services.ingestion_service import IngestionService
from omnirag.storage.sessions import new_session_id
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

SESSION_KEY = "omnirag_session_id"
MESSAGES_KEY = "omnirag_messages"
SELECTED_KEY = "omnirag_selected_documents"
PROCESSED_KEY = "omnirag_processed_uploads"
PENDING_KEY = "omnirag_pending_prompt"
EDITING_KEY = "omnirag_editing_message"
ACTION_ERROR_KEY = "omnirag_message_action_error"
GENERATION_KEY = "omnirag_generation"


def _cached_engine() -> OmniRAGEngine:
    """Return the process engine after checking the current settings.

    ``get_engine`` already owns the process-wide cache.  A second Streamlit
    resource cache kept stale settings and provider instances alive across hot
    reloads, so every access now passes the freshly bootstrapped settings
    through the engine's configuration equality check.
    """
    return get_engine(get_settings())


def engine() -> OmniRAGEngine:
    return _cached_engine()


def settings() -> AppSettings:
    return get_settings()


def ingestion_service() -> IngestionService:
    return IngestionService(engine())


def chat_service() -> ChatService:
    return ChatService(engine())


# --------------------------------------------------------------------------- #
def init_state() -> str:
    """Ensure the session namespace and containers exist. Returns the id."""
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = new_session_id()
        logger.info("Started a new browser session namespace")
    st.session_state.setdefault(MESSAGES_KEY, [])
    st.session_state.setdefault(SELECTED_KEY, None)  # None == all documents
    st.session_state.setdefault(PROCESSED_KEY, set())
    st.session_state.setdefault(PENDING_KEY, None)
    st.session_state.setdefault(EDITING_KEY, None)
    st.session_state.setdefault(ACTION_ERROR_KEY, None)
    st.session_state.setdefault(GENERATION_KEY, {"status": "idle"})
    return st.session_state[SESSION_KEY]


def session_id() -> str:
    return init_state()


# -- messages --------------------------------------------------------------- #
def messages() -> List[ChatMessage]:
    return st.session_state.get(MESSAGES_KEY, [])


def add_message(message: ChatMessage) -> ChatMessage:
    st.session_state.setdefault(MESSAGES_KEY, []).append(message)
    stored = st.session_state[MESSAGES_KEY][-1]
    if get_settings().debug_generation:
        logger.info(
            "Generation lifecycle stage=session_state generation_id=%s message_id=%s "
            "stored_chars=%d",
            stored.debug.get("generation_id", "") if stored.debug else "",
            stored.message_id,
            len(stored.content),
        )
    return stored


def clear_messages() -> None:
    st.session_state[MESSAGES_KEY] = []


def replace_messages(messages: List[ChatMessage]) -> None:
    st.session_state[MESSAGES_KEY] = list(messages)


def editing_message_id() -> Optional[str]:
    return st.session_state.get(EDITING_KEY)


def set_editing_message(message_id: Optional[str]) -> None:
    st.session_state[EDITING_KEY] = message_id


def set_action_error(message: Optional[str]) -> None:
    st.session_state[ACTION_ERROR_KEY] = message


def take_action_error() -> Optional[str]:
    message = st.session_state.get(ACTION_ERROR_KEY)
    st.session_state[ACTION_ERROR_KEY] = None
    return message


def begin_generation(generation_id: str, user_message_id: str) -> None:
    st.session_state[GENERATION_KEY] = {
        "status": "generating",
        "generation_id": generation_id,
        "user_message_id": user_message_id,
    }


def complete_generation() -> None:
    current = dict(st.session_state.get(GENERATION_KEY) or {})
    current["status"] = "complete"
    st.session_state[GENERATION_KEY] = current


def recover_interrupted_generation() -> Optional[Dict[str, Any]]:
    """Mark an interrupted rerun without ever promoting partial text."""
    current = dict(st.session_state.get(GENERATION_KEY) or {})
    if current.get("status") != "generating":
        if current.get("status") == "complete":
            st.session_state[GENERATION_KEY] = {"status": "idle"}
        return None
    current["status"] = "interrupted"
    st.session_state[GENERATION_KEY] = current
    return current


# -- documents -------------------------------------------------------------- #
def documents() -> List[DocumentSummary]:
    return engine().registry.list(session_id())


def ready_documents() -> List[DocumentSummary]:
    return engine().registry.ready_documents(session_id())


def selected_document_ids() -> Optional[List[str]]:
    """Documents participating in the conversation (``None`` == all of them)."""
    selected = st.session_state.get(SELECTED_KEY)
    available = {d.document_id for d in ready_documents()}
    if selected is None:
        return None
    kept = [d for d in selected if d in available]
    return kept or None


def set_selected_documents(document_ids: Optional[List[str]]) -> None:
    st.session_state[SELECTED_KEY] = document_ids


# -- uploads ---------------------------------------------------------------- #
def already_processed(key: str) -> bool:
    return key in st.session_state.get(PROCESSED_KEY, set())


def mark_processed(key: str) -> None:
    st.session_state.setdefault(PROCESSED_KEY, set()).add(key)


def forget_processed() -> None:
    st.session_state[PROCESSED_KEY] = set()


# -- pending prompt (example-prompt buttons) -------------------------------- #
def set_pending_prompt(prompt: Optional[str]) -> None:
    st.session_state[PENDING_KEY] = prompt


def take_pending_prompt() -> Optional[str]:
    prompt = st.session_state.get(PENDING_KEY)
    st.session_state[PENDING_KEY] = None
    return prompt


# -- lifecycle -------------------------------------------------------------- #
def new_chat() -> None:
    """Clear the conversation but keep the indexed documents."""
    clear_messages()


def reset_session() -> None:
    """Remove every document, vector and asset, then start a fresh namespace."""
    current = st.session_state.get(SESSION_KEY)
    if current:
        try:
            engine().clear_session(current)
        except Exception as exc:
            logger.warning("Session cleanup failed: %s", exc)
    st.session_state[SESSION_KEY] = new_session_id()
    clear_messages()
    forget_processed()
    set_selected_documents(None)


__all__ = [
    "add_message",
    "already_processed",
    "begin_generation",
    "chat_service",
    "clear_messages",
    "complete_generation",
    "documents",
    "engine",
    "forget_processed",
    "ingestion_service",
    "init_state",
    "mark_processed",
    "messages",
    "new_chat",
    "ready_documents",
    "recover_interrupted_generation",
    "replace_messages",
    "reset_session",
    "selected_document_ids",
    "session_id",
    "set_pending_prompt",
    "editing_message_id",
    "set_editing_message",
    "set_action_error",
    "take_action_error",
    "set_selected_documents",
    "settings",
    "take_pending_prompt",
]
