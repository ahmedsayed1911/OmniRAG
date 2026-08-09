"""Pure conversation edits used by the Streamlit message actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from omnirag.core.enums import Role
from omnirag.core.models import ChatMessage


@dataclass(frozen=True)
class RegenerationPlan:
    user_index: int
    prompt: str
    history: List[ChatMessage]
    user_message_id: str


def plan_regeneration(
    messages: Sequence[ChatMessage], message_id: str, *, edited_text: str | None = None
) -> RegenerationPlan:
    """Resolve either a user or assistant action to its source user turn."""
    target_index = next(
        (index for index, message in enumerate(messages) if message.message_id == message_id),
        -1,
    )
    if target_index < 0:
        raise ValueError("message no longer exists")

    target = messages[target_index]
    if target.role == Role.USER:
        user_index = target_index
    elif target.role == Role.ASSISTANT:
        user_index = _preceding_user_index(messages, target_index, target.reply_to_message_id)
    else:
        raise ValueError("only user and assistant messages can be regenerated")

    user = messages[user_index]
    prompt = user.content if edited_text is None else edited_text.strip()
    if not prompt:
        raise ValueError("edited prompt cannot be empty")
    return RegenerationPlan(
        user_index=user_index,
        prompt=prompt,
        history=list(messages[:user_index]),
        user_message_id=user.message_id,
    )


def apply_regeneration(
    messages: Sequence[ChatMessage],
    plan: RegenerationPlan,
    answer: ChatMessage,
) -> List[ChatMessage]:
    """Atomically replace the selected turn and truncate dependent turns."""
    original = messages[plan.user_index]
    user = original.model_copy(update={"content": plan.prompt})
    linked_answer = answer.model_copy(update={"reply_to_message_id": user.message_id})
    return [*plan.history, user, linked_answer]


def _preceding_user_index(
    messages: Sequence[ChatMessage], assistant_index: int, reply_to: str | None
) -> int:
    if reply_to:
        linked = next(
            (
                index
                for index in range(assistant_index - 1, -1, -1)
                if messages[index].message_id == reply_to
                and messages[index].role == Role.USER
            ),
            -1,
        )
        if linked >= 0:
            return linked
    for index in range(assistant_index - 1, -1, -1):
        if messages[index].role == Role.USER:
            return index
    raise ValueError("assistant message has no preceding user prompt")


__all__ = ["RegenerationPlan", "apply_regeneration", "plan_regeneration"]
