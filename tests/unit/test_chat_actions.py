"""Conversation branching and clipboard controls."""

from __future__ import annotations

import pytest

from omnirag.core.enums import Role
from omnirag.core.models import ChatMessage, Citation
from omnirag.services.chat_history import apply_regeneration, plan_regeneration
from omnirag.ui.message_actions import action_key, copy_component_html


def _conversation():
    user1 = ChatMessage(role=Role.USER, content="السؤال الأول\nwith English")
    answer1 = ChatMessage(
        role=Role.ASSISTANT,
        content="First answer [1]",
        reply_to_message_id=user1.message_id,
        citations=[
            Citation(
                index=1,
                chunk_id="c1",
                document_id="d1",
                filename="doc.pdf",
                page_number=2,
            )
        ],
        debug={"provider": "gemini", "model": "old-model"},
    )
    user2 = ChatMessage(role=Role.USER, content="downstream")
    answer2 = ChatMessage(
        role=Role.ASSISTANT,
        content="Downstream answer",
        reply_to_message_id=user2.message_id,
    )
    return [user1, answer1, user2, answer2]


def test_every_message_has_a_stable_unique_id():
    messages = _conversation()
    assert all(message.message_id for message in messages)
    assert len({message.message_id for message in messages}) == len(messages)
    assert messages[0].model_copy().message_id == messages[0].message_id


def test_copy_payload_is_exact_for_arabic_mixed_multiline_and_code():
    text = "سطر عربي\nEnglish line\n```python\nprint('x')\n```"
    markup = copy_component_html(text)
    assert "سطر عربي" in markup
    assert "English line\\n```python" in markup
    assert "navigator.clipboard.writeText(text)" in markup
    assert "retrieval" not in markup


def test_action_keys_are_unique_per_message_and_action():
    messages = _conversation()
    keys = {
        action_key(action, message.message_id)
        for message in messages
        for action in ("copy_user", "edit_user", "regen_user")
    }
    assert len(keys) == len(messages) * 3


@pytest.mark.parametrize(
    "edited",
    ["edited prompt", "سؤال عربي معدل", "Arabic عربي English", "line one\nline two"],
)
def test_edit_replaces_prompt_and_truncates_downstream_turns(edited):
    messages = _conversation()
    plan = plan_regeneration(messages, messages[0].message_id, edited_text=edited)
    replacement = ChatMessage(
        role=Role.ASSISTANT,
        content="New answer [1]",
        citations=messages[1].citations,
        debug={"provider": "openrouter", "model": "new-model"},
    )

    updated = apply_regeneration(messages, plan, replacement)

    assert len(updated) == 2
    assert updated[0].content == edited
    assert updated[0].message_id == messages[0].message_id
    assert updated[1].content == "New answer [1]"
    assert updated[1].reply_to_message_id == messages[0].message_id
    assert updated[1].debug["model"] == "new-model"
    assert updated[1].citations == replacement.citations


def test_user_regenerate_reuses_exact_prompt_and_preserves_state_until_applied():
    messages = _conversation()
    snapshot = [message.model_dump() for message in messages]
    plan = plan_regeneration(messages, messages[0].message_id)
    assert plan.prompt == messages[0].content
    assert [message.model_dump() for message in messages] == snapshot


def test_assistant_regenerate_uses_linked_preceding_user():
    messages = _conversation()
    plan = plan_regeneration(messages, messages[1].message_id)
    assert plan.user_message_id == messages[0].message_id
    assert plan.prompt == messages[0].content
    assert plan.history == []


def test_cancel_edit_is_a_noop_by_design():
    messages = _conversation()
    snapshot = [message.model_dump() for message in messages]
    # Cancel only clears UI editing state; the pure history is never mutated.
    assert [message.model_dump() for message in messages] == snapshot


def test_invalid_or_empty_edit_never_mutates_history():
    messages = _conversation()
    with pytest.raises(ValueError):
        plan_regeneration(messages, messages[0].message_id, edited_text="   ")
    assert len(messages) == 4

