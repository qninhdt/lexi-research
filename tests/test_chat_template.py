"""Tests for Qwen3.5 chat template formatting and history sanitization."""

from tau_research.data.prepare_sft import sanitize_history_for_turn, strip_thinking_tags


def test_strip_thinking_tags() -> None:
    text_with_think = "<think>\nLet's check the database first.\n</think>\ncall:find_user(id=123)"
    cleaned = strip_thinking_tags(text_with_think)
    assert cleaned.strip() == "call:find_user(id=123)"
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned


def test_strip_thinking_tags_when_no_tags() -> None:
    plain_text = "I have updated your address to 123 Main St."
    assert strip_thinking_tags(plain_text) == plain_text


def test_sanitize_history_for_turn() -> None:
    raw_history = [
        {"role": "system", "content": "You are a customer service agent."},
        {"role": "user", "content": "I want to cancel order #456."},
        {
            "role": "assistant",
            "content": "<think>\nCheck order status.\n</think>\ncall:get_order(id=456)",
        },
        {"role": "tool", "content": '{"order_id": 456, "status": "pending"}'},
    ]

    sanitized = sanitize_history_for_turn(raw_history)
    assert len(sanitized) == 4
    # Previous assistant message must not have <think>
    assert sanitized[2]["content"] == "call:get_order(id=456)"
    assert "<think>" not in sanitized[2]["content"]
    # User and tool messages preserved
    assert sanitized[1]["content"] == "I want to cancel order #456."
    assert sanitized[3]["content"] == '{"order_id": 456, "status": "pending"}'
