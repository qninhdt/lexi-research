"""Tests for action parsing, thinking extraction, tool arguments, and truncation handling."""

from tau_research.tau.action_parser import parse_model_output


def test_parse_tool_call_with_reasoning() -> None:
    raw = """<think>
Let's find the user order by customer ID.
</think>
call:get_order(order_id="12345")"""

    parsed = parse_model_output(raw)
    assert parsed.reasoning.strip() == "Let's find the user order by customer ID."
    assert parsed.is_tool_call is True
    assert parsed.tool_name == "get_order"
    assert parsed.tool_args == {"order_id": "12345"}
    assert parsed.is_truncated is False


def test_parse_json_tool_call() -> None:
    raw = """<think>Update user address.</think>
```json
{"name": "modify_address", "arguments": {"new_address": "123 Main St"}}
```"""

    parsed = parse_model_output(raw)
    assert parsed.is_tool_call is True
    assert parsed.tool_name == "modify_address"
    assert parsed.tool_args == {"new_address": "123 Main St"}


def test_parse_direct_message() -> None:
    raw = """<think>
No further tool needed, confirming cancellation to user.
</think>
Your order #12345 has been cancelled successfully."""

    parsed = parse_model_output(raw)
    assert parsed.is_tool_call is False
    assert parsed.message == "Your order #12345 has been cancelled successfully."
    assert parsed.tool_name is None


def test_parse_truncated_reasoning_fallback() -> None:
    # Model hit max tokens before closing </think>
    raw = "<think>\nLet me think about this step, first checking the inventory"

    parsed = parse_model_output(raw)
    assert parsed.is_truncated is True
    assert parsed.termination_reason == "truncation"
    assert parsed.is_tool_call is False


def test_parse_legacy_json_args_form_round_trips() -> None:
    """Legacy call:name({json}) targets must keep their argument values."""
    from tau_research.tau.action_parser import parse_model_output

    parsed = parse_model_output('call:get_order_details({"order_id":"#W4466964"})')
    assert parsed.is_tool_call
    assert parsed.tool_name == "get_order_details"
    assert parsed.tool_args == {"order_id": "#W4466964"}
    assert parsed.to_env_action() == "get_order_details(order_id='#W4466964')"


def test_parse_functional_args_with_quotes_and_spaces() -> None:
    from tau_research.tau.action_parser import parse_model_output

    parsed = parse_model_output("find_user_by_name(name='Juan Lopez', zip='12345')")
    assert parsed.tool_args == {"name": "Juan Lopez", "zip": "12345"}
