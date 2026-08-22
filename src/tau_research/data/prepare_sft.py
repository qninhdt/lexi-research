"""Preprocesses synthetic Tau-Bench trajectories into per-turn conversational SFT records."""

from __future__ import annotations

import re
from typing import Any


def strip_thinking_tags(text: str) -> str:
    """Removes <think>...</think> reasoning blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def sanitize_history_for_turn(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strips thinking blocks from previous assistant messages in the conversation history.

    Per Qwen3.5 guidelines, prior reasoning should not persist in multi-turn history.
    """
    sanitized: list[dict[str, Any]] = []
    for msg in history:
        msg_copy = dict(msg)
        if msg_copy.get("role") == "assistant":
            msg_copy["content"] = strip_thinking_tags(str(msg_copy.get("content", "")))
        sanitized.append(msg_copy)
    return sanitized


def format_assistant_message_content(msg: dict[str, Any]) -> str:
    """Formats assistant turn with reasoning and tool_calls/content."""
    parts = []
    reasoning = msg.get("reasoning_content")
    if reasoning:
        parts.append(f"<think>\n{reasoning.strip()}\n</think>")

    content = str(msg.get("content", "")).strip()
    if content:
        parts.append(content)

    tool_calls = msg.get("tool_calls", [])
    if tool_calls and isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "tool")
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                from tau_research.tau.action_parser import format_functional_tool_call

                parts.append(format_functional_tool_call(name, args))
            else:
                parts.append(f"call:{name}({args})")

    return "\n".join(parts)


def convert_trajectory_to_turn_examples(
    trajectory: list[dict[str, Any]],
    system_prompt: str = "You are a helpful customer service assistant for Retail operations.",
) -> list[dict[str, Any]]:
    """Converts a full multi-turn trajectory into N prompt-completion examples.

    Each example consists of:
      prompt: system + history up to turn k with previous reasoning stripped.
      completion: assistant response for turn k (including its <think> reasoning and action).
    """
    examples: list[dict[str, Any]] = []
    current_history: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    for item in trajectory:
        role = item.get("role")
        if role == "system":
            # Allow trajectory-provided system to override default.
            current_history[0] = {
                "role": "system",
                "content": str(item.get("content", system_prompt)),
            }
            continue
        if role == "assistant":
            formatted_content = format_assistant_message_content(item)
            prompt_history = sanitize_history_for_turn(current_history)
            examples.append(
                {
                    "prompt": prompt_history,
                    "completion": [{"role": "assistant", "content": formatted_content}],
                }
            )
            current_history.append({"role": "assistant", "content": formatted_content})
        else:
            current_history.append({"role": role, "content": str(item.get("content", ""))})

    return examples
