"""Action parser for decomposing model output into reasoning, tool calls, and fallback messages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import json_repair


@dataclass
class ParsedAction:
    raw_text: str
    reasoning: str = ""
    is_tool_call: bool = False
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    is_truncated: bool = False
    termination_reason: str | None = None

    def to_env_action(self) -> str:
        """Serialize into a tau2 Gym env.step action string.

        Prefers functional tool-call form ``name(k=v, ...)`` which tau2's
        ``parse_action_string`` understands. Falls back to plain message text.
        """
        if self.is_tool_call and self.tool_name:
            return format_functional_tool_call(self.tool_name, self.tool_args)
        return self.message


def format_functional_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render ``name(k=v, ...)`` for tau2 ``parse_action_string``."""
    if not args:
        return f"{name}()"
    parts: list[str] = []
    for key, value in args.items():
        parts.append(f"{key}={_format_arg_value(value)}")
    return f"{name}({', '.join(parts)})"


def _format_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, dict)):
        return repr(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _split_top_level_args(raw_args: str) -> list[str]:
    """Split kwargs on top-level commas, respecting quotes and nested parens/brackets."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    escape = False
    for ch in raw_args:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue
        if not in_single and not in_double:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        # Undo the escaping applied by format_functional_tool_call so a parsed
        # call re-serializes byte-identically.
        return text[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_call_args(raw_args: str) -> dict[str, Any]:
    """Parses a call argument string into a kwargs dict.

    Accepts ``k=v`` pairs and a single JSON object (e.g. ``{"k": "v"}`` as
    produced by legacy training formats), so a trained call string round-trips
    without losing argument values.
    """
    args: dict[str, Any] = {}
    stripped = raw_args.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json_repair.loads(stripped)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    for part in _split_top_level_args(raw_args):
        if "=" in part:
            key, _, value = part.partition("=")
            args[key.strip()] = _parse_scalar(value)
        elif part.strip().startswith("{"):
            try:
                data = json_repair.loads(part)
            except Exception:
                continue
            if isinstance(data, dict):
                args.update(data)
    return args


def _extract_balanced_call(action_str: str) -> tuple[str, str] | None:
    """Find ``call:name(...)`` with balanced parentheses (supports nested parens in args)."""
    match = re.search(r"call:([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_str)
    if not match:
        return None
    fn_name = match.group(1)
    start = match.end()  # index after opening '('
    depth = 1
    in_single = False
    in_double = False
    escape = False
    i = start
    while i < len(action_str):
        ch = action_str[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\":
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return fn_name, action_str[start:i]
        i += 1
    return None


def parse_tool_string(action_str: str) -> tuple[str | None, dict[str, Any]]:
    """Extract tool name and arguments from call:/JSON/functional syntax."""
    # Pattern 1: call:func_name(kwargs) with balanced parens
    call = _extract_balanced_call(action_str)
    if call is not None:
        fn_name, raw_args = call
        return fn_name, _parse_call_args(raw_args)

    # Pattern 1b: bare functional form name(k=v) (tau2 native)
    bare = re.match(
        r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", action_str.strip(), flags=re.DOTALL
    )
    if bare and bare.group(1) not in {"if", "for", "while", "return"}:
        fn_name = bare.group(1)
        raw_args = bare.group(2).strip()
        # Re-extract with balance from original strip for nested parens
        balanced = _extract_balanced_call(f"call:{action_str.strip()}")
        if balanced is not None:
            fn_name, raw_args = balanced
        return fn_name, _parse_call_args(raw_args)

    # Pattern 2: JSON block ```json {"name": "...", "arguments": {...}} ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", action_str, flags=re.DOTALL)
    if json_match:
        try:
            data = json_repair.loads(json_match.group(1))
            if isinstance(data, dict) and data.get("name"):
                arguments = data.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                return str(data["name"]), arguments
        except Exception:
            pass

    # Pattern 3: Raw JSON object
    stripped = action_str.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json_repair.loads(stripped)
            if isinstance(data, dict) and data.get("name"):
                arguments = data.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                return str(data["name"]), arguments
        except Exception:
            pass

    return None, {}


def parse_model_output(output_text: str) -> ParsedAction:
    """Parses raw completion text into a structured ParsedAction object."""
    if not output_text or not output_text.strip():
        return ParsedAction(
            raw_text=output_text or "",
            is_truncated=True,
            termination_reason="empty_output",
        )

    # Check for unclosed <think> tag (truncation)
    if "<think>" in output_text and "</think>" not in output_text:
        reasoning = output_text.split("<think>", 1)[1].strip()
        return ParsedAction(
            raw_text=output_text,
            reasoning=reasoning,
            is_truncated=True,
            termination_reason="truncation",
        )

    # Generation can start INSIDE the think block when enable_thinking renders
    # an opening "<think>\n" before sampling; only the closing tag appears.
    if "</think>" in output_text and "<think>" not in output_text:
        reasoning_part, _, action_text = output_text.partition("</think>")
        reasoning = reasoning_part.strip()
        action_text = action_text.strip()
        if not action_text:
            return ParsedAction(
                raw_text=output_text,
                reasoning=reasoning,
                is_truncated=True,
                termination_reason="empty_action",
            )
        tool_name, tool_args = parse_tool_string(action_text)
        if tool_name:
            return ParsedAction(
                raw_text=output_text,
                reasoning=reasoning,
                is_tool_call=True,
                tool_name=tool_name,
                tool_args=tool_args,
                message=action_text,
            )
        return ParsedAction(
            raw_text=output_text,
            reasoning=reasoning,
            is_tool_call=False,
            tool_name=None,
            tool_args={},
            message=action_text,
        )

    # Extract thinking content
    reasoning = ""
    action_text = output_text
    think_match = re.search(r"<think>(.*?)</think>", output_text, flags=re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        action_text = output_text[think_match.end() :].strip()

    if not action_text:
        return ParsedAction(
            raw_text=output_text,
            reasoning=reasoning,
            is_truncated=True,
            termination_reason="empty_action",
        )

    tool_name, tool_args = parse_tool_string(action_text)

    if tool_name:
        return ParsedAction(
            raw_text=output_text,
            reasoning=reasoning,
            is_tool_call=True,
            tool_name=tool_name,
            tool_args=tool_args,
            message=action_text,
        )

    return ParsedAction(
        raw_text=output_text,
        reasoning=reasoning,
        is_tool_call=False,
        tool_name=None,
        tool_args={},
        message=action_text,
    )
