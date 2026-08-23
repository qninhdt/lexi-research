"""Generic ``key.subkey=value`` overrides layered onto parsed YAML configs."""

from __future__ import annotations

from typing import Any


def coerce_value(raw: str) -> Any:
    """Converts a CLI string into bool / int / float / None / str."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Applies ``section.key=value`` pairs onto a parsed YAML dict, in place.

    Each override splits on the FIRST ``=``; the key path splits on dots and
    creates missing intermediate sections. Later overrides win.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override expects section.key=value, got {item!r}")
        raw_key, _, raw_value = item.partition("=")
        parts = [p for p in raw_key.strip().split(".") if p]
        if not parts:
            raise ValueError(f"--override has empty key path: {item!r}")
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = coerce_value(raw_value)
    return data
