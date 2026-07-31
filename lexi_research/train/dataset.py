"""Convert processed rows to the exact prompt/completion pairs a student learns.

The tokenised form lives in `collate`, which is what the trainer consumes. This
module is the untokenised view of the same example — used where a tokenizer is
neither available nor wanted, such as inspecting a row or building a prompt for
an external engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .collate import completion_text, training_messages


def training_example(row: Mapping[str, Any], *, nonce: str | None = None) -> dict[str, str]:
    """One prompt-parity-preserving SFT example; bands remain code-derived.

    `nonce` is left unset by default so each call draws a fresh delimiter, the
    distribution inference draws from. A constant would teach one literal token.
    """
    messages = training_messages(row, nonce=nonce)
    return {
        "system": messages[0]["content"],
        "user": messages[1]["content"],
        "completion": completion_text(row),
    }


__all__ = ["training_example"]
