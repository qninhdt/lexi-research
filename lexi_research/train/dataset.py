"""Convert processed rows to the exact prompt/completion pairs a student learns."""

from __future__ import annotations

import json
from typing import Any

from lexi_research.teacher import render_grader_prompt
from lexi_research.teacher.schemas import SenseRef


def training_example(row: dict[str, Any], *, nonce: str = "training") -> dict[str, str]:
    """One prompt-parity-preserving SFT example; bands remain code-derived."""
    messages = render_grader_prompt(
        str(row["target"]),
        SenseRef(definition=str(row["definition"]), pos=str(row["pos"])),
        str(row["text"]),
        nonce=nonce,
    )
    completion = json.dumps(
        {key: row[key] for key in ("correction", "meaning", "feedback")},
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "system": messages[0]["content"],
        "user": messages[1]["content"],
        "completion": completion,
    }


__all__ = ["training_example"]
