"""Backwards-compatible entry point. The surface is `lexi train sft`.

Kept so `python -m lexi_research.train.cli` still works from a shell that has no
console script on its PATH; it forwards, so there is one argument parser rather
than two that can drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from lexi_research.cli import main as lexi_main


def main(argv: Sequence[str] | None = None) -> int:
    return lexi_main(["train", "sft", *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main())
