"""Emit `lexi_colab.ipynb` from the percent-format source, deterministically.

The notebook is generated rather than edited because a hand-edited notebook
drifts from the CLI silently: a flag renamed in the repo leaves a cell that still
passes the old one, and nobody notices until the run fails on Colab. Generating
it means the committed `.ipynb` either matches its source or a test fails.

Deterministic on purpose — no timestamps, no execution counts, no outputs, no
generated ids. Running this twice on an unchanged source produces no diff, which
is what makes the "committed matches source" test meaningful.

    uv run python ops/build-notebook.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "notebooks" / "lexi_colab.py"
NOTEBOOK = SOURCE.with_suffix(".ipynb")

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"

#: Pinned rather than read from the environment: the kernel a Colab notebook
#: declares must not depend on the machine that generated the file.
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Percent-format source into `(kind, lines)` pairs."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    body: list[str] = []

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped == MARKDOWN_MARKER or stripped == CELL_MARKER:
            if body:
                cells.append((kind, body))
            kind = "markdown" if stripped == MARKDOWN_MARKER else "code"
            body = []
            continue
        body.append(line)
    if body:
        cells.append((kind, body))
    return cells


def _clean(kind: str, lines: list[str]) -> list[str]:
    """Strip the comment prefix from markdown, and trim blank edges."""
    if kind == "markdown":
        lines = [line[2:] if line.startswith("# ") else line.removeprefix("#") for line in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _source(lines: list[str]) -> list[str]:
    """nbformat stores source as lines with trailing newlines, except the last."""
    return [line + "\n" for line in lines[:-1]] + lines[-1:] if lines else []


def build(text: str) -> dict[str, object]:
    cells = []
    for kind, lines in split_cells(text):
        body = _clean(kind, lines)
        if not body:
            continue
        cell: dict[str, object] = {
            "cell_type": kind,
            "metadata": {},
            "source": _source(body),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def render(text: str) -> str:
    return json.dumps(build(text), indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed notebook is stale"
    )
    args = parser.parse_args(argv)

    rendered = render(SOURCE.read_text(encoding="utf-8"))
    if args.check:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        if current != rendered:
            print(f"{NOTEBOOK} is stale; run `make -f ops/Makefile notebook`", file=sys.stderr)
            return 1
        print(f"{NOTEBOOK} matches its source")
        return 0

    NOTEBOOK.write_text(rendered, encoding="utf-8")
    print(f"wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
