"""Measure the teacher's tag mix against a human-annotated corpus.

The teacher writes its own learner sentences, so nothing in the pipeline checks
whether the *kinds* of error it produces resemble real ones. Stage A answers that
for free: it is the same 16-tag taxonomy applied by human annotators to real
learner writing, so the two share on disk are directly comparable.

Aggregate rates only — no corpus text — so the output is safe to publish
alongside the dataset, which is the point: a card claiming a distribution should
ship the measurement behind it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_TEACHER = "data/raw/raw_labels.parquet"
DEFAULT_HUMAN = "data/gec/train.parquet"
DEFAULT_OUT = "reports/tag-distribution-reference.json"


def tag_shares(path: str | Path) -> tuple[dict[str, float], int, int]:
    """Share of each tag among all tags, plus row and tag counts."""
    rows = pq.read_table(path, columns=["tags"]).to_pylist()
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in row.get("tags") or []:
            counts[str(tag)] += 1
    total = sum(counts.values())
    if not total:
        raise SystemExit(f"{path} carries no tags to compare")
    return {tag: count / total for tag, count in counts.items()}, len(rows), total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--human", default=DEFAULT_HUMAN)
    parser.add_argument("--human-name", default="W&I+LOCNESS train split")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    mine, teacher_rows, teacher_tags = tag_shares(args.teacher)
    human, human_rows, human_tags = tag_shares(args.human)

    payload = {
        "human_corpus": args.human_name,
        "human_rows": human_rows,
        "human_tags": human_tags,
        "teacher_rows": teacher_rows,
        "teacher_tags": teacher_tags,
        "tags": {
            tag: {
                "teacher_share": round(mine.get(tag, 0.0), 4),
                "human_share": round(human.get(tag, 0.0), 4),
            }
            for tag in sorted(set(mine) | set(human))
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"{'tag':7s} {'teacher':>9s} {'human':>9s} {'ratio':>8s}")
    for tag in sorted(payload["tags"], key=lambda t: -mine.get(t, 0.0)):
        a = mine.get(tag, 0.0)
        b = human.get(tag, 0.0)
        ratio = f"{a / b:.2f}x" if b else "n/a"
        print(f"{tag:7s} {a:8.1%} {b:8.1%} {ratio:>8s}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
