"""Deterministic, leakage-safe dataset splitting by normalized target word."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SplitResult:
    rows: tuple[dict[str, Any], ...]
    contamination: dict[str, int]


def _bucket(group: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{group}".encode()).digest()[:8], "big") % 10_000


def assign_split(group: str, *, seed: int, train: float = 0.8, val: float = 0.1) -> str:
    """Assign an entire target group reproducibly; never use row order."""
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("train and val must be positive and leave a test share")
    point = _bucket(group, seed) / 10_000
    return "train" if point < train else "val" if point < train + val else "test"


def split_rows(rows: Sequence[dict[str, Any]], *, seed: int, version: str) -> SplitResult:
    """Tag rows with a split and quantify repeated texts crossing splits."""
    out: list[dict[str, Any]] = []
    text_splits: dict[str, set[str]] = {}
    for row in rows:
        group = row.get("target_norm") or row.get("lemma_key")
        if not isinstance(group, str) or not group:
            raise ValueError("every row needs target_norm (or lemma_key) for a grouped split")
        split = assign_split(group, seed=seed)
        tagged = {**row, "split": split, "split_version": version, "seed": seed}
        out.append(tagged)
        text = row.get("text")
        if isinstance(text, str):
            digest = hashlib.sha256(text.encode()).hexdigest()
            text_splits.setdefault(digest, set()).add(split)
    crossing = sum(len(splits) > 1 for splits in text_splits.values())
    counts = Counter(row["split"] for row in out)
    return SplitResult(tuple(out), {"cross_split_text_hashes": crossing, **dict(counts)})


def strict_test_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Test rows whose exact text did not occur in train or validation."""
    seen = {
        hashlib.sha256(str(row.get("text", "")).encode()).hexdigest()
        for row in rows
        if row.get("split") in {"train", "val"}
    }
    return [
        row for row in rows
        if row.get("split") == "test"
        and hashlib.sha256(str(row.get("text", "")).encode()).hexdigest() not in seen
    ]


__all__ = ["SplitResult", "assign_split", "split_rows", "strict_test_rows"]
