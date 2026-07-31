"""Deterministic cap-only balancing; rare strata are never discarded."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence


def balance_rows(rows: Sequence[dict[str, Any]], *, max_stratum_share: float, seed: int) -> list[dict[str, Any]]:
    if not 0 < max_stratum_share <= 1:
        raise ValueError("max_stratum_share must be in (0, 1]")
    cap = max(1, int(len(rows) * max_stratum_share))
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("meaning"), row.get("error_spec", "unknown"))
        grouped.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        ranked = sorted(
            grouped[key], key=lambda row: hashlib.sha256(f"{seed}|{row.get('req_uid', '')}".encode()).hexdigest()
        )
        selected.extend(ranked[:cap])
    return selected


def distribution(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(f"{r.get('meaning')}|{r.get('error_spec', 'unknown')}" for r in rows))


__all__ = ["balance_rows", "distribution"]
