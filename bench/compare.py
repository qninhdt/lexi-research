"""B8: the student, a served MoE, and the teacher API, through one harness.

This answers the only question a reader actually has — was any of this worth it
versus just calling a bigger model? — so it deliberately reuses the Phase 5
runner and the Phase 2 harness rather than growing its own. A metric that appears
only in the comparison is a metric nobody tested.

The MoE is inference-only. The serving skills transfer; training one is a
different project with a different budget.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The three systems, and what each one is for.
SYSTEMS = ("student", "moe", "teacher")


class CompareError(ValueError):
    """The comparison cannot be assembled."""


@dataclass(frozen=True)
class SystemResult:
    """One system's quality, speed and economics, measured the same way."""

    system: str
    quality: Mapping[str, Any] = field(default_factory=dict)
    latency: Mapping[str, Any] = field(default_factory=dict)
    cost_per_1k_requests: float | None = None
    skipped: str | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.skipped:
            return {"system": self.system, "skipped": self.skipped}
        return {
            "system": self.system,
            "qwk": self.quality.get("meaning.qwk"),
            "span_tag_f1": self.quality.get("correction.span_tag_f1"),
            "validity_rate": self.quality.get("format.validity_rate"),
            "e2e_p95_s": self.latency.get("e2e_p95_s"),
            "tokens_per_s": self.latency.get("tokens_per_s"),
            "peak_vram_mb": self.latency.get("peak_vram_mb"),
            "cost_per_1k_requests": self.cost_per_1k_requests,
        }


def quality_per_dollar(result: SystemResult, *, slo_s: float) -> float | None:
    """Quality per dollar, at a fixed latency SLO. The axis that decides B8.

    A system that misses the SLO scores nothing rather than scoring well slowly:
    the comparison is between systems that could serve this product, and one that
    cannot is not a cheaper option.
    """
    if result.skipped:
        return None
    p95 = result.latency.get("e2e_p95_s")
    quality = result.quality.get("meaning.qwk")
    cost = result.cost_per_1k_requests
    if p95 is None or quality is None or not cost:
        return None
    if float(p95) > slo_s:
        return 0.0
    return float(quality) / float(cost)


def assemble(
    results: Sequence[SystemResult],
    *,
    slo_s: float,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """One report holding all three, plus the verdict the numbers support."""
    if not results:
        raise CompareError("no systems to compare")
    rows = []
    for result in results:
        row = result.as_dict()
        row["quality_per_dollar"] = quality_per_dollar(result, slo_s=slo_s)
        rows.append(row)

    ranked = [row for row in rows if row.get("quality_per_dollar")]
    ranked.sort(key=lambda row: float(row["quality_per_dollar"]), reverse=True)
    return {
        "slo_s": slo_s,
        "systems": rows,
        "best_quality_per_dollar": ranked[0]["system"] if ranked else None,
        "lineage": dict(lineage),
        "note": (
            "The axis is quality per dollar at a fixed latency SLO, not raw "
            "quality. A larger model that wins on QWK while costing several times "
            "more per request loses for this application, and a system that "
            "misses the SLO is not a cheaper option."
        ),
    }


def write(payload: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return out


__all__ = ["SYSTEMS", "CompareError", "SystemResult", "assemble", "quality_per_dollar", "write"]
