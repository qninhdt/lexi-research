"""Pure fitting helpers for teacher-reference band calibration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from lexi_research.format import BandConfig
from lexi_research.format.tags import CONFUSABLE_PAIRS


def assert_confusable_weights(config: BandConfig) -> None:
    """Fail fast if a known tag confusion could alter a derived band."""
    unequal = [
        (str(a), str(b))
        for a, b in CONFUSABLE_PAIRS
        if config.weight_of(str(a)) != config.weight_of(str(b))
    ]
    if unequal:
        raise ValueError(f"confusable tags have unequal weights: {unequal}")


def fit_thresholds(
    penalties: Sequence[float], references: Sequence[int]
) -> tuple[float, float, float, float]:
    """Fit monotone cut points from ordinal references using adjacent midpoints."""
    if len(penalties) != len(references) or not penalties:
        raise ValueError("non-empty equally sized penalties and references are required")
    means = {}
    for band in range(5):
        values = [
            penalty
            for penalty, reference in zip(penalties, references, strict=True)
            if reference == band
        ]
        if not values:
            raise ValueError(f"reference band {band} is absent")
        means[band] = sum(values) / len(values)
    # Penalty rises as the reference band falls; use a midpoint between neighbours.
    return tuple((means[band] + means[band - 1]) / 2 for band in (4, 3, 2, 1))  # type: ignore[return-value]


def calibration_report(tags: Sequence[Sequence[str]]) -> dict[str, float | int]:
    counts = Counter(tag for row in tags for tag in row)
    total = sum(counts.values())
    return {
        "rows": len(tags),
        "other_rate": counts["other"] / total if total else 0.0,
        "tagged_edits": total,
    }


__all__ = ["assert_confusable_weights", "calibration_report", "fit_thresholds"]
