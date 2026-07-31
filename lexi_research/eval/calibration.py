"""Calibration: is a confident prediction more often right than an unsure one?

A grader that is 90% confident and 60% correct is worse than useless in a product
that shows the band to a learner, and no accuracy number says so. ECE does, and
the reliability diagram shows *where* the miscalibration is.

Bins hold equal counts rather than covering equal widths. The band distribution
here is skewed by design — the sampler weights the middle — so equal-width bins
would leave several nearly empty, and an ECE dominated by bins holding four rows
is noise reported to three decimal places.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class CalibrationError(ValueError):
    """The inputs cannot support a calibration estimate."""


@dataclass(frozen=True)
class Bin:
    """One reliability bin: how confident, how often right, how many."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }


def equal_mass_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> list[Bin]:
    """Split into `bins` groups of near-equal count, ordered by confidence."""
    if len(confidences) != len(correct):
        raise CalibrationError("confidences and outcomes must have equal length")
    if not confidences:
        raise CalibrationError("no predictions to calibrate")
    if bins < 1:
        raise CalibrationError("bins must be positive")

    paired = sorted(zip(confidences, correct, strict=True), key=lambda item: item[0])
    total = len(paired)
    bins = min(bins, total)

    out: list[Bin] = []
    start = 0
    for index in range(bins):
        # Boundaries by index rather than by value, so every bin holds the same
        # count give or take one regardless of how the confidences clump.
        end = round(total * (index + 1) / bins)
        group = paired[start:end]
        start = end
        if not group:
            continue
        values = [confidence for confidence, _ in group]
        hits = [outcome for _, outcome in group]
        out.append(
            Bin(
                lower=values[0],
                upper=values[-1],
                count=len(group),
                mean_confidence=sum(values) / len(values),
                accuracy=sum(hits) / len(hits),
            )
        )
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    """Count-weighted mean gap between confidence and accuracy across bins."""
    grouped = equal_mass_bins(confidences, correct, bins=bins)
    total = sum(item.count for item in grouped)
    return sum(item.count * abs(item.mean_confidence - item.accuracy) for item in grouped) / total


def reliability(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """ECE plus the diagram's bins, ready to plot."""
    grouped = equal_mass_bins(confidences, correct, bins=bins)
    total = sum(item.count for item in grouped)
    return {
        "ece": sum(item.count * abs(item.mean_confidence - item.accuracy) for item in grouped)
        / total,
        "bins": [item.as_dict() for item in grouped],
        "n": total,
        # Positive means overconfident, which is the direction that hurts: a
        # grader that undersells a correct answer costs less than one that
        # oversells a wrong one.
        "mean_confidence": sum(item.count * item.mean_confidence for item in grouped) / total,
        "accuracy": sum(item.count * item.accuracy for item in grouped) / total,
    }


__all__ = [
    "Bin",
    "CalibrationError",
    "equal_mass_bins",
    "expected_calibration_error",
    "reliability",
]
