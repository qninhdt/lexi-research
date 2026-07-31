"""Teacher-fidelity metrics for held-out student predictions."""

from __future__ import annotations

from collections.abc import Sequence

from lexi_research.data.pilot_gate import edit_f1, quadratic_weighted_kappa


def meaning_metrics(predicted: Sequence[int], teacher: Sequence[int]) -> dict[str, float]:
    if len(predicted) != len(teacher):
        raise ValueError("predicted and teacher labels must have equal length")
    if not teacher:
        return {"qwk": 0.0, "exact": 0.0, "within_one": 0.0, "mae": 0.0}
    total = len(teacher)
    return {
        "qwk": quadratic_weighted_kappa(predicted, teacher),
        "exact": sum(a == b for a, b in zip(predicted, teacher, strict=True)) / total,
        "within_one": sum(abs(a - b) <= 1 for a, b in zip(predicted, teacher, strict=True)) / total,
        "mae": sum(abs(a - b) for a, b in zip(predicted, teacher, strict=True)) / total,
    }


def correction_metrics(
    predicted: Sequence[Sequence[tuple[int, int, str]]],
    teacher: Sequence[Sequence[tuple[int, int, str]]],
) -> dict[str, float]:
    if len(predicted) != len(teacher):
        raise ValueError("predicted and teacher edits must have equal length")
    score = (
        sum(edit_f1(a, b) for a, b in zip(predicted, teacher, strict=True)) / len(teacher)
        if teacher
        else 0.0
    )
    return {"edit_f1": score}


__all__ = ["correction_metrics", "meaning_metrics"]
