"""The pilot gate — the cheapest place for this project to fail.

Before spending on ~20K rows, 500 are generated and measured against eight
criteria. Two of them are hard stops, and they are hard stops because they are
the only failures that cannot be repaired anywhere downstream:

**G1, teacher self-consistency.** Every metric in Phase 8 compares the student to
the teacher, so the teacher's agreement with itself is the ceiling on all of
them. A teacher that disagrees with its own earlier grading is emitting noise,
and a student trained on noise produces errors indistinguishable from the
teacher's. This is measurable *without gold labels* — it only requires grading
the same text twice — which is what makes it worth gating on.

**G2, middle-band presence.** If `meaning` comes out bimodal (0 and 4 only), the
dataset has no examples of the region where real learners land. No amount of
training recovers a distribution that was never sampled.

The rest (G3-G7) are prompt-tuning loops: cheap to iterate, ~$1 per pass over
500 rows. They warn; they do not block.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lexi_research.format import MAX_BAND, MIN_BAND

#: The two gates that block. Named here rather than inline so Phase 7's entry
#: check can assert on the same identifiers instead of re-deriving them.
HARD_GATES: frozenset[str] = frozenset({"G1_self_consistency", "G2_band_coverage"})


@dataclass(frozen=True)
class GateResult:
    """One gate measurement, with the threshold it was judged against."""

    name: str
    value: float
    threshold: float
    passed: bool
    blocking: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "blocking": self.blocking,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateReport:
    """Every gate measurement for one pilot run."""

    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        """True when every gate passed — the condition for full generation."""
        return all(result.passed for result in self.results)

    @property
    def blocking_passed(self) -> bool:
        """True when both hard gates passed — the condition Phase 7 checks.

        Separate from `passed` on purpose: a prompt-tuning warning (G3-G7) should
        not block training on data that is already generated, but a noisy teacher
        or a missing middle band must.
        """
        return all(result.passed for result in self.results if result.blocking)

    def failures(self) -> tuple[GateResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocking_passed": self.blocking_passed,
            "gates": [result.as_dict() for result in self.results],
        }

    def write(self, path: str | Path) -> None:
        """Write `pilot-gate.json` — small, no dictionary text, safe for Git."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> GateReport:
        """Load a written report, for Phase 7's entry check."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            results=tuple(
                GateResult(
                    name=str(gate["name"]),
                    value=float(gate["value"]),
                    threshold=float(gate["threshold"]),
                    passed=bool(gate["passed"]),
                    blocking=bool(gate["blocking"]),
                    detail=str(gate.get("detail", "")),
                )
                for gate in payload["gates"]
            )
        )


def quadratic_weighted_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Agreement between two band assignments, chance-corrected.

    QWK rather than plain accuracy because the bands are ordered: confusing 3
    with 4 is a near miss, confusing 0 with 4 is a total disagreement, and an
    accuracy score calls both simply "wrong".

    Returns 1.0 when both sides are constant and identical — perfect agreement
    with no variance to correct for. Plain kappa is undefined there (the
    denominator vanishes), and reporting 0.0 would call two identical rating sets
    maximally inconsistent.
    """
    if len(first) != len(second):
        raise ValueError(f"unequal lengths: {len(first)} vs {len(second)}")
    if not first:
        return 0.0

    bands = list(range(MIN_BAND, MAX_BAND + 1))
    size = len(bands)
    offset = MIN_BAND

    observed = [[0.0] * size for _ in bands]
    for a, b in zip(first, second):
        observed[a - offset][b - offset] += 1

    n = float(len(first))
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(size)) for c in range(size)]

    numerator = 0.0
    denominator = 0.0
    for r in range(size):
        for c in range(size):
            weight = ((r - c) ** 2) / ((size - 1) ** 2)
            expected = row_totals[r] * col_totals[c] / n
            numerator += weight * observed[r][c]
            denominator += weight * expected

    if denominator == 0.0:
        # No expected disagreement: either side is constant. Perfect agreement if
        # the observed disagreement is also zero, otherwise total disagreement.
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def edit_f1(predicted: Sequence[tuple[int, int, str]], gold: Sequence[tuple[int, int, str]]) -> float:
    """F1 over `(start, end, tag)` edit triples.

    Set-based rather than sequence-based: two graders may mark the same edits in
    a different order, and that is not a disagreement. An edit counts as matched
    only when span *and* tag agree — a correct span with the wrong tag moves a
    band, so it is not a hit.
    """
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        # Both graders found a clean sentence. That is agreement, and scoring it
        # 0.0 would penalise the teacher for the easiest case in the dataset.
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0

    matched = len(predicted_set & gold_set)
    if matched == 0:
        return 0.0
    precision = matched / len(predicted_set)
    recall = matched / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def band_distribution(meanings: Sequence[int]) -> dict[int, float]:
    """Share of each band, over all five bands including the empty ones.

    Every band appears in the result even at zero: G2 asks whether a band is
    *missing*, and a dict that silently omits absent keys makes that invisible.
    """
    total = len(meanings)
    counts = dict.fromkeys(range(MIN_BAND, MAX_BAND + 1), 0.0)
    for meaning in meanings:
        counts[meaning] = counts.get(meaning, 0.0) + 1
    if total == 0:
        return counts
    return {band: count / total for band, count in counts.items()}


def evaluate(
    *,
    self_consistency_qwk: float,
    self_consistency_edit_f1: float,
    meanings: Sequence[int],
    format_validity: float,
    mean_distinct2: float,
    other_tag_share: float,
    batch_single_qwk: float,
    manual_review_passed: bool | None = None,
) -> GateReport:
    """Judge one pilot run against all eight gates.

    Thresholds come from the plan and are stated here as literals rather than
    read from `params.yaml`: they are the definition of the gate, not a knob. A
    run that wants a different bar is a decision to record in the plan, not a
    parameter to nudge.
    """
    distribution = band_distribution(meanings)
    middle = sum(distribution[band] for band in (1, 2, 3))
    empty = sorted(band for band, share in distribution.items() if share == 0.0)

    results = [
        GateResult(
            name="G1_self_consistency",
            value=self_consistency_qwk,
            threshold=0.7,
            passed=self_consistency_qwk >= 0.7 and self_consistency_edit_f1 >= 0.6,
            blocking=True,
            detail=(
                f"QWK {self_consistency_qwk:.3f} (>= 0.700), "
                f"edit-F1 {self_consistency_edit_f1:.3f} (>= 0.600). "
                "A teacher that disagrees with itself caps every downstream metric."
            ),
        ),
        GateResult(
            name="G2_band_coverage",
            value=middle,
            threshold=0.4,
            passed=not empty and middle >= 0.4,
            blocking=True,
            detail=(
                f"middle bands {{1,2,3}} hold {middle:.1%} (>= 40%); "
                f"empty bands: {empty or 'none'}. "
                "A bimodal dataset cannot teach the region real learners occupy."
            ),
        ),
        GateResult(
            name="G3_format_validity",
            value=format_validity,
            threshold=0.9,
            passed=format_validity > 0.9,
            blocking=False,
            detail="Share of call-2 outputs passing the six checks.",
        ),
        GateResult(
            name="G4_batch_diversity",
            value=mean_distinct2,
            threshold=0.7,
            passed=mean_distinct2 > 0.7,
            blocking=False,
            detail="Mean distinct-2 per batch. Low means K sentences collapsed into one.",
        ),
        GateResult(
            name="G5_other_tag_share",
            value=other_tag_share,
            threshold=0.05,
            passed=other_tag_share < 0.05,
            blocking=False,
            detail="High `other` means the taxonomy is missing a category (Phase 1).",
        ),
        GateResult(
            name="G6_batch_single_parity",
            value=batch_single_qwk,
            threshold=0.8,
            passed=batch_single_qwk >= 0.8,
            blocking=False,
            detail="K=6 vs K=1 agreement. Low means batching changed the judgement.",
        ),
    ]

    if manual_review_passed is not None:
        results.append(
            GateResult(
                name="G7_manual_review",
                value=1.0 if manual_review_passed else 0.0,
                threshold=1.0,
                passed=manual_review_passed,
                blocking=False,
                detail="Human read of ~50 rows: plausible learner text, useful feedback.",
            )
        )

    return GateReport(results=tuple(results))


def edit_triples(correction: str | None) -> list[tuple[int, int, str]]:
    """Edit triples from a correction string, for `edit_f1`.

    An unparseable correction contributes no edits, which is the right reading
    for agreement: two graders who both gave up agree, and `edit_f1` scores two
    empty sets as 1.0.
    """
    from lexi_research.format import ParseError, parse_correction

    if correction is None:
        return []
    parsed = parse_correction(correction)
    if isinstance(parsed, ParseError):
        return []
    return [(edit.span[0], edit.span[1], edit.tag) for edit in parsed.edits]


def measure_self_consistency(pairs: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """QWK on `meaning` and edit-F1 on `correction` across two grading passes.

    `pairs` comes from `label.self_consistency_pairs`: each row carries both
    passes' output for one text.
    """
    if not pairs:
        return 0.0, 0.0

    first = [int(pair["meaning"]) for pair in pairs]
    second = [int(pair["meaning_pass2"]) for pair in pairs]
    qwk = quadratic_weighted_kappa(first, second)

    scores = [
        edit_f1(edit_triples(pair.get("correction")), edit_triples(pair.get("correction_pass2")))
        for pair in pairs
    ]
    return qwk, sum(scores) / len(scores)


__all__ = [
    "HARD_GATES",
    "GateReport",
    "GateResult",
    "band_distribution",
    "edit_f1",
    "edit_triples",
    "evaluate",
    "measure_self_consistency",
    "quadratic_weighted_kappa",
]
