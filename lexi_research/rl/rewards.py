"""The verifiable reward. Exogenous, and computed by the code eval already uses.

GRPO runs first precisely because this number does not depend on the policy: when
the curve misbehaves, the cause is the trainer or the data, and both are
separately inspectable. JEPO and NRT rewards are functions of theta, so a rising
reward there can mean the model became more confident rather than more correct —
there is no way to tell without a track whose reward cannot drift.

Nothing here reimplements a metric. `edit_f1` and `validate_output` are imported
from the modules the eval harness scores with, because a reward that measured
something slightly different from eval would train the model toward a number
nobody reports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lexi_research.data.pilot_gate import edit_f1
from lexi_research.eval.correction import edit_triples
from lexi_research.format import BandConfig, ValidationError, validate_output
from lexi_research.format.bands import MAX_BAND, MIN_BAND
from lexi_research.format.parser import ParseError, parse_correction

#: Reward scopes — ablation A3.
SCOPES = ("correction_meaning", "full_answer")


class RewardError(ValueError):
    """The reward could not be computed for this pair."""


@dataclass(frozen=True)
class RewardWeights:
    """The four terms of the verifiable reward.

    `strip_mismatch` is subtracted rather than folded into `format_valid` so the
    one failure that cannot be seen downstream — a correction that quietly
    rewords text it did not mark — carries its own penalty and its own number in
    the report.

    Its weight is 1.0, above the whole positive total, on purpose. At 0.5 an
    otherwise-perfect answer that rewrote unmarked text still scored positive,
    which makes the behaviour worth producing. The invariant wanted is stronger:
    a drifted correction is never worth emitting at all.
    """

    edit_f1: float = 0.5
    meaning: float = 0.3
    format_valid: float = 0.2
    strip_mismatch: float = 1.0

    @classmethod
    def from_config(cls, values: Mapping[str, Any] | None) -> RewardWeights:
        if not values:
            return cls()
        unknown = set(values) - {"edit_f1", "meaning", "format_valid", "strip_mismatch"}
        if unknown:
            raise RewardError(f"unknown reward weights {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})

    @property
    def positive_total(self) -> float:
        """What a perfect answer scores, before any penalty."""
        return self.edit_f1 + self.meaning + self.format_valid


@dataclass(frozen=True)
class RewardParts:
    """Every term, so a reward can be read rather than trusted."""

    edit_f1: float
    meaning: float
    format_valid: float
    strip_mismatch: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "edit_f1": self.edit_f1,
            "meaning": self.meaning,
            "format_valid": self.format_valid,
            "strip_mismatch": self.strip_mismatch,
            "total": self.total,
        }


def parse_answer(text: str) -> dict[str, Any] | None:
    """The answer object a rollout produced, or None if it is not one."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _meaning_score(predicted: Any, gold: Any) -> float:
    """1.0 at an exact band, falling linearly to 0 across the full range."""
    if not isinstance(predicted, int) or isinstance(predicted, bool):
        return 0.0
    if not MIN_BAND <= predicted <= MAX_BAND:
        return 0.0
    span = MAX_BAND - MIN_BAND
    return 1.0 - abs(predicted - int(gold)) / span


def _strip_mismatch(correction: Any, text: str) -> float:
    """1.0 when the correction rewrote text it did not mark.

    The one failure invisible downstream: the stripped correction must reproduce
    the learner's sentence exactly, or the model can quietly improve prose it was
    not asked to touch and every span-based metric still agrees with it.
    """
    if correction is None:
        return 0.0
    if not isinstance(correction, str):
        return 1.0
    parsed = parse_correction(correction)
    if isinstance(parsed, ParseError):
        return 1.0
    return 0.0 if parsed.text == text else 1.0


def verifiable_reward(
    prediction: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    config: BandConfig,
    weights: RewardWeights | None = None,
    *,
    scope: str = "correction_meaning",
) -> RewardParts:
    """Score one rollout against the teacher's answer, with code only.

    `scope` is ablation A3. `full_answer` adds a feedback term; the default
    excludes it, because feedback is unverifiable and rewarding it teaches the
    model to chase the teacher's phrasing rather than its grading.
    """
    if scope not in SCOPES:
        raise RewardError(f"reward scope {scope!r}; expected one of {list(SCOPES)}")
    weights = weights or RewardWeights()
    text = str(row["text"])

    if prediction is None:
        return RewardParts(0.0, 0.0, 0.0, 0.0, 0.0)

    checked = validate_output(dict(prediction), text, config)
    valid = 0.0 if isinstance(checked, ValidationError) else 1.0

    f1 = edit_f1(
        edit_triples(prediction.get("correction")),
        edit_triples(row.get("correction")),
    )
    meaning = _meaning_score(prediction.get("meaning"), row.get("meaning", 0))
    mismatch = _strip_mismatch(prediction.get("correction"), text)

    total = (
        weights.edit_f1 * f1
        + weights.meaning * meaning
        + weights.format_valid * valid
        - weights.strip_mismatch * mismatch
    )
    if scope == "full_answer":
        # A3's other arm. Deliberately the same shape as the terms above so the
        # only difference between the arms is what is rewarded, not how.
        from lexi_research.eval.harness import chrf

        predicted_feedback = prediction.get("feedback")
        gold_feedback = row.get("feedback")
        if isinstance(predicted_feedback, str) and isinstance(gold_feedback, str):
            total += weights.format_valid * chrf(predicted_feedback, gold_feedback)

    return RewardParts(
        edit_f1=f1,
        meaning=meaning,
        format_valid=valid,
        strip_mismatch=mismatch,
        total=total,
    )


def normalised_reward(parts: RewardParts, weights: RewardWeights | None = None) -> float:
    """The reward on [0, 1] for a perfect answer, for cross-track comparison."""
    weights = weights or RewardWeights()
    return parts.total / weights.positive_total if weights.positive_total else 0.0


__all__ = [
    "SCOPES",
    "RewardError",
    "RewardParts",
    "RewardWeights",
    "normalised_reward",
    "parse_answer",
    "verifiable_reward",
]
