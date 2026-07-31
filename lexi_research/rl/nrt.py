"""NRT: aggregate the gold-token probabilities a reasoning induces.

`R(z) = f(c_1 .. c_T)` where `c_i = pi(y*_i | x, z, y*_<i)`. Ablation A4 is the
choice of `f`, and the four arms are the ones the source paper compares.

`seq_logp` is exactly single-sample JEPO. That equivalence is asserted by a test
rather than stated here, because the two are implemented independently and a bug
would have to exist in both, identically, to pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .base import RLError

AGGREGATIONS = ("seq_logp", "geo_mean", "arith_mean", "weighted_neglogp")

#: Probabilities below this are floored before a log. A single zero would send
#: the whole reward to -inf and take the group's advantages with it.
PROBABILITY_FLOOR = 1e-12


def _checked(probabilities: Sequence[float]) -> list[float]:
    if not probabilities:
        raise RLError("NRT needs at least one gold-token probability")
    out = []
    for value in probabilities:
        if not 0.0 <= value <= 1.0:
            raise RLError(f"gold-token probability {value} is outside [0, 1]")
        out.append(max(float(value), PROBABILITY_FLOOR))
    return out


def seq_logp(probabilities: Sequence[float]) -> float:
    """Sum of log-probabilities — the same quantity JEPO rewards."""
    return float(sum(math.log(value) for value in _checked(probabilities)))


def geo_mean(probabilities: Sequence[float]) -> float:
    """Length-normalised: `exp(mean log p)`.

    The arm that removes the length preference `seq_logp` carries. A long answer
    is not worse for being long, and under `seq_logp` it always is.
    """
    values = _checked(probabilities)
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def arith_mean(probabilities: Sequence[float]) -> float:
    """Mean probability. Dominated by the easy tokens, which is the point of
    having it as an arm: most of an answer is punctuation and field names."""
    values = _checked(probabilities)
    return float(sum(values) / len(values))


def weighted_neglogp(
    probabilities: Sequence[float],
    base_probabilities: Sequence[float] | None = None,
) -> float:
    """Weight each token by how surprising the *base* model found it.

    Without weighting, a reasoning is rewarded for making `"meaning":` likely —
    which it already was. Weighting by `-log p_base` concentrates the reward on
    the tokens the base model was unsure about, which are the ones a reasoning
    could plausibly have helped with.
    """
    values = _checked(probabilities)
    if base_probabilities is None:
        base = values
    else:
        base = _checked(base_probabilities)
        if len(base) != len(values):
            raise RLError("base probabilities must align token for token")
    weights = [-math.log(value) for value in base]
    total = sum(weights)
    if total <= 0.0:
        # The base model was certain about every token, so no token carries
        # information about the reasoning. Falling back to the unweighted mean
        # keeps the reward defined rather than dividing by zero.
        return sum(math.log(value) for value in values) / len(values)
    return float(
        sum(weight * math.log(value) for weight, value in zip(weights, values, strict=True)) / total
    )


def nrt_reward(
    probabilities: Sequence[float],
    *,
    aggregation: str = "geo_mean",
    base_probabilities: Sequence[float] | None = None,
) -> float:
    """Apply the aggregation named by `rl.nrt.aggregation`. Ablation A4."""
    if aggregation not in AGGREGATIONS:
        raise RLError(f"nrt aggregation {aggregation!r}; expected one of {list(AGGREGATIONS)}")
    if aggregation == "seq_logp":
        return seq_logp(probabilities)
    if aggregation == "geo_mean":
        return geo_mean(probabilities)
    if aggregation == "arith_mean":
        return arith_mean(probabilities)
    return weighted_neglogp(probabilities, base_probabilities)


__all__ = [
    "AGGREGATIONS",
    "PROBABILITY_FLOOR",
    "arith_mean",
    "geo_mean",
    "nrt_reward",
    "seq_logp",
    "weighted_neglogp",
]
