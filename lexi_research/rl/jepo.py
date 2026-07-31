"""JEPO: reward a reasoning by how likely it makes the gold answer.

`R(z) = log pi(correction, meaning | x, z)` — teacher-forced, so no answer is
sampled and the reward needs no verifier. That also makes it endogenous: `R`
depends on theta, so a rising curve can mean the model became more confident
rather than more correct. GRPO exists to be the baseline that cannot do that.

The token log-probabilities arrive from the trainer; this module is the
definition, so it stays exact arithmetic that a test can check by hand.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import RLError


def jepo_reward(gold_logprobs: Sequence[float]) -> float:
    """Sequence log-probability of the gold answer under this reasoning.

    Summed rather than averaged: the quantity the objective is a bound on is the
    joint probability of the answer, and dividing by length would make a long
    answer cheaper to satisfy than a short one.
    """
    if not gold_logprobs:
        raise RLError("JEPO needs at least one gold-token log-probability")
    return float(sum(gold_logprobs))


def multi_sample_bound(rewards: Sequence[float]) -> float:
    """The K-sample lower bound: `log mean exp(R_k)`, in a stable form.

    Optional, and behind a flag, because with K=1 it reduces to the single-sample
    reward and the extra samples cost K forward passes for a bound that is only
    tighter when the reasonings genuinely differ.
    """
    if not rewards:
        raise RLError("no rewards to bound")
    import math

    highest = max(rewards)
    total = sum(math.exp(reward - highest) for reward in rewards)
    return highest + math.log(total / len(rewards))


__all__ = ["jepo_reward", "multi_sample_bound"]
