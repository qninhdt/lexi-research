"""What the three tracks share: advantages, the empty baseline, KL, and the loss.

The design's claim is that GRPO, JEPO and NRT differ *only* in how `R(z)` is
defined. That is only true if everything around `R` is literally the same code,
so the group normalisation, the baseline clipping, the KL term and the combined
loss live here and the tracks supply a reward function.

All of it is plain arithmetic over lists. Torch appears in the trainer that calls
these, not in the definitions — which is what lets the equivalence between NRT's
`seq_logp` and JEPO be tested exactly rather than to a training-run's tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Below this, a group's rewards are treated as constant and the advantages are
#: zeroed. Dividing by a near-zero std turns rounding noise into a large gradient.
STD_FLOOR = 1e-6

ALGORITHMS = ("grpo", "jepo", "nrt")


class RLError(ValueError):
    """The RL configuration or a rollout group is unusable."""


@dataclass(frozen=True)
class Group:
    """One prompt's rollouts: the reasonings sampled for it and their rewards."""

    rewards: tuple[float, ...]
    empty_reward: float = 0.0

    def __post_init__(self) -> None:
        if not self.rewards:
            raise RLError("a rollout group holds no samples")


def baseline_clip(rewards: Sequence[float], empty_reward: float) -> list[float]:
    """`R' = max(0, R(z) - R(empty))` — the improvement reasoning actually bought.

    Mandatory for JEPO and NRT per their source papers, and the primary diagnostic
    for both: if this collapses to zero, reasoning has stopped contributing and
    the run is dead regardless of what the raw reward curve says.
    """
    return [max(0.0, reward - empty_reward) for reward in rewards]


def advantages(rewards: Sequence[float]) -> list[float]:
    """Group-relative advantage: `(R - mean) / std`, zeroed on a constant group.

    A group where every rollout scored the same carries no information about which
    reasoning was better. Normalising it anyway would amplify float noise into a
    gradient, so it contributes nothing instead.
    """
    if not rewards:
        raise RLError("no rewards to normalise")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < STD_FLOOR:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]


def zero_advantage_share(groups: Sequence[Group]) -> float:
    """Share of groups that carried no signal. A rising value means a dead run."""
    if not groups:
        return 0.0
    dead = sum(1 for group in groups if all(a == 0.0 for a in advantages(group.rewards)))
    return dead / len(groups)


def kl_divergence(policy_logprobs: Sequence[float], reference_logprobs: Sequence[float]) -> float:
    """Mean per-token KL estimate between the policy and the frozen reference.

    The k1 estimator — the plain log-ratio — because it is unbiased and this is a
    diagnostic rather than a term being optimised. Logged every step: an
    endogenous reward can rise while the policy walks away from anything the
    reference would recognise, and only this shows it.
    """
    if len(policy_logprobs) != len(reference_logprobs):
        raise RLError("policy and reference log-probabilities must align token for token")
    if not policy_logprobs:
        return 0.0
    return sum(
        policy - reference
        for policy, reference in zip(policy_logprobs, reference_logprobs, strict=True)
    ) / len(policy_logprobs)


def combined_loss(
    ce_loss: Any,
    rl_loss: Any,
    *,
    lambda_rl: float,
    kl: Any = 0.0,
    kl_coefficient: float = 0.0,
) -> Any:
    """`L = CE(answer) + lambda * L_RL(reasoning) + beta * KL`.

    The cross-entropy covers the whole answer including feedback; the RL term
    touches only the reasoning. Keeping them additive rather than interleaved is
    what makes `lambda` a single interpretable knob to sweep.

    Typed loosely because it is the *definition*, called with plain floats by the
    tests that pin its arithmetic and with tensors by the trainer that
    back-propagates through it. One expression, one place, either way.
    """
    return ce_loss + lambda_rl * rl_loss + kl_coefficient * kl


@dataclass(frozen=True)
class Rollout:
    """One sampled reasoning and everything measured about it."""

    reasoning: str
    reward: float
    parts: Mapping[str, float]
    tokens: int


@dataclass(frozen=True)
class StepReport:
    """The RL health panel for one optimiser step.

    Every field here is in the design's §5 list. A reward curve alone is never the
    verdict — `empty_gap` and `kl` are what separate "the model got better" from
    "the model got more confident".
    """

    reward_mean: float
    reward_std: float
    advantage_std: float
    empty_reward: float
    empty_gap: float
    kl: float
    reasoning_tokens: float
    zero_advantage_share: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rl/reward_mean": self.reward_mean,
            "rl/reward_std": self.reward_std,
            "rl/advantage_std": self.advantage_std,
            "rl/empty_reward": self.empty_reward,
            "rl/empty_gap": self.empty_gap,
            "rl/kl": self.kl,
            "rl/reasoning_tokens": self.reasoning_tokens,
            "rl/zero_advantage_share": self.zero_advantage_share,
        }


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def step_report(groups: Sequence[Group], rollouts: Sequence[Rollout], kl: float) -> StepReport:
    """Assemble the health panel from one step's groups and rollouts."""
    rewards = [reward for group in groups for reward in group.rewards]
    if not rewards:
        raise RLError("no rollouts to report on")
    empties = [group.empty_reward for group in groups]
    empty_mean = sum(empties) / len(empties)
    all_advantages = [
        value
        for group in groups
        for value in advantages(baseline_clip(group.rewards, group.empty_reward))
    ]
    return StepReport(
        reward_mean=sum(rewards) / len(rewards),
        reward_std=_std(rewards),
        advantage_std=_std(all_advantages),
        empty_reward=empty_mean,
        empty_gap=sum(rewards) / len(rewards) - empty_mean,
        kl=kl,
        reasoning_tokens=(
            sum(rollout.tokens for rollout in rollouts) / len(rollouts) if rollouts else 0.0
        ),
        zero_advantage_share=zero_advantage_share(groups),
    )


#: A reward function: sampled reasoning plus its context -> a scalar.
RewardFn = Callable[..., float]


def resolve_algorithm(name: str) -> str:
    if name not in ALGORITHMS:
        raise RLError(f"rl algorithm {name!r}; expected one of {list(ALGORITHMS)}")
    return name


def reward_of(algorithm: str) -> Any:
    """The reward callable for a track, imported only when that track is used."""
    resolve_algorithm(algorithm)
    if algorithm == "grpo":
        from .grpo import grpo_reward

        return grpo_reward
    if algorithm == "jepo":
        from .jepo import jepo_reward

        return jepo_reward
    from .nrt import nrt_reward

    return nrt_reward


__all__ = [
    "ALGORITHMS",
    "STD_FLOOR",
    "Group",
    "RLError",
    "RewardFn",
    "Rollout",
    "StepReport",
    "advantages",
    "baseline_clip",
    "combined_loss",
    "kl_divergence",
    "resolve_algorithm",
    "reward_of",
    "step_report",
    "zero_advantage_share",
]
