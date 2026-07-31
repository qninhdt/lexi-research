"""GRPO-RLVR: sample K answers, score each with code.

This track runs first, and the reason is its reward: it is computed by
`lexi_research/format` primitives that Phase 2 already tested, and it does not
depend on the policy at all. When the curve misbehaves the cause is the trainer
or the data, and both can be inspected separately.

The other two tracks have no such property, so building them first would mean
debugging two unknowns with one equation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lexi_research.format import BandConfig

from .rewards import RewardParts, RewardWeights, parse_answer, verifiable_reward


def grpo_reward(
    completion: str,
    row: Mapping[str, Any],
    config: BandConfig,
    *,
    weights: RewardWeights | None = None,
    scope: str = "correction_meaning",
) -> RewardParts:
    """Score a sampled completion. Exogenous: no policy term appears."""
    return verifiable_reward(parse_answer(completion), row, config, weights, scope=scope)


__all__ = ["grpo_reward"]
