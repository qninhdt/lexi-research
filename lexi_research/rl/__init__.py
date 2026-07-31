"""Three RL tracks over one shared reward mask.

They differ only in how `R(z)` is defined; everything around it — the group
advantages, the empty-reasoning baseline, the KL to reference, the combined loss
— is `base.py`, shared literally rather than by convention. That is what makes a
comparison between the tracks a comparison of reward definitions.

`feedback` receives no reward in any track. It is voice and register:
unverifiable, and rewarding it would teach the model to chase teacher phrasing
rather than grading quality. It is still fully supervised by the cross-entropy.
"""

from .base import Group, RLError, StepReport, advantages, baseline_clip, combined_loss, step_report
from .rewards import RewardParts, RewardWeights, verifiable_reward
from .segments import Segments, build_segments, policy_gradient_mask, reward_mask

__all__ = [
    "Group",
    "RLError",
    "RewardParts",
    "RewardWeights",
    "Segments",
    "StepReport",
    "advantages",
    "baseline_clip",
    "build_segments",
    "combined_loss",
    "policy_gradient_mask",
    "reward_mask",
    "step_report",
    "verifiable_reward",
]
