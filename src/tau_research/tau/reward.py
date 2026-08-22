"""Reward extraction and diagnostic metrics calculation for Tau-Bench Retail."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TauReward:
    reward: float
    db_reward: float
    communicate_reward: float
    partial_action_reward: float = 0.0
    is_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "db_reward": self.db_reward,
            "communicate_reward": self.communicate_reward,
            "partial_action_reward": self.partial_action_reward,
            "is_success": self.is_success,
        }


def calculate_outcome_reward(
    db_success: bool,
    communicate_success: bool,
    partial_action: float = 0.0,
) -> TauReward:
    """Calculates official outcome-based binary reward R = R_DB * R_COMM."""
    r_db = 1.0 if db_success else 0.0
    r_comm = 1.0 if communicate_success else 0.0
    r_total = r_db * r_comm
    is_success = r_total == 1.0

    return TauReward(
        reward=r_total,
        db_reward=r_db,
        communicate_reward=r_comm,
        partial_action_reward=partial_action,
        is_success=is_success,
    )
