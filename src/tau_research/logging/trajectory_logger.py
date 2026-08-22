"""W&B trajectory table logger for multi-turn conversational episodes."""

from typing import Any


class TrajectoryLogger:
    """Logs sample multi-turn trajectory tables to Weights & Biases."""

    def __init__(self, project_name: str = "tau-research") -> None:
        self.project_name = project_name

    def log_trajectory_table(
        self,
        trajectories: list[dict[str, Any]],
        step: int,
    ) -> None:
        """Uploads a formatted W&B Table with conversation turns and outcomes."""
        try:
            import wandb

            if wandb.run is None:
                return

            columns = [
                "task_id",
                "num_turns",
                "reward",
                "db_reward",
                "communicate_reward",
                "is_success",
                "transcript",
            ]
            table = wandb.Table(columns=columns)

            for traj in trajectories:
                reward_obj = traj.get("reward")
                reward_val = getattr(reward_obj, "reward", 0.0)
                db_val = getattr(reward_obj, "db_reward", 0.0)
                comm_val = getattr(reward_obj, "communicate_reward", 0.0)
                is_success = getattr(reward_obj, "is_success", False)

                transcript_str = "\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                    for m in traj.get("history", [])
                )

                table.add_data(
                    traj.get("task_id", "unknown"),
                    traj.get("num_turns", 0),
                    reward_val,
                    db_val,
                    comm_val,
                    is_success,
                    transcript_str,
                )

            wandb.log({"trajectories/sample_table": table}, step=step)
        except Exception:
            pass
