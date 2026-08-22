"""Task difficulty profiling and adaptive sampling for Agentic RL."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DifficultyProfile:
    easy_tasks: list[str] = field(default_factory=list)
    learnable_tasks: list[str] = field(default_factory=list)
    hard_tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "easy": self.easy_tasks,
            "learnable": self.learnable_tasks,
            "hard": self.hard_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DifficultyProfile":
        return cls(
            easy_tasks=data.get("easy", []),
            learnable_tasks=data.get("learnable", []),
            hard_tasks=data.get("hard", []),
        )

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def classify_task_difficulty(success_count: int, total_trials: int = 4) -> str:
    """Classifies a task into easy (100%), learnable (25%-75%), or hard (0%)."""
    if total_trials <= 0:
        return "hard"
    if success_count == total_trials:
        return "easy"
    if success_count == 0:
        return "hard"
    return "learnable"


def sample_tasks_by_difficulty(
    profile: DifficultyProfile,
    batch_size: int,
    learnable_weight: float = 0.70,
    easy_weight: float = 0.15,
    hard_weight: float = 0.15,
    seed: int | None = None,
) -> list[str]:
    """Samples tasks with weighted priority toward learnable tasks to maximize reward variance."""
    rng = random.Random(seed)
    sampled: list[str] = []

    n_learnable = int(batch_size * learnable_weight)
    n_easy = int(batch_size * easy_weight)
    n_hard = batch_size - n_learnable - n_easy

    if profile.learnable_tasks and n_learnable > 0:
        sampled.extend(rng.choices(profile.learnable_tasks, k=n_learnable))
    if profile.easy_tasks and n_easy > 0:
        sampled.extend(rng.choices(profile.easy_tasks, k=n_easy))
    if profile.hard_tasks and n_hard > 0:
        sampled.extend(rng.choices(profile.hard_tasks, k=n_hard))

    # Fill remainder if any pool was empty
    all_tasks = profile.learnable_tasks or profile.easy_tasks or profile.hard_tasks
    while len(sampled) < batch_size and all_tasks:
        sampled.append(rng.choice(all_tasks))

    rng.shuffle(sampled)
    return sampled


def profile_task_difficulty(
    policy: Any,
    task_ids: list[str],
    env_factory: Any,
    trials_per_task: int = 4,
    max_turns: int = 8,
) -> "DifficultyProfile":
    """Profiles empirical success per task by rolling out the current policy.

    Runs ``trials_per_task`` episodes against a fresh env per trial and buckets
    tasks into easy (all success), learnable (mixed), and hard (no success).
    """
    from collections import defaultdict

    from tau_research.tau.rollout import run_episode_rollout

    successes: dict[str, int] = defaultdict(int)
    for task_id in task_ids:
        for _trial in range(trials_per_task):
            env = env_factory.create(task_id)
            trajectory = run_episode_rollout(env, policy, max_turns=max_turns)
            if trajectory["reward"].is_success:
                successes[task_id] += 1

    profile = DifficultyProfile()
    for task_id in task_ids:
        bucket = classify_task_difficulty(successes[task_id], trials_per_task)
        if bucket == "easy":
            profile.easy_tasks.append(task_id)
        elif bucket == "learnable":
            profile.learnable_tasks.append(task_id)
        else:
            profile.hard_tasks.append(task_id)
    return profile


def summarize_profile(profile: "DifficultyProfile") -> dict[str, int]:
    """Returns bucket sizes for logging."""
    return {
        "easy": len(profile.easy_tasks),
        "learnable": len(profile.learnable_tasks),
        "hard": len(profile.hard_tasks),
    }
