"""Tests for task difficulty profiling and weighted sampling for Agentic RL."""

from tau_research.training.difficulty import (
    DifficultyProfile,
    classify_task_difficulty,
    sample_tasks_by_difficulty,
)


def test_classify_task_difficulty() -> None:
    assert classify_task_difficulty(success_count=4, total_trials=4) == "easy"
    assert classify_task_difficulty(success_count=3, total_trials=4) == "learnable"
    assert classify_task_difficulty(success_count=2, total_trials=4) == "learnable"
    assert classify_task_difficulty(success_count=1, total_trials=4) == "learnable"
    assert classify_task_difficulty(success_count=0, total_trials=4) == "hard"


def test_sample_tasks_by_difficulty() -> None:
    profile = DifficultyProfile(
        easy_tasks=[f"easy_{i}" for i in range(10)],
        learnable_tasks=[f"learnable_{i}" for i in range(10)],
        hard_tasks=[f"hard_{i}" for i in range(10)],
    )

    sampled = sample_tasks_by_difficulty(profile, batch_size=20, seed=42)
    assert len(sampled) == 20

    # Ensure learnable tasks are sampled heavily
    learnable_count = sum(1 for t in sampled if "learnable" in t)
    assert learnable_count >= 10  # ~70% of batch
