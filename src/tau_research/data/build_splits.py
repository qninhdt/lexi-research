"""Deterministic task-level split builder for SFT training."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tau_research.data.validate_dataset import assert_no_leakage


def split_task_ids_deterministically(
    task_ids: list[str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Splits task IDs into train and val subsets with a fixed seed."""
    sorted_tasks = sorted(list(set(task_ids)))
    rng = random.Random(seed)
    shuffled = sorted_tasks.copy()
    rng.shuffle(shuffled)

    num_train = int(len(shuffled) * train_ratio)
    train_ids = sorted(shuffled[:num_train])
    val_ids = sorted(shuffled[num_train:])
    assert_no_leakage(train_ids, val_ids)
    return train_ids, val_ids


def save_splits(
    train_ids: list[str],
    val_ids: list[str],
    output_dir: str | Path = "artifacts/splits",
    held_out_test_ids: list[str] | None = None,
) -> None:
    """Saves split task ID arrays to JSON artifacts.

    Optionally validates zero overlap against an official held-out test ID list.
    """
    assert_no_leakage(train_ids, val_ids)
    if held_out_test_ids is not None:
        assert_no_leakage(train_ids, held_out_test_ids)
        assert_no_leakage(val_ids, held_out_test_ids)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "sft_train_task_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)

    with open(out_path / "sft_val_task_ids.json", "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)
