"""Dataset validation and label mask computation."""

from typing import Any


def compute_labels_for_prompt_completion(
    prompt_ids: list[int],
    completion_ids: list[int],
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    """Constructs concatenated input_ids and labels with prompt masked to ignore_index (-100)."""
    input_ids = list(prompt_ids) + list(completion_ids)
    labels = [ignore_index] * len(prompt_ids) + list(completion_ids)
    return input_ids, labels


def assert_no_leakage(train_task_ids: list[str], test_task_ids: list[str]) -> None:
    """Verifies zero overlap between train task IDs and test task IDs."""
    overlap = set(train_task_ids).intersection(set(test_task_ids))
    if overlap:
        msg = f"Data Leakage detected! {len(overlap)} task IDs overlap: {overlap}"
        raise ValueError(msg)


def validate_sft_example(example: dict[str, Any]) -> bool:
    """Validates structural correctness of an SFT example dict."""
    if "prompt" not in example or "completion" not in example:
        return False
    if not isinstance(example["prompt"], list) or not isinstance(example["completion"], list):
        return False
    if len(example["completion"]) == 0 or example["completion"][0].get("role") != "assistant":
        return False
    return True
