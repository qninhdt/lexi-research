"""Tests for conversational prompt-completion loss masking and label tensor values."""

import torch

from tau_research.data.validate_dataset import compute_labels_for_prompt_completion


def test_compute_labels_for_prompt_completion() -> None:
    prompt_ids = [100, 101, 102, 103]  # e.g., System + User + Prev tool
    completion_ids = [200, 201, 202]  # Assistant reasoning + Tool call

    input_ids, labels = compute_labels_for_prompt_completion(prompt_ids, completion_ids)

    assert len(input_ids) == len(labels) == 7
    # Prompt tokens must be masked to -100
    for i in range(len(prompt_ids)):
        assert labels[i] == -100
        assert input_ids[i] == prompt_ids[i]

    # Completion tokens must be preserved for training loss
    for j in range(len(completion_ids)):
        idx = len(prompt_ids) + j
        assert labels[idx] == completion_ids[j]
        assert input_ids[idx] == completion_ids[j]


def test_token_level_loss_mask_tensor() -> None:
    prompt_ids = list(range(10, 20))
    completion_ids = list(range(100, 105))

    input_ids, labels = compute_labels_for_prompt_completion(prompt_ids, completion_ids)
    labels_tensor = torch.tensor(labels)

    # Prompt portion has -100
    assert (labels_tensor[: len(prompt_ids)] == -100).all()
    # Completion portion matches completion_ids
    assert (labels_tensor[len(prompt_ids) :] == torch.tensor(completion_ids)).all()
