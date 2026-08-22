"""Tests for SFT configuration loading, dataset formatting, and trainer initialization."""

from pathlib import Path
from unittest.mock import MagicMock

from tau_research.training.train_sft import SFTTrainingConfig, prepare_sft_dataset_for_trainer


def test_sft_training_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
model:
  name_or_path: "Qwen/Qwen3.5-2B"
  torch_dtype: "bfloat16"
  enable_thinking: true

training:
  output_dir: "artifacts/models/test"
  learning_rate: 1.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  bf16: true
"""
    config_file = tmp_path / "sft_test.yaml"
    config_file.write_text(yaml_content)

    cfg = SFTTrainingConfig.from_yaml(config_file)
    assert cfg.model_name == "Qwen/Qwen3.5-2B"
    assert cfg.learning_rate == 1.0e-4
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.gradient_checkpointing is True


def test_prepare_sft_dataset_for_trainer() -> None:
    raw_examples = [
        {
            "prompt": [{"role": "user", "content": "Cancel my order"}],
            "completion": [{"role": "assistant", "content": "<think>Ok</think>Done"}],
        }
    ]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = lambda msgs, tokenize=False: (
        "user: Cancel my order" if msgs[0]["role"] == "user" else "assistant: <think>Ok</think>Done"
    )

    formatted = prepare_sft_dataset_for_trainer(raw_examples, mock_tokenizer)
    assert len(formatted) == 1
    assert "prompt" in formatted[0]
    assert "completion" in formatted[0]
