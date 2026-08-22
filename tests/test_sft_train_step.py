"""Tests for SFT configuration loading, single-pass rendering, and trainer setup."""

from pathlib import Path
from typing import Any

from tau_research.training.train_sft import SFTTrainingConfig, prepare_sft_dataset_for_trainer


def test_sft_training_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
model:
  name_or_path: "Qwen/Qwen3.5-2B"
  enable_thinking: true

dataset:
  train_path: "artifacts/data/areal_sft_train.json"
  val_path: "artifacts/data/areal_sft_val.json"
  max_seq_length: 6144

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

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
    assert cfg.train_path == "artifacts/data/areal_sft_train.json"
    assert cfg.max_seq_length == 6144
    assert cfg.learning_rate == 1.0e-4
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.gradient_checkpointing is True
    assert cfg.lora_r == 16


class FakeQwenTemplate:
    """Fake tokenizer mimicking the Qwen3.5 template's boundary property.

    The final assistant turn (after the last user query) renders as
    ``<|assistant|>`` + generation header + content; the prompt render with
    ``add_generation_prompt=True`` ends with exactly the same header string.
    """

    GEN_HEADER = "<|im_start|>assistant\n<think>\n"

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False, **kw: Any
    ) -> str:
        del tokenize
        enable = kw.get("enable_thinking", True)
        add_gen = kw.get("add_generation_prompt", False)
        header = "THINK_ON" if enable else "THINK_OFF"

        user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
        last_user_idx = user_idxs[-1] if user_idxs else None

        out = header
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and (last_user_idx is None or i > last_user_idx):
                out += "<|assistant|>" + self.GEN_HEADER + m["content"] + "<|im_end|>\n"
            else:
                out += f"<|{m['role']}|>{m['content']}<|end|>\n"
        if add_gen:
            out += "<|assistant|>" + self.GEN_HEADER
        return out


def test_prepare_sft_dataset_boundary_split() -> None:
    raw = [
        {
            "prompt": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "Cancel my order"},
            ],
            "completion": [
                {"role": "assistant", "content": "<think>Ok</think>Done"}
            ],
        }
    ]
    formatted = prepare_sft_dataset_for_trainer(raw, FakeQwenTemplate(), enable_thinking=True)
    assert len(formatted) == 1
    ex = formatted[0]
    # Prompt ends at the generation header; completion is the exact suffix.
    assert ex["prompt"].endswith(FakeQwenTemplate.GEN_HEADER)
    assert ex["prompt"] + ex["completion"] == (
        FakeQwenTemplate().apply_chat_template(
            raw[0]["prompt"] + raw[0]["completion"], enable_thinking=True
        )
    )


def test_prepare_sft_dataset_skips_unrenderable_prompts() -> None:
    class RaisingTemplate(FakeQwenTemplate):
        def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, **kw: Any) -> str:
            if not any(m["role"] == "user" for m in messages):
                raise ValueError("No user query found in messages.")
            return super().apply_chat_template(messages, tokenize, **kw)

    raw = [
        {
            "prompt": [{"role": "system", "content": "policy"}],
            "completion": [{"role": "assistant", "content": "<think>hi</think>Hello!"}],
        },
        {
            "prompt": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hi"},
            ],
            "completion": [{"role": "assistant", "content": "<think>x</think>Yo"}],
        },
    ]
    formatted = prepare_sft_dataset_for_trainer(raw, RaisingTemplate(), enable_thinking=True)
    assert len(formatted) == 1
    assert "Yo" in formatted[0]["completion"]
