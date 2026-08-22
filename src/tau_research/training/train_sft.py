"""SFT Training Pipeline for tau-research with conversational prompt-completion format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SFTTrainingConfig:
    model_name: str
    output_dir: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    bf16: bool
    max_seq_length: int = 4096
    enable_thinking: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTTrainingConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        m = data.get("model", {})
        t = data.get("training", {})
        d = data.get("dataset", {})

        return cls(
            model_name=m.get("name_or_path", "Qwen/Qwen3.5-2B"),
            output_dir=t.get("output_dir", "artifacts/models/qwen3.5-2b-tau-retail-sft"),
            learning_rate=float(t.get("learning_rate", 1e-4)),
            num_train_epochs=int(t.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
            gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
            bf16=bool(t.get("bf16", True)),
            max_seq_length=int(d.get("max_seq_length", 4096)),
            enable_thinking=bool(m.get("enable_thinking", True)),
            seed=int(t.get("seed", 42)),
        )


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool = False,
) -> str:
    """Apply chat template, preferring enable_thinking when the tokenizer supports it."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        # Older tokenizers may not accept enable_thinking.
        return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_sft_dataset_for_trainer(
    raw_examples: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool = True,
) -> list[dict[str, str]]:
    """Converts conversational turn examples into text prompt-completion pairs
    formatted by chat template with enable_thinking from config.
    """
    formatted: list[dict[str, str]] = []
    for ex in raw_examples:
        prompt_msgs = ex["prompt"]
        completion_msgs = ex["completion"]

        prompt_text = _apply_chat_template(
            tokenizer,
            prompt_msgs,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        completion_text = _apply_chat_template(
            tokenizer,
            completion_msgs,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
        )

        formatted.append(
            {
                "prompt": prompt_text,
                "completion": completion_text,
            }
        )
    return formatted
