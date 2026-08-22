"""SFT Training Pipeline: AReaL prompt/completion data into a LoRA reasoning agent.

Rendering contract (single-pass): each example is rendered ONCE as
``prompt + completion`` through the official chat template, and the completion
string is cut at the prompt/assistant boundary. Rendering the completion alone
violates the Qwen3.5 template (it requires a user query) and was the source of
a 100% crash; the single-pass render also guarantees train/inference format
identity, including the pre-emitted ``<think>\n`` from ``enable_thinking``.
"""

from __future__ import annotations

import json
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
    train_path: str = "artifacts/data/areal_sft_train.json"
    val_path: str = "artifacts/data/areal_sft_val.json"
    max_seq_length: int = 8192
    enable_thinking: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    wandb_project: str = "tau-research"
    run_name: str = "qwen35-2b-sft-retail"
    merged_dir: str = "artifacts/models/qwen3.5-2b-tau-retail-sft-merged"
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTTrainingConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        m = data.get("model", {})
        t = data.get("training", {})
        d = data.get("dataset", {})
        lora = data.get("lora", {})
        w = data.get("wandb", {})

        return cls(
            model_name=m.get("name_or_path", "Qwen/Qwen3.5-2B"),
            output_dir=t.get("output_dir", "artifacts/models/qwen3.5-2b-tau-retail-sft"),
            learning_rate=float(t.get("learning_rate", 1e-4)),
            num_train_epochs=int(t.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
            gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
            bf16=bool(t.get("bf16", True)),
            train_path=str(d.get("train_path", "artifacts/data/areal_sft_train.json")),
            val_path=str(d.get("val_path", "artifacts/data/areal_sft_val.json")),
            max_seq_length=int(d.get("max_seq_length", 8192)),
            enable_thinking=bool(m.get("enable_thinking", True)),
            lora_r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("lora_alpha", 32)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(t.get("warmup_ratio", 0.05)),
            logging_steps=int(t.get("logging_steps", 5)),
            eval_steps=int(t.get("eval_steps", 50)),
            save_steps=int(t.get("save_steps", 50)),
            save_total_limit=int(t.get("save_total_limit", 2)),
            seed=int(t.get("seed", 42)),
            wandb_project=str(w.get("project", "tau-research")),
            run_name=str(w.get("run_name", "qwen35-2b-sft-retail")),
            report_to=str(t.get("report_to", "none")),
        )


def load_sft_examples(path: str | Path) -> list[dict[str, Any]]:
    """Loads jsonl prompt/completion records produced by the AReaL converter."""
    examples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


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
        rendered = tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        # Older tokenizers may not accept enable_thinking.
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return str(rendered)


def prepare_sft_dataset_for_trainer(
    raw_examples: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool = True,
) -> list[dict[str, str]]:
    """Converts conversational turn examples into text prompt/completion pairs.

    Each example is rendered once as prompt+completion through the chat
    template; the completion is the exact string suffix after the
    ``add_generation_prompt=True`` render of the prompt. Examples whose prompt
    cannot be rendered (e.g. no user query yet) are skipped.
    """
    formatted: list[dict[str, str]] = []
    for ex in raw_examples:
        prompt_msgs = list(ex["prompt"])
        completion_msgs = list(ex["completion"])
        try:
            prompt_text = _apply_chat_template(
                tokenizer,
                prompt_msgs,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
            full_text = _apply_chat_template(
                tokenizer,
                prompt_msgs + completion_msgs,
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
            )
        except Exception:
            continue
        if not full_text.startswith(prompt_text):
            continue
        formatted.append(
            {
                "prompt": prompt_text,
                "completion": full_text[len(prompt_text) :],
            }
        )
    return formatted


def run_sft_training(
    config: SFTTrainingConfig,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Runs LoRA reasoning SFT with TRL SFTTrainer and merges the final adapter.

    Returns a summary dict with example counts, skip counts, and output paths.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_raw = load_sft_examples(config.train_path)
    val_raw = load_sft_examples(config.val_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_formatted = prepare_sft_dataset_for_trainer(train_raw, tokenizer, config.enable_thinking)
    val_formatted = prepare_sft_dataset_for_trainer(val_raw, tokenizer, config.enable_thinking)

    summary: dict[str, Any] = {
        "train_raw": len(train_raw),
        "train_formatted": len(train_formatted),
        "train_skipped_render": len(train_raw) - len(train_formatted),
        "val_raw": len(val_raw),
        "val_formatted": len(val_formatted),
        "val_skipped_render": len(val_raw) - len(val_formatted),
    }
    if dry_run:
        summary["dry_run"] = True
        return summary
    if not train_formatted:
        raise ValueError(
            f"All {len(train_raw)} training examples failed chat-template rendering; "
            "check the converter output format."
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_kwargs: dict[str, Any] = dict(
        output_dir=config.output_dir,
        max_length=config.max_seq_length,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=max_steps if max_steps else config.num_train_epochs,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        seed=config.seed,
        report_to=[config.report_to] if config.report_to != "none" else [],
        run_name=config.run_name,
    )
    if max_steps:
        # Explicit step budget replaces the epoch schedule for smoke runs.
        del sft_kwargs["num_train_epochs"]
        sft_kwargs["max_steps"] = max_steps

    training_args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_formatted),
        eval_dataset=Dataset.from_list(val_formatted),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    kept = len(trainer.train_dataset)
    if kept == 0:
        raise ValueError(
            "TRL dropped every training example (usually max_seq_length smaller "
            "than the rendered sequences). Raise max_seq_length or shorten data."
        )
    if kept < len(train_formatted):
        print(
            f"[train-sft] warning: TRL kept {kept}/{len(train_formatted)} examples "
            f"after length filtering (max_seq_length={config.max_seq_length})"
        )
    summary["train_after_length_filter"] = kept

    trainer.train()
    adapter_dir = Path(config.output_dir) / "final-adapter"
    trainer.save_model(str(adapter_dir))

    from tau_research.training.merge_adapter import merge_lora_adapter

    merge_lora_adapter(trainer.model, tokenizer, config.merged_dir, seed=config.seed)
    summary["adapter_dir"] = str(adapter_dir)
    summary["merged_dir"] = str(config.merged_dir)
    return summary
