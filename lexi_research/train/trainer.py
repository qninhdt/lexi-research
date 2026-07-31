"""QLoRA training wiring; imports heavy GPU libraries only when training starts."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from .dataset import training_example


def train(
    train_path: str | Path, output_dir: str | Path, *, base_model: str, epochs: int = 2
) -> None:
    """Fine-tune a Qwen-style causal LM with 4-bit QLoRA on processed parquet.

    Requires the optional Colab dependencies in `requirements-colab.txt`; keeping
    them out of the base environment preserves fast, CPU-only CI.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised on Colab
        raise RuntimeError("install requirements-colab.txt before starting QLoRA training") from exc

    rows = pq.read_table(train_path).to_pylist()
    examples = [training_example(row) for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    dataset = Dataset.from_list(
        [{"text": f"{row['system']}\n\n{row['user']}\n\n{row['completion']}"} for row in examples]
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=1e-4,
            bf16=True,
            save_steps=100,
            gradient_checkpointing=True,
            report_to="none" if os.environ.get("WANDB_DISABLED") else "wandb",
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
