---
phase: 4
title: "Baseline & SFT Training Pipeline"
status: completed
priority: P1
dependencies: [2, 3]
---

# Phase 04: Baseline & SFT Training Pipeline

## Overview
Implement the baseline evaluation script for zero-shot `Qwen/Qwen3.5-2B` on held-out τ³ Retail tasks, along with the complete SFT training pipeline using TRL `SFTTrainer` (LoRA, BF16, gradient checkpointing) and an adapter merging utility to create `qwen3.5-2b-tau-retail-sft-merged`.

## Requirements
- Functional:
  - `train_sft.py`: TRL `SFTTrainer` setup loading preprocessed per-turn dataset, initializing LoRA adapter ($r=16, \alpha=32$, `target_modules="all-linear"`), configuring cosine learning rate scheduler ($1\times 10^{-4}$), gradient accumulation steps, and W&B logging.
  - `merge_adapter.py`: Merge base model weights with SFT LoRA adapter into a single standalone model directory (`artifacts/models/qwen3.5-2b-tau-retail-sft-merged`).
  - Baseline evaluation harness setup to record pre-training performance of base Qwen3.5-2B.
- TDD / Test Suite:
  - `tests/test_sft_train_step.py`: CPU-based unit test executing a 2-step mock training run with a dummy model/tokenizer to verify loss computation and checkpoint saving.
  - `tests/test_merge_adapter.py`: Test creating a mock PEFT model, saving an adapter, merging it with base weights, and asserting identical output logits.

## Related Code Files
- Create:
  - `src/tau_research/training/__init__.py`
  - `src/tau_research/training/train_sft.py`
  - `src/tau_research/training/merge_adapter.py`
  - `src/tau_research/logging/__init__.py`
  - `src/tau_research/logging/wandb_callbacks.py`
  - `tests/test_sft_train_step.py`
  - `tests/test_merge_adapter.py`
- Modify:
  - `configs/sft.yaml`

## Implementation Steps
1. **Write TDD Tests First**:
   - Create `tests/test_sft_train_step.py` with tiny synthetic dataset and miniature model to test SFTTrainer step execution.
   - Create `tests/test_merge_adapter.py` asserting weight equivalence before and after merge.
2. Implement `wandb_callbacks.py` tracking train/val loss, learning rate, GPU memory allocation (`gpu/memory_allocated_gb`), and step throughput.
3. Implement `train_sft.py`:
   - Load preprocessed dataset from `artifacts/splits/`.
   - Setup tokenizer and model with BF16 and PEFT config.
   - Setup `SFTConfig` with `dataset_text_field=None`, `completion_only_loss=True`, and checkpoint intervals.
   - Execute training and save `best-val-loss` and `final-sft-adapter`.
4. Implement `merge_adapter.py`:
   - Load base model and LoRA adapter.
   - Execute `model.merge_and_unload()`.
   - Save merged model and tokenizer to destination path.

## Success Criteria
- [ ] `uv run pytest tests/test_sft_train_step.py tests/test_merge_adapter.py` passes.
- [ ] SFT trainer executes without OOM on 24GB VRAM with max sequence length 4096.
- [ ] Merged model loads cleanly into standalone vLLM / PyTorch pipelines without requiring adapter flags.
