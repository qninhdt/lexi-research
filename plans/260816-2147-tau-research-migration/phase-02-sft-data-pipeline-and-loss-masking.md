---
phase: 2
title: "SFT Data Pipeline & Loss Masking"
status: completed
priority: P1
dependencies: [1]
---

# Phase 02: SFT Data Pipeline & Loss Masking

## Overview
Implement the SFT data processing pipeline that downloads `fuvty/tau-bench-synthetic`, filters for successful Retail trajectories (`reward == 1.0`), decomposes multi-turn trajectories into per-turn conversational training examples with prior `<think>` reasoning stripped, and generates leakage-free train/val splits partitioned by `task_id`.

## Requirements
- Functional:
  - `build_splits.py`: Split synthetic tasks by `task_id` (90% train / 10% val, seed=42) and save split task IDs to `artifacts/splits/`.
  - `prepare_sft.py`: Process synthetic trajectories into conversational prompt-completion pairs:
    - `prompt`: `[system, user_turn_1, assistant_action_1 (no think), tool_res_1, ...]`
    - `completion`: `[assistant_current_turn (think + action/message)]`
  - `profile_lengths.py`: Profile token distributions (P50, P90, P95, P99, max) with Qwen3.5 tokenizer.
  - `validate_dataset.py`: Assert that test leakage is zero and labels for prompt tokens are masked to `-100`.
- TDD / Test Suite:
  - `tests/test_chat_template.py`: Validate that Qwen3.5 chat template formats tool definitions, strips old thinking from history, and retains current turn thinking.
  - `tests/test_loss_mask.py`: Validate that `DataCollatorForCompletionOnlyLM` / TRL conversational template masks all prompt tokens (`labels == -100`) and only trains on assistant completion tokens.
  - `tests/test_no_test_leakage.py`: Verify that zero task IDs from the official τ³ Retail test split appear in SFT train/val splits.

<!-- Red Team Finding 4 Fix: Token-Level Label Mask Verification -->
- **Token-Level Loss Masking Guard**:
  - `tests/test_loss_mask.py` must include exact token ID verification asserting `labels[i] == -100` for all tokens prior to `<think>` start and `labels[i] == input_ids[i]` for all tokens inside `<think>...</think>` and tool calls.

## Related Code Files
- Create:
  - `src/tau_research/data/__init__.py`
  - `src/tau_research/data/build_splits.py`
  - `src/tau_research/data/prepare_sft.py`
  - `src/tau_research/data/profile_lengths.py`
  - `src/tau_research/data/validate_dataset.py`
  - `tests/test_chat_template.py`
  - `tests/test_loss_mask.py`
  - `tests/test_no_test_leakage.py`
- Modify:
  - `configs/sft.yaml`

## Implementation Steps
1. **Write TDD Tests First**:
   - Create `tests/test_chat_template.py` with mock Qwen3.5 conversation turns verifying `<think>` removal in multi-turn history.
   - Create `tests/test_loss_mask.py` verifying exact token-level label tensor values (`-100` on prompt, `>= 0` on completion including `<think>` tokens).
   - Create `tests/test_no_test_leakage.py` comparing synthetic task IDs against official test task IDs.
2. Implement `build_splits.py` with deterministic hashing / seeded shuffle over synthetic `task_id`s.
3. Implement `prepare_sft.py`:
   - Load `fuvty/tau-bench-synthetic`.
   - Filter `domain == "retail"`, `reward == 1.0`, `termination == "normal"`.
   - Iterate turns: build prompt with sanitized previous history and target completion with current thinking + action.
4. Implement `profile_lengths.py` using `rich` console output for token percentiles.
5. Implement `validate_dataset.py` as an executable gate before training starts.

## Success Criteria
- [ ] `uv run pytest tests/test_chat_template.py tests/test_loss_mask.py tests/test_no_test_leakage.py` passes 100%.
- [ ] Synthetic trajectories successfully decompose into per-turn conversational dataset.
- [ ] No `task_id` overlap between SFT train, SFT val, and held-out τ³ test split.
