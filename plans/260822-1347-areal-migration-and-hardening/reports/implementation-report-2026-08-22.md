# Implementation Report — AReaL Migration & Hardening

**Date**: 2026-08-22 | **Commits**: 5bbec1c..28a4e58 (8 commits)

## What changed

| Commit | Content |
|---|---|
| 5bbec1c | Baseline: legacy lexi removal + tau scaffolding as-is |
| bf58901 | Plan documents |
| 22eacf5 | AReaL converter, single-pass render (Bug A fix), real SFT entrypoint, JSON-args parser (Bug B fix) |
| 8fe2c0d | AgentGymEnv factory, reward_info parsing, real difficulty profiler |
| 1d3da85 | Policy loaders (HF/vLLM), decoding plumbing, Pass^k, paired bootstrap CI |
| 4b002d1 | Multi-turn GRPO rollout_func with env_mask token masking |
| d16f17a | Real pipeline scripts + profile-difficulty CLI + README |
| 28a4e58 | Full dataset conversion + decontamination audit + 8192 context cap |

## Acceptance criteria results

1. **Render pass**: 21/21 fixture examples and full-data dry-run render 100% through the real Qwen3.5 tokenizer (was 0%).
2. **Tool-call round-trip**: legacy `call:name({json})` and canonical `name(k=v)` both preserve arguments; unit-tested.
3. **Real training path**: `train-sft` instantiates TRL SFTTrainer (`completion_only_loss`, LoRA all-linear); `train-grpo` wires GRPOTrainer with custom rollout_func; dry-run modes validate wiring without weights.
4. **Real env**: TauEnvFactory loads official split_tasks.json — verified 74 train / 40 test disjoint IDs; reward_info JSON parsed for DB/COMMUNICATE.
5. **No fabricated artifacts**: scripts call CLIs only; smoke_test.sh passes end-to-end on CPU.
6. **Gates**: ruff clean, mypy strict clean (46 files), 56 tests passing.

## Data artifacts produced

- `artifacts/data/areal_sft_{train,val}.json`: **10,196 / 1,091 examples**, dialog-disjoint (892/100), from 33,531 rows (11,287 retail verified-success kept).
- `artifacts/evaluation/decontamination_report.json`: **0 flagged pairs** vs official test split (8-gram Jaccard, threshold 0.05).
- `artifacts/data/length_profile.json`: p90=6528 / p95=7250 → `max_seq_length=8192` (drops 1.5%).

## Known limitations / next steps

- SFT/GRPO full runs happen on Colab L4 (local box has RTX 3050 only); smoke gates cover wiring, not VRAM behavior at 8k.
- GRPO difficulty-weighted repetition is static (profile-based); TRL-side zero-variance dynamic filtering relies on loss_type=dapo defaults.
- flash-attn wheel in colab extras points to a third-party release — verify checksum before use.
