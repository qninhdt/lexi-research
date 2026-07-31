---
title: "Lexi Lab — research and inference"
description: "Turn lexi-research into a complete lab: a full CLI surface, a model-agnostic QLoRA trainer, three RL tracks (GRPO/JEPO/NRT) sharing one reward mask, a metric harness with W&B panels, and an inference benchmark across engines, quantisations and speculative decoding."
status: in-progress
priority: P1
size: L
tags: [ml, distillation, rl, grpo, jepo, nrt, inference, vllm, mlops]
created: 2026-08-01
---

# Lexi Lab — research and inference

## Overview

Design doc: [`docs/lexi-lab-design.md`](../../docs/lexi-lab-design.md) — authoritative
for the loss architecture, ablation matrix, metrics, and hardware.
Parent design: [`docs/grader-distillation-design.md`](../../docs/grader-distillation-design.md)
— still authoritative for the I/O contract, taxonomy, and band derivation.

The repo already has a sound format core, teacher pipeline, and serving shim. This
plan adds everything downstream of the data: a real trainer, RL, evaluation, and
an inference lab, on one MLOps spine.

**Goal is learning + resume.** Not cost reduction, not latency, not a leaderboard
number. Scope decisions are made against that goal, not against a product.

## Non-negotiables

1. **No Python in a notebook.** Every action is a `lexi …` subcommand. The Colab
   notebook is a launcher: clone, install, secrets, `dvc pull`, invoke, push
   artifact. An experiment that needs a code change gets a commit, not a cell.
2. **No module names a model.** The base model is a value in `params.yaml`.
   Loading reads the checkpoint's own `config.architectures`, prompt format comes
   from the tokenizer's chat template, and LoRA targets are resolved by role from
   the loaded module tree. Training a different model must never be a code
   change. See design §2.
3. **`feedback` never receives reward signal** in any RL track — only SFT. See
   design §3. This is what makes the three tracks comparable.
4. **Phase 2 (eval harness) precedes Phase 4 (RL).** A negative RL result from a
   trusted harness is a result; from an untrusted one it is indistinguishable
   from a bug.
5. **Cut ablations before cutting `serve/` and `bench/`.** The engineering half is
   what stops the work reading as research-only.

## Phases

| # | Title | Depends on | Delivers | Status |
|---|---|---|---|---|
| 0 | [CLI surface and trainer rewrite](phase-00-cli-surface-and-trainer-rewrite.md) | — | `lexi` entry point, model-agnostic trainer, 50-row fixture, `lexi smoke` green on CPU | done |
| 1 | [DVC pipeline and MLOps spine](phase-01-dvc-pipeline-and-mlops-spine.md) | 0 | all stages wired, W&B lineage, Colab notebook, CI smoke job | done |
| 2 | [Eval harness and W&B panels](phase-02-eval-harness-and-wandb-panels.md) | 1 | full metric suite, teacher-as-judge, custom panels | done |
| 3 | [SFT and architecture ablations](phase-03-sft-and-architecture-ablations.md) | 2 | sweep runner, resume, in-loop eval; A2/A6/A7 arms defined | infra done, runs need a GPU |
| 4 | [RL — GRPO, JEPO, NRT](phase-04-rl-grpo-jepo-nrt.md) | 3 | three tracks on one mask, all green on CPU; A1/A3/A4 arms defined | impl done, runs need a GPU |
| 5 | [Inference lab](phase-05-inference-lab.md) | 2 | engine adapters, open-loop bench harness; B1–B7 defined | harness done, sweeps need a GPU |
| 6 | [MoE comparison and write-up](phase-06-moe-comparison-and-writeup.md) | 4, 5 | B8, model card, W&B report | pending |

**Phase 5 depends only on Phase 2.** Run it in parallel with 3 and 4 — train on
Colab while benchmarking on the rented GPU.

```
0 ── 1 ── 2 ─┬─ 3 ── 4 ─┬─ 6
             └─ 5 ──────┘
```

## Acceptance criteria for the plan as a whole

- `lexi smoke` runs the entire pipeline on a 50-row fixture on CPU and exits 0.
- `lexi smoke --gpu` does the same on a GPU with the checkpoint in
  `train.base_model` and prints how much of that model the adapter reached.
- Switching base model is an edit to `train.base_model` and nothing else.
- Every ablation in design §4 is runnable as a single `lexi` invocation with a
  config override — no code edit required to change an arm.
- Every number in the final report traces to a W&B run, which traces to a DVC
  stage hash, which traces to a commit.
- `notebooks/lexi_colab.ipynb` contains no cell that defines a function or class.

## Open questions (resolve during execution, not before)

- Does PEFT attach cleanly to a Gated-DeltaNet stack's `in_proj_*` projections?
  Phase 3's first GPU run answers this empirically; the answer decides A6's arms.
- Which nightly digests of vLLM and SGLang support the reference model? Phase 5
  pins them.
- Is `lambda` for the RL term stable across the three tracks, or does each need
  its own? Phase 4 sweeps it on the 50-row fixture first.
- Training examples measure ~1250 whitespace tokens, almost all of it the rubric.
  Phase 3 should measure whether a shortened system prompt costs accuracy — at
  this length the prompt dominates both the sequence budget and the prefill cost.
