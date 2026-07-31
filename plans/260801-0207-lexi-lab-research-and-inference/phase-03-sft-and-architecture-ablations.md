---
phase: 3
title: "SFT and architecture ablations"
status: pending
priority: P1
size: M
dependencies: [2]
---

# Phase 3: SFT and architecture ablations

## Overview

First real training run, and the three ablations that are properties of the
*architecture and the loss* rather than of the method. They come before RL because
RL inherits every one of these choices — running A6 after A4 would mean every RL
arm was trained with an unknown LoRA placement.

Ablations here: **A2** (thinking), **A6** (LoRA target and rank), **A7** (loss mask).

## Requirements

**Functional**

- `lexi train sft` trains a real adapter on `train.parquet`, validates on
  `val.parquet`, checkpoints to Drive or W&B, and resumes after a Colab kill.
- Each ablation arm is a single `--override`. Changing an arm never edits code.
- Every run logs the resolved LoRA target-module list and count — not just the
  preset name — so an arm can be audited after the fact.
- `lexi train sweep --ablation a6` enumerates the arms and launches them in order,
  resuming from wherever it stopped.

**Non-functional**

- Fits Colab A100 40 GB and degrades to T4 16 GB through config alone.
- Deterministic given seed plus config; the seed is in the lineage dict.
- A killed session loses at most one checkpoint interval.

## Ablation arms

**A7 — loss mask (run first, 2 arms).** Completion-only vs full-sequence. This is
the cheapest arm and it validates the Phase 0 fix on real data. If full-sequence
is not measurably worse, something is wrong with the mask, not with the finding.

**A2 — thinking (3 arms).** `on`, `off`, `forced-empty` (thinking enabled but the
model is trained to emit `<think></think>`). The third arm separates *reasoning
helps* from *the `<think>` scaffold alone helps*, and without it a win for `on` is
uninterpretable. Report tokens generated per request alongside quality — reasoning
that buys 0.01 QWK for 4× the tokens is a loss for this task.

**A6 — LoRA placement and rank (7 arms).** Presets `attn`, `attn+mlp`,
`all-linear` crossed with rank 8 and 64 — rank 32 is the Phase 0 default and is
already measured — plus one arm pinning the legacy `q/k/v/o_proj` + MLP name
list, which is what quantifies the cost of the bug Phase 0 removed on whatever
architecture this run uses. Role-resolved placement on a hybrid linear-attention
stack has no prior art, so report trainable-parameter count and peak VRAM
alongside quality: the interesting result may be *equal quality at fewer
parameters*, not higher quality.

The first GPU run of this phase is also `make smoke-gpu`, which is where the
question Phase 0 could not answer on CPU gets settled — whether PEFT and the
quantiser attach to this architecture's projections at all, and to how much of
it. The resolved coverage is printed before the first optimiser step.

Arms run in that order. A7 is a correctness check, A2 changes the data format, A6
is the widest sweep and should run last with the other two settled.

## Files

**Create**

- `lexi_research/train/sft.py` — trainer built on the Phase 0 helpers
- `lexi_research/train/callbacks.py` — in-loop eval, qualitative W&B table, checkpoint/resume
- `lexi_research/train/sweep.py` — ablation arm enumeration and resumable launching
- `ops/ablations/a2-thinking.yaml`, `a6-lora.yaml`, `a7-mask.yaml` — arm definitions
- `tests/train/test_sweep.py`, `tests/train/test_callbacks.py`

**Modify**

- `lexi_research/train/trainer.py` — becomes a thin dispatch to `sft.py` / later `rl/`
- `params.yaml` — real training hyperparameters, checkpoint interval, eval interval
- `dvc.yaml` — `sft` stage
- `notebooks/lexi_colab.py` — the sweep invocation

## Implementation steps

1. **`sft.py` on the Phase 0 helpers.** No new masking or templating logic here;
   if something is missing, it belongs in `collate.py` where it is already tested.
2. **In-loop eval.** Every N steps, run the Phase 2 harness on a fixed val subset
   and log to W&B, including the qualitative table. Watching only loss for hours
   and discovering at the end that format validity collapsed is the failure mode
   this prevents.
3. **Checkpoint and resume.** Save adapter plus optimiser state plus RNG state at
   an interval; `--resume auto` picks the latest. Verify by killing a run and
   restarting — resume that has never been exercised does not work.
4. **Sweep runner.** Arms enumerated from a YAML file; state written after each
   arm so a killed sweep resumes at the next one. Each arm is its own W&B run,
   grouped by ablation name.
5. **Run A7, then A2, then A6**, writing findings into the phase report as each
   completes rather than at the end.

## Tests

| Test | Asserts |
|---|---|
| `test_sweep.py::test_arms_enumerated` | the A6 YAML expands to exactly 6 arms with the expected override sets |
| `test_sweep.py::test_resume_skips_completed` | a sweep resumed after 2 of 6 arms launches only the remaining 4 |
| `test_callbacks.py::test_eval_interval` | the harness is invoked at the configured step interval, not every step |
| `test_callbacks.py::test_checkpoint_roundtrip` | adapter + optimiser + RNG state save and reload to identical values |
| `make smoke` | still green — the SFT stage runs 2 steps on the fixture |

## Acceptance

- One adapter trained on real data, pushed as a W&B artifact with `band_config.json`.
- Phase 2 report generated for it, with every metric normalised to the ceiling.
- A7, A2, A6 complete, each a W&B group with a parallel-coordinates panel.
- A killed and resumed run reaches the same final metrics as an uninterrupted one,
  within seed noise.
- Findings written to `plans/…/reports/phase-03-findings.md`, including the
  trainable-parameter and VRAM columns for A6.

## Risks

| Risk | Handling |
|---|---|
| PEFT or the quantiser rejects this architecture's projections | `make smoke-gpu` is the first run of this phase and surfaces it in minutes; if `all-linear` fails, A6 loses arms and the failure is a documented finding |
| Colab session limits make the A6 sweep painful | Sweep state is resumable per arm; each arm is sized to fit one session |
| Thinking arms change sequence length enough to change effective batch size | Report tokens-per-step alongside quality; hold token budget rather than step count constant where they conflict |
| In-loop eval slows training badly | Fixed small val subset, interval in config; the subset is fixed across arms so numbers stay comparable |
