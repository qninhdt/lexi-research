---
phase: 3
title: "SFT and architecture ablations"
status: blocked-on-hardware
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

**Created**

- `lexi_research/train/callbacks.py` — in-loop eval, qualitative W&B table, checkpoint resolution
- `lexi_research/train/sweep.py` — ablation arm enumeration and resumable launching
- `ops/ablations/a2-thinking.yaml`, `a6-lora.yaml`, `a7-mask.yaml` — arm definitions
- `tests/train/test_sweep.py`, `tests/train/test_callbacks.py`
- `plans/…/reports/phase-03-findings.md`

**Modified**

- `lexi_research/train/collate.py` — `train.thinking` with three arms; `forced-empty` supervises an empty reasoning block
- `lexi_research/train/trainer.py` — resume, in-loop eval, val rows
- `lexi_research/cli/__init__.py` — `lexi train sweep`; `--val`, `--ceiling`, `--resume` on `sft`
- `params.yaml` — `train.thinking`, `eval_steps`, `eval_subset`
- `dvc.yaml` — `sft` stage takes val and the ceiling
- `notebooks/lexi_colab.py` — the sweep and scoring invocations

**Deviation.** The plan called for a new `sft.py` with `trainer.py` reduced to a
dispatcher. `trainer.py` already *is* the SFT trainer built on the Phase 0
helpers; renaming it would churn every DVC dep and every import for no behaviour
change. The dispatcher arrives when there is a second trainer to dispatch to,
which is Phase 4.

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

Infrastructure — done and exercised on CPU:

- [x] Each arm is a single `--override`; changing one never edits code. A test
      resolves every override of every arm against `params.yaml`, so an arm
      cannot silently fail to change anything.
- [x] `lexi train sweep --ablation a6` enumerates 7 arms and launches them in
      order, recording state after each so a killed session resumes at the next.
- [x] Every run logs the resolved target-module list and coverage, not just the
      preset name.
- [x] `--resume auto` picks the latest checkpoint by step, and an explicit
      missing checkpoint raises rather than silently starting from zero.
- [x] In-loop eval fires on the configured interval and is switchable off.
- [x] `make smoke` still green; `uv run pytest` 632.

Experiments — blocked on a GPU and a real dataset:

- [ ] One adapter trained on real data, pushed as a W&B artifact with its band config.
- [ ] Phase 2 report generated for it, normalised to the ceiling.
- [ ] A7, A2, A6 complete, each a W&B group.
- [ ] A killed and resumed run reaching the same final metrics as an
      uninterrupted one, within seed noise. Resume is unit-tested; that it
      *converges* the same is not, and cannot be until there is a real run.
- [ ] Findings filled in at `plans/…/reports/phase-03-findings.md`, which
      currently states plainly that every row is pending.

## Risks

| Risk | Handling |
|---|---|
| PEFT or the quantiser rejects this architecture's projections | `make smoke-gpu` is the first run of this phase and surfaces it in minutes; if `all-linear` fails, A6 loses arms and the failure is a documented finding |
| Colab session limits make the A6 sweep painful | Sweep state is resumable per arm; each arm is sized to fit one session |
| Thinking arms change sequence length enough to change effective batch size | Report tokens-per-step alongside quality; hold token budget rather than step count constant where they conflict |
| In-loop eval slows training badly | Fixed small val subset, interval in config; the subset is fixed across arms so numbers stay comparable |
