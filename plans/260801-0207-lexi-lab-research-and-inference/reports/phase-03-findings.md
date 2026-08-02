# Phase 3 findings — SFT and architecture ablations

**Status: infrastructure complete, runs pending hardware.**

Everything an arm needs is in place and exercised on CPU: the three ablation
definitions, the sweep runner with resume, in-loop evaluation, checkpoint
resolution, and the `forced-empty` arm that makes A2 interpretable. What is
missing is a GPU and a real dataset, so the tables below are the shape the
results will take, not results.

Nothing here is a placeholder for a number that was measured and lost. Every row
marked *pending* has never been run.

## What can be stated now

- **The completion is 3.2% of the sequence** on the 50-row fixture, measured by
  `lexi smoke`. That is the size of the defect A7 quantifies: the trainer this
  work replaced spent roughly 97% of its gradient reproducing a rubric that is
  supplied verbatim at inference. A7's `full-sequence` arm should therefore be
  *measurably* worse; if it is not, the mask is not doing what it claims and the
  finding is about the mask.
- **Seven A6 arms enumerate cleanly** from `ops/ablations/a6-lora.yaml`: three
  presets crossed with rank 8 and 64, plus the legacy `q/k/v/o_proj` name list at
  rank 32. Every override in every arm resolves against `params.yaml`, so no arm
  can silently fail to change anything — a test asserts this.
- **`train.thinking` has three values, not two.** A win for `on` over `off` is
  uninterpretable without `forced-empty`: it could mean reasoning helps, or that
  opening a `<think>` block alone shifts the distribution the answer is sampled
  from.

## A7 — loss mask

| Arm | QWK | span+tag F1 | validity | Notes |
|---|---|---|---|---|
| `completion-only` | pending | pending | pending | |
| `full-sequence` | pending | pending | pending | |

## A2 — thinking

Report generated tokens per request alongside quality. Reasoning that buys
0.01 QWK for four times the tokens is a loss for this task.

| Arm | QWK | span+tag F1 | tokens/request | Notes |
|---|---|---|---|---|
| `on` | pending | pending | pending | |
| `off` | pending | pending | pending | |
| `forced-empty` | pending | pending | pending | |

## A6 — LoRA placement and rank

The interesting result may be *equal quality at fewer parameters*, so the
parameter and VRAM columns are not decoration.

| Arm | Modules | Attention layers | Trainable params | Peak VRAM | QWK |
|---|---|---|---|---|---|
| `attn` r8 | pending | pending | pending | pending | pending |
| `attn+mlp` r8 | pending | pending | pending | pending | pending |
| `all-linear` r8 | pending | pending | pending | pending | pending |
| `attn` r64 | pending | pending | pending | pending | pending |
| `attn+mlp` r64 | pending | pending | pending | pending | pending |
| `all-linear` r64 | pending | pending | pending | pending | pending |
| legacy name list r32 | pending | pending | pending | pending | pending |

The legacy arm is the one that turns "the old target list was a bug" from an
assertion into a number. On a stack whose projections are named conventionally it
should tie with `attn+mlp`; on one whose are not, the gap is the cost of the
defect.

## Open question this phase must settle

`make smoke-gpu` is the first run. It answers what CPU could not: whether PEFT
and the quantiser attach to this architecture's projections at all, and to how
much of the stack. The resolved coverage prints before the first optimiser step,
so a near-empty adapter is visible in seconds rather than after an epoch.

If `all-linear` fails on the reference model, A6 loses arms and that failure is
itself the finding — recorded here rather than quietly dropped.

## A second question, raised by Phase 0

A rendered example is ~1250 whitespace tokens, almost all of it the rubric. At
that length the prompt dominates both the sequence budget and the prefill cost.
Whether a shortened system prompt costs accuracy is worth one arm, and it is
cheap to add: the prompt is a template, so the arm is an override.
