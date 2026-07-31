---
phase: 4
title: "RL — GRPO, JEPO, NRT"
status: blocked-on-hardware
priority: P1
size: L
dependencies: [3]
---

# Phase 4: RL — GRPO, JEPO, NRT

## Overview

Three reinforcement-learning tracks over one shared reward mask, in a fixed order.
Ablations: **A1** (method), **A3** (reward mask), **A4** (NRT aggregation).

Design references: [`docs/lexi-lab-design.md`](../../docs/lexi-lab-design.md) §3,
and `idea.md` for the JEPO and NRT derivations.

## The shared mask (design §3)

```
segment              SFT-CE   RL-reward   RL-policy-grad
prompt x               —          —             —
<think> z </think>     —          —            yes
gold correction       yes        yes            —
gold meaning          yes        yes            —
gold feedback         yes         —             —
```

```
L_total = CE(correction + meaning + feedback) + lambda * L_RL(z)
```

`feedback` is excluded from reward in every track. It is voice and register:
unverifiable, and rewarding it teaches the model to chase teacher phrasing rather
than grading quality. It is still fully supervised. Because all three tracks share
this mask, differences between them are attributable to the reward definition
alone — that is the whole point of the ordering below.

## Order is not negotiable

**GRPO first.** Its reward is *exogenous* — computed by `lexi_research/format`
code that Phase 2 already tested. When the curve misbehaves, the cause is the
trainer or the data, and both are separately inspectable.

JEPO and NRT rewards are *endogenous*: `R` is a function of `theta`. Reward rising
can mean the model became more confident rather than more correct. There is no way
to tell without a track whose reward cannot drift. Building them first means
debugging two unknowns with one equation.

## Track definitions

| Track | R(z) | Sampling |
|---|---|---|
| GRPO-RLVR | `w1*edit_F1 + w2*(1 - abs(dMeaning)/4) + w3*format_valid - w4*strip_mismatch` | sample K completions, score each with code |
| JEPO | `log pi(correction, meaning \| x, z)` | sample K reasonings, teacher-force the gold answer after each |
| NRT | `f(c_1..c_T)` over gold-token probabilities, `c_i = pi(y*_i \| x, z, y*_<i)` | as JEPO, then aggregate |

NRT aggregations for A4: `seq-logp` (equivalent to single-sample JEPO — the
equivalence is a **test**, not a claim), `geo-mean`, `arith-mean`,
`weighted(-log p_base)`.

Stabilisation, mandatory for JEPO and NRT:

- empty-reasoning baseline `R' = max(0, R(z) - R(empty))`
- group-relative advantage `A_k = (R'_k - mean(R')) / std(R')`
- separate small format-supervision CE on `<think>` / `</think>` markers only
- reference-policy KL, logged every step

`R(empty)` is not only a baseline, it is the primary diagnostic: if the gap
collapses to zero, reasoning has stopped contributing and the run is dead
regardless of what the reward curve says.

## Requirements

**Functional**

- `lexi train rl --algo {grpo,jepo,nrt}` with all three sharing one `Trainer`
  subclass and differing only in `compute_reward`.
- `--override rl.reward_scope={correction_meaning,full_answer}` switches A3.
- `--override rl.nrt.aggregation={seq_logp,geo_mean,arith_mean,weighted_neglogp}`
  switches A4.
- Every run logs the full RL health panel from design §5.
- All three algorithms run 2 steps inside `lexi smoke` on CPU with a tiny model.

**Non-functional**

- Rewards are computed by the same `lexi_research/format` code the eval harness
  uses. A reward function that reimplements edit-F1 would let the model be
  rewarded for something eval does not measure.
- `lambda` is swept on the 50-row fixture before any full run.

## Files

**Create**

- `lexi_research/rl/base.py` — shared trainer, mask construction, advantage, KL
- `lexi_research/rl/rewards.py` — verifiable reward from `format` primitives
- `lexi_research/rl/grpo.py`, `jepo.py`, `nrt.py`
- `lexi_research/rl/segments.py` — locate reasoning / correction / meaning / feedback token spans
- `ops/ablations/a1-method.yaml`, `a3-reward-scope.yaml`, `a4-nrt-aggregation.yaml`
- `tests/rl/test_segments.py`, `test_rewards.py`, `test_masks.py`, `test_equivalence.py`, `test_advantage.py`

**Modify**

- `params.yaml` — `rl.*`
- `dvc.yaml` — `rl` stage
- `ops/Makefile` — smoke covers all three algorithms

## Implementation steps

1. **`segments.py` first, and test it hardest.** Everything downstream depends on
   correctly locating four token spans in a rendered sequence. A wrong boundary
   here silently trains the wrong thing and no curve will look unusual. Test on
   both thinking and non-thinking renders, and on the `correction: null` case.
2. **`rewards.py` from `format` primitives.** Import `edit_f1` and `validate_output`;
   do not reimplement. Test that reward is exactly 1.0 for a perfect prediction and
   monotonically decreasing as edits are corrupted.
3. **`base.py`**: group sampling, advantage normalisation, KL, the combined loss,
   and logging. Reward computation is abstract.
4. **GRPO.** Run it to convergence on the fixture before writing another line of
   the other two.
5. **JEPO.** Teacher-forced scoring path, multi-sample lower bound optional behind
   a flag.
6. **NRT** with the four aggregations, plus the empty-reasoning baseline and
   clipping.
7. **`lambda` sweep on the fixture**, then A1, then A3, then A4.

## Tests

| Test | Asserts |
|---|---|
| `test_segments.py::test_spans_thinking` | reasoning, correction, meaning, feedback spans located exactly on a thinking render |
| `test_segments.py::test_spans_non_thinking` | same on a non-thinking render, with an empty reasoning span |
| `test_segments.py::test_null_correction` | a `correction: null` row yields a valid correction span, not an empty one |
| `test_masks.py::test_feedback_excluded_from_reward` | feedback tokens carry zero reward weight in all three tracks |
| `test_masks.py::test_policy_grad_only_on_reasoning` | non-reasoning positions receive zero policy gradient |
| `test_rewards.py::test_perfect_prediction_scores_one` | exact match yields 1.0 |
| `test_rewards.py::test_monotone_degradation` | reward decreases as gold edits are progressively corrupted |
| `test_equivalence.py::test_nrt_seqlogp_equals_jepo` | NRT with `seq_logp` and single-sample JEPO produce identical rewards and gradients on a fixed batch, to numerical tolerance |
| `test_advantage.py::test_group_normalisation` | known reward group produces hand-computed advantages |
| `test_advantage.py::test_empty_baseline_clipping` | rewards below `R(empty)` clip to zero improvement |

`test_equivalence.py` is the single most valuable test in this phase: it verifies
two independently written implementations against each other, so a bug must exist
in both, identically, to pass.

## Acceptance

Implementation — done and exercised on CPU:

- [x] All three algorithms complete 2 steps in `lexi smoke` on CPU, on the same
      tiny model, through one shared loop.
- [x] Every RL run logs the `R(empty)` gap and KL to reference, alongside the
      rest of the design §5 health panel.
- [x] `test_nrt_seqlogp_equals_jepo` passes, at every length and with a floored
      zero probability.
- [x] `--override rl.reward_scope=…` switches A3; `--override
      rl.nrt_aggregation=…` switches A4. Both are config, not code.
- [x] Rewards are computed by the same `format` primitives the eval harness uses
      — asserted by a test that reads the source for a reimplementation.
- [x] `uv run pytest` green — 697.

Experiments — blocked on a GPU, a trained SFT baseline, and a real dataset:

- [ ] The `lambda` sweep on the fixture, per track.
- [ ] A1, A3, A4 complete, each a W&B group.
- [ ] Findings filled in at `plans/…/reports/phase-04-findings.md`, **including a
      null result if that is what happens**. The file currently records what the
      CPU gate shows and marks every experimental row pending.

## Risks

| Risk | Handling |
|---|---|
| **RL does not beat SFT** | Expected on ~20 k rows of distillation data. Acceptable because Phase 2 preceded this. Report it plainly with ceiling-normalised numbers; a well-measured null result is a stronger artifact than an unmeasured win |
| Endogenous reward rises without capability rising | Track the `R(empty)` gap and KL to reference; a reward curve alone is never the verdict |
| Reasoning collapses to empty or to a fixed template | Log reasoning-length histogram and token entropy per step; NRT's format-supervision CE and the empty baseline exist for exactly this |
| Segment location is subtly wrong | Tested first and hardest; the equivalence test would also break |
| `lambda` needs different values per track | Swept per track on the fixture; the chosen value is in the run config and reported |
| GRPO reward hacking — high edit-F1 with unusable output | `format_valid` and `strip_mismatch` terms; the qualitative W&B table is inspected before trusting any arm |
