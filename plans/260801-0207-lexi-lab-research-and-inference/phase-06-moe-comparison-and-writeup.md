---
phase: 6
title: "MoE comparison and write-up"
status: pending
priority: P2
size: M
dependencies: [4, 5]
---

# Phase 6: MoE comparison and write-up

## Overview

The last ablation, **B8**, and the artifacts that make the whole lab legible to
someone who did not build it: the model card, the W&B report, and an honest
statement of what is and is not claimable.

B8 answers the only question a reader actually has: *was any of this worth it
versus just calling a bigger model?*

## B8 — three-way comparison

| System | Configuration |
|---|---|
| Student 4B | best adapter from Phase 4, best serving config from Phase 5 |
| MoE 30B-A3B | `Qwen3-30B-A3B` at AWQ-int4 (24 GB tier) or FP8 (48 GB tier), zero-shot on the same prompt |
| Teacher API | the endpoint that generated the labels — the ceiling by construction |

Measured on the same test split, with the same Phase 2 harness:

- QWK, edit-F1, format validity — quality
- p50/p95 latency, tokens per second — speed
- Peak VRAM, cost per 1 000 requests — economics

The interesting axis is **quality per dollar at a fixed latency SLO**, not raw
quality. A 30B MoE that beats the student on QWK while costing 8× more per request
loses for this application, and saying so plainly is the point of the comparison.

The MoE is inference-only. No MoE training in this project — the serving skills
(expert placement, offload, memory) are what transfer, and training one is a
different project with a different budget.

## Requirements

**Functional**

- `lexi bench compare --systems student,moe,teacher` runs all three through the
  same harness and emits one comparison report.
- The MoE runs through the same engine adapter interface from Phase 5; no
  special-cased code path.
- `MODEL_CARD.md` is generated from the eval report, not hand-written, so it
  cannot drift from the numbers.
- A W&B Report assembles the panels from every phase into one narrative.

**Non-functional**

- Every number in the model card traces to a report JSON, which carries lineage
  back to a commit.
- Claims are bounded to what was actually measured. The parent design already
  states that only *distillation fidelity* is claimable — no human gold set, no
  literature-comparable benchmark. That constraint holds here and is restated
  prominently rather than buried.

## Files

**Create**

- `bench/compare.py` — three-way orchestration and report assembly
- `lexi_research/report/model_card.py` — generate `MODEL_CARD.md` from report JSON
- `plans/…/reports/phase-06-findings.md`
- `docs/results.md` — the narrative write-up

**Modify**

- `MODEL_CARD.md` — becomes generated output
- `README.md` — status, results summary, how to reproduce
- `dvc.yaml` — `compare`, `model_card` stages

## Implementation steps

1. Rent the burst tier for a bounded window; run B8 within it and release. The MoE
   is the only thing that needs the larger card.
2. `compare.py` reusing the Phase 5 runner and the Phase 2 harness. No new metric
   code — a metric that appears only in the comparison is a metric nobody tested.
3. Model card generation, including the limitations section verbatim from the
   parent design §13 and this design §11.
4. W&B Report assembling the panels across phases into a readable narrative.
5. `docs/results.md`: what was found, what was not, what would be next.

## Model card must state

- Trained on teacher-generated data; **no human gold set**. Every number is
  fidelity to a teacher, not accuracy against ground truth.
- Train and test are both teacher-generated; the real-learner distribution is
  unverified. This is the largest validity threat and it was not addressed.
- Band thresholds live in `band_config.json` and ship with the adapter. A
  checkpoint without it produces meaningless bands.
- `feedback` quality is measured only by a weak proxy, deliberately.
- Whether RL beat SFT, stated plainly either way, with ceiling-normalised numbers.

## Acceptance

- B8 complete, all three systems on one plot.
- `MODEL_CARD.md` generated, not hand-edited; regenerating produces no diff.
- W&B Report published, linked from the README.
- `docs/results.md` written, including negative results.
- A reader can reproduce any headline number from the repo plus a GPU.

## Risks

| Risk | Handling |
|---|---|
| The MoE simply wins outright | Then that is the finding, and the honest framing is quality-per-dollar at a fixed SLO. A distillation project that concludes "for this task, a served MoE was better" is a stronger artifact than one that hides the comparison |
| Burst-tier rental overruns | Bounded window, B8 scripted and dry-run on the daily tier first |
| Model card drifts from the numbers | Generated from report JSON; regeneration diff is in acceptance |
| Overclaiming in the write-up | Limitations section is copied from the design docs, not paraphrased; "fidelity, not accuracy" appears in the README summary, not only deep in the card |
