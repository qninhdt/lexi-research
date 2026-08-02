# Phase 6 findings — MoE comparison and write-up

**Status: comparison and card generator complete, B8 pending a burst-tier rental.**

## What is settled

- **The model card is generated, not written.** `lexi report model-card` renders
  it from the eval report JSON, and a test asserts regeneration produces no diff.
  A hand-edited card drifts from the numbers the moment a run is repeated, and
  the drift is invisible because both are prose.
- **The limitations are copied verbatim, not paraphrased.** A test asserts every
  one appears in the card exactly. Paraphrase across a few revisions is how
  "fidelity to a teacher" becomes "accuracy", which is the one claim this project
  cannot make.
- **A dirty tree is admitted in the card.** A result produced from uncommitted
  code cannot be reproduced from its SHA, and saying so costs one line.
- **B8's axis is quality per dollar at a fixed SLO**, and a system that misses
  the SLO scores zero rather than scoring well slowly — it cannot serve this
  product at all, so it is not a cheaper option. A test pins the hand-computed
  ranking where the higher-quality system loses.

## B8 — three-way comparison

| System | QWK | span+tag F1 | p95 | tok/s | VRAM | cost / 1k | quality per $ |
|---|---|---|---|---|---|---|---|
| Student 4B | pending | pending | pending | pending | pending | pending | pending |
| MoE 30B-A3B | pending | pending | pending | pending | pending | pending | pending |
| Teacher API | pending | pending | pending | pending | — | pending | pending |

The MoE is inference-only. The serving skills — expert placement, offload, memory
— are what transfer; training one is a different project with a different budget.

**If the MoE simply wins, that is the finding.** A distillation project that
concludes "for this task, a served MoE was better" is a stronger artifact than
one that omits the comparison. The honest framing is the quality-per-dollar
column, not the QWK column.

## The write-up

`docs/results.md` is written now rather than after the runs, so the claims it is
allowed to make are fixed before the numbers arrive. It states what is claimable,
lists the five defects the build itself surfaced, and names the result most
likely to appear — that RL does not beat SFT on this data — before it has been
measured.

## Remaining

- [ ] Rent the burst tier for a bounded window, run B8, release.
- [ ] Regenerate `MODEL_CARD.md` from a real eval report.
- [ ] Assemble the W&B Report across phases and link it from the README.
