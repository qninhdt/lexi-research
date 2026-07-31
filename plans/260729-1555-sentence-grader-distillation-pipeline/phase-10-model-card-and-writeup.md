---
phase: 10
title: "Model card and writeup"
status: pending
priority: P2
effort: "2d"
dependencies: [8, 9]
---

# Phase 10: Model card and writeup

## Overview

Produce the artifacts that make the work legible to someone who did not build it: a
model card stating honestly what the model does and does not know, and a technical
writeup that names the method correctly and reports what was measured versus what
was deferred.

## Requirements

**Functional**

- Model card: intended use, training data provenance, metrics, limitations
- Technical writeup: method, decisions, results, deferred items
- Reproduction instructions: exact lineage tuple → results
- Optional: HF Hub publish (adapter + config bundle)

**Non-functional**

- Every number traceable to a W&B run or DVC artifact
- No claim beyond what was measured
- Cambridge-derived text stays private

## Architecture

### Naming the method correctly

The writeup must call this what it is: **sequence-level knowledge distillation with
rejection sampling**. Not "knowledge distillation" unqualified — classical KD
(Hinton 2015) trains against soft logit distributions with KL divergence, which this
does not do (the teacher does not expose logprobs, Phase 2). Nearest published
references are Kim & Rush 2016 (sequence-level KD) and Self-Instruct (teacher
generates inputs).

Getting this wrong is the single easiest thing for a reader to catch, and it costs
credibility disproportionately.

### The three honest limitations

These are stated prominently, not buried:

1. **No human gold labels.** Every metric is agreement with the teacher, not
   correctness. The model reproduces teacher judgments including teacher mistakes.
   Ceiling = teacher.
2. **Synthetic input distribution.** Both training and test sentences were written by
   an LLM simulating learners, not by learners. Whether real learner text is graded
   equivalently is **unmeasured**. This is the largest validity threat.
3. **Two of five fields are not learned.** `grammar` and `naturalness` are computed
   from `correction` by a weighted formula. Their quality is a function of edit
   detection plus a hand-set weight table, not of model capability.

Stating these plainly is worth more than the metrics. A reader who finds an
unstated limitation discounts everything; a reader who sees them named upfront trusts
the numbers that are there.

### What the writeup claims

| Claim | Supported? |
|---|---|
| Reproduces teacher grading at QWK X on held-out senses | yes |
| Detects grammatical errors at F1 Y against teacher annotations | yes |
| Emits valid structured output at Z% | yes |
| Runs at N ms p50, M GB VRAM | yes |
| Grades learner sentences correctly | **no — no gold** |
| Better than any published system | **no — no shared benchmark** |

### Deferred, with the path forward

- Human gold set → would convert fidelity claims into accuracy claims
- Real learner corpus input (W&I+LOCNESS, Lang-8) → would address distribution shift
- Public benchmark comparison (ERRANT on BEA-2019) → would give an external number
- `define` task → schema already supports it
- Integration into `pycil` → shim exists, app-side wiring not done

## Related Code Files

- Create: `MODEL_CARD.md`
- Create: `docs/writeup.md`
- Create: `docs/reproduction.md`
- Create: `README.md` (repo root — overview, quickstart, architecture)
- Create: `scripts/publish_hf.py` (optional)

## Implementation Steps

1. Model card: intended use, out-of-scope use, training data (provenance + private
   status), evaluation results, the three limitations, ethical considerations.
2. Writeup: problem, why finetuning is justified here (grading input is unbounded and
   uncacheable, unlike generation), method with correct naming, key design decisions
   and their rationale, results, limitations, future work.
3. Document the design decisions worth explaining to a reader:
   - Bands computed from `correction` rather than emitted — structural consistency,
     tunable without retraining
   - Two-call generation — labels must be a function of text, not of the request
   - Confusable tags share weights — noise tolerance by construction
   - Split by target word — prevents leakage across the many rows per word
4. `reproduction.md`: lineage tuple → exact commands.
5. Root README: what this is, quickstart, architecture diagram, repo layout.
6. Optional HF publish: adapter + `band_config.json` as one bundle, card attached.
   Never publish Cambridge-derived text.

## Success Criteria

- [ ] Model card complete; three limitations stated prominently
- [ ] Writeup names the method correctly with references
- [ ] Every reported number traceable to a W&B run or DVC artifact
- [ ] No claim exceeds what was measured
- [ ] Reproduction instructions verified by following them from a clean clone
- [ ] Root README lets a newcomer understand and run the project
- [ ] No Cambridge-derived text in any published artifact

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Overclaiming (accuracy when only fidelity was measured) | high | Explicit claim table; review every number against its source |
| Method mislabeled as classical KD | medium | Correct naming with references; state the difference |
| Limitations buried | medium | Prominent placement in both card and writeup |
| Cambridge text published | medium | Publish adapter + config only; no dataset rows |
| Reproduction instructions untested | low | Follow them from a clean clone before declaring done |
