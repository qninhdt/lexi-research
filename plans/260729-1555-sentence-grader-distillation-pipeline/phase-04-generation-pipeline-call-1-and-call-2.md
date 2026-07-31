---
phase: 4
title: "Generation pipeline: call 1 and call 2"
status: pending
priority: P1
effort: "3d"
dependencies: [2, 3]
---

# Phase 4: Generation pipeline — call 1 and call 2

## Overview

The two-call pipeline that produces the dataset. Call 1 is a **diversifier** that
writes learner sentences. Call 2 is the **teacher** that grades them with the exact
inference prompt. Only call 2's output becomes ground truth.

This is the phase where distillation validity is either established or lost.

## Requirements

**Functional**

- Call 1: batch K specs per sense → K learner sentences
- Call 2: batch K sentences per sense → K gradings, using the frozen inference prompt
- Specs (`meaning_req`, `error_spec`, `profile_id`) persist as **metadata only**
- Resumable: interrupt at any point, restart without duplicating spend
- Per-element validation — one bad element must not discard a whole batch
- Diversity measurement (distinct-n) per call-1 batch
- Cost and token accounting per stage

**Non-functional**

- ~2.5K + ~2.5K calls must survive network faults, rate limits, provider hiccups
- Idempotent by `req_uid`; safe to re-run
- No prompt drift: call 2 loads the same frozen file the trainer/server use

## Architecture

### Why two calls — the reason, stated once

Single-call self-labeling produces labels that describe the **instruction**, not
the **sentence**:

```
prompt : "write a sentence with meaning=2"
output : { "text": "The room is bright and airy.", "meaning": 2 }
```

That sentence uses `bright` = "full of light" correctly — true `meaning` is 4. The
model emitted `2` because it was asked for `2`. The label never passed through the
text. Training on it teaches: *this correct sentence scores 2*.

A better rubric does not fix this, because **no grading step occurs** — the label
flows from instruction to output, bypassing the sentence. And the failure is
invisible: correct and incorrect rows look identical.

Call 2 severs that path. It sees only `{target, sense, text}` and never the spec,
so its label is a **function of the text**.

### Flow

```
senses_pool.parquet  +  profiles.json
        │  sample_batches(seed)
        ▼
batch_specs.parquet          sense × K specs
        │
        ▼  CALL 1 — diversifier (knows spec)
raw_texts.parquet            {req_uid, text}   spec kept as METADATA
        │
        ▼  CALL 2 — teacher (blind to spec, frozen inference prompt)
raw_labels.parquet           {correction, meaning, feedback}  ← GROUND TRUTH
```

### Batching

Both calls batch by sense — the `{target, definition, pos}` context is shared
across K elements, so input tokens amortize.

`K = 6`. Bounded deliberately: larger K raises intra-batch correlation, and call 2
must stay close to single-item inference behavior (see Phase 2's parity check).

Call 1 output: JSON array of K strings.
Call 2 output: JSON array of K grading objects, index-aligned to input.

If a call-2 response has the wrong element count, the batch is retried once, then
split into singletons rather than discarded.

### Spec grid — coverage target, not quota

Two independent axes:

| Axis | Values |
|---|---|
| `meaning_req` | 0 · 1 · 2 · 3 · 4 |
| `error_spec` | none · 1 error · 2–3 errors · many · unreadable |

Profile is sampled as a third dimension, not gridded.

**Sampling weighted toward `meaning_req` ∈ {1,2,3}.** Unweighted, the teacher
defaults to writing either clean sentences (`4`) or nonsense (`0`), leaving the
middle — where real learners live and where grading is hardest — empty.

Elements inside one batch draw **different grid cells**, which forces textual
variety structurally rather than relying on prompt wording alone.

### Specs are metadata, not labels

`meaning_req` is retained in the dataset but **never used as a label and never
used to filter rows**. There is no drop rule: whatever call 1 writes is a valid
learner sentence, and call 2 labels what is actually there.

Its value is diagnostic. `meaning_req` vs `meaning` measures **how well call 1's
prompt steers the output** — a free signal from two numbers already on hand. Cells
with large systematic divergence indicate prompt weaknesses, not bad rows.

### Diversity guard

Intra-batch correlation is a real side effect of batching: models tend to write K
near-identical sentences varying a few words.

Measure `distinct-2` and `distinct-3` per batch. Batches below threshold are
flagged in `generation-report.json` (not auto-discarded — the number should first
inform the prompt). Prompt wording is the primary defense; measurement verifies it
actually worked.

### Resumability

Both stages write **JSONL append-only** keyed by `req_uid` before Parquet
conversion. On restart, completed `req_uid`s are read and skipped. A crash at call
1.8K of 2.5K costs nothing.

Cost ledger per stage: calls, prompt/completion tokens, retries, estimated spend →
`cost-report.json`.

## Related Code Files

- Create: `lexi_research/data/sample_batches.py`
- Create: `lexi_research/data/generate.py` (call 1)
- Create: `lexi_research/data/label.py` (call 2)
- Create: `lexi_research/data/diversity.py` (distinct-n)
- Create: `lexi_research/data/jsonl_store.py` (resumable append store)
- Create: `tests/data/{test_sample_batches,test_diversity,test_jsonl_store}.py`
- Create: `tests/data/test_generate_label_fakes.py` (fake teacher, no network)
- Consumes: `prompts/grade_use_in_sentence.v1.txt` (frozen, Phase 2)

## Implementation Steps

1. `sample_batches.py`: deterministic sampling from `senses_pool` × grid ×
   profiles; weighted toward middle `meaning`; distinct cells within a batch;
   emit `batch_specs.parquet` with `req_uid` = `sha256(sense_uid|spec|seed)[:16]`.
2. `jsonl_store.py`: append-only writer + `completed_ids()` reader.
3. `generate.py` (call 1): render diversifier prompt from spec + profile; batch
   K; parse JSON array; per-element validation (non-empty, contains target or is
   an intentional `unreadable` cell); append JSONL.
4. `label.py` (call 2): load the **frozen** inference prompt; batch K texts;
   parse array; index-align; per-element structural validation; append JSONL.
   On element-count mismatch → retry once → fall back to singletons.
5. `diversity.py`: distinct-n per batch → report.
6. Convert JSONL → `raw_texts.parquet` / `raw_labels.parquet`; join on `req_uid`.
7. Cost/token ledger → `cost-report.json`.
8. Tests with a fake teacher: batching, resume-after-crash, element-count
   mismatch fallback, per-element rejection isolation, sampling determinism.
9. **Pilot: 500 rows.** Then inspect (see gate below).
10. DVC stages `generate` and `label`; push to R2.

## Pilot gate — mandatory before full generation

Run 500 rows and inspect before spending on 20K:

| # | Measurement | Threshold | If it fails |
|---|---|---|---|
| G1 | **Teacher self-consistency** — re-grade the same 200 texts in a second pass, shuffled order | **QWK ≥ 0.7** on `meaning`; edit-level F1 ≥ 0.6 on `correction` | **STOP.** Revise the rubric anchors and the frozen prompt (Phase 2), re-run G1. Do **not** proceed to generation — an inconsistent teacher cannot be distilled |
| G2 | `meaning` distribution | all 5 bands present; bands {1,2,3} ≥ 40% combined | **STOP.** Reweight sampling; revise call-1 prompt |
| G3 | Format validity (call 2) | > 90% | Tighten prompt / structured decode |
| G4 | distinct-2 per batch | > 0.7 | Revise call-1 prompt; lower K |
| G5 | Tag distribution | `other` < 5% | Revisit taxonomy (Phase 1) |
| G6 | Batch-vs-single parity | QWK ≥ 0.8 | Lower K, or unbatch call 2 |
| G7 | Manual read of ~50 rows | plausible learner text; feedback actually useful | Prompt revision |

**G1 and G2 are hard stops.** They are the two failures that cannot be repaired
downstream:

- **G1 low** → the teacher disagrees with itself, so its labels are noise. Every
  metric in Phase 8 is bounded by teacher self-consistency, so a low G1 caps the
  whole project before a single training step runs. This is measurable **without
  gold** — it only requires grading the same text twice.
- **G2 failing** → the dataset is bimodal (band 0 and 4 only). A model trained on
  it cannot grade the middle, which is exactly where real learners land. No amount
  of training fixes a distribution that has no middle.

Step 10 (full generation) is blocked until every row passes. **Phase 7 (training)
is blocked on G1 and G2 regardless of how much data exists** — record both numbers
in `pilot-gate.json` and reference them in the Phase 7 entry criteria. Discovering
either problem after Phase 8 means having paid for generation *and* training.

Two failure modes here differ in cost: G3–G7 are prompt-tuning loops (cheap, ~$1
per iteration on 500 rows). G1 is a **rubric design failure** — it means §7 of the
design doc needs concrete anchors, not that the prompt needs rewording.

## Success Criteria

- [ ] Pilot 500 rows passes every gate row above (G1–G7)
- [ ] `pilot-gate.json` records every gate measurement with pass/fail, committed
      to Git (small, no Cambridge text) so Phase 7 can assert on it
- [ ] G1 (teacher self-consistency) and G2 (band coverage) explicitly recorded —
      these are the two hard stops that also gate Phase 7
- [ ] Full run ~20K rows completed and in R2
- [ ] Resume verified: kill mid-run, restart, zero duplicate spend
- [ ] `meaning_req` never appears in any training-label path (test asserts this)
- [ ] Call 2 loads the frozen prompt file — byte-identical to the serving prompt
- [ ] `generation-report.json`: band distribution, tag distribution, diversity,
      `meaning_req` vs `meaning` divergence per cell
- [ ] `cost-report.json` totals within the planned budget
- [ ] All tests hermetic (fake teacher; no network in CI)

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Intra-batch correlation → low real diversity | high | K=6; distinct cells per batch; distinct-n measured and reported |
| Batching shifts call-2 grading vs single-item inference | high | Phase 2 parity check; fallback to unbatched call 2 |
| Middle `meaning` bands stay empty | high | Weighted sampling; pilot gate blocks progress |
| Interrupted long run wastes spend | medium | JSONL resume keyed by `req_uid` |
| Element misalignment corrupts labels | medium | Count check → retry → singleton fallback |
| Spec leaks into labels | medium | Structural test asserts spec columns absent from label path |
| Teacher writes unnatural "learner" text | accepted | Deferred validity limitation; recorded in the model card |
