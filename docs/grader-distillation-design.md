# Design — Sentence Grader Distillation

**Status:** design approved, implementation not started
**Date:** 2026-07-29
**Repo:** `lexi-research` (standalone; `lexi-ai` and `pycil` are read-only references)
**Goal:** learning + resume. Not cost reduction, not latency.

---

## 1. Scope

**In:** dataset generation (teacher LLM) → QLoRA train → eval → serving shim → model card.

**Out:** any code change in `lexi-ai` or `pycil`. `lexi-ai/data` (SQLite, 150MB) is read **read-only**, exported once to parquet. This repo does not depend on the `lexi_ai` package.

**Out:** pass/fail verdict. The model **measures**; the app **decides** (thresholds live in `pycil`, tunable without retraining).

**One task only:** `use_in_sentence`. Task `define` deferred — no call site exists in `pycil`.

**Teacher:** any OpenAI-compatible endpoint, runtime config. Never hardcoded. No `logprobs` available (commercial provider) → hard labels, cross-entropy.

---

## 2. Task

| Task | What | Call site |
|---|---|---|
| `use_in_sentence` | Grade a learner sentence using a target word in one specific sense | `lexi_ai/questions/types/use_in_sentence.py` → `grade_rubric` |

Grammar/naturalness are **criteria**, not tasks.

---

## 3. I/O contract

### Input

```json
{
  "target": "bright",
  "sense": { "definition": "full of light", "pos": "adjective" },
  "text": "The room have bright light in morning."
}
```

### Output — what the model emits

```json
{
  "correction": "The room [have>has:agr] bright light in [>the:art] morning.",
  "meaning": 4,
  "feedback": "Correct use of 'bright', but check subject-verb agreement and the missing article."
}
```

Three fields. `feedback` is **one sentence, English**.

### Derived by code, not by the model

```
grammar     = threshold(penalty(Correctness tags))
naturalness = threshold(penalty(Usage tags))
```

See §6. The model never emits these.

---

## 4. `correction` format

Re-emit the **whole sentence**, marking edits inline.

Syntax: `[` original `>` replacement `:` tag `]`

| Operation | Syntax | Example |
|---|---|---|
| Replace | `[A>B:tag]` | `[speak>speaks:agr]` |
| Delete | `[A>:tag]` | `the [the>:art] very` |
| Insert | `[>B:tag]` | `went [>to the:art] store` |

- Clean sentence → verbatim, **zero token overhead**.
- Unparseable sentence → `correction: null` (do not fabricate edits).
- Parse: `\[([^\]>]*)>([^\]:]*):([a-z]+)\]` — one regex, one pass. Text outside `[...]` is preserved.
- Escape: a literal `[` in learner text → `\[`.
- Insert `[>B:tag]` stands alone between spaces; position is explicit in the string.

**Why whole-sentence re-emit:** the frontend renders it directly, with no span-matching against the original. Tags are short and from a closed set, so display messages live in a UI lookup table (i18n-able) rather than being generated per edit.

**Reference:** GECToR (Omelianchuk et al., 2020 — Grammarly Research) uses token tagging rather than rewriting, and does **not** have the model emit error types (ERRANT assigns them afterwards). We use a decoder LM, so emitting `:tag` inline is cheaper than building a separate ERRANT classifier. Grammarly's production output format is not public — this design is our own, not a reproduction.

---

## 5. Taxonomy — 16 tags

| Group | Weight | Tags |
|---|---|---|
| Correctness | 1 | `punc` `sp` `art` `num` `poss` |
| | 2 | `prep` `part` `agr` `tense` `form` `pron` |
| | 3 | `order` |
| Usage | 2 | `coll` `word` |
| | 3 | `unnat` |
| — | 2 | `other` |

`sp` spelling (absorbs contractions: `dont`) · `agr` subject-verb agreement · `tense` tense · `form` word form, absorbs MORPH/INFL (`eloquent`→`eloquently`) · `art` article/determiner · `prep` preposition · `part` phrasal particle (`look up`/`look on`) · `num` number/countability · `poss` possessive · `pron` pronoun · `order` word order · `punc` punctuation · `coll` collocation (`do a decision`) · `word` word choice · `unnat` unnatural phrasing (absorbs register + wordiness) · `other` catch-all

**No tag for wrong meaning.** Wrong meaning does not live in a span and cannot be fixed by replacement → it lives in the `meaning` band.

**Load-bearing property:** tags that are **easy to confuse carry the same weight**. `word`↔`coll` (2/2), `prep`↔`part` (2/2) → confusion does not shift the band. This is what makes band-derivation tolerant of label noise. **Any taxonomy change must preserve this property.**

`other` exists to **measure what the taxonomy is missing**. Without it the teacher forces mismatches into the nearest tag, producing hidden label noise.

---

## 6. Band derivation (code, not model output)

```
penalty(group)  = Σ weight(tag ∈ group) / √(word_count)
grammar         = threshold(penalty(Correctness))
naturalness     = threshold(penalty(Usage))
correction null → grammar = 0, skip the formula
```

`√(word_count)` normalization: 2 errors in 6 words is worse than 2 errors in 30 words.

**Why derive instead of letting the model emit bands:**
- **Structural consistency** — the same error set always yields the same score. A model emitting bands disagrees with itself.
- Explicit anchors (the weight table) instead of a vague rubric.
- **Tunable without retraining.** Model-emitted labels are frozen into the weights.

**The weights and thresholds are initial design values, not calibrated.** Calibrate after the dataset exists (§10), and ship `band_config.json` **as part of the model artifact** — a checkpoint without it produces meaningless bands.

---

## 7. `meaning` rubric (model emits)

| Band | Criterion |
|---|---|
| 4 | Correct target sense, no drift |
| 3 | Correct sense, slight nuance drift |
| 2 | Ambiguous — readable as another sense |
| 1 | Right word, drifted to a near sense |
| 0 | Wrong sense entirely, or target not used |

---

## 8. Dataset generation — two calls, distinct roles

Every label comes from the LLM. Nothing is heuristically labelled.

| Field | Source | Kind |
|---|---|---|
| `target`, `sense.definition`, `sense.pos` | `SELECT` from Cambridge SQLite | read |
| `text` | **call 1** | generated |
| `correction`, `meaning`, `feedback` | **call 2** | generated → **ground truth** |
| `grammar`, `naturalness` | formula over `correction` | code |

### The two calls

```
CALL 1 — diversifier
  in:  target + sense + K specs {profile, meaning_req, error_spec}
  out: K sentences
  role: produce varied text. Specs are METADATA, never labels.

CALL 2 — teacher
  in:  target + sense + text          ← exactly the inference prompt
  out: {correction, meaning, feedback}
  role: GROUND TRUTH
```

**Why call 2 is mandatory.** In a single call the model both receives the request and self-reports the label, so the label is a *copy of the instruction*, not a reading of the text. Asked for `meaning=2`, a frontier model routinely writes a sentence whose true `meaning` is 4 — and still reports 2. That row teaches the student that a correct sentence scores low, and there is no way to detect it afterwards because the output looks identical either way. A better rubric does not help: no grading step occurs at all. Call 2 makes the label a **function of the text**.

**Prompt parity (load-bearing).** Call 2's prompt must be *exactly* the prompt used at inference — same file, hashed into the manifest. If data generation uses extra examples or guidance absent at inference, the student learns a function it will never be asked to compute.

Batching is the only permitted divergence: call 2 grades K texts at once, inference grades one. **Verify** `K=8` and `K=1` produce the same labels on a sample; if they diverge, batching is affecting grading — lower K or drop batching for call 2.

**No drop rule.** Specs are not labels, so `|meaning_req − meaning|` has no correctness meaning. It remains a **diagnostic**: it measures how well call 1's prompt steers the distribution. Cells with poor steering → fix the prompt, not the data.

### Learner profiles

`profiles.json`, versioned in Git, hashed into the manifest. Sampled deterministically by seed.

```json
{
  "id": "vi-b1-articles",
  "l1": "Vietnamese",
  "cefr": "B1",
  "error_bias": ["art", "agr", "tense"],
  "length": "short",
  "traits": "omits articles; overuses present simple; word-for-word phrasing"
}
```

`l1` is the learner's **first language**, which shapes the error profile — output is always English. ~12–16 profiles across 4 CEFR levels × several L1 families (Vietnamese, Japanese, Spanish, Arabic — four distinct error signatures) × length. Each profile must be **observably distinct**; no decorative variants.

### Generation grid

Two axes, sampled per batch:

| Axis | Values |
|---|---|
| `meaning_req` | 0 · 1 · 2 · 3 · 4 |
| `error_spec` | none · 1 error · 2–3 errors · many · unparseable |

Quota weighted toward `meaning ∈ {1,2,3}` — the middle band is where real learners land and where grading is hardest. Unweighted, the model defaults to clean sentences (4) or total misses (0).

Coverage is measured on the **observed** `meaning` from call 2, not on the requested value.

### Batching

One call = one sense × K samples; the sense context is shared. `K = 6–8`. For ~20K rows: ~2.5K calls per stage instead of 20K.

**Risk:** samples within a batch correlate — the model tends to write K variations of one template. Mitigations: cap `K ≤ 8`; spread **different** grid cells within a batch to force divergence; measure distinct-n per batch and reject over-homogeneous batches. Validate **per element**, not per batch, so one bad sample does not discard 8 rows.

### Not in v1

- Junk input (empty, target absent, verbatim example copy, wrong language) — rule-detectable → the app rejects it before the model.
- Prompt injection — `guarded_messages` in `lexi_ai/llm.py` already wraps the user turn in a nonce delimiter.
- Real learner corpora (W&I+LOCNESS / Lang-8).
- Any heuristic generation path (cross-sense pairing, corruption, collocation/synonym swap).

### Volume

~15–30K rows.

---

## 9. Post-decode validation

1. `correction` fully parses
2. Every tag ∈ the 16
3. **Strip markup == input `text`** — mandatory. Without it the model can silently alter untouched parts of the sentence (the specific risk of whole-sentence re-emit)
4. `meaning` ∈ 0–4
5. No empty edit `[>:tag]`
6. `feedback` non-empty, single sentence

Fail → drop row (dataset build) / retry (inference).

---

## 10. Source data

`lexi-ai/data`, SQLite, pinned by sha256. Read-only.

| Measure | Count |
|---|---|
| Senses with definition + POS | 199,842 |
| After POS filtering (lexical only) | **196,748** |
| Senses with CEFR label | 16,817 |
| Core examples (`is_extra=0`) | 300,922 |

**`entries.pos` is dirty** and needs normalization at export: `''`, `'V'`, `'adj'`, plus non-lexical entries (`abbreviation`, `symbol`, `prefix`, `suffix`, `combining form`, `written abbreviation`). Keep only lexical POS.

Cambridge-derived text stays **private** (redistribution rights unverified) — DVC remote only, never Git, never HF Hub.

---

## 11. Split · Eval · Train

### Split

By **target word**, not by row. One word yields many rows → row-splitting leaks. Additional check: identical sentence hashes crossing splits.

### Eval

No human gold (deferred — §13). Measurable:

| Metric | Meaning |
|---|---|
| QWK `meaning` student vs teacher (held-out) | fidelity |
| Exact / ±1 accuracy per band | breakdown by region |
| `correction` P/R/F1 by span+tag | vs teacher labels |
| Format validity rate | constrained-decode quality |
| Tag distribution; `other` % | taxonomy adequacy |
| Tag confusion matrix (double-grade) | pairs diverging **across weight tiers** → recalibrate |
| Teacher self-consistency | upper bound on every metric above |
| Latency p50/p95, VRAM | operations |

**Only "distillation fidelity" is claimable.** No literature-comparable number (BEA-2019/ERRANT dropped). State this in the model card.

### Train

- Base: Qwen2.5-7B-Instruct, **QLoRA** 4-bit nf4. Colab Pro (A100/L4) / Kaggle T4.
- `unsloth` or `trl` + `peft` — no hand-written training loop.
- Structured decode to enforce the JSON schema.
- Balance: cap easy strata; report band distribution before/after.
- Ablation: 7B vs 1.5B (latency story for later).

Correct name for the procedure: **sequence-level distillation with rejection sampling**. Not classical KD — no soft targets, no KL, no temperature (provider withholds logprobs).

---

## 12. Serving

The model returns 3 fields. **Nobody computes `grammar`/`naturalness`** if a client calls vLLM directly.

```
pycil / lexi-ai
   │  OpenAI-compatible /chat/completions
   ▼
serve/ shim (FastAPI)      ← loads band_config.json
   │  parse correction → penalty → grammar, naturalness
   ▼
vLLM + LoRA adapter
```

The shim preserves the deployment promise: `pycil` changes only `llm_base_url`, `llm_api_key`, `llm_model`.

---

## 13. Risks

| Risk | Level | Handling |
|---|---|---|
| **No human gold** — every number anchors to the teacher; the student inherits its bias | high | Deliberately deferred. State in model card. Build a gold set later |
| **Train and test are both teacher-generated** — real-learner distribution unverified | high | Deferred. This is the largest validity threat |
| Band weights/thresholds uncalibrated | medium | Lives in code; fixable without retraining |
| `naturalness` is not fully span-local — a sentence can be locally fine yet globally odd | medium | `unnat` over a wide span is an approximation, not an equivalence |
| Teacher not self-consistent → distillation meaningless | medium | Measure before bulk generation |
| Batch correlation in call 1 | medium | K≤8, mixed grid cells, distinct-n rejection |
| Batching shifts call 2's labels vs inference | medium | Verify K=8 vs K=1 parity |
| Model silently edits untouched text | medium | Validation #3 |
| No unseen words (all targets in Cambridge 113K) | low | Production `generate_fenced` creates new words. Note as a limitation |
| Output schema ≠ production `Judgment{correct,score,feedback}` | low | Serving shim maps it |

---

## 14. Deferred, with intent

- Human gold set → upgrades claims from *fidelity* to *accuracy*.
- Real learner corpora as input.
- Task `define`.
- Distribution-match evidence vs real learners.

---

## Unresolved

- `threshold(penalty)` → band cut points: no values yet; calibrate after the dataset exists.
- Rows per (target, sense): diversity vs cost.
- When to build the gold set, how many samples, who grades.
- Whether to run the 1.5B ablation for a latency story.
