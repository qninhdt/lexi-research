---
phase: 7
title: "QLoRA training on Colab"
status: pending
priority: P1
effort: "3d"
dependencies: [4, 5, 6]
---

# Phase 7: QLoRA training on Colab

## Overview

Fine-tune Qwen2.5-7B-Instruct with QLoRA to imitate the teacher's grading function.
Runs on Colab Pro / Kaggle from a `git clone` + `dvc pull`, tracked in W&B, resumable
across session kills.

## Requirements

**Functional**

- Train a LoRA adapter on `train.parquet`, validate on `val.parquet`
- Training input is the **exact inference prompt** (Phase 2 contract); target is the
  teacher's JSON output
- Loss on completion tokens only, not on the prompt
- W&B logs loss curves, val metrics, config, and full lineage
- Checkpoint to Drive/W&B and resume from the latest checkpoint after a kill
- Ablations: 7B vs 1.5B, full output vs `meaning`-only

**Non-functional**

- Entry point is a CLI (`lexi-research train`), not a notebook. The notebook is a
  three-cell launcher: clone, pull, invoke
- Fits Colab Pro A100 40GB and degrades to T4 16GB via config only
- Deterministic given seed + config; every run records its lineage tuple

## Entry gate — do not start this phase until all hold

<!-- Updated: Validation Session 1 - hard gate before training -->

Training on data that failed the Phase 4 pilot gate wastes GPU hours and produces a
model whose weakness is only discovered in Phase 8. Verify from
`generation-report.json` and `calibration-report.json`:

| Condition | Source | Required |
|---|---|---|
| Teacher self-consistency (`meaning` QWK, re-grade) | Phase 4 gate | ≥ 0.7 |
| Teacher self-consistency (`correction` edit-F1) | Phase 4 gate | ≥ 0.6 |
| All 5 `meaning` bands present | Phase 4 gate | yes |
| Middle bands (1–3) share | Phase 4 gate | ≥ 40% |
| Batch-vs-single parity | Phase 2/4 | QWK ≥ 0.8 |
| `band_config.json` calibrated and committed | Phase 6 | yes |

Self-consistency is the **ceiling on every downstream number**: a student cannot be
measured as more faithful than the teacher is to itself. If QWK < 0.7 the correct
action is revising the rubric or prompt (Phase 2), not training.

## Architecture

### Why CLI-first, notebook-as-launcher

A notebook that contains logic cannot be tested, diffed, or reproduced. All logic
lives in the package; the notebook only clones the repo, pulls data, and calls the
CLI. This is the difference between a homework artifact and a pipeline — and it is
what makes the same code run locally, on Colab, and in CI.

```
notebooks/train_colab.ipynb
  cell 1: pip install -e . ; dvc remote credentials from Colab secrets
  cell 2: dvc pull data/processed/{train,val}.parquet
  cell 3: !lexi-research train --config configs/qwen7b-qlora.yaml
```

### Stack

`trl.SFTTrainer` + `peft` + `bitsandbytes`. No hand-written training loop — a custom
loop here would be strictly worse than the maintained one and would signal the
opposite of production judgement.

`unsloth` is a drop-in speed/VRAM win on single-GPU Colab but pins its own
torch/transformers builds, which has broken reproducibility across Colab image
updates. Decision: **`trl`+`peft` as the default path**; unsloth behind a config
flag, tested but not required.

### Prompt/target construction

```
messages = [ system(GRADER_SYSTEM), user(render(target, sense, text)) ]
target   = json.dumps({correction, meaning, feedback})
```

`GRADER_SYSTEM` and `render` are imported from the same module Phase 2 defines and
Phase 9 serves. There is no training-specific prompt. Completion-only loss via
`DataCollatorForCompletionOnlyLM` so the model is not scored on reproducing the
prompt.

### Config

```yaml
model:
  id: Qwen/Qwen2.5-7B-Instruct
  revision: <immutable-sha>          # pinned, never a moving tag
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
train:
  max_seq_length: 1024
  per_device_batch_size: 4
  gradient_accumulation_steps: 8     # effective 32
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  num_epochs: 2
  bf16: true
  gradient_checkpointing: true
  seed: 42
eval:
  eval_steps: 100
  save_steps: 100
  save_total_limit: 3
  metric_for_best_model: val_format_validity
```

`max_seq_length: 1024` is a claim, not a guess to leave unchecked — Phase 3 profiles
token lengths and this is set from that report. Sequences are short (one sentence in,
one short JSON out), which is why 7B QLoRA fits comfortably.

### Resumability

Colab kills sessions. Non-negotiable requirements:

- `save_steps: 100` to a Drive-mounted or W&B-artifact path
- `--resume` finds the latest checkpoint and continues optimizer + scheduler state
- W&B `resume="allow"` with a deterministic run id derived from the config hash, so a
  resumed run appends to the same run instead of forking a new one

### Validation during training

Beyond val loss, compute cheap task metrics at each eval step:

- **format validity rate** — parses, tags in the closed set, strip-equals-input
- **`meaning` exact + ±1** against teacher labels

Format validity is the primary early-stopping metric. Val loss can improve while the
model emits unparseable markup, and an unparseable output is worthless regardless of
loss.

### Ablations

| Run | Purpose |
|---|---|
| 7B full output | Primary |
| 1.5B full output | Latency/quality tradeoff for a future deployment story |
| 7B `meaning`-only | Isolates how much `correction` generation costs `meaning` accuracy |

Ablations run after the primary config is settled — not a Cartesian sweep.

## Related Code Files

- Create: `lexi_research/train/__init__.py`
- Create: `lexi_research/train/dataset.py` (parquet → chat-format, completion masking)
- Create: `lexi_research/train/trainer.py` (SFTTrainer wiring, callbacks)
- Create: `lexi_research/train/callbacks.py` (task-metric eval callback)
- Create: `lexi_research/train/cli.py`
- Create: `configs/qwen7b-qlora.yaml`, `configs/qwen1.5b-qlora.yaml`
- Create: `notebooks/train_colab.ipynb` (launcher only)
- Create: `tests/train/test_dataset_masking.py`

## Implementation Steps

1. `dataset.py`: parquet → chat messages using the Phase 2 renderer; assert prompt
   parity with the serving path in a test.
2. Completion-only collator; unit-test that prompt tokens are masked (`-100`).
3. `trainer.py`: 4-bit load, LoRA attach, SFTTrainer, W&B init with lineage config.
4. `callbacks.py`: format-validity + `meaning` accuracy at each eval step.
5. Checkpoint/resume wiring; deterministic W&B run id from config hash.
6. Sanity gate: overfit 64 examples to near-zero loss. If this fails, stop — the
   data or masking is wrong, and no amount of training fixes it.
7. Subset smoke run (500 rows, 1 epoch) end to end including resume-after-kill.
8. Full run on 7B; monitor in W&B.
9. Ablations: 1.5B, `meaning`-only.
10. Upload best adapter + `band_config.json` as a versioned W&B artifact.

## Success Criteria

- [ ] `lexi-research train --config ...` runs unmodified on Colab Pro A100 and T4
- [ ] Notebook contains no logic beyond clone/pull/invoke
- [ ] 64-example overfit reaches near-zero loss
- [ ] Kill-and-resume verified: run continues in the same W&B run, same optimizer state
- [ ] W&B records the full lineage tuple and per-eval task metrics
- [ ] Prompt parity test passes (train prompt == serving prompt, byte-identical)
- [ ] Best adapter exported as a W&B artifact **bundled with** `band_config.json`
- [ ] Ablation results recorded

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Colab session killed mid-run | high | save_steps 100 + `--resume` + W&B resume; verified in step 7 |
| Prompt drift between train and serve | high | Shared renderer module + byte-equality test |
| Loss improves while output stays unparseable | medium | Format validity as the early-stopping metric |
| Colab image update breaks pinned deps | medium | Pinned versions in `requirements-colab.txt`; unsloth optional, not required |
| Trained on stale dataset version | medium | `dvc pull` by hash; dataset hash logged to W&B and refused if mismatched |
| 7B QLoRA OOM on T4 | low | Config-only degrade: batch 1, accum 32, seq 512; documented in the config file |
