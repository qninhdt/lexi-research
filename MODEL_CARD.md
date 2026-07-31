# Sentence Grader Distillation — Model Card

## Status

No trained adapter has been released. This repository implements the pipeline and
its format contract; teacher generation, calibration, QLoRA training, and held-out
evaluation remain required before a model artifact can be claimed.

## Intended use

Given a target word, one dictionary sense, and a learner-written English sentence,
the future student model will emit an inline correction, a 0–4 meaning band, and
one sentence of feedback. The serving layer derives grammar and naturalness bands
from correction tags. It does not decide pass/fail.

## Method

This is **sequence-level knowledge distillation with rejection sampling**: one
teacher call diversifies learner-like text and a second, prompt-identical teacher
call grades that text. It is not classical soft-target/KL distillation because the
teacher does not expose logits.

## Data and provenance

The source sense pool is exported read-only from a Cambridge-derived SQLite
database. Dictionary text is private and is not published in Git, model hubs, or
reports. Source identity, prompt hashes, DVC hashes, split version, seed, teacher
model, and base-model revision are recorded as run lineage.

## Limitations

1. **No human gold labels.** Future metrics are teacher-fidelity, not correctness.
2. **Synthetic learner distribution.** Training and test text are teacher-written,
   so generalisation to real learners is unmeasured.
3. **Two fields are computed.** Grammar and naturalness derive from correction tags
   and a calibrated weight configuration rather than being learned directly.

Do not use a future model artifact for high-stakes assessment, diagnosis, or an
automatic learner progression decision without human oversight and validation on
real learner data.

## Evaluation status

The repository contains QWK, exact/±1, MAE, edit-F1, format-validity, and
self-consistency measurement code. There are no measured adapter results yet.

## Deployment constraints

The future serving shim is designed for a private network. It will not include
authentication in v1; do not expose it publicly without an authentication layer.
