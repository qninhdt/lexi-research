---
phase: 5
title: "Inference lab"
status: pending
priority: P1
size: L
dependencies: [2]
---

# Phase 5: Inference lab

## Overview

The engineering half. Engine adapters behind one interface, a load-generating
benchmark harness, and ablations **B1–B7**. Runs on a rented GPU, in parallel with
Phases 3 and 4 on Colab.

This phase depends only on Phase 2 (the metric harness) because quality must be
measured alongside latency — a throughput number without a quality number is
meaningless, and a quantisation that gains 40% throughput while losing 0.1 QWK is
a decision, not a win.

## Hardware

Rent by the hour, not by the month.

| Tier | Card | Share of use | Unlocks |
|---|---|---|---|
| Daily | RTX 4090 24 GB | ~80% | FP8 (Ada), AWQ, all of B1–B7 |
| Burst | L40S 48 GB (or A100 80 GB) | occasional | Phase 6's MoE at FP8/bf16, large batch |

RTX A6000 48 GB is Ampere and has **no FP8** — avoid it despite the memory. If
only one card can be rented, take the L40S 48 GB.

The harness reads VRAM as a config parameter. Any assumption of a specific card
in the code is a defect.

## Requirements

**Functional**

- `lexi serve up --engine {vllm,sglang,hf} --adapter <path|wandb://…>` starts a
  server behind the existing shim, unchanged in its contract.
- `lexi bench run --engine … --concurrency 1,4,16,64,128 --duration 60` produces a
  report JSON with TTFT, TPOT, e2e p50/p95/p99, tokens per second, peak VRAM,
  goodput under an SLO, and cost per 1 000 requests.
- Every benchmark run also emits quality metrics from the Phase 2 harness on the
  same configuration. Latency and quality are never reported separately.
- Nightly engine builds are pinned by digest and recorded in the report lineage.

**Non-functional**

- The load generator is open-loop (fixed arrival rate), not closed-loop. A
  closed-loop generator with a fixed worker count cannot show a saturation knee —
  it silently backs off, and the queue depth that a real user experiences never
  appears in the numbers.
- Warm-up requests are discarded and the count is in the report.
- Every run records GPU model, driver, engine digest, and quantisation.

## Ablation arms

| # | Axis | Arms | Note |
|---|---|---|---|
| B1 | Engine | vLLM nightly · SGLang nightly · `transformers serve` | the HF baseline exists to quantify what the engines actually buy |
| B2 | Quantisation | bf16 · FP8 · AWQ-int4 · GPTQ | the deliverable is a Pareto plot, not a winner |
| B3 | Adapter | merged weights · runtime multi-LoRA | measures LoRA-switching cost, which decides whether one server can host several adapters |
| B4 | Decoding | free · constrained (xgrammar/outlines) · retry loop | constrained decoding trades latency for format validity; the shim's retry loop is the third option and may beat both |
| B5 | Speculative | none · **MTP head** (where the checkpoint ships one) · ngram | the reference model carries `mtp_num_hidden_layers: 1`, so MTP costs a flag; ngram is the arm that works on any model; report acceptance rate, not just speedup |
| B6 | Concurrency | 1 · 4 · 16 · 64 · 128 | find the knee; report goodput under SLO, not raw throughput |
| B7 | Prefix cache | on · off | the system prompt is fixed and long, so this should be the largest single win — measure it rather than assuming |

B4 connects directly to a metric the harness already computes: format validity
rate. Constrained decoding should drive it to 1.0; the question is what it costs.

## Files

**Create**

- `bench/runner.py` — open-loop load generator, percentile accumulation
- `bench/report.py` — report JSON, Pareto assembly, W&B panel push
- `bench/engines/vllm.py`, `sglang.py`, `hf.py` — launch, health-check, teardown
- `bench/engines/base.py` — the interface all three satisfy
- `serve/adapter.py` — route the existing shim to any engine
- `ops/engines/*.lock` — pinned nightly digests
- `ops/docker-compose.bench.yml`
- `tests/bench/test_runner.py`, `test_report.py`, `tests/serve/test_adapter.py`

**Modify**

- `serve/backend.py` — construct from an engine adapter rather than a hardcoded HTTP client
- `serve/Dockerfile` — engine as a build argument
- `dvc.yaml` — `bench` stage
- `params.yaml` — `bench.*`, `serve.engine`

## Implementation steps

1. **`engines/base.py` interface first**: launch, wait-for-ready, OpenAI-compatible
   URL, teardown, capability flags (`supports_fp8`, `supports_lora`, `supports_mtp`).
   Capability flags rather than try/except, so an unsupported arm is skipped and
   *reported as skipped* instead of silently failing.
2. **`hf.py` first**, not vLLM. It is the slowest and the simplest, so the harness
   is debugged against something that cannot fail for interesting reasons.
3. **`runner.py`** with a synthetic prompt set drawn from the real test split, so
   prompt-length distribution matches production.
4. **Percentiles from raw samples**, not from a running approximation. At these
   request volumes there is no reason to approximate, and p99 from a t-digest on
   a few thousand samples is not trustworthy.
5. **vLLM nightly**, pinned. Expect breakage; debugging it is the exercise. Record
   the digest that works.
6. **SGLang nightly**, same treatment.
7. **B7 → B6 → B2 → B5 → B4 → B3 → B1** in that order: cheapest and most
   independent first, so early results inform the expensive sweeps.
8. **Pareto assembly** and the W&B inference panels from Phase 2.

## Tests

| Test | Asserts |
|---|---|
| `test_runner.py::test_open_loop_arrival` | request arrival follows the configured rate regardless of response latency |
| `test_runner.py::test_percentiles_exact` | p50/p95/p99 from a known sample set match hand-computed values |
| `test_runner.py::test_warmup_excluded` | warm-up requests are absent from the reported statistics |
| `test_report.py::test_pareto_frontier` | a known quality/latency set yields the hand-computed frontier |
| `test_report.py::test_skipped_arm_recorded` | an arm skipped for missing capability appears in the report as skipped, never as absent |
| `test_adapter.py::test_shim_contract_unchanged` | the existing `tests/serve/` suite passes against every engine adapter |

## Acceptance

- All three engines serve the adapter behind the unchanged shim contract.
- B1–B7 complete on the rented GPU, or are explicitly recorded as skipped with a
  reason.
- A Pareto plot of quality against latency across B2 arms, with the frontier drawn.
- `lexi bench run` reproduces its own numbers within noise on a repeat run, and
  the report states the observed run-to-run variance.
- Findings in `plans/…/reports/phase-05-findings.md`.

## Risks

| Risk | Handling |
|---|---|
| vLLM or SGLang nightly lacks the model type or breaks mid-sweep | Pin working digests in `ops/engines/*.lock`; the HF baseline always works, so no arm is fully blocked. Debugging this **is** the inference-engineering exercise, not an obstacle to it |
| GPU rental cost grows without bound | Cheapest-first ordering; every sweep is scoped to a duration in config; the report includes cost per 1 000 requests so spend is visible |
| Benchmark numbers are not reproducible | Repeat run in acceptance; variance reported rather than hidden |
| Quality regressions from quantisation go unnoticed | Every bench arm also runs the Phase 2 harness; a latency number is never reported alone |
| 24 GB is too small for a later MoE arm | B8 is deliberately deferred to Phase 6, where the burst tier is rented for a short window |
