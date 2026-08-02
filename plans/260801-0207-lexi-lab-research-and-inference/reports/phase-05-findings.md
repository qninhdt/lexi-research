# Phase 5 findings — inference lab

**Status: harness complete, sweeps pending a rented GPU.**

The engine layer, the load generator and the statistics are implemented and
tested. B1–B7 cannot run here: they need a GPU, and two of the three engines are
nightly builds that do not install on a CPU box.

## What is settled

- **The load generator is open-loop.** Arrivals follow a schedule fixed before
  the run starts, and a test asserts two calls agree. A closed-loop generator
  with a fixed worker count silently backs off as the server slows, so the queue
  never builds and the report shows graceful degradation right up until
  production discovers otherwise.
- **Percentiles come from the raw samples**, hand-checked against 1..100:
  p50 = 50.5, p95 = 95.05, p99 = 99.01. At a few thousand requests there is no
  reason to approximate, and a p99 from a running digest over that many samples
  is not a number worth defending.
- **Warm-up is discarded and counted.** A test puts a nine-second cold start in
  the sample set and asserts it leaves p95 untouched.
- **Capability flags, not try/except.** An engine that cannot do FP8 produces an
  arm marked `skipped` with the reason attached. A crash leaves a hole a reader
  fills with a guess; a silent fallback to bf16 puts a bf16 number under an FP8
  heading.
- **Goodput is separate from throughput.** Throughput counts work done, goodput
  counts work that arrived inside the SLO, and only the second is what a user
  gets.
- **The shim contract is unchanged.** `serve/engine.py` builds the backend from
  an engine adapter, and the existing `tests/serve/` suite passes against it.

## B1 — engine

| Engine | digest | tok/s | e2e p95 | QWK | Notes |
|---|---|---|---|---|---|
| `transformers` baseline | pending | pending | pending | pending | always available; the number the others are "faster than" |
| vLLM nightly | pending | pending | pending | pending | digest recorded in the report, not assumed |
| SGLang nightly | pending | pending | pending | pending | |

## B2 — quantisation

The deliverable is a Pareto plot with the frontier drawn explicitly, not a
winner. A scatter of a dozen arms invites everyone to pick the point nearest
their prior.

| Quantisation | tok/s | e2e p95 | QWK / ceiling | peak VRAM |
|---|---|---|---|---|
| bf16 | pending | pending | pending | pending |
| FP8 | pending | pending | pending | pending |
| AWQ int4 | pending | pending | pending | pending |
| GPTQ | pending | pending | pending | pending |

FP8 needs Ada or newer. On an Ampere card this arm is *skipped with a reason*,
which is why the card model is in the report lineage.

## B3–B7

| Axis | Arms | Status |
|---|---|---|
| B3 adapter | merged weights vs runtime multi-LoRA | pending |
| B4 decoding | free vs constrained vs retry loop | pending |
| B5 speculative | none vs MTP head vs ngram | pending — MTP depends on the checkpoint shipping a head; ngram is the arm that works anywhere |
| B6 concurrency | 1 · 4 · 16 · 64 · 128 | pending — the knee is the number, not the peak |
| B7 prefix cache | on vs off | pending — the system prompt is ~1250 tokens and fixed, so this should be the largest single win. Measure it rather than assuming it |

## Order

B7 → B6 → B2 → B5 → B4 → B3 → B1: cheapest and most independent first, so early
results inform the expensive sweeps. Every sweep is scoped to a duration in
config and the report carries cost per 1 000 requests, so spend stays visible
while it is being incurred rather than afterwards.

## Reproducibility

`bench.repeat_runs` is 2 and the report states the observed run-to-run spread. A
benchmark that cannot reproduce itself within noise is not measuring the system,
and the honest place to find that out is before the numbers are in a write-up.
