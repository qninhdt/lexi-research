# Design — Lexi Lab (research + inference)

**Status:** approved 2026-08-01
**Supersedes nothing.** Extends [`grader-distillation-design.md`](grader-distillation-design.md),
which stays authoritative for I/O contract, taxonomy, and band derivation.
**Goal:** learning + resume. Two tracks — research (train) and engineering (inference) —
on one MLOps spine.

---

## 1. Problem statement

The repo has a working format core, teacher pipeline, and serving shim, but:

- `train/trainer.py` is a stub: hardcoded hyperparams that ignore `params.yaml`,
  no chat template, no completion-only loss mask, no validation, no resume.
- No RL, no ablation harness, no inference benchmark, no engine layer.
- DVC wires 1 of ~12 stages.
- Everything downstream of the data assumes one base model. The reference model
  changed once already, and each change invalidated assumptions carried over
  from the previous one — so **the base model is a parameter, not a premise**.

## 2. Model independence (a constraint, not a feature)

No module names a model, an architecture, or a module path. A different base
model is a value change in `params.yaml`; it is never a code change. Three rules
carry this:

| Concern | Rule |
|---|---|
| Loading | The checkpoint's own `config.architectures` names its class, so the loader asks rather than assumes. `AutoModelForCausalLM` is the fallback, not the default |
| Prompt format | Rendering goes through the tokenizer's own chat template. Nothing in this repo formats a turn |
| LoRA placement | Targets are resolved from the loaded model's module tree by **role**, never from a name list |

Role resolution walks `named_modules()`, keeps what looks like a linear
projection (`in_features` / `out_features` — so a depthwise conv, an embedding
and a norm are excluded by their shape rather than by their name), attributes
each to a decoder layer, and classifies it as attention or feed-forward from its
path. Which path components mean what is `train.layout` in `params.yaml`, so an
architecture these conventions misread is a config edit. Presets are role sets:

| Preset | Roles |
|---|---|
| `attn` | every attention projection, whatever the container is called |
| `attn+mlp` | plus every feed-forward projection, including MoE experts |
| `all-linear` | plus projections matching no known convention |

An explicit pattern list remains available for a stack the conventions misread.
Resolution raises when a preset or a pattern matches nothing: an adapter attached
to zero modules trains, converges, logs a falling loss, and teaches the model
nothing.

### 2.1 Reference model

`Qwen/Qwen3.5-4B` is the first model trained, and the one the numbers in the
report come from. Its facts are recorded because they are the shape the design
was tested against — not because anything depends on them.

| Property | Value | Consequence |
|---|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (vision-language) | `AutoModelForCausalLM` is the wrong loader — which is why the loader reads `config.architectures` |
| Density | Dense, **not MoE** | MoE stays an inference-comparison subject only |
| Layers | 32: 24 `linear_attention` (Gated DeltaNet) + 8 `full_attention` (`full_attention_interval: 4`) | see below |
| Linear-attn modules | `linear_attn.{in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj}`, `conv1d` | none match the conventional `q/k/v/o_proj` list — which is why targets are resolved by role |
| Vision tower | present | excluded from LoRA by `train.layout.excluded_markers`; adapting it for a text task is waste |
| Context | 262 144 native, ~1 M with YaRN | not a constraint here |
| Prompt length | a rendered example measures ~1250 whitespace tokens | `max_seq_len` must clear ~2 K after subword splitting; the old 1024 dropped every row |
| Thinking | on by default, emits `<think>…</think>`, **no `/nothink` soft switch** | reaches the template as `train.enable_thinking`; ablation A2 |
| MTP head | `mtp_num_hidden_layers: 1` | speculative decoding available for free (B5) |
| Engines | vLLM / SGLang **nightly only** | pin nightly digests; treat breakage as inference-eng practice |
| License | Apache-2.0 | adapter publishable |

**Load-bearing finding.** A hardcoded `q/k/v/o_proj` + MLP list adapts the
attention of 8 of these 32 layers and none of the 24 Gated-DeltaNet ones, while
producing a run indistinguishable from a healthy one. Role resolution removes the
bug; the placement question it exposed stays as ablation A6, because LoRA
placement on hybrid linear-attention architectures has no established recipe.

## 3. Loss architecture (shared by every RL variant)

`feedback` carries **no** reward signal in any track. It is voice and register:
unverifiable, and rewarding it would teach the model to chase teacher phrasing
rather than grading quality. It is still fully supervised by SFT.

```
segment              SFT-CE   RL-reward   RL-policy-grad
prompt x               —          —             —
<think> z </think>     —          —            yes      <- only reasoning gets policy gradient
gold correction       yes        yes            —
gold meaning          yes        yes            —
gold feedback         yes         —             —       <- SFT only
```

```
L_total = CE(correction + meaning + feedback) + lambda * L_RL(z)
```

Three reward definitions over the identical mask, so the tracks are directly
comparable:

| Track | R(z) | Notes |
|---|---|---|
| GRPO-RLVR | `w1*edit_F1 + w2*(1 - abs(dMeaning)/4) + w3*format_valid - w4*strip_mismatch` | reward is **exogenous**; samples an answer and scores it with code |
| JEPO | `log pi(correction, meaning \| x, z)` | teacher-forced; no answer sampling |
| NRT | `f(c_1..c_T)` over gold-token probabilities | `f` in {seq-logp, geo-mean, arith-mean, weighted(-log p)} |

**Order is not negotiable: GRPO first.** JEPO/NRT rewards are endogenous (`R`
depends on `theta`), so rising reward can mean rising confidence rather than
rising skill. Without an exogenous baseline there is nothing to debug against.

Stabilisation for JEPO/NRT (mandatory, per the source papers): empty-reasoning
baseline `R' = max(0, R(z) - R(empty))`, group-relative advantage normalisation,
and a separate small format-supervision CE on the `<think>`/`</think>` markers
only.

## 4. Ablations

### 4.1 Research track

Required — these carry the narrative:

| # | Axis | Arms | Question |
|---|---|---|---|
| A1 | Method | SFT · +GRPO · +JEPO · +NRT-geo · +NRT-wlogp | does RL beat SFT here, and which variant |
| A2 | Thinking | on · off · forced-empty | is reasoning worth its token cost for this task |
| A3 | Reward mask | {correction+meaning} · {full answer incl. feedback} | tests the §3 claim directly |
| A4 | NRT `f` | seq-logp · geo-mean · arith-mean · weighted(-log p) | reproduces the NRT ablation on a new domain |
| A7 | Loss mask | completion-only · full-sequence | quantifies the harm of the current stub |

Wanted — engineering credibility:

| # | Axis | Arms |
|---|---|---|
| A5 | Data scale | 1 k · 5 k · 20 k rows (learning curve) |
| A6 | LoRA target | `attn` · `attn+mlp` · `all-linear` · the legacy `q/k/v/o_proj` name list · rank 8/32/64 |
| A8 | Init | instruction-tuned vs base checkpoint of the same family |

A3 and A6 have no prior art — A3 because it is specific to this output schema,
A6 because role-resolved placement on a hybrid linear-attention stack is
unstudied. They are the defensible contributions. The legacy-name-list arm of A6
is what quantifies the cost of the bug this design removes, on whatever
architecture the run uses.

### 4.2 Inference track

| # | Axis | Arms | Measured by |
|---|---|---|---|
| B1 | Engine | vLLM (nightly) · SGLang (nightly) · `transformers serve` baseline | tok/s, p95 |
| B2 | Quantisation | bf16 · FP8 · AWQ-int4 · GPTQ | quality-vs-throughput Pareto |
| B3 | Adapter | merged weights · runtime multi-LoRA | LoRA-switching latency cost |
| B4 | Decoding | free · constrained (xgrammar/outlines) · retry loop | format validity vs latency |
| B5 | Speculative | none · **MTP head** · ngram | acceptance rate, speedup |
| B6 | Concurrency | 1 · 4 · 16 · 64 · 128 | throughput knee, goodput under SLO |
| B7 | Prefix cache | on · off | fixed long system prompt should win big |
| B8 | Model class | student-4B · MoE 30B-A3B · teacher API | cost per 1 k requests vs quality |

## 5. Metrics

| Group | Metrics |
|---|---|
| Meaning | QWK · exact · within-1 · MAE · per-band breakdown · ECE with reliability diagram |
| Correction | span+tag P/R/F1 · span-only F1 (separates span errors from tag errors) · tag confusion matrix · cross-weight-tier confusion rate |
| Format | validity rate · strip-markup identity rate · JSON parse rate · retry count |
| Taxonomy | `other` rate · tag distribution vs teacher |
| Feedback | no hard metric. Teacher-as-judge pairwise win-rate, chrF as a weak proxy, **labelled weak in every report** |
| Ceiling | teacher self-consistency — an upper bound on every metric above |
| RL health | reward mean/std · advantage std · KL to reference · reasoning length · token entropy · share of zero-advantage groups · `R(empty)` gap |
| Serving | TTFT · TPOT · e2e p50/p95/p99 · tok/s · peak VRAM · goodput under SLO · cost per 1 k requests |

## 6. MLOps spine

| Component | Decision |
|---|---|
| DVC | complete the pipeline: `export -> sample -> generate -> label -> validate -> balance -> split -> calibrate -> sft -> rl -> eval -> bench` |
| Config | keep `params.yaml` + DVC params. No Hydra — the existing mechanism already hashes params into stage signatures |
| W&B | run-to-DVC-hash lineage, Artifacts for adapter + `band_config.json`, Model Registry with a promotion gate, Sweeps for A5/A6, a Report per ablation |
| CI | keep the existing pure-CPU job; add a 50-row full-pipeline smoke on a tiny randomly-initialised model, 2 optimiser steps |
| Serving | existing shim gains an engine-adapter layer so vLLM / SGLang / HF sit behind one interface |
| Reproducibility | seed everywhere; `nvidia-smi` and library versions recorded into the run config; `dvc.lock` is the source of truth |

### W&B panels

Defaults are not sufficient. Required:

- **Training** — loss split into CE and RL components, reward mean with std band,
  KL to reference, reasoning-length histogram over steps, LR, grad-norm.
- **Eval** — tag-by-tag confusion heatmap, reliability diagram, per-band
  exact/within-1 stacked bars, band distribution before and after balancing.
- **Ablation** — parallel-coordinates over A1–A8 against each metric; Pareto
  scatter of quality against latency with an explicit frontier line.
- **Qualitative** — a `wandb.Table` of input / gold / prediction / reasoning /
  diff, filterable by band and tag. This is the panel used most during debugging.
- **Inference** — latency CDF (not mean bars), throughput against concurrency with
  the SLO line drawn, VRAM timeline.

## 7. Hardware

Rent by the hour, not by the month.

| Tier | Card | Share of use | Unlocks |
|---|---|---|---|
| Daily | RTX 4090 24 GB | ~80% | FP8 (Ada), AWQ, all of B1–B7 |
| Burst | L40S 48 GB (or A100 80 GB) | occasional | B8 with MoE at FP8/bf16, large batch |

RTX A6000 48 GB is Ampere and therefore has **no FP8** — avoid it. If only one
card can be rented, take the L40S 48 GB. Training stays on Colab (A100/L4).

The benchmark harness must run on both tiers from config: VRAM is a parameter,
never an assumption.

## 8. Layout

```
lexi_research/
  format/  teacher/  data/     unchanged - already sound
  cli/
    __init__.py     the `lexi` entry point; every stage is a subcommand
    config.py       params.yaml + --override, frozen after load
    smoke.py        the acceptance gate
  train/
    collate.py      chat template + completion-only label mask
    modules.py      LoRA targets resolved by role from the loaded model
    trainer.py      SFT; every hyperparameter from config, no model named
    callbacks.py    W&B tables, in-loop eval, resume
  rl/
    base.py         shared Trainer subclass and mask logic
    rewards.py      edit_f1 / meaning / format -> verifiable reward
    jepo.py nrt.py  latent-reasoning trainers
  eval/
    harness.py      full metric suite -> JSON + W&B
    judge.py        teacher-as-judge for feedback
bench/
  runner.py         load generator; TTFT / TPOT / p95
  engines/          vllm.py sglang.py hf.py
serve/              + engine adapter
ops/                Makefile, docker-compose, smoke fixtures
```

## 9. Acceptance gate

`make smoke` — one command, must be green before any real experiment:

```
50-row fixture
  -> dvc repro (every stage)
  -> sft, 2 steps            (tiny model, CPU)
  -> rl, 2 steps x 3 variants
  -> eval harness -> metrics.json with every field populated
  -> serve up -> 5 requests -> valid grades
  -> bench 30 s -> latency report
```

On a GPU, `make smoke-gpu` repeats this against the checkpoint named in
`train.base_model`, on 50 rows for one epoch. It answers the one question a tiny
stand-in cannot: whether PEFT and the quantiser actually attach to *that*
architecture's modules, and to how much of it. The resolved coverage is printed
before the first step, so a near-empty adapter is visible in seconds rather than
after a long run.

## 10. Phases

| Phase | Content | Depends on |
|---|---|---|
| 0 | `lexi` CLI surface; rewrite `trainer.py` (loader from `config.architectures`, chat template, completion-only mask, role-resolved LoRA targets, honour `params.yaml`), pin deps, 50-row smoke fixture | — |
| 1 | Complete the DVC pipeline, W&B lineage, CI smoke job | 0 |
| 2 | Eval harness with the full metric suite and W&B panels | 1 |
| 3 | Real SFT plus ablations A2, A6, A7 | 2 |
| 4 | RL: GRPO baseline, then JEPO, then NRT; ablations A1, A3, A4 | 3 |
| 5 | Inference lab: engine, quantisation, decoding, speculative decoding, bench harness (on the VPS) | 2 |
| 6 | MoE comparison B8, model card, W&B report | 4, 5 |

Phase 5 depends only on Phase 2, so it runs **in parallel** with 3 and 4 — train
on Colab while benchmarking on the VPS.

## 11. Risks

| Risk | Level | Handling |
|---|---|---|
| **A1 returns "RL does not beat SFT"** on 20 k distillation rows | high | Acceptable *only* because Phase 2 precedes Phase 4. A negative result from a trusted harness is presentable; a negative result from an unvalidated one is indistinguishable from a bug |
| PEFT / bitsandbytes fail on an unconventional attention implementation | medium | `make smoke-gpu` surfaces it on day one, not after a long run |
| A future base model's layout defeats the role conventions | medium | Resolution raises rather than adapting nothing; `train.layout` and an explicit pattern list are the two escapes, both config-only |
| vLLM / SGLang nightly breaks mid-experiment | medium | Pin nightly digests; debugging this *is* the inference-engineering exercise |
| Endogenous RL reward rises without skill rising | medium | GRPO baseline first; track `R(empty)` gap and KL to reference |
| Scope loss: cutting `serve/` and `bench/` when time runs short | medium | These are what stop the work reading as research-only. Cut ablations before cutting engineering |
| Every number still anchors to the teacher, no human gold | high | Inherited from the parent design; restated in the model card |

## 12. Operator experience (hard requirement)

**No Python is written inside a notebook.** Every action is a CLI subcommand with
explicit flags; the Colab notebook is a launcher of at most a handful of cells
(clone, install, secrets, `dvc pull`, invoke, push artifact). If an experiment
needs a code change, that change lands in the repo and is version-controlled —
never in a cell.

Single entry point, one subcommand per pipeline stage:

```
lexi data export | sample | generate | label | validate | balance | split | calibrate
lexi train sft   --config params.yaml [--override train.lora_r=64]
lexi train rl    --algo {grpo,jepo,nrt} --config params.yaml
lexi eval run    --adapter <path|wandb-artifact> --split test
lexi bench run   --engine {vllm,sglang,hf} --concurrency 1,4,16,64
lexi serve up    --engine vllm --adapter <path>
lexi smoke       [--gpu]
```

Notebook shipped as `notebooks/lexi_colab.ipynb`, generated from a tracked source
so the committed `.ipynb` never carries execution output or drifts from the CLI.

## 13. Out of scope

Multi-GPU / FSDP / DeepSpeed, data pipelines above ~1 M rows, and any
public-benchmark-comparable number (BEA-2019 / ERRANT remain dropped, per the
parent design §11).

RAG, agents, and tool-calling are deliberately absent: they are already covered
elsewhere in the author's work, and bolting them onto a distillation lab would
blur both.
