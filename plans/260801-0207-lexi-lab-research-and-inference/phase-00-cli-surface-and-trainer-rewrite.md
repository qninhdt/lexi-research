---
phase: 0
title: "CLI surface and trainer rewrite"
status: done
priority: P1
size: M
dependencies: []
---

# Phase 0: CLI surface and trainer rewrite

## Overview

Two jobs. First, give the repo a single `lexi` entry point so no experiment ever
requires writing Python in a notebook. Second, replace `lexi_research/train/trainer.py`,
which was a stub with four independent defects, none of which could be diagnosed
while the others were present.

The rewrite is **model-agnostic by construction**: nothing under `train/` names a
model, an architecture, or a module path. That is a constraint, not a feature —
the base model has already changed once, and every assumption carried over from
the previous one had to be found by hand.

Nothing here needs a GPU except the final gate.

## The four defects in the old trainer

| Defect | Consequence | Fixed by |
|---|---|---|
| `AutoModelForCausalLM` on a checkpoint whose class is something else | wrong loader, and silently so for any non-plain-causal-LM head | ask the checkpoint: `config.architectures` names its class |
| training text is `f"{system}\n\n{user}\n\n{completion}"` | no chat template — the student learns a format it will never be served with, breaking the prompt-parity property the parent design calls load-bearing | `collate.build_example` renders through `apply_chat_template` |
| `SFTTrainer` with no completion-only collator | loss is computed over the prompt as well as the answer | token-position mask, from the template's assistant mask or a checked prefix |
| `target_modules = [q,k,v,o,gate,up,down]_proj` | on the reference model, adapts the attention of 8 of 32 layers and none of the 24 linear-attention ones — and would be silently wrong on any other family too | `modules.resolve_target_modules` selects by role from the loaded module tree |
| hyperparameters hardcoded | `params.yaml` silently ignored, so DVC's param hashing was a lie | every value read through `cli.config` |

## Requirements

**Functional**

- `lexi` console script with subcommand groups `data`, `train`, `eval`, `bench`,
  `serve`, `smoke`. Every stage of the pipeline is reachable as a subcommand.
- All configuration flows from `params.yaml`; `--override key.path=value` applies
  dotted overrides for sweeps without editing the file.
- The trainer loads the checkpoint with the class it declares, renders training
  text through `tokenizer.apply_chat_template`, masks loss to completion tokens
  only, and resolves LoRA targets by **inspecting the loaded model**.
- `lexi smoke` runs the full pipeline on a 50-row fixture and exits non-zero on
  any failure.

**Non-functional**

- `lexi smoke` runs on CPU with a tiny randomly-initialised model, in CI, without
  network access and without touching Cambridge-derived data.
- Importing `lexi_research.train` must not import `torch`. Every heavy import
  stays inside the function that needs it so CI stays fast and CPU-only.
- No module under `lexi_research/` names a model family, an architecture, or a
  module path. Asserted by tests over four synthetic architectures.

## Files

**Created**

- `lexi_research/cli/__init__.py` — argparse dispatch, subcommand registry
- `lexi_research/cli/__main__.py` — `python -m lexi_research.cli`, for runtimes
  that clone rather than install
- `lexi_research/cli/config.py` — load `params.yaml`, apply `--override`, freeze
- `lexi_research/cli/smoke.py` — the acceptance gate, including the throwaway
  tokenizer and tiny model it trains
- `lexi_research/train/modules.py` — LoRA target resolution by model introspection
- `lexi_research/train/collate.py` — chat-template rendering + completion-only mask
- `ops/fixtures/smoke_50.jsonl` — 50 rows spanning every meaning band and every tag
- `ops/fixtures/build_smoke_fixture.py` — builds and validates the fixture
- `ops/Makefile` — `make smoke`, `make smoke-gpu`, `make lint`, `make test`
- `tests/train/test_modules.py`, `tests/train/test_collate.py`,
  `tests/train/test_trainer.py`, `tests/cli/test_config.py`, `tests/cli/test_cli.py`

**Modified**

- `lexi_research/train/trainer.py` — rewritten around the new helpers
- `lexi_research/train/dataset.py` — now the untokenised view of `collate`
- `lexi_research/train/cli.py` — a thin forward into `lexi_research.cli`
- `pyproject.toml` — `[project.scripts] lexi`, `pyyaml`, a `smoke` dependency
  group pinned to CPU torch wheels, mypy overrides for the training stack
- `params.yaml` — `train.target_modules`, `train.layout`, `train.enable_thinking`,
  `train.completion_only`, `train.max_steps`, `train.load_in_4bit`,
  `train.max_drop_fraction`, a `smoke` section; `max_seq_len` raised from 1024
- `.github/workflows/test.yml` — a second job that runs the gate on CPU torch
- `run_colab_train.sh`, `README.md`, `docs/lexi-lab-design.md`

## How target resolution works

No name list. `resolve_target_modules` walks `named_modules()` and keeps modules
that look like a linear projection — `in_features` / `out_features`, the
convention every `nn.Linear` and every quantised drop-in follows. A depthwise
convolution, an embedding and a norm are excluded by their shape rather than by
their name. Each match is attributed to a decoder layer and classified as
attention or feed-forward from its path. Presets are role sets:

| Preset | Roles |
|---|---|
| `attn` | every attention projection, whatever its container is called |
| `attn+mlp` | plus every feed-forward projection, including MoE experts |
| `all-linear` | plus projections matching no known convention |

Which path components mean what is `train.layout` in `params.yaml`, so an
architecture the conventions misread is a config edit. An explicit pattern list
stays available as the second escape. Resolution raises when a preset or a
pattern matches nothing.

## Tests

Tests came **before** the corresponding implementation.

| Test | Asserts |
|---|---|
| `test_collate.py::test_labels_mask_prompt` | every label before the assistant turn is `-100`, and the unmasked span decodes exactly to the completion — run against both a template that provides an assistant mask and one that does not |
| `test_collate.py::test_chat_template_parity` | the prompt half is what `serve/backend.py` sends, message for message, and its tokens are a prefix of the full render |
| `test_collate.py::test_thinking_flag_changes_render` | `enable_thinking` reaches the template and changes the render |
| `test_collate.py::test_a_template_with_no_generation_block_falls_back` | the ordinary case — no published template carries one — takes the fallback instead of raising |
| `test_collate.py::test_a_misplaced_generation_block_raises` | a template that declares a generation block and marks nothing is a broken template, distinct from one that has none |
| `test_collate.py::test_an_asymmetric_template_still_masks_correctly` | a template that renders the generation prompt differently from the assistant turn — Qwen under `enable_thinking=false` — still masks |
| `test_collate.py::test_the_fallback_holds_against_a_real_tokenizer` | the stub models an API; this proves the API is what the stub says it is |
| `test_modules.py::test_every_preset_resolves_on_every_architecture` | dense, hybrid linear-attention, MoE and vision-language trees all resolve |
| `test_modules.py::test_the_legacy_pattern_list_reaches_a_quarter_of_the_attention` | the old defect stays measurable |
| `test_modules.py::test_a_moe_router_is_never_adapted_by_a_preset` | routers are excluded from every preset and reachable by an explicit pattern |
| `test_modules.py::test_a_vision_tower_is_never_adapted` | sub-towers that are not the language model are excluded |
| `test_modules.py::test_layout_is_configuration_not_code` | an unconventional stack is reachable without touching the source |
| `test_modules.py::test_unknown_pattern_raises` | a pattern matching nothing raises rather than training a no-op adapter |
| `test_trainer.py::test_dropping_more_than_the_ceiling_raises` | length-biased data loss stops the run rather than skewing it |
| `test_config.py::test_override_unknown_key_raises` | `--override train.lora_rr=64` fails |
| `test_config.py::test_override_types` | `train.lora_r=64` yields `int`, not `"64"` |
| `test_cli.py::test_commands_from_later_phases_exit_non_zero` | a stub cannot make the gate pass |

## Acceptance

- [x] `uv run pytest` green — 511 with the training stack, 509 + 2 skipped without.
- [x] `mypy` strict green.
- [x] `ruff check` / `ruff format --check` green on everything this phase touched.
- [x] `make -f ops/Makefile smoke` exits 0 on CPU end-to-end over the 50-row
      fixture with a tiny randomly-initialised model and 2 optimiser steps.
- [x] The gate runs in CI, in a second job that installs the `smoke` group.
- [x] `lexi train sft --override train.lora_r=8` changes the run without a file edit.
- [ ] `make smoke-gpu` — needs a GPU; runs at the start of Phase 3.

## Findings

- **`max_seq_len: 1024` would have dropped every row.** A rendered example
  measures ~1250 whitespace tokens before subword splitting, almost all of it the
  rubric. Raised to 4096. Whether the rubric can be shortened without costing
  accuracy is a Phase 3 question — at this length the prompt dominates both the
  sequence budget and the prefill cost.
- **The completion is ~3% of the sequence.** Measured by the gate. That is the
  size of the loss-mask defect: the old trainer spent ~97% of its gradient on
  reproducing a rubric that is supplied verbatim at inference. A7 will quantify
  what that cost in accuracy.
- **Trusting an all-zero assistant mask would have failed on every real
  checkpoint.** Asked for an assistant mask, `transformers` warns and returns
  zeros when the template has no `{% generation %}` block — it does not raise. No
  published template (Qwen, Llama, Mistral, Gemma) carries one. Detection now
  reads the template string, the same way `transformers` does, and the gate
  exercises both paths and asserts they agree.
- **Prompt-render-as-prefix is not a sound way to find the boundary.** A template
  is free to render an assistant turn differently from the generation prompt it
  emits for the same turn; Qwen does exactly that under `enable_thinking=false`.
  The fallback builds the sequence as prompt + answer instead, so the boundary is
  exact by construction and ablation A2's no-thinking arm stays runnable.
- **Over-long examples raise rather than truncate**, and dropping is capped at
  `train.max_drop_fraction`. Cutting the tail teaches unterminated JSON; cutting
  the head drops the rubric. Dropping is length-biased, so past a threshold the
  survivors are a different distribution and the run stops.
- **MoE routers are excluded from every preset.** A router is a linear layer and
  would otherwise be adapted by `all-linear`, perturbing expert assignment. It
  stays reachable by an explicit pattern, so A6 can ablate it.
- **The training nonce is now fresh per example.** The old code passed a constant
  `nonce="training"`, which would have taught the student one literal delimiter
  while inference draws a random one each request.
- `ruff check` reports 51 pre-existing violations in `data/` and
  `lexi_research/data/`, untouched by this phase. Left alone rather than folded
  into this diff — `make lint` is therefore red on arrival.

## Risks

| Risk | Handling |
|---|---|
| PEFT or bitsandbytes rejects an unconventional projection | `make smoke-gpu` discovers it in minutes, not after a long run. Resolution prints coverage before the first step |
| The role conventions misread a future architecture | Resolution raises rather than adapting nothing; `train.layout` and an explicit pattern list are the escapes, both config-only |
| Under `enable_thinking=true` the answer follows the reasoning marker with no reasoning in it | The SFT data carries no reasoning trace, so the arm trains the model to open a think block and immediately answer. Phase 3 owns what A2's `on` arm should actually contain |
| The parquet path is exercised by no test that also trains | `test_trainer.py` covers `load_rows`; the first parquet training run is Phase 1's DVC stage |
