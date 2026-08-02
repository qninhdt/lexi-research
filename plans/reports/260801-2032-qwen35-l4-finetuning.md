---
title: Qwen3.5-4B L4 fine-tuning research
date: 2026-08-01 20:32 +07:00
status: complete
scope: single NVIDIA L4, text-only QLoRA SFT
---

# Qwen3.5-4B L4 fine-tuning research

## Summary

For this repository's text-only SFT workload, the best verified default is
PyTorch SDPA plus the Qwen3.5 DeltaNet kernels. External FlashAttention 2 has
not shown a material steady-state speedup on one L4: it covers only the
full-attention blocks, while the measured backends were effectively tied after
warm-up.

The next configuration to validate is smaller LoRA rank, a 2048-token safety
cap, and optional Liger kernels. Keep gradient checkpointing on: disabling it
OOMed on the 22 GiB L4 at roughly 1.5k-token examples.
`torch.compile` is implemented as an opt-in experiment, but remains disabled
by default after the L4 A/B below.
The RL loop also has an end-to-end GPU smoke for both target models; that smoke
validates plumbing and memory, not reward quality.

## Scope and current baseline

- Model: `Qwen/Qwen3.5-4B`
- Hardware: NVIDIA L4, 22 GiB
- Stack: Torch 2.7.1+cu126, Transformers 5.14.1, bitsandbytes 0.50,
  PEFT 0.20
- Fine-tuning: text-only QLoRA, BF16 compute, batch 1, gradient accumulation,
  all-linear LoRA, selective completion logits, non-reentrant checkpointing
- Dataset smoke shape: 50 examples, 1,487–1,529 tokens

## Findings

### 1. Qwen3.5 is a hybrid model

The official model card describes 32 language layers arranged as eight groups of
three Gated DeltaNet layers followed by one Gated Attention layer. The DeltaNet
heads have dimension 128; the full-attention heads have dimension 256. The model
also has a padded 248,320-token vocabulary.

Sources: [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

This explains why FlashAttention 2 is not a universal accelerator here: it can
only replace the full-attention path. The majority path is DeltaNet.

### 2. The high-value kernels are FLA and causal-conv1d

Transformers documents `fla` and `causal_conv1d` as the optional fast kernels for
Qwen3.5's DeltaNet path and warns that the fallback is slower and more memory
hungry. The L4 environment has both extensions importing successfully.

Source: [Transformers Qwen3.5 usage notes](https://huggingface.co/docs/transformers/model_doc/qwen3_5#usage-tips-and-notes).

### 3. FlashAttention 2 was tested with a cold-start correction

`flash-attn==2.8.3` built successfully on the L4 with Torch 2.7.1/CUDA 12.6.
The first comparison used only two optimizer steps, so it included a large
first-step compile/warm-up cost and was not a valid steady-state comparison:

| Model | SDPA | FlashAttention 2 | Result |
| --- | ---: | ---: | --- |
| Qwen3.5-4B | 214.6 s | 280.2 s | FA2 was ~31% slower |
| Gemma 4 E4B-it | 144.4 s | failed | head dimension exceeded FA2's 256 limit |

That Qwen result was an observation from a cold run, not evidence that FA2 is
always slower. To correct for it, the same Qwen3.5 workload was run for 12
optimizer steps with `grad_accum=1`, in both backend orders:

| Order | Backend | Total | First step | Steps 2–12 |
| --- | --- | ---: | ---: | ---: |
| SDPA → FA2 | SDPA | 193.8 s | ~167.6 s | ~2.39 s/step |
| SDPA → FA2 | FA2 | 45.21 s | ~19.09 s | ~2.37 s/step |
| FA2 → SDPA | FA2 | 45.28 s | ~19.08 s | ~2.38 s/step |
| FA2 → SDPA | SDPA | 45.46 s | ~19.12 s | ~2.39 s/step |

The steady-state difference is under 1% on this smoke workload. The first
step is dominated by compilation/warm-up, which is why the original two-step
result was misleading. The long run is sufficient to reject the earlier
“FA2 is 31% slower” conclusion, but not a substitute for a production-length
benchmark on the real corpus.

The likely explanation for the tie is that only one quarter of Qwen3.5's
attention blocks use full attention, while the run also pays FA2 integration and
dtype/cast overhead under checkpointed QLoRA. This is an inference from the
architecture and benchmark.

### 4. `torch.compile` is currently a regression on this L4 path

The repository now exposes `train.torch_compile`,
`train.torch_compile_backend`, and `train.torch_compile_mode`; the default is
off. With `torch_compile=true`, Transformers handed the model to TorchInductor
and the run completed without OOM:

| Backend | Steps | Train runtime | Late-step behaviour |
| --- | ---: | ---: | --- |
| SDPA, compile off | 20 | 62.55 s | ~2.2–2.3 s/step |
| SDPA + Inductor, `mode=default` | 20 | 85.07 s | ~2.3–2.4 s/step |

The compiled run spent about 30.3 seconds in its first step and still had
multiple slow compilation steps afterward. It was about 36% slower over this
20-step smoke run, with no observed steady-state win. Inductor also reported
that the L4 had too few SMs for its max-autotune GEMM path. Keep the flag
available for experiments on a longer, fixed-shape corpus, but do not enable it
in the default Colab recipe.

### 5. RL now runs on both models after enabling checkpointing in the policy phase

The first Qwen RL smoke reached sampling and reward calculation but OOMed at
the first policy backward: the loop had attached gradient checkpointing while
leaving the freshly loaded model in evaluation mode. The fix switches to
`eval()` for rollout/reward phases and `train()` for policy backward, so
Transformers activates checkpointing where gradients are needed.

With the default GRPO group size of 4, 256 reasoning-token cap, SDPA, QLoRA,
and two optimizer steps on the 50-row fixture:

| Model | Result | Peak allocated | Peak reserved | Wall time |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-4B | 2/2 steps, 8 rollouts | 7.9 GiB | 8.6 GiB | 266.8 s |
| Gemma 4 E4B-it | 2/2 steps, 8 rollouts | 19.8 GiB | 20.1 GiB | 356.0 s |

Both loops saved an adapter and `rl-report.json` successfully. Gemma fits but
has little headroom on a 22 GiB L4; a longer real run may need a smaller
reasoning cap, group size, or sequence cap. The smoke rewards were mostly zero
and groups had zero advantage because this is an untrained fixture, so these
runs prove execution rather than learning quality.

### 6. Official dense-model recipe points to different knobs

The current ms-swift Qwen3.5 recipe uses the Transformers backend for dense
models, BF16, `lora_rank=8`, `lora_alpha=32`, `all-linear`, `max_length=2048`,
length grouping, multiple data-loader workers, and recommends packing or
padding-free batching. It also lists FA2 and Liger as available acceleration
options. Its example uses four 20 GiB GPUs, so its throughput numbers are not a
single-L4 guarantee.

Source: [ms-swift Qwen3.5 best practices](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md).

## Recommended L4 recipe

Keep these settings as the starting point:

```yaml
train:
  load_in_4bit: true
  bnb_4bit_use_double_quant: false
  attn_implementation: sdpa
  text_only: true
  per_device_batch_size: 1
  gradient_checkpointing: true
  gradient_checkpointing_use_reentrant: false
  lora_r: 8
  lora_alpha: 32
  target_modules: all-linear
  max_seq_len: 2048
  selective_logits: true
  tf32: true
  dataloader_num_workers: 2
  dataloader_persistent_workers: true
  dataloader_prefetch_factor: 2
```

Why:

1. `sdpa` is the measured winner for this single-L4 workload.
2. `fla` and `causal-conv1d` accelerate the dominant DeltaNet path.
3. Rank 8/alpha 32 follows the current dense Qwen3.5 recipe and reduces LoRA
   optimizer/adapter work relative to the repository's current rank 32.
4. 2048 is enough for the current 1.5k-token examples and avoids paying for
   accidental long outliers. It is a data policy, not a model context limit.
5. Checkpointing is required for this sequence length on 22 GiB; it is slower,
   but the no-checkpoint path OOMed.

## What to benchmark next

The next useful A/B order is:

1. `lora_r=32` vs `lora_r=8`, same data and steps; compare both runtime and
   validation quality.
2. `use_liger_kernel=true` with SDPA; Transformers supports the flag, and the
   Qwen3.5 recipe lists Liger as a memory optimization, but it has not yet been
   verified in this repository.
3. Packing/padding-free batching only if the real dataset has broad length
   variance. The current smoke rows are already tightly clustered, so packing
   will not materially improve that benchmark.

Do not add FlashAttention 2 to the default Colab requirements unless a new
benchmark with the real dataset reverses the result. Keep the explicit loader
override available for experiments.

## Unresolved questions

- Does rank 8 preserve the project's format/meaning metrics on the real corpus?
- Does Liger support every Qwen3.5 module used by this Transformers version
  without changing the custom selective-logits loss path?
- How wide is the real sequence-length distribution, and therefore how much can
  packing save?

---

## Addendum — 2026-08-01 23:30 +07:00: batch size and eval batching (measured)

Two changes were made after the findings above, and both were measured on a
fresh Colab L4 rather than reasoned about.

### 7. `per_device_batch_size: 1` was justified by an obsolete calculation

`params.yaml` pinned batch 1 with the comment that Qwen3.5's padded 248,320-token
vocabulary "makes four long sequences exceed an L4's memory even in 4-bit". That
was true of the full-logits path, but `selective_logits` narrows the projected
window from the whole ~1.5k-token sequence to the ~80-token supervised answer.
At batch 4 that is roughly 0.6 GiB of logits rather than ~11 GiB.

Measured over the same number of sequences, 2 optimizer steps, text-only QLoRA:

| Model | Batch x accum | Wall | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-4B | 4 x 8 | 171.3 s | 9.54 GiB | 10.61 GiB |
| Qwen3.5-4B | 8 x 4 | 237.6 s | 13.89 GiB | 15.49 GiB |
| Gemma 4 E4B-it | 1 x 32 | 220.1 s | 20.58 GiB | 20.62 GiB |
| Gemma 4 E4B-it | 4 x 8 | out of memory | — | — |

An earlier 3-step run at the same effective batch gave 349.4 s at batch 1
against 321.8 s at batch 4 for Qwen, in the same direction.

The Gemma rows are from a *fresh* kernel with 21.66 GiB free, and that detail
matters. Each arm above ran as a separate subprocess, but a warm Colab kernel
retains device memory from earlier work in the session: a first attempt measured
Gemma at batch 1 taking 770 s, and a later end-to-end run in that same dirtied
kernel OOMed Gemma even at batch 1 while reporting 10.08 GiB already held by the
parent process. Both were artifacts of the session rather than of the setting.
Re-measured clean, batch 1 completes in 220 s, and batch 4 still runs out of
memory with 20.92 GiB allocated by PyTorch itself — so the batch-4 failure is
real, not residue. The rule this implies: verify a Gemma memory claim in a fresh
kernel, because Gemma's ~1.4 GiB of headroom is smaller than the residue a warm
Colab kernel can hold.

Conclusions:

1. Qwen3.5-4B is fastest at batch 4. Batch 8 is *slower* than batch 4 despite
   fitting, so this is a sweet spot rather than a monotonic curve — worth
   remembering before raising it further.
2. Gemma 4 E4B-it cannot use it. It already peaks at 20.6 of 22 GiB at batch 1,
   and batch 4 dies in `completion_only_loss` while allocating 336 MiB.
3. Therefore the batch size is a per-model fact, not a global default. The
   `params.yaml` default stays at the value that fits the tighter model
   (1 x 32), and `run_colab_train.sh` raises it to 4 x 8 for Qwen. Both pairs
   multiply to an effective batch of 32, so the learning rate carries over.

The Gemma figure also reframes the earlier RL result: Gemma's RL smoke peaked at
19.8 GiB, and SFT peaks at 20.6 GiB. Gemma has no memory headroom on this card
in either loop, so any further Gemma optimisation has to *reduce* memory rather
than trade memory for speed.

### 8. In-loop eval generated one row at a time

`predict_rows` looped rows and called `model.generate` once per row. With
`eval_subset: 32` and `eval_steps: 200`, every in-loop evaluation was 32
sequential decode loops, each paying its own prefill and kernel launches.

It now left-pads rows into batches of `eval.batch_size` (default 8). Left
padding is required rather than incidental: every row's final prompt token must
sit in the same column, because that is the position `generate` continues from.

Measured on the L4, 16 fixture rows, greedy, 128 new tokens:

| Path | Wall | Peak allocated |
| --- | ---: | ---: |
| One row at a time | 207.9 s | 3.62 GiB |
| Batched at 8 | 39.3 s | 5.34 GiB |

That is a 5.3x speedup on the eval path for 1.7 GiB more peak memory.

#### Batched generation is not bit-identical, and that is expected

This was checked rather than assumed, because a silent change to eval output
would corrupt every metric downstream. Greedy decoding, 128 new tokens:

- Decoding one row alone, twice: **8/8 token-identical**. The model is
  deterministic, so any difference below is caused by batching.
- Batch of 8 against those solo runs: **5/8 token-identical**. The three that
  drifted first differed at tokens 13, 16, and 120.
- Parsed answer JSON, batched against solo: **8/8 identical**.

The cause is reduction order inside the bf16 matmuls changing with batch shape,
which can flip a greedy argmax at a near-tie. The scored artifact — the parsed
answer — was unaffected on this sample, but `raw` completion text can differ
token-for-token between a batched and a serial run even at temperature 0.
`eval.batch_size: 1` restores byte-stable completions for an investigation that
needs them.

### 9. `max_seq_len` lowered 4096 -> 2048

Real examples tokenise to about 1.5k. Batches pad to their own longest row
rather than to this limit, so lowering it does not speed up a typical batch;
what it does is bound the worst case, where one 4k outlier would stretch every
sequence batched with it. `max_drop_fraction` remains the guard.

### Still open

- `lora_r` is unchanged at 32. Finding 6 recommends 8 from the ms-swift dense
  recipe, and it would cut adapter and optimizer work, but rank governs adapter
  capacity and is ablation material — it is a quality decision, not a speed
  cleanup, so it was left for an explicit call.
- Liger kernels remain unverified.
- Gemma 4 E4B-it needs a memory-reduction pass, not a speed pass. Candidates:
  a smaller `eval.batch_size` during in-loop eval, and a lower reasoning cap.

---

## Addendum — 2026-08-02: kernel options for Qwen3.5 on one L4

### 10. The architecture explains the FlashAttention result

Read from the checkpoint's own config rather than inferred:

```
num_hidden_layers      = 32
layer_types            = [linear_attention x3, full_attention] x8
full_attention_interval = 4
head_dim               = 256
vocab_size             = 248320
hidden_size            = 2560
```

Two consequences:

1. **Only 8 of 32 layers are full attention.** FlashAttention can accelerate a
   quarter of the stack; the other 24 layers are Gated DeltaNet and run on
   `fla` + `causal-conv1d`, which are already installed. A kernel that touches
   25% of the layers cannot produce a large end-to-end win, which is consistent
   with the sub-1% steady-state tie measured in finding 3.
2. **`head_dim = 256` is exactly FlashAttention 2's upper limit**, which is why
   Gemma 4 E4B-it failed to run under FA2 while Qwen did.

The 248,320-entry vocabulary over hidden size 2560 also means the LM head is
roughly 635M parameters — about a sixth of a 4B model. That, not attention, is
where the remaining head-room on this workload sits, and it is what makes
`selective_logits` and Liger's chunked cross-entropy relevant.

### 11. Two settings are load-bearing for memory, not just speed

Measured at batch 4 on a clean L4 (21.66 GiB free), 4 optimizer steps:

| Arm | Result |
| --- | --- |
| `selective_logits=false` | **out of memory** — tried to allocate 5.63 GiB |
| `gradient_checkpointing=false` | **out of memory** — 19.97 GiB already allocated |

Both were previously described as speed trade-offs. At batch 4 they are neither
optional nor tunable: turning either off does not run at all. `selective_logits`
in particular is what *makes* batch 4 possible, so the two changes in the
previous addendum are coupled rather than independent.

### 12. Timing from a fixed-order single-shot sweep was not usable

The first sweep ran six arms once each, in a fixed order, and reported the
baseline at 519.1 s against 295.2 s for the arm that ran last. Arms later in the
order were uniformly faster regardless of what they changed, which is the
signature of warm-up or thermal drift on a shared Colab GPU — the same failure
that made the original FA2 comparison in finding 3 misleading.

Those numbers are therefore withheld rather than reported. Memory results above
survive because an out-of-memory failure is not a stopwatch reading.

Two further attempts did not recover a usable timing comparison:

1. An interleaved sweep (4 arms x 3 rounds) was abandoned after round 1 — about
   four minutes per run, dominated by re-loading a 4B checkpoint per arm.
2. A corrected sweep timing only `train_runtime` (3 arms x 2 rounds, warm cache)
   ran all six arms successfully, but the extraction read the final save
   directory's `trainer_state.json`, which does not carry `train_runtime`; every
   arm returned null, and the session was lost before the value could be
   recovered from the checkpoint states.

What that run did establish, from peak memory, is that **Liger measurably
reduces memory** and the attention backend does not change it at all:

| Arm (batch 4, 6 steps) | Peak allocated, round 0 / round 1 |
| --- | --- |
| SDPA | 9.63 / 9.61 GiB |
| FlashAttention 2 | 9.61 / 9.62 GiB |
| Liger (selective logits off) | 9.41 / 9.41 GiB |

Liger is reproducibly ~0.2 GiB lighter across both rounds, which is consistent
with a chunked cross-entropy over the 248k-entry head. FA2 and SDPA are
identical to within noise, which again matches the architecture: FA2 changes
only the 8 full-attention layers, and it changes their memory profile
negligibly under checkpointed QLoRA.

**So there is still no trustworthy step-time comparison between SDPA, FA2, and
Liger at batch 4.** The cheap way to get one, if it is ever worth the time: a
single process that loads the model once and times N steps per backend in a
loop, rather than a fresh subprocess and a fresh 4B load per arm. Most of the
per-arm cost was loading, not training.

### 13. Model loading was never the bottleneck it was assumed to be

The previous section blamed the abandoned sweep's cost on "re-loading a 4B
checkpoint per arm". That was an assumption, and measuring it showed it was
wrong. Split into its parts, on an L4 with the checkpoint already in the HF
cache:

| Stage | Seconds |
| --- | ---: |
| `resolve_model_class` (AutoConfig read) | ~6.0 |
| `AutoTokenizer.from_pretrained` | ~2.4 |
| `from_pretrained` (4-bit materialise + dispatch) | ~5.2 |
| **Total warm load** | **~13.5** |

A cold download of the full checkpoint, with `hf_transfer` enabled, took a
further **39.7 s once**. So an arm that took roughly four minutes spent about 13
seconds loading. The cost was the training steps, not the load, and the
"one process, load once" advice above is worth far less than it appeared.

Two candidate load optimisations were tested and both rejected:

| Arm | Total load, 3 interleaved runs | Verdict |
| --- | --- | --- |
| `device_map="auto"` (current) | 13.59 / 13.55 / 13.49 s | baseline |
| `device_map="cuda:0"` | 13.56 / 13.51 / 13.39 s | no difference |

A first single-shot pass had suggested `cuda:0` cut `resolve_model_class` from
10.3 s to 6.2 s, which looked like Accelerate's multi-device planner being
skipped. Repeating the arms in A/B/A/B order showed `resolve_class` is ~6.0 s in
every arm including `auto`: the extra 4 s was one-off CUDA and library
initialisation paid by whichever arm ran first in a fresh process. This is the
same first-run artifact that invalidated the finding-12 sweep, caught this time
because the arms were repeated.

`device_map` is therefore left as `auto`. It is correct for multi-GPU and costs
nothing measurable on one L4.

What is worth keeping from this:

- **`hf_transfer` is not installed and no HF cache location is configured.** The
  39.7 s download is per fresh Colab VM, and every arm/run of every future
  session pays it again. Adding `hf_transfer` to the requirements and pointing
  `HF_HOME` at a mounted Drive path would remove that repeat cost. `colab
  drivemount` exists for exactly this.
- `resolve_model_class` reads `AutoConfig`, and `from_pretrained` reads it again.
  That duplicate read is real but small relative to everything else, and
  removing it would mean threading the config through the loader for ~1 s.

### 14. Qwen's spare VRAM is not spendable, and Gemma's weight is not a defect

Qwen peaks at 9.5 of 22 GiB, which looks like 12 GiB left on the table while
`gradient_checkpointing` is still paying recomputation for memory nobody needs.
Two ways to spend it were tested, and neither works.

**Disabling checkpointing OOMs, at every batch size tried:**

| Arm | Result |
| --- | --- |
| batch 4, checkpointing off | out of memory (both repeats) |
| batch 2, checkpointing off | out of memory (both repeats) |

Peak *allocated* is not the constraint — activation memory during backward is,
and it is transient enough not to show up in the steady-state figure. The 12 GiB
gap is headroom the backward pass needs, not headroom going unused.

**Widening the batch is within noise.** Batch 4 against batch 6, checkpointing
on, two interleaved repeats each, normalised per sequence so the arms compare:

| Arm | sec/sequence, r0 | sec/sequence, r1 | Peak |
| --- | ---: | ---: | ---: |
| batch 4 x accum 8 | 2.316 | 2.347 | 9.62 GiB |
| batch 6 x accum 6 | 2.505 | 2.100 | 11.78 GiB |

Batch 6 varies from 2.10 to 2.51 s/sequence between two identical runs, a spread
wider than its distance from batch 4. On a shared Colab GPU the run-to-run noise
exceeds the effect, so batch 6 cannot be called faster; it just costs 2.2 GiB
more. **Batch 4 stays.** This is also the first trustworthy step-time number in
this report — roughly 2.3 s per sequence — obtained by normalising per sequence
and repeating each arm, which is what findings 12 and 13 lacked.

**Gemma's memory was investigated and is correct.** Its 20.6 GiB looked
suspicious next to Qwen's 9.5, and the suspicion was that `text_only` silently
failed for it — every log line in the project reads `removed model.visual`,
which is Qwen's tower name. Inspecting the module trees disproved that:

| Model | Media towers removed | Language-model params |
| --- | --- | ---: |
| Qwen3.5-4B | `model.visual` (168.9M) | 2,421M |
| Gemma 4 E4B-it | `vision_tower`, `audio_tower`, `embed_vision`, `embed_audio` (247.1M) | **5,476M** |

`text_only` works on both, and removes *more* from Gemma. Gemma is simply a much
larger language model — 5,476M against 2,421M, roughly 2.3x — so 20.6 GiB is the
expected cost of it, not a leak. Any further Gemma work has to reduce what the
language stack itself needs; the towers are already gone.

A `load_in_4bit=false` arm was also attempted and could not run: the installed
`torchao` 0.10.0 is incompatible with this Transformers version's bf16 LoRA path.
Untested rather than rejected.

---

## Addendum — 2026-08-02: the run does not fit in a session, which outranks the speed work

### 15. The real run is ~21 hours, so survivability matters more than throughput

Computed from `params.yaml` at the configured scale, using the measured
2.3 s/sequence from finding 14:

```
sample.full_senses 3400 x sample.k 6      = 20,400 texts
x split.train 0.8                         = 16,320 rows
x train.epochs 2                          = 32,640 sequences
x 2.3 s                                   ≈ 20.9 hours
```

A Colab session ends well before that. This reframes every optimisation above:
batching eval saves ~0.47 h of 21 h (about 2%), and the batch-size change saves
roughly 5%. Neither matters if the run cannot reach the end.

Three defects in the launcher made a long run unrecoverable, all now fixed:

1. **`--resume` was never passed.** The CLI has accepted `--resume auto` (its own
   default) since Phase 3, and `resolve_resume` picks the newest checkpoint, but
   `run_colab_train.sh` never forwarded it — so relaunching after a killed
   session silently restarted at step 0. Now `RESUME` (default `auto`).
2. **Checkpoints only left the VM after training finished.** Step 5 downloads
   the output directory once the run completes, so a session killed at hour 6
   took its ten checkpoints with it. `REMOTE_OUTPUT_DIR` now points `--output`
   at a Drive mount, and the Trainer writes each `save_steps` checkpoint to
   storage that outlives the VM.
3. **`trap cleanup EXIT` destroyed the VM on *any* exit** — including a failed
   step or a Ctrl-C — deleting the VM-local checkpoints at exactly the moment
   they mattered. The trap now inspects the exit status: it stops the VM on
   success, and on failure leaves it running, printing how to resume, inspect,
   or stop it, plus a note that it costs compute units meanwhile. Provisioning
   reuses a live session so a resume can reach those checkpoints, and setup is
   idempotent (`git fetch` when the checkout exists, rather than a clone that
   would fail).

Resume was verified rather than assumed, on an L4:

| Phase | Command | Result |
| --- | --- | --- |
| 1 | `max_steps=3`, `--resume none` | `global_step` 3, `checkpoint-3` |
| 2 | `max_steps=6`, `--resume auto` | `global_step` **6**, `checkpoint-3` + `checkpoint-6` |

A broken resume would have re-run steps 1–3 and finished at 3. It finished at 6,
so the second launch continued from the checkpoint.

`REMOTE_OUTPUT_DIR` itself is **not** exercised against a real Drive mount:
`colab drivemount` needs interactive Google authorisation, which a headless
session cannot complete. The path handling and validation are tested; the mount
is not.

### 16. `lora_r` lowered 32 -> 8 at the user's direction

Finding 6 recommended rank 8 / alpha 32 from the ms-swift dense Qwen3.5 recipe,
and previous addenda deliberately left it alone as a quality decision. The user
chose rank 8. Alpha moves with it to 32 so the LoRA scaling `alpha / r` stays at
4 — changing rank alone would also change the effective update magnitude and
confound a rank comparison with a learning-rate one.

This is a capacity reduction, not a free win. It is untested against the real
corpus, so if format validity or meaning agreement regress against a rank 32
arm, rank is the first thing to restore.

### What to do next

1. **Build the dataset.** `data/clean/` does not exist, so there is nothing to
   train on yet. A full run needs 20,400 teacher-generated texts, which is spend
   and wall-clock in its own right, and it gates everything else.
2. **Then a resumable full run** with `REMOTE_OUTPUT_DIR` on Drive, relaunching
   after each session ends.
3. Optional, cheap when a GPU is already up: rank 8 against rank 32 on quality,
   and `load_in_4bit=false` once `torchao` is upgraded.
