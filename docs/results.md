# Results — Lexi Lab

**Status: the lab is built and green end to end on CPU. No experimental result
exists yet, because every one of them needs a GPU and a real dataset.**

This document is written now rather than after the runs so that the claims it is
allowed to make are fixed before the numbers arrive. What follows is what was
built, what it already showed, and what the experiments will and will not be able
to say.

## What is claimable, and what is not

The parent design settled this and it does not change: every number this project
can produce is **fidelity to a teacher, not accuracy against ground truth**.

- There is no human gold set. Train and test are both teacher-generated.
- The real-learner distribution is unverified. This is the largest validity
  threat in the project and it was not addressed.
- Metrics are reported as a fraction of teacher self-consistency, because a
  student cannot exceed the agreement its teacher has with itself.
- `feedback` has no verifiable ground truth at all, and its two proxies are
  labelled weak everywhere they appear.

A reader looking for "how good is this grader" will not find it here. They will
find how closely a 4B student reproduces a frontier grader on the distribution
that grader generated.

## What the build already showed

These came out of building the harness, not out of running experiments, and each
one changed the code:

1. **The old trainer supervised the prompt.** The completion is 3.2% of a
   rendered sequence, so roughly 97% of the gradient went into reproducing a
   rubric that is supplied verbatim at inference. A7 will quantify what that cost.
2. **`max_seq_len: 1024` would have dropped every row.** A rendered example is
   ~1250 whitespace tokens before subword splitting. Raised to 4096.
3. **A hardcoded LoRA target list adapts a fraction of some architectures.** On a
   hybrid linear-attention stack it reaches the attention of a quarter of the
   layers and none of the rest, while producing a run that trains, logs a falling
   loss, and saves an adapter. Targets are now resolved by role from the loaded
   module tree, and the coverage prints before the first optimiser step.
4. **Trusting a chat template's assistant mask would have failed on every real
   checkpoint.** Asked for one, `transformers` warns and returns a mask of zeros
   when the template has no `{% generation %}` block — and no published template
   has one.
5. **The strip-mismatch reward penalty was too small to work.** At the design's
   0.5 an otherwise-perfect answer that quietly rewrote unmarked text still scored
   positive, so the behaviour was worth producing. It is now larger than the whole
   positive total.

Each of these is the kind of defect that produces a plausible number rather than
a crash. That is the argument for building the harness before the experiments,
and it is the argument this project would make again.

## What the experiments will say

| Ablation | Question | Status |
|---|---|---|
| A7 | What does supervising the prompt cost? | arms defined, needs a GPU |
| A2 | Is reasoning worth its token cost here? | arms defined, needs a GPU |
| A6 | Where should LoRA attach, and at what rank? | arms defined, needs a GPU |
| A1 | Does RL beat SFT? | tracks implemented, need an SFT baseline |
| A3 | Does rewarding feedback hurt the verifiable fields? | arms defined |
| A4 | Which NRT aggregation? | arms defined |
| B1–B7 | Engine, quantisation, decoding, speculation, concurrency, cache | harness done, needs a rented GPU |
| B8 | Was any of this worth it versus calling a bigger model? | comparison implemented |

## The result most likely to appear

**RL probably does not beat SFT on ~20k rows of distillation data.** That is the
expected outcome, not a hedge, and the whole ordering of this project exists to
make it presentable: the eval harness was built and validated in Phase 2, against
hand-computed values, before any RL code existed. A null result from a trusted
harness is a result. From an untrusted one it is indistinguishable from a bug.

If that is what happens it will be stated plainly, with ceiling-normalised
numbers, in the model card and here.

## What would be next

- A human-labelled test set, even a small one. It is the only thing that converts
  every number here from fidelity into accuracy.
- Real learner sentences rather than teacher-generated ones, for the test split at
  minimum.
- The rubric is ~1250 tokens and dominates both the sequence budget and the
  prefill cost. Whether a shorter one costs accuracy is one cheap arm and nobody
  has run it.
