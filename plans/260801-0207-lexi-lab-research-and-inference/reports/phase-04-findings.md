# Phase 4 findings — RL: GRPO, JEPO, NRT

**Status: three tracks implemented and running, ablations pending hardware.**

All three complete two optimiser steps inside `lexi smoke` on CPU against a
randomly-initialised model, which is the acceptance criterion this phase could
meet without a GPU. The A1/A3/A4 numbers cannot exist yet: they need a trained
SFT baseline and a real dataset.

## What is settled

- **`seq_logp` NRT and single-sample JEPO produce identical rewards**, to 1e-12,
  at every sequence length tested and including a floored zero probability. The
  two were written independently — JEPO sums log-probabilities it is handed, NRT
  takes probabilities and logs them itself — so a bug would have to exist in
  both, identically, to pass. This is the check that makes A1 and A4 meet at a
  known point.
- **The three objectives never share a token.** Cross-entropy covers the whole
  answer, reward covers correction and meaning, policy gradient covers the
  reasoning. A test asserts the masks are disjoint where they must be, so a
  difference between two tracks stays attributable to its reward definition.
- **Switching `rl.reward_scope` moves exactly the feedback span and nothing
  else**, which is what makes A3 one axis rather than two.
- **The strip-mismatch penalty had to be raised to 1.0.** At the design's 0.5, an
  otherwise-perfect answer that quietly rewrote unmarked text still scored +0.3 —
  positive, so worth producing. The penalty now exceeds the whole positive total,
  making a drifted correction never worth emitting. This is the one failure mode
  invisible to every span-based metric downstream.

## What the CPU gate already shows

Running the tiny random model through all three tracks is not a result about the
method, but two of its readings are worth recording because they are what a real
run should *not* look like:

| Track | reward mean | empty gap | dead groups |
|---|---|---|---|
| GRPO | 0.000 | 0.000 | 100% |
| JEPO | -296.6 | -0.154 | 0% |
| NRT (geo-mean) | 0.0013 | 7e-06 | 0% |

GRPO's reward is flat zero because a random model never emits parseable JSON, so
every rollout scores the same and every group is dead — exactly the diagnostic
`zero_advantage_share` exists for. JEPO's reward is a raw sequence
log-probability, so its scale is the answer length; that is why the tracks are
compared through normalised rewards and never by raw curve.

The negative empty gaps say sampled reasoning made the gold answer *less* likely
than an empty block did. On a random model that is noise. On a trained one it
would mean the run is dead, whatever the reward curve is doing.

## A1 — method

| Arm | QWK / ceiling | span+tag F1 / ceiling | reward mean | empty gap | KL |
|---|---|---|---|---|---|
| SFT | pending | pending | — | — | — |
| +GRPO | pending | pending | pending | pending | pending |
| +JEPO | pending | pending | pending | pending | pending |
| +NRT geo-mean | pending | pending | pending | pending | pending |
| +NRT weighted(-log p) | pending | pending | pending | pending | pending |

**A null result here is the expected outcome and will be reported as one.** RL
beating SFT on ~20k rows of distillation data is not the default expectation.
That is acceptable only because Phase 2 built and validated the harness before
any of this code existed; a well-measured null result is a stronger artifact than
an unmeasured win.

## A3 — reward scope

| Arm | correction F1 | meaning QWK | feedback chrF (weak) |
|---|---|---|---|
| `correction_meaning` | pending | pending | pending |
| `full_answer` | pending | pending | pending |

The design's claim is that rewarding feedback costs the verifiable fields. If
`full_answer` wins on chrF and loses on correction F1, the claim holds and the
default is right. If it wins on both, the claim was wrong and that is the finding.

## A4 — NRT aggregation

| Aggregation | reward mean | QWK | reasoning tokens |
|---|---|---|---|
| `seq_logp` | pending | pending | pending |
| `geo_mean` | pending | pending | pending |
| `arith_mean` | pending | pending | pending |
| `weighted_neglogp` | pending | pending | pending |

`seq_logp` must reproduce the A1 JEPO arm. A disagreement between them is a bug,
not a finding.

## Before any full run

`lambda` is swept on the 50-row fixture, per track. There is no reason the three
should agree on it, and a shared value chosen from one track's sweep would make
the other two look worse than they are. The chosen value goes in the run config
and is reported next to the numbers it produced.
