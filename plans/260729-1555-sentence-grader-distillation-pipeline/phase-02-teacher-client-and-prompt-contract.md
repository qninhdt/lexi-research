---
phase: 2
title: "Teacher client and prompt contract"
status: complete
priority: P1
effort: "3d"
dependencies: [1]
---

# Phase 2: Teacher client and prompt contract

## Overview

Build the OpenAI-compatible teacher client (retry, rate limit, resume, cost
tracking) and author the **two prompts**: call 1 the diversifier, call 2 the
teacher/grader.

The single most important constraint in this whole plan lives here: **call 2's
prompt is byte-identical to the prompt the finetuned student will see at
inference.** That identity is what makes this distillation rather than
"training on data a teacher happened to produce". If the two drift, the student
learns a function it will never be asked to compute.

## Requirements

**Functional**

- `TeacherClient` over any OpenAI-compatible `/chat/completions` endpoint
- Structured output via `chat.completions.parse`, with a `function_calling`
  fallback for loose proxies
- Retry with exponential backoff on transient failures; typed exhaustion error
- Concurrency limit + rate limiting; configurable
- **Resumable**: a 2,500-call job must survive interruption without re-spending
- Per-call cost + token accounting, aggregated to a run report
- Prompt registry that loads templates from disk and hashes them
- `render_grader_prompt()` — the single function used by call 2, eval, and serving

**Non-functional**

- Prompt templates are **versioned artifacts in Git**, hashed into the manifest
- Prompt hash change → DVC invalidates the generation stage
- No secrets in code; API key from env
- Tests hermetic — a fake client, no network

## Architecture

### Client

```
lexi_research/teacher/
├── client.py        TeacherClient, retry, concurrency, cost
├── cache.py         content-addressed response cache (resume)
├── prompts/
│   ├── grader_system.jinja        ← CALL 2 = INFERENCE PROMPT
│   ├── grader_user.jinja          ← CALL 2 = INFERENCE PROMPT
│   ├── diversify_system.jinja     ← CALL 1 only
│   └── diversify_user.jinja       ← CALL 1 only
├── registry.py      template loading + hashing
└── schemas.py       pydantic I/O models
```

Reuse the proven shape from `lexi-ai/lexi_ai/llm.py` (read-only reference):
a narrow `parse(messages, schema)` protocol, lazy client construction, retry with
`base_delay * 2**attempt`. Do not import `lexi_ai` — copy the pattern, keep this
repo standalone.

### Resume via content-addressed cache

```
key = sha256(model + prompt_hash + serialized_request)
```

Cache hit → return without a network call. This makes the pipeline **idempotent**
and interruption-safe: rerun the same stage and only the missing calls fire. It
also makes cost predictable when iterating on downstream stages.

Store as JSONL sharded by key prefix. Not DVC-tracked (it is a cache, not an
artifact), but its hit rate is reported.

### Call 2 — the grader prompt (authoritative)

System prompt content:

- Role: grade a learner's sentence for a target word used in a specific sense
- The `meaning` rubric, 0–4, with anchors (Phase 2 authors these; see below)
- The `correction` format spec: `[A>B:tag]`, three operations, whole sentence
  re-emitted, `null` when unreadable
- The 16-tag taxonomy with one-line definitions
- Hard rule: **do not alter any text outside a bracket group**
- `feedback`: exactly one sentence, English
- Untrusted-input boundary: wrap the learner text in a nonce-delimited block and
  instruct that its content is data, never instructions (same technique as
  `lexi_ai.llm.guarded_messages`)

User prompt: `target`, `sense.definition`, `sense.pos`, and the learner `text`
inside the nonce block.

Output schema:

```json
{ "correction": "string | null", "meaning": 0, "feedback": "string" }
```

### `meaning` rubric with anchors

Anchored bands, because a rubric without anchors cannot be applied consistently —
that is the failure mode that would poison every label:

| Band | Criterion | Anchor (`bright` = "full of light") |
|---|---|---|
| 4 | Correct sense, no drift | *The room was bright and airy.* |
| 3 | Correct sense, slight nuance drift | *Her bright lamp helped me read.* (leans to the lamp, not the space) |
| 2 | Ambiguous — readable as another sense | *She had a bright future in the sunny office.* |
| 1 | Right word, slipped to a near sense | *She's a bright student.* (= clever) |
| 0 | Wrong sense entirely, or target absent | *The music was bright yesterday.* |

Anchors ship in the prompt as few-shot exemplars. **The exemplars are part of the
inference prompt**, so they must be present in both call 2 and serving.

### Call 1 — the diversifier (NOT a label source)

Produces `text` only. Conditioned on a learner profile and a target band/error
spec. Those knobs are **diversity parameters**, retained as metadata for coverage
analysis — never as labels.

Batched: one sense, K specs, K sentences out. **`K = 6` (fixed project-wide)**.

Output schema: `{ "sentences": [{"spec_id": "...", "text": "..."}] }`

### Batching parity check

Call 2 is batched (K texts per call) while inference is single. That is the one
permitted deviation from prompt identity, and it must be **measured, not assumed**:
grade ~40 sentences at `K=6` and again at `K=1`; if `meaning` disagrees or
`correction` F1 drops materially, reduce K or unbatch call 2.

This check runs in Phase 4's pilot and gates full generation.

## Related Code Files

- Create: `lexi_research/teacher/{__init__,client,cache,registry,schemas}.py`
- Create: `lexi_research/teacher/prompts/{grader_system,grader_user,diversify_system,diversify_user}.jinja`
- Create: `tests/teacher/{test_client,test_cache,test_registry,test_prompt_parity}.py`
- Read-only reference: `/home/qninh/projects/lexi-ai/lexi_ai/llm.py`

## Implementation Steps

1. `schemas.py`: `GraderOutput`, `DiversifyBatch`, `TeacherConfig`, `CallStats`.
2. `client.py`: async client, `parse()` with json_schema + function_calling modes,
   retry, `asyncio.Semaphore` concurrency cap, token/cost accumulation.
3. `cache.py`: content-addressed JSONL store; `get`/`put`; hit-rate stats.
4. `registry.py`: load `.jinja` from the package, render, expose
   `prompt_hash()` over all templates. Single `render_grader_prompt(target, sense,
   text) -> list[ChatMsg]` used by call 2, eval, and serving alike.
5. Author `grader_system.jinja`: role, anchored rubric, correction spec, taxonomy,
   no-drift rule, feedback rule, nonce boundary.
6. Author `grader_user.jinja`: target/sense/text with nonce-wrapped learner text.
7. Author `diversify_*.jinja`: profile-conditioned, K specs, JSON array out.
8. Tests: fake client (no network); retry exhaustion; cache hit avoids the call;
   cost accounting; **prompt-parity test asserting call 2 and serving call the same
   `render_grader_prompt`** (guards against drift by construction, not convention).
9. CLI: `python -m lexi_research.teacher.probe` — one round trip against the
   configured endpoint to confirm creds, structured-output mode, and latency.

## Success Criteria

- [ ] `probe` succeeds against the user's endpoint and reports which
      structured-output mode works
- [ ] Retry, concurrency cap, and cost accounting unit-tested with a fake client
- [ ] Cache: rerunning an identical batch issues **zero** network calls
- [ ] `prompt_hash()` stable across runs, changes when any template changes
- [ ] Prompt-parity test passes: exactly one code path renders the grader prompt
- [ ] Rubric anchors present in the prompt for all five bands
- [ ] mypy strict + ruff clean; no secrets in tracked files

## Risk Assessment

| Risk | Mitigation |
|---|---|
| **Call 2 prompt drifts from inference prompt → not distillation** | One shared render function; parity test; prompt hash in manifest |
| Provider ignores strict json_schema, returns loose JSON | `function_calling` fallback mode (proven necessary in `lexi-ai`) |
| Batching changes grading behaviour | Explicit K=6 vs K=1 parity measurement gating Phase 4 |
| Interrupted long job re-spends budget | Content-addressed cache; resume is the default path |
| Rubric anchors bias the teacher toward exemplar wording | Anchors are short and span all bands; Phase 4 measures the resulting band distribution |
| Rate limits / 429 storms | Semaphore + backoff + jitter; configurable concurrency |
| Prompt injection inside learner text | Nonce-delimited untrusted block, mirroring `guarded_messages` |
