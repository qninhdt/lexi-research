---
phase: 9
title: "Serving shim"
status: pending
priority: P1
effort: "2d"
dependencies: [6, 7]
---

# Phase 9: Serving shim

## Overview

Expose the trained adapter as an OpenAI-compatible endpoint that returns the **full**
five-field grade. Closes the gap between what the model emits (3 fields) and what a
consumer needs (5 fields), so switching `llm_base_url` is genuinely all that is
required downstream.

## Requirements

**Functional**

- OpenAI-compatible `POST /v1/chat/completions` accepting the Phase 2 prompt
- Parse model output, compute `grammar` / `naturalness` from the calibrated config,
  return `{correction, meaning, grammar, naturalness, feedback}`
- Retry on format-validation failure, bounded, then a typed error
- `GET /healthz` (liveness) and `GET /readyz` (model loaded, config loaded)
- Report which `band_config.json` version is active
- Docker Compose: shim + vLLM backend

**Non-functional**

- Backend-agnostic: vLLM, llama.cpp, or Ollama behind the same interface
- Structured JSON logs; latency and retry-rate metrics
- Config via env vars, no hardcoded paths

## Architecture

### Why a shim exists at all

The model emits `correction`, `meaning`, `feedback`. `grammar` and `naturalness` are
**computed**, not generated (Phase 6). Point a consumer straight at vLLM and it
receives three fields and silently lacks two. The shim is where the band function
lives.

```
consumer  ──OpenAI-compatible──►  shim  ──►  vLLM + LoRA adapter
                                    │
                                    └── band_config.json → grammar, naturalness
```

### The coupling that must not break

**`band_config.json` and the adapter are one deployable unit.** A checkpoint served
with the wrong calibration produces plausible but wrong bands — the worst failure
mode, because nothing errors. Mitigations:

- Both artifacts carry the same `bundle_id`; the shim refuses to start on mismatch
- `/readyz` reports adapter revision and config version
- Every response carries the config version so a bad deploy is traceable from logs

### Request handling

1. Receive request; extract `{target, sense, text}`
2. Forward to backend with the Phase 2 prompt (identical to training — the module is
   shared, not copied)
3. Validate output against the Phase 2 validator
4. On failure: retry up to `MAX_RETRIES` (default 2). Persistent failure returns a
   typed error, never a fabricated grade
5. On success: compute bands, return five fields

### Failure posture

Returning a wrong grade is worse than returning an error — a learner's FSRS state
would be mutated from a fabricated verdict. Every unrecoverable path returns a typed
error and lets the caller decide. This mirrors how `grade_rubric` already returns
`pending` when no judge is available.

### No auth in v1 — stated explicitly

The shim ships **without authentication**. Acceptable only because it binds to
localhost or a private Docker network. Exposing it publicly without an auth layer
would let anyone consume GPU. Documented in the README as a deployment constraint,
not left implicit.

## Related Code Files

- Create: `serve/__init__.py`
- Create: `serve/app.py` (FastAPI)
- Create: `serve/backend.py` (OpenAI-compatible client to vLLM/llama.cpp)
- Create: `serve/bands.py` (imports Phase 6 calculator — no reimplementation)
- Create: `serve/config.py` (pydantic-settings)
- Create: `serve/schemas.py`
- Create: `serve/Dockerfile`
- Create: `docker-compose.yml`
- Create: `tests/serve/test_app.py`, `tests/serve/test_bundle_guard.py`
- Create: `serve/README.md`

## Implementation Steps

1. `config.py`: env-driven settings (backend URL, adapter path, config path, retries).
2. `backend.py`: async client to the inference backend, timeout + retry.
3. `app.py`: the endpoint, wiring parse → validate → retry → band compute.
4. Bundle guard: compare `bundle_id` in adapter metadata against config; refuse
   startup on mismatch.
5. `/healthz`, `/readyz` with adapter and config versions.
6. Structured logging: request id, latency, retry count, validity outcome.
7. Dockerfile + Compose with vLLM and the adapter mounted.
8. Tests against a fake backend: happy path, malformed output + retry, persistent
   failure → typed error, bundle mismatch → refuse start.
9. README: how to run, env vars, the no-auth constraint, how to swap backends.

## Success Criteria

- [ ] `docker compose up` serves a working endpoint
- [ ] Response contains all five fields with correct band computation
- [ ] Band code imported from Phase 6, not duplicated
- [ ] Prompt module shared with training, not copied
- [ ] Bundle mismatch refuses startup with a clear error
- [ ] Malformed output retried; persistent failure returns typed error, never a
      fabricated grade
- [ ] `/readyz` reports adapter revision and config version
- [ ] Tests cover happy path, retry, hard failure, bundle guard
- [ ] README states the no-auth constraint explicitly

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| **Adapter/config version drift → plausible wrong bands** | high | `bundle_id` guard refuses startup; version in every response |
| Prompt drift between training and serving | high | Single shared module; parity test |
| Unauthenticated endpoint exposed publicly | high | Localhost/private-network binding; documented constraint |
| Band logic reimplemented in shim | medium | Import from Phase 6; test asserts identical output |
| Fabricated grade on parse failure | medium | Typed error only, never a default grade |
| Backend lock-in to vLLM | low | OpenAI-compatible client interface |
