# Journal — AReaL Migration & Pipeline Hardening (2026-08-22)

## Decisions

1. **SFT dataset switched fuvty → AReaL-tau2-data** after verifying Simia's public release has zero thinking traces (the project's reasoning-SFT design hard-requires them). AReaL ships verified per-turn thinking labels, Apache-2.0, 11.3k retail examples across ~1k dialogs vs fuvty's 7k from only 124 task templates.
2. **RL stays on the official τ² Retail train split** (74 tasks) instead of AReaL's synthetic RL split: zero integration cost through AgentGymEnv + official verifier, cleaner data-provenance story.
3. **max_seq_length=8192** chosen from measured distribution (p95=7250); the plan's own >5%-drop rule eliminated 4096 (41.8% drop) and 6144 (13.4%).

## Lessons

- **Single-pass chat rendering**: rendering prompt and completion separately through the Qwen3.5 template crashes (template demands a user query) AND drifts from inference formatting. Render `prompt+completion` once, cut at the generation header — the suffix is then byte-identical to what sampling will see at decode time.
- **Manual inter-turn appends for multi-turn RL**: re-rendering history after each env turn breaks byte-alignment because the template strips prior-turn think blocks. Append the exact raw strings (`<|im_end|>\n`, `<tool_response>` block, generation header) and let TRL's `env_mask` separate model vs environment tokens.
- **TRL silently drops over-length examples**: an empty-dataloader error with no message. Always log kept/total after trainer construction.
- **Scaffolding tests can pass while being wrong**: two committed tests never ran against real behavior (one expected reward=1.0 from a policy that never cancelled; one mocked away the very kwargs that caused Bug A).

## Verification highlights

- Full conversion: 33,531 rows → 11,287 kept → 10,196/1,091 dialog-disjoint split.
- Decontamination audit: 0 flagged 8-gram pairs vs 40 official test tasks.
- Wiring test: real 3-step SFT backward pass on GPU (0.5B stand-in), loss 2.004, adapter merge OK.
- Gates: ruff + mypy strict + 56 pytest all green at every commit.
