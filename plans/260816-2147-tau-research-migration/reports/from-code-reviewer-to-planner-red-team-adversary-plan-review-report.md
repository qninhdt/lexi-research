# Red Team Review Report: `tau-research` Migration Plan

**Plan Directory**: `plans/260816-2147-tau-research-migration/`  
**Date**: 2026-08-16  
**Lenses Applied**: Security Adversary, Assumption Destroyer, Failure Mode Analyst, Scope & Complexity Critic  
**Verification Tier**: Full  

---

## 1. Executive Summary

The implementation plan is technically solid, properly aligned with `idea.md`, and adopts a clean TDD posture. However, adversarial stress-testing identified 6 critical/high-priority failure modes that must be explicitly guarded in the phase files before execution begins:

1. **VRAM OOM & vLLM Memory Collision on 24GB L4**: Memory budget collision during GRPO backward step if vLLM KV cache allocation is not strictly capped with sleep mode.
2. **Qwen3.5 Thinking Loop & Truncation Handling**: Potential rollout deadlock/crash when reasoning length hits hard limit without producing a closing `</think>` tag or tool action.
3. **Zero Reward Variance in GRPO Advantage Computation**: Stalling of policy updates when all $G=4$ rollouts for a task yield identical rewards ($0,0,0,0$ or $1,1,1,1$).
4. **Chat Template Response Marker Masking Ambiguity**: Risk of `DataCollatorForCompletionOnlyLM` masking target thinking tokens if tokenizer template doesn't explicitly mark the start of assistant reasoning.
5. **State Pollution in tau2-bench Gym Database**: Reusing SQLite DB instances across parallel rollouts or repeated trials without atomic reset.
6. **API Key & Credential Leakage in Colab Scripts**: Exposing `OPENAI_API_KEY` or `WANDB_API_KEY` in shell scripts or W&B run configs.

---

## 2. Adversarial Findings & Adjudication

### Finding 1: vLLM KV Cache & Optimizer State Collision on 24GB L4
- **Severity**: Critical
- **Lens**: Failure Mode Analyst
- **Location**: Phase 05, section "GRPO Training"
- **Flaw**: Running `GRPOTrainer` + PyTorch backward pass + colocated vLLM inference on a single 24GB L4 GPU risks instant CUDA OOM if vLLM memory utilization exceeds ~0.25 while gradient checkpointing buffers activations.
- **Failure Scenario**: During the first GRPO update step after rollout collection, PyTorch allocates optimizer states and backward gradients, crashing with `CUDA out of memory. Tried to allocate X GiB`.
- **Disposition**: **Accept**
- **Rationale**: Real hardware constraint on L4. Explicit bounds and sleep-mode activation must be enforced in `configs/grpo.yaml` and verified in Phase 05.
- **Suggested Fix**: Mandate `vllm_gpu_memory_utilization: 0.20`, `vllm_enable_sleep_mode: true`, `beta: 0.0`, `per_device_train_batch_size: 1`, and `gradient_accumulation_steps: 16` in `configs/grpo.yaml`.

---

### Finding 2: Thinking Loop & Truncation Handling in Action Parser
- **Severity**: High
- **Lens**: Assumption Destroyer
- **Location**: Phase 03, section "action_parser.py" & Phase 05, section "Rollout Loop"
- **Flaw**: Qwen3.5-2B in thinking mode is known to occasionally enter repetitive reasoning loops ("Wait, let me rethink..."). If `max_generated_tokens_per_turn` (1024) is reached, the completion string is truncated mid-thought without emitting `</think>` or a valid tool call.
- **Failure Scenario**: `action_parser.py` throws an unhandled parsing exception or attempts to step the Gym with an empty action, crashing the entire training loop.
- **Disposition**: **Accept**
- **Rationale**: Validated behavior from Qwen model card. Truncation must be caught gracefully and translated into a structured fallback action (`termination_reason: "truncation"` with penalty reward).
- **Suggested Fix**: Add explicit fallback logic in `action_parser.py` for unclosed `<think>` tags and truncated outputs, returning a standardized `TruncatedAction` that steps the env to terminal state with zero reward and logs `tau/truncation_rate`.

---

### Finding 3: Zero-Variance Reward Batch Handling in GRPO
- **Severity**: High
- **Lens**: Failure Mode Analyst
- **Location**: Phase 05, section "train_grpo.py"
- **Flaw**: If all $G=4$ generations for a sampled task return the same reward (e.g. all failed $\to 0,0,0,0$), standard deviation $\sigma_R = 0$. Standard GRPO advantage normalization divides by zero or produces zero advantage, wasting optimizer steps.
- **Failure Scenario**: Over 50% of optimizer steps produce zero gradient updates, causing the policy to diverge or stagnate.
- **Disposition**: **Accept**
- **Rationale**: Critical RL signal issue documented in TRL. Difficulty profiling helps, but online handling during training is also mandatory.
- **Suggested Fix**: Add dynamic batch filtering in `rollout_func` / trainer callbacks to track `frac_reward_zero_std` and resample tasks if zero-variance persists for $> 3$ consecutive batches.

---

### Finding 4: Tokenizer Chat Template Response Marker & Masking Verification
- **Severity**: High
- **Lens**: Security Adversary / Fact Checker
- **Location**: Phase 02, section "tests/test_loss_mask.py"
- **Flaw**: If `tokenizer.apply_chat_template(enable_thinking=True)` inserts special tokens (`<|im_start|>assistant\n<think>`) and `DataCollatorForCompletionOnlyLM` uses standard string matching for `response_template="<|im_start|>assistant\n"`, the `<think>` reasoning tokens might be incorrectly included in the prompt or masked out if formatting shifts between transformers versions.
- **Failure Scenario**: Model trains only on the tool call and ignores the thinking process, or trains on user prompt tokens, causing massive test degradation.
- **Disposition**: **Accept**
- **Rationale**: Crucial TDD check. We must assert token-level labels in `tests/test_loss_mask.py` with exact token ID verification.
- **Suggested Fix**: Write a deterministic token-level test asserting `labels[i] == -100` for all token IDs before `<think>` and `labels[i] == input_ids[i]` for all token IDs within `<think>...</think>` and tool calls.

---

### Finding 5: SQLite Database Isolation & Reset in Tau Environment
- **Severity**: High
- **Lens**: Failure Mode Analyst
- **Location**: Phase 03, section "rollout.py" & Phase 06, section "evaluate_tau.py"
- **Flaw**: Tau-bench environments maintain internal database state (orders, products, users). If an episode mutates the DB (e.g. cancels an order) and the environment is reused without a full deep-copy / reset of the initial state, subsequent trials or parallel rollouts read corrupted state.
- **Failure Scenario**: Evaluation Pass$^1$ fluctuates wildly and produces non-reproducible benchmark scores.
- **Disposition**: **Accept**
- **Rationale**: Benchmark integrity requires atomic state isolation per episode.
- **Suggested Fix**: Ensure `env.reset()` explicitly verifies clean DB state restoration from the original fixture for every new task/trial, asserted in `tests/test_tau_rollout.py`.

---

### Finding 6: Secret & API Key Sanitization in Colab Automation
- **Severity**: Medium
- **Lens**: Security Adversary
- **Location**: Phase 07, section "Colab Automation"
- **Flaw**: Scripts running in Colab might accidentally print or log `OPENAI_API_KEY` / `WANDB_API_KEY` into bash logs or W&B run configs.
- **Failure Scenario**: Secret exposure in public Colab notebook runs or shared logs.
- **Disposition**: **Accept**
- **Rationale**: Security best practice.
- **Suggested Fix**: Ensure `scripts/setup_colab.sh` and `configs/*.yaml` resolve secrets exclusively from environment variables (`os.environ.get(...)`) and never output them to console or artifacts.

---

## 3. Adjudication Summary

| # | Finding | Severity | Disposition | Target Phase File |
|---|---|---|---|---|
| 1 | vLLM KV Cache & Optimizer Memory Collision | Critical | **Accept** | `phase-05-grpo-training-and-difficulty-profiling.md` |
| 2 | Thinking Loop & Truncation Fallback Handling | High | **Accept** | `phase-03-tau-env-action-parser-and-rollout-adapter.md` |
| 3 | Zero-Variance Reward Batch Handling | High | **Accept** | `phase-05-grpo-training-and-difficulty-profiling.md` |
| 4 | Chat Template & Loss Mask Token-Level Verification | High | **Accept** | `phase-02-sft-data-pipeline-and-loss-masking.md` |
| 5 | SQLite DB State Isolation & Reset Integrity | High | **Accept** | `phase-03-tau-env-action-parser-and-rollout-adapter.md` |
| 6 | Secret Sanitization in Scripts & Configs | Medium | **Accept** | `phase-07-colab-automation-and-smoke-verification.md` |

Total Findings: **6** (1 Critical, 4 High, 1 Medium) | Accepted: **6** | Rejected: **0**
