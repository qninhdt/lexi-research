"""Multi-turn agentic ``rollout_func`` for TRL GRPOTrainer.

Each episode keeps ONE growing token sequence. Agent-generated tokens are
appended directly from the sampling backend (mask=1); between turns, the exact
raw strings the Qwen3.5 template would render for tool responses / user replies
are appended manually (mask=0 via ``env_mask``, which TRL maps onto its tool
masking path). Manual appends avoid re-render drift: the chat template strips
prior-turn think blocks, so re-rendering the whole history would no longer be
byte-identical to what the model actually conditioned on while generating.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Exact renderings copied from the Qwen3.5 chat template.
ASSISTANT_TURN_END = "<|im_end|>\n"
TOOL_RESPONSE_TEMPLATE = "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n"
USER_MESSAGE_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n"
GENERATION_HEADER = "<|im_start|>assistant\n<think>\n"


class GenerationBackend:
    """Samples one assistant turn given a token-prefix prompt."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def complete(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        """Returns (generated token ids, their logprobs) for one turn."""
        args = self.trainer.args
        if getattr(self.trainer, "vllm_generation", None) is not None and getattr(
            args, "use_vllm", False
        ):
            _p, completion_ids, logprobs, _t = self.trainer.vllm_generation.generate(
                prompts=[prompt_ids],
                images=None,
                num_generations=1,
            )
            # TRL's VLLMGeneration returns per-token top-k logprobs shaped
            # (batch, seq_len, num_logprobs); policy loss wants the sampled
            # token's logprob, i.e. index 0 of the last axis.
            seq_logprobs = logprobs[0]
            flat: list[float] = []
            for entry in seq_logprobs:
                flat.append(float(entry[0]) if isinstance(entry, (list, tuple)) else float(entry))
            return list(completion_ids[0]), flat

        # CPU / non-vLLM fallback: greedy-free sampling straight off the policy.
        import torch

        model = self.trainer.model
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=args.temperature > 0,
            temperature=max(float(args.temperature), 1e-4),
            top_p=float(getattr(args, "top_p", 1.0)),
            top_k=int(getattr(args, "top_k", 0)) or None,
            max_new_tokens=int(args.max_completion_length),
            pad_token_id=self.trainer.processing_class.eos_token_id,
        )
        new_ids = output[0][input_ids.shape[1] :].tolist()
        return new_ids, [0.0] * len(new_ids)


def run_grpo_episode(
    task_id: str,
    env: Any,
    backend: GenerationBackend,
    tokenizer: Any,
    system_prompt: str,
    max_turns: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Runs one multi-turn episode, tracking model vs environment token spans."""
    from tau_research.tau.action_parser import parse_model_output
    from tau_research.tau.rollout import strip_role_prefix

    obs, info = env.reset()
    base_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": strip_role_prefix(str(obs))},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    sequence_ids: list[int] = list(tokenizer(base_text, add_special_tokens=False)["input_ids"])
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_mask: list[int] = []
    completion_budget = max_completion_tokens

    terminated = False
    truncated = False
    turn_count = 0
    final_reward = 0.0
    final_reward_info: Any = None

    def append_env_text(text: str) -> None:
        nonlocal sequence_ids, completion_ids, logprobs, env_mask, completion_budget
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids[: max(0, completion_budget)]
        sequence_ids.extend(ids)
        completion_ids.extend(ids)
        logprobs.extend([0.0] * len(ids))
        env_mask.extend([0] * len(ids))
        completion_budget -= len(ids)

    while not (terminated or truncated) and turn_count < max_turns and completion_budget > 0:
        turn_count += 1
        gen_ids, gen_logprobs = backend.complete(sequence_ids)
        gen_ids = gen_ids[:completion_budget]
        gen_logprobs = gen_logprobs[: len(gen_ids)]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=False)
        raw_output = raw_output.split("<|im_end|>")[0]

        # Sampling backends include the terminating <|im_end|>/EOS token in
        # output ids; drop it so the manual turn-end below is the ONLY one.
        eos_like = {
            getattr(tokenizer, "eos_token_id", None),
            tokenizer.convert_tokens_to_ids("<|im_end|>")
            if hasattr(tokenizer, "convert_tokens_to_ids")
            else None,
        }
        while gen_ids and gen_ids[-1] in eos_like:
            gen_ids.pop()
            if len(gen_logprobs) == len(gen_ids) + 1:
                gen_logprobs.pop()

        sequence_ids.extend(gen_ids)
        completion_ids.extend(gen_ids)
        logprobs.extend(gen_logprobs)
        env_mask.extend([1] * len(gen_ids))
        completion_budget -= len(gen_ids)

        # Close the assistant turn exactly like the template does.
        append_env_text(ASSISTANT_TURN_END)

        parsed = parse_model_output(raw_output)
        if parsed.is_truncated:
            truncated = True
            break

        action_payload = parsed.to_env_action()
        obs, _reward_val, terminated, env_truncated, step_info = env.step(action_payload)
        truncated = bool(env_truncated)

        if terminated:
            from tau_research.tau.rollout import parse_reward_info

            official = parse_reward_info(step_info) if isinstance(step_info, dict) else None
            final_reward = float(official.reward) if official else float(_reward_val)
            final_reward_info = official
            break

        content = strip_role_prefix(str(obs))
        next_text = (
            TOOL_RESPONSE_TEMPLATE.format(content)
            if parsed.is_tool_call
            else USER_MESSAGE_TEMPLATE.format(content)
        ) + GENERATION_HEADER
        append_env_text(next_text)

    if not terminated and completion_budget <= 0:
        # Ran out of the episode token budget: treat as truncation failure.
        truncated = True

    return {
        "task_id": task_id,
        "prompt_ids": sequence_ids[: len(sequence_ids) - len(completion_ids)],
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "reward": final_reward,
        "reward_info": final_reward_info,
        "num_turns": turn_count,
        "truncated": truncated or turn_count >= max_turns,
    }


def make_tau_rollout_func(
    tokenizer: Any,
    env_factory: Any,
    num_generations: int,
    max_turns: int = 8,
    max_completion_tokens: int = 4096,
) -> Callable[[list[Any], Any], dict[str, Any]]:
    """Builds the TRL ``rollout_func``: G independent episodes per prompt/task.

    Returns the TRL-required keys plus ``env_mask`` (model vs environment token
    spans) and ``tau_rewards`` / ``tau_db`` / ``tau_comm`` forwarded to reward
    functions.
    """

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        from tau_research.tau.env_factory import build_system_prompt
        from tau_research.tau.reward import TauReward

        backend = GenerationBackend(trainer)

        prompt_ids_out: list[list[int]] = []
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_masks: list[list[int]] = []
        tau_rewards: list[float] = []
        tau_db: list[float] = []
        tau_comm: list[float] = []

        policy_cache: dict[str, str] = {}
        for row in prompts:
            task_id = row["prompt"] if isinstance(row, dict) else str(row)
            # Policy text is static per task: read it once per prompt via one
            # reset (each reset spawns an orchestrator thread + a user-sim API
            # call), then reuse for every generation episode.
            if task_id not in policy_cache:
                probe_env = env_factory.create(str(task_id))
                _obs, info = probe_env.reset()
                policy_text = str(info.get("policy") or "")
                policy_cache[str(task_id)] = (
                    build_system_prompt(policy_text)
                    if policy_text
                    else "You are a helpful customer service assistant for Retail operations."
                )
            system_prompt = policy_cache[str(task_id)]

            for _gen in range(num_generations):
                episode_env = env_factory.create(str(task_id))
                episode = run_grpo_episode(
                    task_id=str(task_id),
                    env=episode_env,
                    backend=backend,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    max_turns=max_turns,
                    max_completion_tokens=max_completion_tokens,
                )
                reward_obj = episode.get("reward_info") or TauReward(
                    reward=episode["reward"],
                    db_reward=episode["reward"],
                    communicate_reward=episode["reward"],
                    is_success=episode["reward"] >= 1.0,
                )
                prompt_ids_out.append(episode["prompt_ids"])
                completion_ids_out.append(episode["completion_ids"])
                logprobs_out.append(episode["logprobs"])
                env_masks.append(episode["env_mask"])
                tau_rewards.append(float(reward_obj.reward))
                tau_db.append(float(reward_obj.db_reward))
                tau_comm.append(float(reward_obj.communicate_reward))

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            "env_mask": env_masks,
            "tau_rewards": tau_rewards,
            "tau_db": tau_db,
            "tau_comm": tau_comm,
        }

    return rollout_func


def tau_outcome_reward(
    completions: list[Any], tau_rewards: list[float], **kwargs: Any
) -> list[float]:
    """TRL reward function reading the official outcome reward from rollouts."""
    del completions, kwargs
    return [float(r) for r in tau_rewards]
