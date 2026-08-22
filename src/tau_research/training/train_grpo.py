"""Agentic Reinforcement Learning Pipeline with TRL GRPOTrainer and custom rollout_func."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tau_research.training.difficulty import (
    DifficultyProfile,
    sample_tasks_by_difficulty,
)


@dataclass
class GRPOTrainingConfig:
    model_name: str
    output_dir: str
    learning_rate: float
    num_generations: int
    max_completion_length: int
    vllm_gpu_memory_utilization: float
    vllm_enable_sleep_mode: bool
    beta: float
    loss_type: str = "dapo"
    max_turns: int = 8
    seed: int = 42
    resume_from_checkpoint: str | None = None
    save_steps: int = 50
    difficulty_profile_path: str | None = None
    learnable_weight: float = 0.70
    easy_weight: float = 0.15
    hard_weight: float = 0.15
    resample_on_zero_variance: bool = True
    report_to: str = "none"
    max_consecutive_zero_variance_batches: int = 3
    domain: str = "retail"
    rl_split: str = "train"
    user_model: str = "gpt-4.1-mini"
    user_temperature: float = 0.7
    temperature: float = 1.0
    top_p: float = 0.95
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    merged_dir: str = "artifacts/models/qwen3.5-2b-tau-retail-grpo-merged"

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOTrainingConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        m = data.get("model", {})
        t = data.get("training", {})
        env = data.get("environment", {})
        diff = data.get("difficulty_sampling", {})
        weights = diff.get("weights", {}) if isinstance(diff, dict) else {}

        return cls(
            model_name=m.get(
                "name_or_path",
                "artifacts/models/qwen3.5-2b-tau-retail-sft-merged",
            ),
            output_dir=t.get("output_dir", "artifacts/models/qwen3.5-2b-tau-retail-grpo"),
            learning_rate=float(t.get("learning_rate", 1e-5)),
            num_generations=int(t.get("num_generations", 4)),
            max_completion_length=int(t.get("max_completion_length", 1536)),
            vllm_gpu_memory_utilization=float(t.get("vllm_gpu_memory_utilization", 0.20)),
            vllm_enable_sleep_mode=bool(t.get("vllm_enable_sleep_mode", True)),
            beta=float(t.get("beta", 0.0)),
            loss_type=str(t.get("loss_type", "dapo")),
            max_turns=int(env.get("max_turns", 8)),
            seed=int(t.get("seed", 42)),
            resume_from_checkpoint=t.get("resume_from_checkpoint"),
            save_steps=int(t.get("save_steps", 50)),
            difficulty_profile_path=(diff.get("profile_path") if isinstance(diff, dict) else None),
            learnable_weight=float(weights.get("learnable", 0.70)),
            easy_weight=float(weights.get("easy", 0.15)),
            hard_weight=float(weights.get("hard", 0.15)),
            resample_on_zero_variance=bool(diff.get("resample_on_zero_variance", True)),
            report_to=str(t.get("report_to", "none")),
            max_consecutive_zero_variance_batches=int(
                diff.get("max_consecutive_zero_variance_batches", 3)
            ),
            domain=str(env.get("domain", "retail")),
            rl_split=str(env.get("rl_split", "train")),
            user_model=str(env.get("user_simulator", {}).get("model", "gpt-4.1-mini")),
            user_temperature=float(env.get("user_simulator", {}).get("temperature", 0.7)),
            temperature=float(t.get("temperature", 1.0)),
            top_p=float(t.get("top_p", 0.95)),
            use_vllm=bool(t.get("use_vllm", True)),
            vllm_mode=str(t.get("vllm_mode", "colocate")),
        )


def check_zero_variance_reward_batch(rewards: list[float]) -> bool:
    """Returns True if reward standard deviation is 0.0 across generation group."""
    if not rewards or len(rewards) <= 1:
        return True
    return float(np.std(rewards)) == 0.0


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages: (r - mean) / (std + eps), matching GRPO."""
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    denom = std if std > 1e-8 else 1.0
    return [float((r - mean) / denom) for r in rewards]


def format_rollout_batch_for_grpo(
    prompt_ids: list[list[int]],
    completion_ids: list[list[int]],
    rewards: list[float],
    logprobs: list[list[float]] | None = None,
    advantages: list[float] | None = None,
    returns: list[float] | None = None,
) -> dict[str, Any]:
    """Formats rollout episode tensors into the contract expected by GRPOTrainer.

    Always includes TRL-required keys: prompt_ids, completion_ids, logprobs,
    advantages, returns, and rewards.
    """
    n = len(rewards)
    if n != len(prompt_ids) or n != len(completion_ids):
        raise ValueError(
            f"Batch length mismatch: prompts={len(prompt_ids)} "
            f"completions={len(completion_ids)} rewards={n}"
        )

    if logprobs is None:
        logprobs = [[0.0] * len(c) for c in completion_ids]
    if advantages is None:
        advantages = compute_group_advantages(rewards)
    if returns is None:
        returns = list(rewards)

    if len(logprobs) != n or len(advantages) != n or len(returns) != n:
        raise ValueError("logprobs/advantages/returns must match batch size")

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "advantages": advantages,
        "returns": returns,
        "rewards": rewards,
    }


def load_difficulty_profile(path: str | Path | None) -> DifficultyProfile | None:
    if path is None:
        return None
    profile_path = Path(path)
    if not profile_path.exists():
        return None
    import json

    with open(profile_path, encoding="utf-8") as f:
        data = json.load(f)
    return DifficultyProfile.from_dict(data)


def select_training_tasks(
    config: GRPOTrainingConfig,
    batch_size: int,
    fallback_task_ids: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Sample a GRPO batch with difficulty-weighted priority when a profile exists."""
    profile = load_difficulty_profile(config.difficulty_profile_path)
    if profile is None:
        if not fallback_task_ids:
            return []
        rng = np.random.default_rng(seed if seed is not None else config.seed)
        idx = rng.choice(len(fallback_task_ids), size=min(batch_size, len(fallback_task_ids)))
        return [fallback_task_ids[int(i)] for i in idx]

    return sample_tasks_by_difficulty(
        profile,
        batch_size=batch_size,
        learnable_weight=config.learnable_weight,
        easy_weight=config.easy_weight,
        hard_weight=config.hard_weight,
        seed=seed if seed is not None else config.seed,
    )


def build_grpo_trainer_kwargs(config: GRPOTrainingConfig) -> dict[str, Any]:
    """Map config into TRL GRPOTrainer / TrainingArguments-compatible kwargs."""
    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "num_generations": config.num_generations,
        "max_completion_length": config.max_completion_length,
        "beta": config.beta,
        "loss_type": config.loss_type,
        "seed": config.seed,
        "save_steps": config.save_steps,
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "vllm_enable_sleep_mode": config.vllm_enable_sleep_mode,
    }
    if config.resume_from_checkpoint:
        kwargs["resume_from_checkpoint"] = config.resume_from_checkpoint
    return kwargs


def resolve_resume_checkpoint(config: GRPOTrainingConfig) -> str | None:
    """Return an existing checkpoint path for resume, or None."""
    if config.resume_from_checkpoint:
        path = Path(config.resume_from_checkpoint)
        return str(path) if path.exists() else None
    output = Path(config.output_dir)
    if not output.exists():
        return None
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def weight_tasks_by_difficulty(
    task_ids: list[str],
    profile_path: str | None,
    learnable_weight: float = 0.70,
) -> list[str]:
    """Repeats task IDs so learnable tasks occupy ~their weight of the pool."""
    profile = load_difficulty_profile(profile_path)
    if profile is None:
        return list(task_ids)

    buckets = {
        "learnable": set(profile.learnable_tasks),
        "easy": set(profile.easy_tasks),
        "hard": set(profile.hard_tasks),
    }
    # A stale or foreign profile (e.g. mock-named tasks) must never silently
    # distort sampling weights: require real overlap with the actual pool.
    known = buckets["learnable"] | buckets["easy"] | buckets["hard"]
    overlap = len(known & set(task_ids))
    if overlap == 0:
        print(
            f"[train-grpo] warning: difficulty profile has 0/{len(known)} IDs matching "
            f"the {len(task_ids)} official train tasks; ignoring it. Regenerate via "
            "'tau-research profile-difficulty'."
        )
        return list(task_ids)
    if overlap < len(set(task_ids)) // 2:
        print(
            f"[train-grpo] warning: difficulty profile covers only {overlap}/"
            f"{len(set(task_ids))} train tasks; uncovered tasks fall into the easy bucket."
        )
    other_weight = (1.0 - learnable_weight) / 2.0
    repeats = {
        "learnable": max(1, round(learnable_weight * 10)),
        "easy": max(1, round(other_weight * 10)),
        "hard": max(1, round(other_weight * 10)),
    }
    weighted: list[str] = []
    for tid in task_ids:
        bucket = next((name for name, ids in buckets.items() if tid in ids), "easy")
        weighted.extend([tid] * repeats.get(bucket, 1))
    return weighted


def run_grpo_training(
    config: GRPOTrainingConfig,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Runs online multi-turn GRPO against official tau2 train tasks."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from tau_research.tau.env_factory import TauEnvFactory
    from tau_research.tau.grpo_rollout import make_tau_rollout_func, tau_outcome_reward

    factory = TauEnvFactory(
        domain=config.domain,
        split=config.rl_split,
        user_model=config.user_model,
        user_temperature=config.user_temperature,
    )

    task_ids = factory.iter_task_ids()
    weighted_ids = weight_tasks_by_difficulty(
        task_ids, config.difficulty_profile_path, config.learnable_weight
    )
    summary: dict[str, Any] = {"train_tasks": len(task_ids), "weighted_rows": len(weighted_ids)}
    if dry_run:
        summary["dry_run"] = True
        return summary

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_rows = [{"prompt": tid} for tid in weighted_ids]

    rollout_func = make_tau_rollout_func(
        tokenizer,
        factory,
        num_generations=config.num_generations,
        max_turns=config.max_turns,
        max_completion_tokens=config.max_completion_length,
    )

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        loss_type=config.loss_type,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        save_steps=config.save_steps,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        report_to=[config.report_to] if config.report_to != "none" else [],
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        logging_steps=5,
    )
    if max_steps:
        grpo_kwargs["max_steps"] = max_steps
    # NOTE: TrainingArguments.resume_from_checkpoint is NOT consumed by
    # Trainer.train() automatically; it must be passed explicitly below.
    resume_path = resolve_resume_checkpoint(config)

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[tau_outcome_reward],
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=Dataset.from_list(train_rows),
        rollout_func=rollout_func,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(config.merged_dir)
    summary["merged_dir"] = config.merged_dir
    return summary
