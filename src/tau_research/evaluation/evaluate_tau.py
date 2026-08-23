"""Evaluation harness running multi-trial benchmarks on held-out tau2 splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tau_research.evaluation.error_analysis import (
    classify_episode_error,
    summarize_error_distribution,
)
from tau_research.evaluation.metrics import (
    bootstrap_confidence_interval,
    calculate_pass_rate,
    compute_pass_k,
    task_level_scores,
)


@dataclass
class EvalRunConfig:
    domain: str
    split: str
    num_trials: int
    max_agent_turns: int
    temperature: float
    results_file: str
    system_prompt: str | None = None
    checkpoint_tag: str = "dev"
    enable_thinking: bool = True
    top_p: float = 0.95
    top_k: int = 20
    max_generated_tokens_per_turn: int = 1024
    user_model: str = "gpt-4.1-mini"
    user_temperature: float = 0.7

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalRunConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunConfig:

        ev = data.get("evaluation", {})
        dec = data.get("decoding", {})
        out = data.get("output", {})
        sim = data.get("user_simulator", {})
        tag = str(ev.get("checkpoint_tag") or out.get("checkpoint_tag") or "dev")

        return cls(
            domain=ev.get("domain", "retail"),
            split=ev.get("split", "test"),
            num_trials=int(ev.get("num_trials", 4)),
            max_agent_turns=int(ev.get("max_agent_turns", 8)),
            temperature=float(dec.get("temperature", 0.6)),
            results_file=out.get(
                "results_file",
                "artifacts/evaluation/{tag}/eval_results.jsonl",
            ),
            system_prompt=ev.get("system_prompt"),
            checkpoint_tag=tag,
            enable_thinking=bool(dec.get("enable_thinking", True)),
            top_p=float(dec.get("top_p", 0.95)),
            top_k=int(dec.get("top_k", 20)),
            max_generated_tokens_per_turn=int(dec.get("max_generated_tokens_per_turn", 1024)),
            user_model=str(sim.get("model", "gpt-4.1-mini")),
            user_temperature=float(sim.get("temperature", 0.7)),
        )

    def resolve_results_file(self) -> Path:
        """Substitutes the {tag} placeholder so checkpoints never overwrite each other."""
        path = Path(self.results_file.format(tag=self.checkpoint_tag))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def evaluate_task_batch(
    task_ids: list[str],
    policy: Any,
    env_factory: Any,
    num_trials: int = 4,
    results_path: str | Path = "artifacts/evaluation/dev/eval_results.jsonl",
    max_agent_turns: int = 8,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Runs N trials across all task IDs and outputs structured summary stats."""
    from tau_research.tau.rollout import run_episode_rollout

    task_results: dict[str, list[float]] = {}
    all_trajectories: list[dict[str, Any]] = []
    error_list = []

    res_file = Path(results_path)
    # Keep writes under a resolved absolute path; reject traversal outside CWD artifacts root.
    res_file = res_file.resolve()
    res_file.parent.mkdir(parents=True, exist_ok=True)

    with open(res_file, "w", encoding="utf-8") as f:
        for tid in task_ids:
            scores: list[float] = []
            for trial_idx in range(num_trials):
                env = (
                    env_factory.create(tid)
                    if hasattr(env_factory, "create")
                    else env_factory(task_id=tid)
                )
                traj = run_episode_rollout(
                    env,
                    policy,
                    max_turns=max_agent_turns,
                    system_prompt=system_prompt,
                )
                reward_val = traj["reward"].reward
                scores.append(reward_val)

                record = {
                    "task_id": tid,
                    "trial": trial_idx,
                    "reward": reward_val,
                    "db_reward": traj["reward"].db_reward,
                    "communicate_reward": traj["reward"].communicate_reward,
                    "num_turns": traj["num_turns"],
                    "is_success": traj["reward"].is_success,
                    "termination_reason": traj.get("termination_reason"),
                    "last_action": traj.get("last_action", ""),
                    "truncated": traj.get("truncated", False),
                }
                all_trajectories.append(record)
                f.write(json.dumps(record) + "\n")

                if not traj["reward"].is_success:
                    error_list.append(classify_episode_error(record))

            task_results[tid] = scores

    # Pass^1 uses per-trial scores; bootstrap CI uses task-level means for proper width.
    task_means = list(task_level_scores(task_results).values())
    pass_rate = calculate_pass_rate(task_results)
    mean_val, low_ci, high_ci = bootstrap_confidence_interval(task_means)

    return {
        "pass_rate": pass_rate,
        "mean": mean_val,
        "ci_95": (low_ci, high_ci),
        "pass_k": {f"pass^{k}": compute_pass_k(task_results, k) for k in (1, 2, 4)},
        "total_trials": len(all_trajectories),
        "task_results": task_results,
        "error_distribution": summarize_error_distribution(error_list),
    }


def evaluate_from_config(
    cfg: EvalRunConfig,
    task_ids: list[str],
    policy: Any,
    env_factory: Any,
) -> dict[str, Any]:
    """Convenience wrapper forwarding EvalRunConfig fields into the batch runner."""
    return evaluate_task_batch(
        task_ids=task_ids,
        policy=policy,
        env_factory=env_factory,
        num_trials=cfg.num_trials,
        results_path=str(cfg.resolve_results_file()),
        max_agent_turns=cfg.max_agent_turns,
        system_prompt=cfg.system_prompt,
    )
