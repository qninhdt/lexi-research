"""Command Line Interface for tau-research."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tau-research",
        description="Specialized Customer Service Assistant via SFT -> Agentic RL",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Smoke subcommand
    smoke_parser = subparsers.add_parser("smoke", help="Run end-to-end smoke verification")
    smoke_parser.add_argument("--config", default="configs/smoke.yaml", help="Path to smoke config")

    # Convert AReaL SFT data
    convert_parser = subparsers.add_parser(
        "convert-areal", help="Convert AReaL-tau2-data SFT jsonl into training records"
    )
    convert_parser.add_argument("--input", required=True, help="Path to tau2_sft_train.jsonl")
    convert_parser.add_argument("--out-dir", default="artifacts/data")
    convert_parser.add_argument("--train-ratio", type=float, default=0.9)
    convert_parser.add_argument("--seed", type=int, default=42)

    # Decontamination audit
    audit_parser = subparsers.add_parser(
        "audit-decontamination", help="Audit AReaL vs official test split overlap"
    )
    audit_parser.add_argument("--input", required=True, help="Path to tau2_sft_train.jsonl")
    audit_parser.add_argument("--threshold", type=float, default=0.05)
    audit_parser.add_argument(
        "--output", default="artifacts/evaluation/decontamination_report.json"
    )

    # SFT training
    sft_parser = subparsers.add_parser("train-sft", help="Run LoRA reasoning SFT")
    sft_parser.add_argument("--config", default="configs/sft.yaml", help="Path to SFT config")
    sft_parser.add_argument("--max-steps", type=int, default=None, help="Override step budget")
    sft_parser.add_argument(
        "--dry-run", action="store_true", help="Load and render data only, no model load"
    )
    sft_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the newest checkpoint in output_dir if present",
    )

    # Difficulty profiling
    prof_parser = subparsers.add_parser(
        "profile-difficulty",
        help="Bucket official train tasks by empirical success of a policy",
    )
    prof_parser.add_argument("--model-path", required=True)
    prof_parser.add_argument("--config", default="configs/grpo.yaml")
    prof_parser.add_argument(
        "--output", default="artifacts/splits/rl_train_difficulty_profile.json"
    )
    prof_parser.add_argument("--trials", type=int, default=4)
    prof_parser.add_argument("--max-tasks", type=int, default=None)

    # GRPO training
    grpo_parser = subparsers.add_parser(
        "train-grpo", help="Run online multi-turn GRPO on official tau2 train tasks"
    )
    grpo_parser.add_argument("--config", default="configs/grpo.yaml")
    grpo_parser.add_argument("--max-steps", type=int, default=None)
    grpo_parser.add_argument("--dry-run", action="store_true")

    # Evaluation
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run multi-trial tau2 evaluation for a checkpoint"
    )
    eval_parser.add_argument("--config", default="configs/eval.yaml")
    eval_parser.add_argument(
        "--model-path", required=True, help="Base/SFT-merged/RL-merged model path"
    )
    eval_parser.add_argument("--tag", default="dev", help="Checkpoint tag for results dir")
    eval_parser.add_argument(
        "--policy", choices=["hf", "vllm"], default="hf", help="Policy backend"
    )
    eval_parser.add_argument("--split", default=None, help="Override split (train/test)")
    eval_parser.add_argument("--limit", type=int, default=None, help="Only first N tasks")
    eval_parser.add_argument(
        "--num-trials", type=int, default=None, help="Override trials per task"
    )

    args = parser.parse_args()

    if args.command == "smoke":
        # Real CPU gate: converter on fixture data, then render dry-run.
        from tau_research.data.load_areal_sft import convert_file
        from tau_research.training.train_sft import SFTTrainingConfig, run_sft_training

        manifest = convert_file("tests/fixtures/areal_sample.jsonl", "/tmp/tau_smoke_data")
        assert manifest["train_examples"] > 0, "converter produced no examples"
        cfg = SFTTrainingConfig.from_yaml(args.config)
        cfg.train_path = "/tmp/tau_smoke_data/areal_sft_train.json"
        cfg.val_path = "/tmp/tau_smoke_data/areal_sft_val.json"
        summary = run_sft_training(cfg, dry_run=True)
        assert summary["train_formatted"] > 0, "no training example rendered"
        assert summary["train_skipped_render"] == 0, (
            f"{summary['train_skipped_render']} examples failed chat-template rendering"
        )
        print("[tau-research] Smoke verification passed.")
        sys.exit(0)
    elif args.command == "convert-areal":
        from tau_research.data.load_areal_sft import convert_file

        manifest = convert_file(args.input, args.out_dir, args.train_ratio, args.seed)
        print(json.dumps(manifest, indent=2))
    elif args.command == "audit-decontamination":
        from tau_research.data.audit_decontamination import run_audit

        report = run_audit(args.input, threshold=args.threshold)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        flagged = len(report["flagged_pairs"])
        print(
            f"Audited {report['areal_dialogs_audited']} dialogs vs "
            f"{report['official_test_tasks']} test tasks: {flagged} flagged pairs -> {out_path}"
        )
    elif args.command == "train-sft":
        from tau_research.training.train_sft import SFTTrainingConfig, run_sft_training

        cfg = SFTTrainingConfig.from_yaml(args.config)
        summary = run_sft_training(
            cfg, max_steps=args.max_steps, dry_run=args.dry_run, resume=args.resume
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "profile-difficulty":
        from tau_research.evaluation.policies import HFChatPolicy
        from tau_research.tau.env_factory import TauEnvFactory
        from tau_research.training.difficulty import profile_task_difficulty, summarize_profile
        from tau_research.training.train_grpo import GRPOTrainingConfig

        grpo_cfg = GRPOTrainingConfig.from_yaml(args.config)
        prof_policy = HFChatPolicy(args.model_path)
        factory = TauEnvFactory(
            domain=grpo_cfg.domain,
            split=grpo_cfg.rl_split,
            user_model=grpo_cfg.user_model,
            user_temperature=grpo_cfg.user_temperature,
        )
        task_ids = factory.iter_task_ids()
        if args.max_tasks:
            task_ids = task_ids[: args.max_tasks]

        profile = profile_task_difficulty(
            prof_policy, task_ids, factory, trials_per_task=args.trials
        )
        profile.save(args.output)
        print(f"Profile saved to {args.output}: {summarize_profile(profile)}")
    elif args.command == "train-grpo":
        from tau_research.training.train_grpo import GRPOTrainingConfig, run_grpo_training

        grpo_cfg = GRPOTrainingConfig.from_yaml(args.config)
        grpo_summary = run_grpo_training(grpo_cfg, max_steps=args.max_steps, dry_run=args.dry_run)
        print(json.dumps(grpo_summary, indent=2))
    elif args.command == "evaluate":
        from tau_research.evaluation.evaluate_tau import EvalRunConfig, evaluate_from_config
        from tau_research.tau.env_factory import TauEnvFactory

        eval_cfg = EvalRunConfig.from_yaml(args.config)
        if args.tag:
            eval_cfg.checkpoint_tag = args.tag
        if args.split:
            eval_cfg.split = args.split
        if args.num_trials:
            eval_cfg.num_trials = args.num_trials

        common: dict[str, Any] = dict(
            temperature=eval_cfg.temperature,
            top_p=eval_cfg.top_p,
            top_k=eval_cfg.top_k,
            max_new_tokens=eval_cfg.max_generated_tokens_per_turn,
            enable_thinking=eval_cfg.enable_thinking,
        )
        policy: Any
        if args.policy == "vllm":
            from tau_research.evaluation.policies import VLLMChatPolicy

            policy = VLLMChatPolicy(args.model_path, **common)
        else:
            from tau_research.evaluation.policies import HFChatPolicy

            policy = HFChatPolicy(args.model_path, **common)

        factory = TauEnvFactory(
            domain=eval_cfg.domain,
            split=eval_cfg.split,
            user_model=eval_cfg.user_model,
            user_temperature=eval_cfg.user_temperature,
        )
        task_ids = factory.iter_task_ids()
        if args.limit:
            task_ids = task_ids[: args.limit]

        results = evaluate_from_config(eval_cfg, task_ids, policy, factory)
        ci_low, ci_high = results["ci_95"]
        print(
            f"[evaluate:{args.tag}] Pass^1={results['pass_rate']:.2%} "
            f"95% CI=[{ci_low:.2%}, {ci_high:.2%}] over {len(task_ids)} tasks"
        )
        print(f"Pass^k: {results['pass_k']}")
        print(f"Results: {eval_cfg.resolve_results_file()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
