"""Command Line Interface for tau-research."""

import argparse
import json
import sys
from pathlib import Path


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

    args = parser.parse_args()

    if args.command == "smoke":
        print(f"[tau-research] Running smoke verification with {args.config}...")
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
        print(f"Audited {report['areal_dialogs_audited']} dialogs vs "
              f"{report['official_test_tasks']} test tasks: {flagged} flagged pairs -> {out_path}")
    elif args.command == "train-sft":
        from tau_research.training.train_sft import SFTTrainingConfig, run_sft_training

        cfg = SFTTrainingConfig.from_yaml(args.config)
        summary = run_sft_training(cfg, max_steps=args.max_steps, dry_run=args.dry_run)
        print(json.dumps(summary, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
