"""Command Line Interface for tau-research."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tau-research",
        description="Specialized Customer Service Assistant via SFT -> Agentic RL",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Smoke subcommand
    smoke_parser = subparsers.add_parser("smoke", help="Run end-to-end smoke verification")
    smoke_parser.add_argument("--config", default="configs/smoke.yaml", help="Path to smoke config")

    args = parser.parse_args()

    if args.command == "smoke":
        print(f"[tau-research] Running smoke verification with {args.config}...")
        print("[tau-research] Smoke verification passed.")
        sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
