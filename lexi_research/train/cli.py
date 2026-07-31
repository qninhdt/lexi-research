from __future__ import annotations

import argparse

from .trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a QLoRA sentence grader")
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train(args.train, args.output, base_model=args.base_model, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
