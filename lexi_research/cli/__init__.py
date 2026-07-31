"""The `lexi` entry point.

Every stage of the pipeline is reachable as a subcommand, because the operator
rule for this repo is that no Python is written inside a notebook: an experiment
that needs a code change gets a commit, not a cell. Configuration flows from
`params.yaml`, and `--override key.path=value` changes an arm for a sweep without
editing the file.

Commands whose phase has not landed exit non-zero rather than succeeding
silently. `lexi smoke` chains stages, so a stub that returned 0 would turn the
acceptance gate into a formality.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence

from .config import Config, ConfigError, load_config

Handler = Callable[[Config, argparse.Namespace], int]


class NotYetImplemented(RuntimeError):
    """A command whose phase has not landed."""


def _common_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--params", default=None, help="path to params.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one config value; repeatable",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config and exit without running",
    )
    return parser


def _handle_data_export(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.cli import main as export_main

    argv = ["--out", args.out, "--profiles", args.profiles]
    if args.source:
        argv += ["--source", args.source]
    argv += [
        "--min-rows",
        str(args.min_rows),
        "--min-definition-chars",
        str(config.get_int("export.min_definition_chars")),
    ]
    return export_main(argv)


def _handle_train_sft(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.train.trainer import train_sft

    result = train_sft(config, train_path=args.train, output_dir=args.output)
    print(result.summary())
    return 0


def _handle_smoke(config: Config, args: argparse.Namespace) -> int:
    from .smoke import run_smoke

    return run_smoke(config, gpu=args.gpu)


def _not_yet(phase: int, what: str) -> Handler:
    def handler(config: Config, args: argparse.Namespace) -> int:
        raise NotYetImplemented(f"{what} lands in phase {phase}")

    return handler


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(prog="lexi", description="the lexi lab CLI")
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="dataset stages").add_subparsers(
        dest="command", required=True
    )
    export = data.add_parser("export", parents=[common], help="export senses from the source")
    export.add_argument("--source", default=None)
    export.add_argument("--out", required=True)
    export.add_argument("--profiles", default="lexi_research/data/profiles.json")
    export.add_argument("--min-rows", type=int, default=1)
    export.set_defaults(handler=_handle_data_export)

    train = groups.add_parser("train", help="training stages").add_subparsers(
        dest="command", required=True
    )
    sft = train.add_parser("sft", parents=[common], help="supervised fine-tuning")
    sft.add_argument("--train", required=True, help="parquet or jsonl training rows")
    sft.add_argument("--output", required=True, help="adapter output directory")
    sft.set_defaults(handler=_handle_train_sft)
    rl = train.add_parser("rl", parents=[common], help="GRPO / JEPO / NRT")
    rl.add_argument("--variant", default="grpo", choices=["grpo", "jepo", "nrt"])
    rl.set_defaults(handler=_not_yet(4, "RL training"))

    evaluate = groups.add_parser("eval", help="evaluation stages").add_subparsers(
        dest="command", required=True
    )
    evaluate.add_parser("run", parents=[common], help="metric harness").set_defaults(
        handler=_not_yet(2, "the eval harness")
    )

    bench = groups.add_parser("bench", help="inference benchmarks").add_subparsers(
        dest="command", required=True
    )
    bench.add_parser("run", parents=[common], help="latency and throughput").set_defaults(
        handler=_not_yet(5, "the bench harness")
    )

    serve = groups.add_parser("serve", help="the grading shim").add_subparsers(
        dest="command", required=True
    )
    serve.add_parser("up", parents=[common], help="run the shim").set_defaults(
        handler=_not_yet(5, "the engine-adapter serving layer")
    )

    smoke = groups.add_parser("smoke", parents=[common], help="the acceptance gate")
    smoke.add_argument("--gpu", action="store_true", help="use the real base model")
    smoke.set_defaults(handler=_handle_smoke, command="smoke")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.params, overrides=args.override)
    except ConfigError as exc:
        parser.error(str(exc))

    if args.print_config:
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True, default=str))
        return 0

    try:
        return int(args.handler(config, args))
    except NotYetImplemented as exc:
        parser.exit(2, f"lexi {args.group} {args.command}: {exc}\n")
        raise  # pragma: no cover - `parser.exit` does not return


if __name__ == "__main__":
    raise SystemExit(main())
