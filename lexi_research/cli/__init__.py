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


def _report(stage: str, config: Config, payload: dict[str, object]) -> int:
    """Print a stage report and record it, with the lineage that dates it."""
    from lexi_research.tracking import collect, start

    run = start(config, stage=stage, lineage=collect(config.as_dict(), stage=stage))
    run.summary(payload)
    run.finish()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


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


def _handle_data_sample(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_sample

    return _report(
        "sample", config, run_sample(config, pool=args.pool, out=args.out, full=args.full)
    )


def _handle_data_generate(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_generate

    return _report(
        "generate",
        config,
        run_generate(config, specs=args.specs, out=args.out, cache=args.cache),
    )


def _handle_data_label(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_label

    return _report(
        "label", config, run_label(config, texts=args.texts, out=args.out, cache=args.cache)
    )


def _handle_data_process(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_process

    return _report(
        "process",
        config,
        run_process(
            config,
            texts=args.texts,
            labels=args.labels,
            out=args.out,
            band_config=args.band_config,
        ),
    )


def _handle_data_calibrate(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_calibrate

    return _report(
        "calibrate",
        config,
        run_calibrate(config, rows_path=args.rows, out=args.out, band_config=args.band_config),
    )


def _handle_train_sft(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.tracking import collect, start
    from lexi_research.train.trainer import train_sft

    lineage = collect(config.as_dict(), stage="sft")
    run = start(config, stage="sft", lineage=lineage)
    try:
        result = train_sft(config, train_path=args.train, output_dir=args.output, run=run)
        # The adapter and the band config version together: a checkpoint without
        # the config that derives its bands produces meaningless bands.
        run.log_artifact(
            config.get_str("tracking.adapter_artifact"),
            [result.output_dir, args.band_config],
            metadata={"lineage": lineage, "targets": result.targets.summary()},
        )
        run.summary({"examples": result.examples, "dropped": result.dropped})
    finally:
        run.finish()
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

    sample = data.add_parser("sample", parents=[common], help="draw senses, build call-1 batches")
    sample.add_argument("--pool", default="data/pool/senses_pool.parquet")
    sample.add_argument("--out", default="data/batches")
    sample.add_argument(
        "--full", action="store_true", help="sample.full_senses instead of sample.pilot_senses"
    )
    sample.set_defaults(handler=_handle_data_sample)

    generate = data.add_parser("generate", parents=[common], help="call 1: write sentences")
    generate.add_argument("--specs", default="data/batches/batch_specs.parquet")
    generate.add_argument("--out", default="data/raw")
    generate.add_argument("--cache", default=".cache/teacher")
    generate.set_defaults(handler=_handle_data_generate)

    label = data.add_parser("label", parents=[common], help="call 2: grade sentences")
    label.add_argument("--texts", default="data/raw/raw_texts.parquet")
    label.add_argument("--out", default="data/raw")
    label.add_argument("--cache", default=".cache/teacher")
    label.set_defaults(handler=_handle_data_label)

    process = data.add_parser(
        "process", parents=[common], help="validate, balance and split in one pass"
    )
    process.add_argument("--texts", default="data/raw/raw_texts.parquet")
    process.add_argument("--labels", default="data/raw/raw_labels.parquet")
    process.add_argument("--out", default="data/clean")
    process.add_argument("--band-config", default=None)
    process.set_defaults(handler=_handle_data_process)

    calibrate = data.add_parser(
        "calibrate", parents=[common], help="place band cut points on the real distribution"
    )
    calibrate.add_argument("--rows", default="data/clean/train.parquet")
    calibrate.add_argument("--out", default="band_config.json")
    calibrate.add_argument("--band-config", default=None)
    calibrate.set_defaults(handler=_handle_data_calibrate)

    train = groups.add_parser("train", help="training stages").add_subparsers(
        dest="command", required=True
    )
    sft = train.add_parser("sft", parents=[common], help="supervised fine-tuning")
    sft.add_argument("--train", required=True, help="parquet or jsonl training rows")
    sft.add_argument("--output", required=True, help="adapter output directory")
    sft.add_argument("--band-config", default="band_config.json")
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
