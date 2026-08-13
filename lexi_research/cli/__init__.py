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
from pathlib import Path
from typing import Any

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
        "sample",
        config,
        run_sample(
            config, pool=args.pool, out=args.out, full=args.full, senses=args.senses
        ),
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


def _handle_data_pilot_gate(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_pilot_gate

    return _report(
        "pilot_gate",
        config,
        run_pilot_gate(
            config,
            texts=args.texts,
            labels=args.labels,
            generate_report=args.generate_report,
            out=args.out,
            cache=args.cache,
        ),
    )


def _handle_data_publish(config: Config, args: argparse.Namespace) -> int:
    """Publish the teacher-generated dataset. Stage A is excluded by licence."""
    del config
    from lexi_research.data.publish_hf import main as publish_main

    argv = ["--repo-id", args.repo_id]
    if args.private:
        argv.append("--private")
    if args.dry_run:
        argv.append("--dry-run")
    if args.card_out:
        argv += ["--card-out", args.card_out]
    return publish_main(argv)


def _handle_data_gec_import(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.data.stages import run_gec_import

    return _report(
        "gec_import", config, run_gec_import(config, corpus=args.corpus, out=args.out)
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


def _handle_eval_predict(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.eval.harness import iter_rows, write_predictions
    from lexi_research.eval.predict import load_for_inference, predict_rows
    from lexi_research.format import BandConfig

    model, tokenizer = load_for_inference(config, args.adapter)
    rows = list(iter_rows(args.rows))
    predictions = predict_rows(
        config,
        rows,
        model=model,
        tokenizer=tokenizer,
        band_config=BandConfig.from_json(args.band_config),
        max_retries=config.get_int("eval.max_retries"),
    )
    path = write_predictions(predictions, args.out)
    print(f"wrote {len(predictions)} predictions to {path}")
    return 0


def _handle_eval_score(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.eval.harness import load_ceiling, read_predictions, score
    from lexi_research.format import BandConfig
    from lexi_research.tracking import collect, start
    from lexi_research.tracking.panels import log_confusion, log_qualitative

    lineage = collect(config.as_dict(), stage="eval")
    rows = read_predictions(args.predictions)
    report = score(
        rows,
        stage="eval",
        split=args.split,
        lineage=lineage,
        ceiling=load_ceiling(args.ceiling),
        band_config=BandConfig.from_json(args.band_config),
        calibration_bins=config.get_int("eval.calibration_bins"),
    )
    written = report.write(args.out)

    run = start(config, stage="eval", lineage=lineage)
    try:
        run.summary(report.flat())
        log_qualitative(run, rows)
        confusion = report.groups["correction"].get("confusion")
        if isinstance(confusion, dict):
            log_confusion(run, confusion)
    finally:
        run.finish()

    print(report.markdown())
    print(f"wrote {written}")
    return 0


def _train_once(
    config: Config,
    args: argparse.Namespace,
    *,
    stage: str,
    output_dir: str,
    group: str | None = None,
) -> Any:
    """One SFT run, with lineage, in-loop eval, and the artifact it publishes."""
    from lexi_research.eval.harness import iter_rows, load_ceiling
    from lexi_research.format import BandConfig
    from lexi_research.tracking import collect, start
    from lexi_research.train.trainer import train_sft

    band_config = BandConfig.from_json(args.band_config)
    val_rows = None
    ceiling = None
    if args.val:
        subset = config.get_int("train.eval_subset")
        val_rows = list(iter_rows(args.val))[:subset]
        ceiling = load_ceiling(args.ceiling) if args.ceiling else {}

    run_config = config.with_overrides([f"tracking.group={group}"]) if group else config
    lineage = collect(run_config.as_dict(), stage=stage)
    run = start(run_config, stage=stage, lineage=lineage, output_dir=output_dir)

    try:
        result = train_sft(
            run_config,
            train_path=args.train,
            output_dir=output_dir,
            run=run,
            resume=args.resume,
            val_rows=val_rows,
            band_config=band_config,
            ceiling=ceiling,
        )
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
    return result


def _handle_train_sft(config: Config, args: argparse.Namespace) -> int:
    result = _train_once(config, args, stage="sft", output_dir=args.output)
    print(result.summary())
    return 0


def _handle_train_sweep(config: Config, args: argparse.Namespace) -> int:
    """Run every arm of an ablation, resuming from wherever the last session died."""
    from lexi_research.train.sweep import (
        SweepState,
        default_ablation_path,
        iter_arms,
        load_ablation,
        summarise,
    )

    ablation = load_ablation(args.definition or default_ablation_path(args.ablation))
    state = SweepState.load(Path(args.output) / f"{ablation.key}-state.json")
    print(summarise(ablation, state), flush=True)

    for arm in iter_arms(ablation, state, resume=not args.restart):
        print(f"\n=== {arm.name}: {' '.join(arm.overrides)}", flush=True)
        arm_config = config.with_overrides(arm.overrides)
        result = _train_once(
            arm_config,
            args,
            stage=ablation.key,
            output_dir=str(Path(args.output) / arm.name),
            group=ablation.key,
        )
        state.record(
            arm,
            {
                "steps": result.steps,
                "examples": result.examples,
                "dropped": result.dropped,
                "targets": result.targets.summary(),
                "output_dir": str(result.output_dir),
            },
        )
        print(result.summary(), flush=True)

    print(f"\n{summarise(ablation, state)}")
    return 0


def _handle_train_rl(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.format import BandConfig
    from lexi_research.rl.trainer import train_rl
    from lexi_research.tracking import collect, start

    run_config = config.with_overrides([f"rl.algo={args.algo}"]) if args.algo else config
    stage = f"rl-{run_config.get_str('rl.algo')}"
    lineage = collect(run_config.as_dict(), stage=stage)
    run = start(run_config, stage=stage, lineage=lineage, output_dir=args.output)

    try:
        result = train_rl(
            run_config,
            train_path=args.train,
            output_dir=args.output,
            band_config=BandConfig.from_json(args.band_config),
            run=run,
        )
        run.summary(result.last.as_dict())
    finally:
        run.finish()
    print(result.summary())
    return 0


def _handle_report_model_card(config: Config, args: argparse.Namespace) -> int:
    from lexi_research.report.model_card import generate

    written = generate(
        args.report,
        args.out,
        base_model=config.get_str("train.base_model"),
        rl_verdict=args.rl_verdict,
        comparison_path=args.comparison,
    )
    print(f"wrote {written}")
    return 0


def _handle_bench_run(config: Config, args: argparse.Namespace) -> int:
    """Benchmark one engine across concurrency levels, and skip loudly."""
    import json as _json

    from bench.engines import build as build_engine
    from bench.engines import skip_reason
    from bench.runner import BenchResult
    from lexi_research.tracking import collect, start

    name = args.engine or config.get_str("bench.engine")
    engine = build_engine(name, config.get_str("train.base_model"), args.adapter)
    reason = skip_reason(engine.capabilities(), quantisation=args.quantisation, features=())

    levels = (
        [int(value) for value in args.concurrency.split(",")]
        if args.concurrency
        else [int(value) for value in config.get("bench.concurrency")]
    )
    lineage = collect(config.as_dict(), stage=f"bench-{name}")
    run = start(config, stage=f"bench-{name}", lineage=lineage)
    results = []
    try:
        for level in levels:
            result = BenchResult(
                engine=name,
                quantisation=args.quantisation,
                concurrency=level,
                skipped=reason,
                lineage=lineage,
                cost_per_hour=config.get_float("bench.cost_per_hour"),
            )
            if reason is None:
                result.samples = _bench_arm(config, engine, args, level)
            payload = result.statistics(slo_s=config.get_float("bench.slo_s"))
            results.append({"concurrency": level, **payload})
            run.log(
                {
                    f"bench/{key}": value
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }
            )
    finally:
        run.finish()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _json.dumps(
            {
                "engine": name,
                "quantisation": args.quantisation,
                "lineage": lineage,
                "arms": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(_json.dumps(results, indent=2, default=str))
    if reason:
        print(f"skipped: {reason}")
    print(f"wrote {out}")
    return 0


def _bench_arm(config: Config, engine: Any, args: argparse.Namespace, level: int) -> list[Any]:
    """Issue the schedule against a launched engine and time every response."""
    import time

    from bench.runner import Sample, arrival_schedule
    from lexi_research.eval.harness import iter_rows

    launched = engine.launch(quantisation=args.quantisation)
    rows = list(iter_rows(args.rows))
    if not rows:
        raise RuntimeError(f"{args.rows} holds no rows to benchmark against")
    duration = args.duration if args.duration is not None else config.get_float("bench.duration_s")
    schedule = arrival_schedule(
        rate_per_s=level,
        duration_s=duration,
        warmup=config.get_int("bench.warmup_requests"),
    )
    warmups = config.get_int("bench.warmup_requests")

    samples: list[Sample] = []
    origin = time.monotonic()
    try:
        for index, offset in enumerate(schedule):
            # Open loop: sleep until this request's scheduled instant rather than
            # until the previous response arrived.
            delay = origin + offset - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            started = time.monotonic()
            tokens, ttft = _one_request(engine, launched, rows[index % len(rows)], config)
            samples.append(
                Sample(
                    started_s=started - origin,
                    ttft_s=ttft,
                    latency_s=time.monotonic() - started,
                    output_tokens=tokens,
                    warmup=index < warmups,
                )
            )
    finally:
        engine.shutdown()
    return samples


def _one_request(engine: Any, launched: Any, row: Any, config: Config) -> tuple[int, float]:
    """One graded request. In-process for the baseline, HTTP for a served engine."""
    import time

    from lexi_research.train.collate import training_messages

    started = time.monotonic()
    if launched.base_url.startswith("inprocess://"):
        import torch

        prompt = engine.tokenizer.apply_chat_template(
            training_messages(row), tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(engine.model.device)
        with torch.no_grad():
            generated = engine.model.generate(
                prompt,
                max_new_tokens=config.get_int("eval.max_new_tokens"),
                do_sample=False,
                pad_token_id=engine.tokenizer.pad_token_id,
            )
        return int(generated.shape[-1] - prompt.shape[-1]), time.monotonic() - started

    import httpx

    response = httpx.post(
        f"{launched.base_url}/chat/completions",
        json={
            "model": "lexi",
            "messages": training_messages(row),
            "temperature": 0,
            "max_tokens": config.get_int("eval.max_new_tokens"),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
    return tokens, time.monotonic() - started


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
    sample.add_argument(
        "--senses",
        type=int,
        default=None,
        help=(
            "draw exactly this many senses, overriding the configured counts. "
            "The draw is nested, so raising it re-draws the same senses first and "
            "only the new ones cost teacher calls"
        ),
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

    pilot = data.add_parser(
        "pilot-gate",
        parents=[common],
        help="re-grade a sample and evaluate the automatic pilot gates",
    )
    pilot.add_argument("--texts", default="data/raw/raw_texts.parquet")
    pilot.add_argument("--labels", default="data/raw/raw_labels.parquet")
    pilot.add_argument("--generate-report", default="data/raw/generate-report.json")
    pilot.add_argument("--out", default="reports")
    pilot.add_argument("--cache", default=".cache/teacher")
    pilot.set_defaults(handler=_handle_data_pilot_gate)

    gec = data.add_parser(
        "gec-import",
        parents=[common],
        help="stage A: convert a learner corpus into the edit format (no teacher)",
    )
    gec.add_argument("--corpus", default="data/corpora/wi_locness")
    gec.add_argument("--out", default="data/gec")
    gec.set_defaults(handler=_handle_data_gec_import)

    publish = data.add_parser(
        "publish",
        parents=[common],
        help="publish the teacher-generated dataset to the Hugging Face Hub",
    )
    publish.add_argument("--repo-id", required=True, help="e.g. your-name/lexi-grader-sft")
    publish.add_argument("--private", action="store_true")
    publish.add_argument(
        "--dry-run", action="store_true", help="print the card and the upload list only"
    )
    publish.add_argument("--card-out", default=None)
    publish.set_defaults(handler=_handle_data_publish)

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
    sft.add_argument("--val", default=None, help="rows for in-loop evaluation")
    sft.add_argument("--ceiling", default=None, help="teacher self-consistency artifact")
    sft.add_argument("--resume", default="auto", help="`auto`, `none`, or a checkpoint directory")
    sft.set_defaults(handler=_handle_train_sft)

    sweep = train.add_parser("sweep", parents=[common], help="run every arm of an ablation")
    sweep.add_argument("--ablation", default="a7", help="ablation key, e.g. a2, a6, a7")
    sweep.add_argument("--definition", default=None, help="explicit path to an arm YAML")
    sweep.add_argument("--train", required=True)
    sweep.add_argument("--output", default="runs/sweeps")
    sweep.add_argument("--band-config", default="band_config.json")
    sweep.add_argument("--val", default=None)
    sweep.add_argument("--ceiling", default=None)
    sweep.add_argument("--resume", default="auto")
    sweep.add_argument(
        "--restart", action="store_true", help="re-run arms already recorded as complete"
    )
    sweep.set_defaults(handler=_handle_train_sweep)
    rl = train.add_parser("rl", parents=[common], help="GRPO / JEPO / NRT")
    rl.add_argument("--algo", default=None, choices=["grpo", "jepo", "nrt"])
    rl.add_argument("--train", required=True)
    rl.add_argument("--output", default="runs/rl/adapter")
    rl.add_argument("--band-config", default="band_config.json")
    rl.set_defaults(handler=_handle_train_rl)

    evaluate = groups.add_parser("eval", help="evaluation stages").add_subparsers(
        dest="command", required=True
    )
    predict = evaluate.add_parser(
        "predict", parents=[common], help="generate predictions (the only GPU step)"
    )
    predict.add_argument("--rows", default="data/clean/test.parquet")
    predict.add_argument("--adapter", default=None)
    predict.add_argument("--out", default="reports/predictions.jsonl")
    predict.add_argument("--band-config", default="data/clean/band_config.json")
    predict.set_defaults(handler=_handle_eval_predict)

    score = evaluate.add_parser(
        "score", parents=[common], help="score a predictions file (CPU only)"
    )
    score.add_argument("--predictions", default="reports/predictions.jsonl")
    score.add_argument("--split", default="test")
    score.add_argument("--out", default="reports/eval.json")
    score.add_argument("--band-config", default="data/clean/band_config.json")
    score.add_argument(
        "--ceiling",
        default=None,
        help="teacher self-consistency artifact; every metric is normalised by it",
    )
    score.set_defaults(handler=_handle_eval_score)

    bench = groups.add_parser("bench", help="inference benchmarks").add_subparsers(
        dest="command", required=True
    )
    bench_run = bench.add_parser("run", parents=[common], help="latency and throughput")
    bench_run.add_argument("--engine", default=None, choices=["hf", "vllm", "sglang"])
    bench_run.add_argument("--adapter", default=None)
    bench_run.add_argument("--rows", default="data/clean/test.parquet")
    bench_run.add_argument("--quantisation", default="bf16")
    bench_run.add_argument("--concurrency", default=None, help="comma-separated arrival rates")
    bench_run.add_argument("--duration", type=float, default=None)
    bench_run.add_argument("--out", default="reports/bench.json")
    bench_run.set_defaults(handler=_handle_bench_run)

    serve = groups.add_parser("serve", help="the grading shim").add_subparsers(
        dest="command", required=True
    )
    serve.add_parser("up", parents=[common], help="run the shim").set_defaults(
        handler=_not_yet(5, "the engine-adapter serving layer")
    )

    report = groups.add_parser("report", help="generated artifacts").add_subparsers(
        dest="command", required=True
    )
    card = report.add_parser(
        "model-card", parents=[common], help="generate MODEL_CARD.md from the eval report"
    )
    card.add_argument("--report", default="reports/eval-sft.json")
    card.add_argument("--comparison", default="reports/compare.json")
    card.add_argument("--out", default="MODEL_CARD.md")
    card.add_argument(
        "--rl-verdict",
        default="Not yet measured: the RL arms need a GPU and a trained SFT baseline.",
        help="stated plainly either way, per the design",
    )
    card.set_defaults(handler=_handle_report_model_card)

    smoke = groups.add_parser("smoke", parents=[common], help="the acceptance gate")
    smoke.add_argument("--gpu", action="store_true", help="use the real base model")
    smoke.set_defaults(handler=_handle_smoke, command="smoke")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
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
