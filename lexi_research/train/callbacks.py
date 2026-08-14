"""In-loop evaluation and resume.

Watching loss for three hours and finding at the end that format validity
collapsed is the failure this file prevents. Loss falls smoothly while a model
learns to emit prose instead of JSON; only the harness notices, so the harness
runs during training rather than after it.

The subset is fixed and small, and fixed *across arms* — an ablation whose arms
were scored on different rows is not an ablation. Small because in-loop eval that
doubles the wall-clock gets switched off, and an eval nobody runs is worth
nothing.

Resume is exercised rather than assumed: `--resume auto` picks the newest
checkpoint, and a run killed mid-sweep restarts there instead of at step zero.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: `checkpoint-1200` — the layout `transformers.Trainer` writes.
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """The newest checkpoint under `output_dir`, by step rather than by mtime.

    By step because a resumed run rewrites files: the most recently touched
    directory is not necessarily the furthest along.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates = [
        (int(match.group(1)), path)
        for path in root.iterdir()
        if path.is_dir() and (match := _CHECKPOINT.match(path.name))
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def resolve_resume(output_dir: str | Path, resume: str | None) -> str | None:
    """`auto` finds the latest checkpoint; a path is taken as given.

    `auto` on an empty directory returns None rather than raising: the first run
    of a sweep arm has nothing to resume from, and that is not an error.
    """
    if resume is None or resume == "none":
        return None
    if resume == "auto":
        found = latest_checkpoint(output_dir)
        return str(found) if found else None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    return str(path)


def build_eval_callback(
    *,
    model: Any | None = None,
    config: Any,
    run: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    band_config: Any,
    ceiling: Mapping[str, Any],
    every_steps: int,
) -> Any:
    """A `TrainerCallback` that scores a fixed subset every `every_steps`.

    Built inside a function so importing this module costs no transformers
    import; the class has to subclass one of theirs.
    """
    import transformers

    from lexi_research.eval.harness import score
    from lexi_research.eval.predict import predict_rows
    from lexi_research.tracking.panels import log_qualitative

    class InLoopEval(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.history: list[dict[str, float]] = []
            self.model = model

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step)
            if every_steps <= 0 or step == 0 or step % every_steps:
                return
            target_model = kwargs.get("model") or self.model
            if target_model is None:
                return
            self.run_once(target_model, step)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step)
            if self.history and int(self.history[-1].get("step", -1)) == step:
                return
            target_model = kwargs.get("model") or self.model
            if target_model is not None:
                self.run_once(target_model, step)

        def run_once(self, model: Any, step: int) -> dict[str, float]:
            was_training = model.training
            model.eval()
            try:
                predictions = predict_rows(
                    config,
                    rows,
                    model=model,
                    tokenizer=tokenizer,
                    band_config=band_config,
                    max_retries=0,
                )
                report = score(
                    predictions,
                    stage="sft-in-loop",
                    split="val",
                    lineage={"git": {}, "config_sha256": "in-loop", "libraries": {}},
                    ceiling=ceiling,
                    band_config=band_config,
                )
                flat = report.flat()
                run.log({f"val/{key}": value for key, value in flat.items()}, step=step)
                log_qualitative(run, predictions)
                self.history.append({"step": float(step), **flat})
                # Printed as well as logged: a Colab session whose W&B key is
                # absent still has to show whether the run is going anywhere.
                print(
                    f"in-loop eval @ {step} — "
                    f"validity: {flat.get('format.validity_rate', 0):.2f} │ "
                    f"band exact: {flat.get('meaning.exact', 0):.2f} │ "
                    f"correction F1: {flat.get('correction.span_tag_f1', 0):.2f}",
                    flush=True,
                )
                return flat
            finally:
                if was_training:
                    model.train()

    return InLoopEval()


def build_correction_eval_callback(
    *,
    model: Any | None = None,
    config: Any,
    run: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    every_steps: int,
) -> Any:
    """A `TrainerCallback` that evaluates Grammar Correction using unified correction metrics."""
    import torch
    import transformers

    from lexi_research.eval.correction import evaluate_correction_pairs
    from lexi_research.train.corrector_prompt import render_corrector_prompt

    class InLoopCorrectionEval(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.history: list[dict[str, float]] = []
            self.model = model

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step)
            if every_steps <= 0 or step == 0 or step % every_steps:
                return
            target_model = kwargs.get("model") or self.model
            if target_model is None:
                return
            self.run_once(target_model, step)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step)
            if self.history and int(self.history[-1].get("step", -1)) == step:
                return
            target_model = kwargs.get("model") or self.model
            if target_model is not None:
                self.run_once(target_model, step)

        def run_once(self, model: Any, step: int) -> dict[str, float]:
            was_training = model.training
            model.eval()
            try:
                device = next(model.parameters()).device
                subset_size = config.get_int("train.eval_subset")
                eval_subset = rows[:subset_size] if subset_size > 0 else rows[:32]
                enable_thinking = (
                    config.get_str("train.thinking") == "on"
                    if "thinking" in config.section("train")
                    else False
                )

                predictions: list[str] = []
                references: list[str] = []
                sample_logs: list[dict[str, str]] = []

                for row in eval_subset:
                    raw_input = row.get("input") or row.get("text") or ""
                    gold = (
                        row.get("output")
                        or row.get("target")
                        or row.get("correction")
                        or ""
                    ).strip()
                    messages = render_corrector_prompt(raw_input)
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    except TypeError:
                        prompt_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
                    prediction = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    predictions.append(prediction)
                    references.append(gold)
                    if len(sample_logs) < 3:
                        sample_logs.append({
                            "input": raw_input,
                            "prediction": prediction,
                            "gold": gold,
                            "exact": str(bool(prediction == gold)),
                        })

                metrics = evaluate_correction_pairs(predictions, references)
                val_metrics = {f"val/{k}": v for k, v in metrics.items()}
                run.log(val_metrics, step=step)
                self.history.append({"step": float(step), **val_metrics})

                # Log qualitative samples with inline visual diff to W&B
                from lexi_research.tracking.panels import log_correction_samples

                sample_records = [
                    {
                        "input": row.get("input") or row.get("text") or "",
                        "prediction": p,
                        "gold": g,
                        "exact": p == g,
                    }
                    for row, p, g in zip(eval_subset, predictions, references)
                ]
                log_correction_samples(run, sample_records, step=step)

                print(
                    f"\n[Correction Eval @ Step {step}] Exact Match: {metrics['correction.exact_match']:.1%} │ "
                    f"Char Sim: {metrics['correction.char_similarity']:.1%} │ "
                    f"Span F1: {metrics['correction.span_only_f1']:.1%}",
                    flush=True,
                )
                if sample_logs:
                    sample = sample_logs[0]
                    print(
                        f"  ├ Input:  {sample['input'][:70]}\n"
                        f"  ├ Pred:   {sample['prediction'][:70]}\n"
                        f"  └ Gold:   {sample['gold'][:70]} (Exact: {sample['exact']})",
                        flush=True,
                    )
                return val_metrics
            finally:
                if was_training:
                    model.train()

    return InLoopCorrectionEval()


def build_progress_callback() -> Any:

    """A beautiful, rich progress and step callback for training logs."""
    import time
    import torch
    import transformers

    try:
        from rich.console import Console
        from rich.panel import Panel

        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    class PrettyProgressCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.console = Console(force_terminal=True) if HAS_RICH else None
            self.start_time: float | None = None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.start_time = time.perf_counter()
            if HAS_RICH and self.console:
                max_steps = state.max_steps if state.max_steps > 0 else "Epoch-based"
                save_steps = getattr(args, "save_steps", "N/A")
                log_steps = getattr(args, "logging_steps", "N/A")
                self.console.print()
                self.console.print(
                    Panel(
                        f"[bold cyan]🚀 Lexi SFT Training Pipeline Started[/bold cyan]\n"
                        f"[dim]Total Steps:[/dim] [bold yellow]{max_steps}[/bold yellow] │ "
                        f"[dim]Save Interval:[/dim] [bold green]{save_steps} steps[/bold green] │ "
                        f"[dim]Logging Interval:[/dim] [bold magenta]{log_steps} step(s)[/bold magenta]",
                        title="[bold white]Training Pipeline[/bold white]",
                        border_style="cyan",
                    )
                )

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs or not HAS_RICH or not self.console:
                return

            step = int(state.global_step)
            max_steps = int(state.max_steps) if state.max_steps and state.max_steps > 0 else 1
            pct = min(100.0, (step / max_steps) * 100) if max_steps > 0 else 0.0

            loss = logs.get("loss")
            lr = logs.get("learning_rate")

            loss_str = f"[bold green]{loss:.4f}[/bold green]" if loss is not None else "[dim]N/A[/dim]"
            lr_str = f"[bold cyan]{lr:.2e}[/bold cyan]" if lr is not None else "[dim]N/A[/dim]"

            vram_str = "[dim]N/A[/dim]"
            if torch.cuda.is_available():
                alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_str = f"[bold yellow]{alloc_gb:.1f} GB[/bold yellow] / [dim]{total_gb:.1f} GB[/dim]"

            elapsed = time.perf_counter() - (self.start_time or time.perf_counter())
            sps = step / elapsed if elapsed > 0.01 else 0.0

            filled = int(pct // 5)
            progress_bar = f"[{'█' * filled}{'░' * (20 - filled)}]"

            self.console.print(
                f"[bold magenta]Step {step:5d}/{max_steps}[/bold magenta] "
                f"[cyan]{progress_bar}[/cyan] [bold white]{pct:5.1f}%[/bold white] │ "
                f"Loss: {loss_str} │ LR: {lr_str} │ VRAM: {vram_str} │ Speed: [bold green]{sps:.2f} step/s[/bold green]"
            )

    return PrettyProgressCallback()


__all__ = ["build_eval_callback", "build_progress_callback", "latest_checkpoint", "resolve_resume"]

