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
    if resume is None:
        return None
    r_str = str(resume).strip().lower()
    if r_str in ("none", "false", "off", "no", "0", "null"):
        return None
    if r_str in ("auto", "true", "yes", "on", "1", "latest", "resume"):
        found = latest_checkpoint(output_dir)
        return str(found) if found else None

    path = Path(resume)
    if not path.exists():
        rel_candidate = Path(output_dir) / resume
        if rel_candidate.exists():
            return str(rel_candidate)
        raise FileNotFoundError(
            f"Checkpoint {path} does not exist (also checked in {output_dir})"
        )
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
    val_examples: Sequence[Any] | None = None,
    every_steps: int,
) -> Any:
    """A `TrainerCallback` that evaluates Grammar Correction using unified correction metrics."""
    import torch
    import transformers

    from lexi_research.train.corrector_prompt import render_corrector_prompt

    def _select_hardest_indices(source_rows: Sequence[Mapping[str, Any]], k: int = 16) -> list[int]:
        import re

        tag_regex = re.compile(
            r"\[(.*?):(sp|agr|tense|form|art|prep|part|num|poss|pron|order|punc|coll|word|unnat|other)\]"
        )
        scored: list[tuple[float, int]] = []
        for i, r in enumerate(source_rows):
            inp = str(r.get("input") or r.get("text") or "").strip()
            gt = str(r.get("output") or r.get("target") or r.get("correction") or "").strip()
            if inp == gt:
                continue
            tags = tag_regex.findall(gt)
            num_edits = len(tags)
            unique_tags = len(set(t[1] for t in tags))
            length = len(inp.split())
            score_val = num_edits * 10.0 + unique_tags * 2.0 + min(length, 40) * 0.1
            scored.append((score_val, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored[:k]]

    class InLoopCorrectionEval(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.history: list[dict[str, float]] = []
            self.model = model
            self.fixed_hardest_indices = _select_hardest_indices(rows, k=16)

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
                eval_subset = rows[:subset_size] if subset_size > 0 else rows
                enable_thinking = (
                    config.get_str("train.thinking") == "on"
                    if "thinking" in config.section("train")
                    else False
                )

                # 1. Compute Validation Cross-Entropy Loss
                val_loss: float | None = None
                use_cuda_autocast = torch.cuda.is_available() and device.type == "cuda"
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                configured_eval_bs = (
                    config.get_int("train.eval_batch_size")
                    if "eval_batch_size" in config.section("train")
                    else (32 if torch.cuda.is_available() else 4)
                )

                if val_examples:
                    from lexi_research.train.trainer import collate_batch

                    # Use a lightweight micro-batch for full-sequence loss to prevent logits VRAM spike (vocab=152k)
                    loss_batch_size = min(configured_eval_bs, 8 if torch.cuda.is_available() else 4)
                    total_loss = 0.0
                    total_items = 0
                    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
                    with torch.inference_mode():
                        for i in range(0, len(val_examples), loss_batch_size):
                            chunk = val_examples[i : i + loss_batch_size]
                            batch = collate_batch(chunk, pad_id)
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            labels = batch["labels"].to(device)
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_autocast):
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                )
                            if outputs.loss is not None:
                                total_loss += float(outputs.loss.item()) * len(chunk)
                                total_items += len(chunk)
                    if total_items > 0:
                        val_loss = total_loss / total_items
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 2. Compute Generation Predictions
                all_prompts: list[str] = []
                references: list[str] = []
                for row in eval_subset:
                    raw_input = row.get("input") or row.get("text") or ""
                    gold = (
                        row.get("output")
                        or row.get("target")
                        or row.get("correction")
                        or ""
                    ).strip()
                    references.append(gold)
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
                    all_prompts.append(prompt_text)

                orig_padding_side = tokenizer.padding_side
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                eval_batch_size = configured_eval_bs
                predictions: list[str] = []
                from tqdm.auto import tqdm

                with tqdm(
                    total=len(all_prompts),
                    desc=f"🔍 Validation @ Step {step}",
                    unit="sent",
                    leave=False,
                ) as pbar:
                    with torch.inference_mode():
                        for i in range(0, len(all_prompts), eval_batch_size):
                            batch_prompts = all_prompts[i : i + eval_batch_size]
                            batch_inputs = tokenizer(
                                batch_prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=config.get_int("train.max_seq_len"),
                            ).to(device)

                            gen_max_tokens = (
                                config.get_int("train.max_new_tokens")
                                if "max_new_tokens" in config.section("train")
                                else 256
                            )
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_autocast):
                                batch_outputs = model.generate(
                                    **batch_inputs,
                                    max_new_tokens=gen_max_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                )
                            prompt_len = batch_inputs["input_ids"].shape[1]
                            for out_seq in batch_outputs:
                                gen_ids = out_seq[prompt_len:]
                                pred_text = tokenizer.decode(
                                    gen_ids, skip_special_tokens=True
                                ).strip()
                                predictions.append(pred_text)
                            pbar.update(len(batch_prompts))

                tokenizer.padding_side = orig_padding_side

                from lexi_research.eval.correction import evaluate_span_predictions
                from lexi_research.format.span_converter import render_spans_to_markup

                raw_inputs = [
                    str(row.get("input") or row.get("text") or "").strip()
                    for row in eval_subset
                ]

                # Render span predictions back to canonical inline markup for visual diffing
                rendered_predictions = [
                    render_spans_to_markup(raw, pred)
                    for raw, pred in zip(raw_inputs, predictions)
                ]

                span_metrics = evaluate_span_predictions(raw_inputs, predictions, references)

                val_metrics: dict[str, float] = {}
                if val_loss is not None:
                    val_metrics["val/loss"] = val_loss
                val_metrics["val/full_edit_f05"] = span_metrics["correction.full_edit_f05"]
                val_metrics["val/span_f05"] = span_metrics["correction.span_f05"]
                val_metrics["val/clean_accuracy"] = span_metrics["correction.clean_accuracy"]
                val_metrics["val/valid_output_rate"] = span_metrics["correction.valid_output_rate"]

                run.log(val_metrics, step=step)
                self.history.append({"step": float(step), **val_metrics})

                # Log qualitative samples with inline visual diff to W&B (fixed 16 hardest samples)
                from lexi_research.tracking.panels import log_correction_samples

                target_indices = (
                    self.fixed_hardest_indices
                    if self.fixed_hardest_indices
                    else list(range(min(16, len(eval_subset))))
                )
                fixed_hardest_records = [
                    {
                        "input": raw_inputs[idx],
                        "raw_spans": predictions[idx],
                        "prediction": rendered_predictions[idx],
                        "gold": references[idx],
                        "exact": rendered_predictions[idx] == references[idx],
                    }
                    for idx in target_indices
                    if idx < len(eval_subset) and idx < len(rendered_predictions)
                ]
                log_correction_samples(run, fixed_hardest_records, step=step, limit=16)

                loss_header = f"Loss: {val_loss:.4f} │ " if val_loss is not None else ""
                print(
                    f"\n[Correction Eval @ Step {step} over {len(eval_subset)} samples] "
                    f"{loss_header}"
                    f"Full Edit F0.5: {val_metrics['val/full_edit_f05']:.1%} │ "
                    f"Span F0.5: {val_metrics['val/span_f05']:.1%} │ "
                    f"Clean Acc: {val_metrics['val/clean_accuracy']:.1%} │ "
                    f"Valid Rate: {val_metrics['val/valid_output_rate']:.1%}",
                    flush=True,
                )
                for idx in range(min(3, len(fixed_hardest_records))):
                    sample = fixed_hardest_records[idx]
                    status = "✓ EXACT" if sample["exact"] else "✗ DIFF"
                    print(
                        f"  [{status}] In:   {sample['input']}\n"
                        f"         Pred: {sample['prediction']}\n"
                        f"         Gold: {sample['gold']}",
                        flush=True,
                    )
                return val_metrics
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

