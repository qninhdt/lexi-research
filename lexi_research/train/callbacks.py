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

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            step = int(state.global_step)
            if every_steps <= 0 or step == 0 or step % every_steps:
                return
            model = kwargs.get("model")
            if model is None:
                return
            self.run_once(model, step)

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
                    f"validity {flat.get('format.validity_rate', 0):.2f}, "
                    f"meaning exact {flat.get('meaning.exact', 0):.2f}",
                    flush=True,
                )
                return flat
            finally:
                if was_training:
                    model.train()

    return InLoopEval()


__all__ = ["build_eval_callback", "latest_checkpoint", "resolve_resume"]
