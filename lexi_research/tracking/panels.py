"""W&B panels, defined in code so they reproduce.

A panel clicked together in the UI belongs to one workspace and one person. The
next run opens on defaults, and the plot that made a result legible has to be
rebuilt from memory. Defining them here means a run carries its own views.

Two choices are deliberate. Latency is a CDF, never a bar of means — the mean
hides the tail, and the tail is the only part a user notices. And the qualitative
table comes first in the list because it is the panel used most while debugging:
a metric says something regressed, the table says what the model actually wrote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: `wandb.Table` columns for the panel read most often during debugging.
QUALITATIVE_COLUMNS = (
    "req_uid",
    "text",
    "gold_correction",
    "predicted_correction",
    "gold_meaning",
    "predicted_meaning",
    "reasoning",
    "gold_feedback",
    "predicted_feedback",
    "valid",
    "retries",
)

PANEL_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "training": (
        {"title": "loss", "keys": ("train/loss", "train/ce_loss", "train/rl_loss"), "kind": "line"},
        {"title": "reward", "keys": ("rl/reward_mean",), "band": "rl/reward_std", "kind": "line"},
        {"title": "KL to reference", "keys": ("rl/kl",), "kind": "line"},
        {"title": "reasoning length", "keys": ("rl/reasoning_tokens",), "kind": "histogram"},
        {"title": "optimiser", "keys": ("train/learning_rate", "train/grad_norm"), "kind": "line"},
    ),
    "eval": (
        {"title": "tag confusion", "keys": ("eval/confusion",), "kind": "heatmap"},
        {"title": "reliability", "keys": ("eval/reliability",), "kind": "line", "diagonal": True},
        {"title": "per-band accuracy", "keys": ("eval/per_band",), "kind": "stacked_bar"},
        {"title": "band distribution", "keys": ("data/before", "data/after"), "kind": "bar"},
    ),
    "ablation": (
        {"title": "arms", "keys": ("ablation/*",), "kind": "parallel_coordinates"},
        {"title": "quality vs latency", "keys": ("eval/qwk", "bench/p95_ms"), "kind": "scatter"},
    ),
    "qualitative": ({"title": "predictions", "keys": ("eval/table",), "kind": "table"},),
    "inference": (
        # A CDF, not a mean: the mean hides the tail, and the tail is what a user
        # experiences.
        {"title": "latency CDF", "keys": ("bench/latency_ms",), "kind": "cdf"},
        {"title": "throughput vs concurrency", "keys": ("bench/tokens_per_s",), "kind": "line"},
        {"title": "VRAM", "keys": ("bench/vram_mb",), "kind": "line"},
    ),
}


def qualitative_rows(predictions: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Predictions as table rows, in `QUALITATIVE_COLUMNS` order."""
    rows: list[list[Any]] = []
    for row in predictions:
        prediction = row.get("prediction") or {}
        rows.append(
            [
                row.get("req_uid"),
                row.get("text"),
                row["gold"].get("correction"),
                prediction.get("correction"),
                row["gold"].get("meaning"),
                prediction.get("meaning"),
                row.get("reasoning"),
                row["gold"].get("feedback"),
                prediction.get("feedback"),
                bool(prediction),
                row.get("retries", 0),
            ]
        )
    return rows


def compute_html_diff(pred: str | None, gold: str | None) -> str:
    """Generate a clean, standard git-style inline diff between model prediction and gold target."""
    if pred is None or gold is None:
        return "<span>N/A</span>"
    p_str = pred.strip()
    g_str = gold.strip()
    if p_str == g_str:
        import html

        return f'<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.6;">{html.escape(p_str)}</div>'

    import difflib
    import html
    import re

    # Tokenize words, whitespace, and punctuation
    p_tokens = [t for t in re.split(r"(\s+|[^\w\s])", p_str) if t]
    g_tokens = [t for t in re.split(r"(\s+|[^\w\s])", g_str) if t]

    matcher = difflib.SequenceMatcher(None, p_tokens, g_tokens)
    parts: list[str] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        p_chunk = "".join(p_tokens[i1:i2])
        g_chunk = "".join(g_tokens[j1:j2])

        if op == "equal":
            parts.append(html.escape(p_chunk))
        elif op == "replace":
            parts.append(
                f'<span style="background-color:#fee2e2;color:#b91c1c;text-decoration:line-through;padding:1px 3px;border-radius:2px;">-{html.escape(p_chunk.strip())}</span>'
                f'<span style="background-color:#dcfce7;color:#15803d;font-weight:600;padding:1px 3px;border-radius:2px;">+{html.escape(g_chunk.strip())}</span>'
            )
        elif op == "delete":
            # Extra in pred
            parts.append(
                f'<span style="background-color:#fee2e2;color:#b91c1c;text-decoration:line-through;padding:1px 3px;border-radius:2px;">-{html.escape(p_chunk.strip())}</span>'
            )
        elif op == "insert":
            # Missing from pred, in gold
            parts.append(
                f'<span style="background-color:#dcfce7;color:#15803d;font-weight:600;padding:1px 3px;border-radius:2px;">+{html.escape(g_chunk.strip())}</span>'
            )

    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.6;">'
        + "".join(parts)
        + "</div>"
    )


CORRECTION_SAMPLE_COLUMNS = ("Diff (Git-style: -Prediction / +Gold)",)


def log_correction_samples(
    run: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    step: int | None = None,
    limit: int = 50,
) -> None:
    """Publish qualitative GEC samples with inline HTML visual diff to W&B (1 column)."""
    if not getattr(run, "active", False):
        return
    import wandb

    rows = []
    for s in samples[:limit]:
        pred = str(s.get("prediction", "") or "")
        gold = str(s.get("gold", "") or "")

        html_diff = compute_html_diff(pred, gold)
        rows.append([wandb.Html(html_diff)])

    table = wandb.Table(columns=list(CORRECTION_SAMPLE_COLUMNS), data=rows)
    run.log({"val/samples": table}, step=step)


def log_qualitative(
    run: Any, predictions: Sequence[Mapping[str, Any]], *, limit: int = 200
) -> None:
    """Publish the debugging table. A no-op when the run records nothing."""
    if not getattr(run, "active", False):
        return
    import wandb

    table = wandb.Table(
        columns=list(QUALITATIVE_COLUMNS), data=qualitative_rows(predictions[:limit])
    )
    run.log({"eval/table": table})


def log_confusion(run: Any, confusion: Mapping[str, int]) -> None:
    """Publish the tag-by-tag confusion counts as a heatmap-shaped table."""
    if not getattr(run, "active", False):
        return
    import wandb

    rows = [[*key.split("->"), count] for key, count in sorted(confusion.items())]
    run.log({"eval/confusion": wandb.Table(columns=["gold", "predicted", "count"], data=rows)})


def log_reliability(run: Any, bins: Sequence[Mapping[str, Any]]) -> None:
    if not getattr(run, "active", False):
        return
    import wandb

    rows = [[item["mean_confidence"], item["accuracy"], item["count"]] for item in bins]
    run.log(
        {"eval/reliability": wandb.Table(columns=["confidence", "accuracy", "count"], data=rows)}
    )


def log_per_band(run: Any, band_metrics: Mapping[str, Any]) -> None:
    """Publish per-meaning-band metrics as a formatted wandb.Table."""
    if not getattr(run, "active", False):
        return
    import wandb

    rows: list[list[Any]] = []
    for band_key, metrics in sorted(band_metrics.items()):
        if isinstance(metrics, Mapping):
            rows.append(
                [
                    str(band_key),
                    metrics.get("accuracy", 0.0),
                    metrics.get("mae", 0.0),
                    metrics.get("count", 0),
                ]
            )
    if rows:
        table = wandb.Table(columns=["band", "accuracy", "mae", "count"], data=rows)
        run.log({"eval/per_band": table})


def log_eval_overview(run: Any, eval_metrics: Mapping[str, Any]) -> None:
    """Publish a complete evaluation metric suite to W&B run summary and metrics."""
    if not getattr(run, "active", False):
        return

    payload: dict[str, Any] = {}
    for key in (
        "qwk",
        "exact_match",
        "edit_f1",
        "format_validity",
        "mae",
        "self_consistency",
        "retries",
    ):
        if key in eval_metrics:
            payload[f"eval/{key}"] = eval_metrics[key]

    if payload:
        run.log(payload)
        run.summary(payload)


def log_hardware_summary(
    run: Any, *, peak_vram_gb: float | None = None, throughput_tokens_per_s: float | None = None
) -> None:
    """Publish hardware resource usage and throughput metrics."""
    if not getattr(run, "active", False):
        return

    metrics: dict[str, Any] = {}
    if peak_vram_gb is not None:
        metrics["system/peak_vram_gb"] = peak_vram_gb
    if throughput_tokens_per_s is not None:
        metrics["system/throughput_tokens_per_s"] = throughput_tokens_per_s

    if metrics:
        run.log(metrics)
        run.summary(metrics)


__all__ = [
    "PANEL_GROUPS",
    "QUALITATIVE_COLUMNS",
    "log_confusion",
    "log_eval_overview",
    "log_hardware_summary",
    "log_per_band",
    "log_qualitative",
    "log_reliability",
    "qualitative_rows",
]

