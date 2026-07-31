"""Generate `MODEL_CARD.md` from the report JSON, never by hand.

A hand-written card drifts from the numbers the moment a run is repeated, and the
drift is invisible because both look like prose. Generating it means the card
either matches a report that traces to a commit, or regeneration produces a diff
and CI says so.

The limitations are copied from the design documents rather than paraphrased. A
paraphrase of "this measures fidelity to a teacher, not accuracy against ground
truth" is how that sentence turns into "accuracy" over a few revisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Verbatim from the parent design §13 and this design §11. Not paraphrased, and
#: not summarised — every one of these is a threat to the validity of every
#: number in the card above it.
LIMITATIONS = (
    "Trained on teacher-generated data with **no human gold set**. Every number "
    "here is fidelity to a teacher, not accuracy against ground truth.",
    "Train and test are both teacher-generated, so the real-learner distribution "
    "is unverified. This is the largest validity threat in the project and it was "
    "not addressed.",
    "Band thresholds live in `band_config.json` and ship with the adapter. A "
    "checkpoint without it produces meaningless bands.",
    "`feedback` quality is measured only by weak proxies — chrF and a "
    "teacher-as-judge win-rate — and both are labelled weak wherever they appear.",
    "Metrics are reported as a fraction of teacher self-consistency. A student "
    "cannot exceed the agreement its teacher has with itself, so a raw number "
    "compared against 1.0 would understate the result.",
)

#: Rows the card leads with. Ordered by what a reader asks first.
HEADLINE = (
    ("meaning", "qwk", "Meaning band agreement (QWK)"),
    ("meaning", "exact", "Meaning band exact match"),
    ("correction", "span_tag_f1", "Correction edit F1 (span + tag)"),
    ("correction", "span_only_f1", "Correction edit F1 (span only)"),
    ("format", "validity_rate", "Output passes all six format checks"),
)


class ModelCardError(ValueError):
    """The report cannot support a model card."""


def _metric(report: Mapping[str, Any], group: str, key: str) -> Mapping[str, Any] | None:
    value = report.get("metrics", {}).get(group, {}).get(key)
    return value if isinstance(value, Mapping) and "value" in value else None


def _table(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Metric | Value | Of teacher ceiling | Reliability |",
        "|---|---|---|---|",
    ]
    for group, key, label in HEADLINE:
        metric = _metric(report, group, key)
        if metric is None:
            continue
        fraction = metric.get("fraction_of_ceiling")
        lines.append(
            f"| {label} | {float(metric['value']):.4f} | "
            f"{'—' if fraction is None else f'{float(fraction):.1%}'} | "
            f"{metric['reliability']} |"
        )
    if len(lines) == 2:
        raise ModelCardError("the report carries none of the headline metrics")
    return lines


def render(
    report: Mapping[str, Any],
    *,
    base_model: str,
    rl_verdict: str,
    comparison: Sequence[Mapping[str, Any]] = (),
) -> str:
    """The card. Every number comes from `report`; nothing is typed in twice."""
    lineage = report.get("lineage", {})
    git = lineage.get("git", {}) if isinstance(lineage, Mapping) else {}
    ceiling = report.get("ceiling", {})

    lines = [
        "# Model card — lexi grader",
        "",
        "**This card is generated from `reports/eval-sft.json` by "
        "`lexi report model-card`. Do not edit it by hand: a hand-edited card "
        "drifts from the numbers and the drift is invisible.**",
        "",
        "## What it is",
        "",
        f"A LoRA adapter over `{base_model}`, distilled from a frontier grader. It "
        "takes a target word, its dictionary sense, and a learner sentence, and "
        "returns an inline correction, a meaning band, and one sentence of "
        "feedback. `grammar` and `naturalness` are derived from the correction by "
        "code rather than emitted by the model.",
        "",
        "## Results",
        "",
        f"Split: `{report.get('split', 'unknown')}`, {report.get('rows', 0)} rows.",
        "",
        *_table(report),
        "",
        f"Teacher self-consistency ceiling: meaning QWK "
        f"{ceiling.get('meaning_qwk', 'unknown')}, correction edit F1 "
        f"{ceiling.get('correction_edit_f1', 'unknown')}.",
        "",
        "### Did reinforcement learning beat supervised fine-tuning?",
        "",
        rl_verdict,
        "",
    ]

    if comparison:
        lines += [
            "### Against the alternatives",
            "",
            "| System | Quality (QWK) | p95 latency | Cost / 1k requests |",
            "|---|---|---|---|",
        ]
        for row in comparison:
            lines.append(
                f"| {row.get('system')} | {row.get('qwk', '—')} | "
                f"{row.get('e2e_p95_s', '—')} | {row.get('cost_per_1k_requests', '—')} |"
            )
        lines += [
            "",
            "The axis that decides this is quality per dollar at a fixed latency "
            "SLO, not raw quality. A larger model that wins on QWK while costing "
            "several times more per request loses for this application.",
            "",
        ]

    lines += [
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
        "## Provenance",
        "",
        f"- Commit: `{git.get('sha') or 'unknown'}`"
        + (" (**tree was dirty**)" if git.get("dirty") else ""),
        f"- Config hash: `{lineage.get('config_sha256', 'unknown')}`",
        f"- DVC lock: `{lineage.get('dvc_lock_sha256') or 'unknown'}`",
        "- Every number above traces to the report JSON, which traces to this "
        "commit through the DVC stage that produced it.",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate(
    report_path: str | Path,
    destination: str | Path,
    *,
    base_model: str,
    rl_verdict: str,
    comparison_path: str | Path | None = None,
) -> Path:
    from lexi_research.eval.report import load

    report = load(report_path)
    comparison: Sequence[Mapping[str, Any]] = ()
    if comparison_path and Path(comparison_path).exists():
        payload = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
        comparison = payload.get("systems", [])

    out = Path(destination)
    out.write_text(
        render(report, base_model=base_model, rl_verdict=rl_verdict, comparison=comparison),
        encoding="utf-8",
    )
    return out


__all__ = ["HEADLINE", "LIMITATIONS", "ModelCardError", "generate", "render"]
