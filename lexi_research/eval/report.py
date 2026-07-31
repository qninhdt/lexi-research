"""The report: every number, what it is worth, and what it was measured against.

Two rules shape this file.

A metric with no reliability tag gets read as fact. `feedback` has no verifiable
ground truth — chrF measures surface overlap with one teacher phrasing, and a
judge win-rate measures one model's taste — so both carry `reliability: "weak"`
in the JSON and the markdown renderer prints the tag next to the number. A reader
who has to look up whether a metric is trustworthy will not.

And a number without its ceiling is not interpretable. A student at 0.72 QWK
against a teacher that scores 0.74 against itself has nearly saturated the signal
available; the same 0.72 against a ceiling of 0.95 has not. Every headline metric
is therefore reported twice — absolute, and as a fraction of the ceiling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STRONG = "strong"
WEAK = "weak"

#: Metrics with no verifiable ground truth. Named here rather than at each call
#: site so the set is auditable in one place.
WEAK_METRICS = frozenset({"chrf", "judge_win_rate", "judge_discard_rate"})

#: Metrics reported as a fraction of the teacher's agreement with itself.
CEILING_KEYS = {
    "meaning.qwk": "meaning_qwk",
    "correction.span_tag_f1": "correction_edit_f1",
}


class ReportError(ValueError):
    """The report cannot be assembled or is not self-contained."""


@dataclass
class Metric:
    """One number, its reliability, and its share of the ceiling."""

    name: str
    value: float
    reliability: str = STRONG
    ceiling: float | None = None

    @property
    def fraction_of_ceiling(self) -> float | None:
        if self.ceiling is None or self.ceiling <= 0:
            return None
        return self.value / self.ceiling

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"value": self.value, "reliability": self.reliability}
        if self.ceiling is not None:
            payload["ceiling"] = self.ceiling
            payload["fraction_of_ceiling"] = self.fraction_of_ceiling
        return payload


@dataclass
class Report:
    """A complete evaluation, interpretable without W&B and without the repo."""

    stage: str
    split: str
    rows: int
    lineage: Mapping[str, Any]
    ceiling: Mapping[str, Any]
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_group(self, name: str, values: Mapping[str, Any]) -> None:
        tagged: dict[str, Any] = {}
        for key, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                tagged[key] = value
                continue
            metric = Metric(
                name=f"{name}.{key}",
                value=float(value),
                reliability=WEAK if key in WEAK_METRICS else STRONG,
                ceiling=self._ceiling_for(f"{name}.{key}"),
            )
            tagged[key] = metric.as_dict()
        self.groups[name] = tagged

    def _ceiling_for(self, path: str) -> float | None:
        key = CEILING_KEYS.get(path)
        if key is None:
            return None
        value = self.ceiling.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "split": self.split,
            "rows": self.rows,
            "ceiling": dict(self.ceiling),
            "lineage": dict(self.lineage),
            "metrics": self.groups,
            "notes": list(self.notes),
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out

    def flat(self) -> dict[str, float]:
        """`group.metric -> value`, for logging to a run's summary."""
        out: dict[str, float] = {}
        for group, values in self.groups.items():
            for key, payload in values.items():
                if isinstance(payload, Mapping) and "value" in payload:
                    out[f"{group}.{key}"] = float(payload["value"])
        return out

    def markdown(self) -> str:
        """The table that goes in the model card. Weak metrics say so, inline."""
        lines = [
            f"## Evaluation — {self.stage} / {self.split} ({self.rows} rows)",
            "",
            "| Metric | Value | Of ceiling | Reliability |",
            "|---|---|---|---|",
        ]
        for group, values in sorted(self.groups.items()):
            for key, payload in sorted(values.items()):
                if not isinstance(payload, Mapping) or "value" not in payload:
                    continue
                fraction = payload.get("fraction_of_ceiling")
                lines.append(
                    f"| {group}.{key} | {float(payload['value']):.4f} | "
                    f"{'—' if fraction is None else f'{float(fraction):.1%}'} | "
                    f"{payload['reliability']} |"
                )
        if self.notes:
            lines += ["", "### Notes", ""] + [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def check_self_contained(payload: Mapping[str, Any]) -> None:
    """A report that cannot be read on its own is a report that will be misread."""
    for key in ("stage", "split", "rows", "ceiling", "lineage", "metrics"):
        if key not in payload:
            raise ReportError(f"report is missing {key!r}")
    lineage = payload["lineage"]
    for key in ("git", "config_sha256", "libraries"):
        if key not in lineage:
            raise ReportError(f"report lineage is missing {key!r}")
    for group, values in payload["metrics"].items():
        for name, metric in values.items():
            if isinstance(metric, Mapping) and "value" in metric and "reliability" not in metric:
                raise ReportError(f"metric {group}.{name} carries no reliability tag")


def load(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    check_self_contained(payload)
    return payload


__all__ = [
    "CEILING_KEYS",
    "STRONG",
    "WEAK",
    "WEAK_METRICS",
    "Metric",
    "Report",
    "ReportError",
    "check_self_contained",
    "load",
]
