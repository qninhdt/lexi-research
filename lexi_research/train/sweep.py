"""Ablation sweeps: arms enumerated from YAML, launched in order, resumable.

Two properties matter more than the runner being clever.

An arm is a set of `--override`s and nothing else. The moment changing an arm
needs a code edit, the arms stop being comparable — the diff between two runs is
no longer only the axis under test.

And state is written after every arm. Colab kills sessions, and a sweep that
restarts from the beginning after four of seven arms costs more GPU-hours than
the sweep itself. Resume skips what is recorded as done and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml


class SweepError(ValueError):
    """The ablation definition is unusable."""


@dataclass(frozen=True)
class Arm:
    """One run: a name, and the overrides that distinguish it from the others."""

    name: str
    overrides: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "overrides": list(self.overrides)}


@dataclass(frozen=True)
class Ablation:
    """One axis, its arms, and the reason it is being measured."""

    key: str
    question: str
    arms: tuple[Arm, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "arms": [arm.as_dict() for arm in self.arms],
        }


def _override_strings(values: Mapping[str, Any]) -> tuple[str, ...]:
    out = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            rendered = ",".join(str(item) for item in value)
        else:
            rendered = str(value)
        out.append(f"{key}={rendered}")
    return tuple(sorted(out))


def load_ablation(path: str | Path) -> Ablation:
    """Read an arm definition.

    Two shapes, because two shapes are what ablations actually look like. `arms`
    lists them explicitly, for an axis whose arms are not a product. `grid` takes
    the cross product of several keys, for one that is — A6 is rank crossed with
    placement, and writing out six entries by hand invites one of them to drift.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SweepError(f"{path} does not hold a mapping")
    for key in ("key", "question"):
        if key not in payload:
            raise SweepError(f"{path} is missing {key!r}")

    arms: list[Arm] = []
    for entry in payload.get("arms") or []:
        if "name" not in entry or "overrides" not in entry:
            raise SweepError(f"{path}: every arm needs a name and overrides")
        arms.append(Arm(name=str(entry["name"]), overrides=_override_strings(entry["overrides"])))

    grid = payload.get("grid")
    if grid:
        keys = sorted(grid)
        for combination in product(*(grid[key] for key in keys)):
            values = dict(zip(keys, combination, strict=True))
            suffix = "-".join(str(value).replace("/", "_") for value in combination)
            arms.append(Arm(name=f"{payload['key']}-{suffix}", overrides=_override_strings(values)))

    if not arms:
        raise SweepError(f"{path} defines no arms")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise SweepError(f"{path} defines duplicate arm names: {sorted(names)}")
    return Ablation(key=str(payload["key"]), question=str(payload["question"]), arms=tuple(arms))


@dataclass
class SweepState:
    """What has finished, on disk, so a killed session resumes at the next arm."""

    path: Path
    completed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> SweepState:
        resolved = Path(path)
        if not resolved.exists():
            return cls(path=resolved)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(path=resolved, completed=dict(payload.get("completed", {})))

    def is_done(self, arm: Arm) -> bool:
        return arm.name in self.completed

    def record(self, arm: Arm, result: Mapping[str, Any]) -> None:
        self.completed[arm.name] = {"overrides": list(arm.overrides), **dict(result)}
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"completed": self.completed}, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def pending(ablation: Ablation, state: SweepState) -> list[Arm]:
    return [arm for arm in ablation.arms if not state.is_done(arm)]


def iter_arms(
    ablation: Ablation,
    state: SweepState,
    *,
    resume: bool = True,
) -> Iterator[Arm]:
    for arm in ablation.arms:
        if resume and state.is_done(arm):
            continue
        yield arm


def default_ablation_path(key: str, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[2] / "ops" / "ablations"
    matches = sorted(root.glob(f"{key}-*.yaml")) + sorted(root.glob(f"{key}.yaml"))
    if not matches:
        raise SweepError(f"no ablation definition for {key!r} under {root}")
    return matches[0]


def available(root: Path | None = None) -> list[str]:
    root = root or Path(__file__).resolve().parents[2] / "ops" / "ablations"
    return sorted({path.stem.split("-")[0] for path in root.glob("*.yaml")})


def summarise(ablation: Ablation, state: SweepState) -> str:
    done = sum(1 for arm in ablation.arms if state.is_done(arm))
    return f"{ablation.key}: {done}/{len(ablation.arms)} arms complete — {ablation.question}"


def arm_names(ablation: Ablation) -> Sequence[str]:
    return [arm.name for arm in ablation.arms]


__all__ = [
    "Ablation",
    "Arm",
    "SweepError",
    "SweepState",
    "arm_names",
    "available",
    "default_ablation_path",
    "iter_arms",
    "load_ablation",
    "pending",
    "summarise",
]
