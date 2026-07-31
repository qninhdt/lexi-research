"""W&B, behind an interface that works with W&B absent.

Two things make tracking rot: an offline path nobody exercises, and a hard
dependency that turns a missing key into a failed run. So `tracking.mode:
disabled` is a first-class mode that imports nothing and is what CI runs, and
every call site talks to the same handle either way rather than branching on
whether tracking is on.

Adapters and `band_config.json` version together as one artifact. A checkpoint
without its band config produces meaningless bands, so shipping them separately
would make a broken pairing possible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: `disabled` skips W&B entirely, `offline` records to disk for later sync, and
#: `online` needs a key. The names are W&B's own, minus the guessing.
MODES = ("online", "offline", "disabled")


class TrackingError(RuntimeError):
    """Tracking was asked for and could not be provided."""


@dataclass
class Run:
    """A handle that behaves the same whether or not anything is recorded."""

    stage: str
    mode: str
    lineage: Mapping[str, Any] = field(default_factory=dict)
    _run: Any = None

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return str(getattr(self._run, "url", "")) or None if self.active else None

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if self.active:
            self._run.log(dict(metrics), step=step)

    def summary(self, values: Mapping[str, Any]) -> None:
        if self.active:
            self._run.summary.update(dict(values))

    def log_artifact(
        self,
        name: str,
        paths: Sequence[str | Path],
        *,
        kind: str = "model",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish one artifact holding every path given.

        Called with the adapter directory *and* `band_config.json` together, which
        is the point: they are one versioned thing.
        """
        if not self.active:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=kind, metadata=dict(metadata or {}))
        for path in paths:
            resolved = Path(path)
            if not resolved.exists():
                raise TrackingError(f"artifact {name!r} references missing path {resolved}")
            if resolved.is_dir():
                artifact.add_dir(str(resolved))
            else:
                artifact.add_file(str(resolved))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self.active:
            self._run.finish()
            self._run = None


def resolve_mode(config: Any) -> str:
    """`tracking.mode`, falling back to disabled when no key is present.

    An online run without `WANDB_API_KEY` fails inside W&B's login flow, which on
    a headless box means a run that hangs rather than one that says why.
    """
    import os

    mode = str(config.get_str("tracking.mode"))
    if mode not in MODES:
        raise TrackingError(f"tracking.mode={mode!r}; expected one of {list(MODES)}")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        return "disabled"
    return mode


def start(config: Any, *, stage: str, lineage: Mapping[str, Any]) -> Run:
    """Open a run for `stage`, or a handle that records nothing."""
    mode = resolve_mode(config)
    if mode == "disabled":
        return Run(stage=stage, mode=mode, lineage=lineage)

    try:
        import wandb
    except ImportError as exc:
        raise TrackingError(
            f"tracking.mode={mode!r} needs wandb; install it or set tracking.mode=disabled"
        ) from exc

    handle = wandb.init(
        project=config.get_str("tracking.project"),
        entity=config.get_str("tracking.entity") or None,
        group=config.get_str("tracking.group") or None,
        job_type=stage,
        mode=mode,
        config=dict(lineage),
    )
    return Run(stage=stage, mode=mode, lineage=lineage, _run=handle)


__all__ = ["MODES", "Run", "TrackingError", "resolve_mode", "start"]
