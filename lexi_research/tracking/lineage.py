"""What a run was: the code, the config, and the machine.

Lineage is the MLOps property this repo is after — a number in a report resolves
to a W&B run, which resolves to a DVC stage hash, which resolves to a commit. So
every stage collects the same dict, logs it into the run config, and writes it
into its report. A report is then interpretable months later without W&B, and a
W&B run is interpretable without the repo checked out.

Nothing here imports a training library. Versions come from the installed
distribution metadata and the GPU from `nvidia-smi`, so collecting lineage on a
CPU box with no torch costs nothing and never fails.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Libraries whose version changes results. Absent ones report `null` rather than
#: being omitted, so a report always has the same shape.
TRACKED_DISTRIBUTIONS = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "lexi-research",
    "peft",
    "torch",
    "transformers",
    "trl",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def git_state(root: Path | None = None) -> dict[str, Any]:
    """The commit a run came from, and whether the tree was clean at the time.

    `dirty` is not decoration: a result produced from an uncommitted tree cannot
    be reproduced from the SHA, and saying so in the report is cheaper than
    discovering it later.
    """
    root = root or repo_root()
    sha = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "sha": sha,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "dirty": bool(status) if status is not None else None,
    }


def file_sha256(path: str | Path) -> str | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def library_versions() -> dict[str, str | None]:
    """Installed versions, read from distribution metadata rather than imports."""
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str | None] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def gpu_state() -> dict[str, Any]:
    """GPU name, memory and driver, or nulls on a machine without one."""
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        cwd=repo_root(),
    )
    if not output:
        return {"devices": [], "driver": None}
    devices = []
    driver = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        devices.append({"name": parts[0], "memory": parts[1]})
        driver = parts[2]
    return {"devices": devices, "driver": driver}


def config_hash(values: Mapping[str, Any]) -> str:
    """A stable digest of the resolved config, including any `--override`.

    Sorted and separator-pinned, so the same config hashes the same across
    machines and Python versions.
    """
    payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect(
    resolved_config: Mapping[str, Any],
    *,
    stage: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Everything needed to place a result in time, code and hardware."""
    root = root or repo_root()
    return {
        "stage": stage,
        "git": git_state(root),
        "dvc_lock_sha256": file_sha256(root / "dvc.lock"),
        "params_sha256": file_sha256(root / "params.yaml"),
        "config_sha256": config_hash(resolved_config),
        "config": dict(resolved_config),
        "libraries": library_versions(),
        "gpu": gpu_state(),
    }


__all__ = [
    "TRACKED_DISTRIBUTIONS",
    "collect",
    "config_hash",
    "file_sha256",
    "git_state",
    "gpu_state",
    "library_versions",
    "repo_root",
]
