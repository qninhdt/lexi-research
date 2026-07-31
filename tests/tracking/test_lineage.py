"""Lineage — what makes a number in a report traceable months later.

The failure these guard against is silent: a report that looks complete but has
no commit in it, or a `disabled` tracking mode that quietly tries to reach the
network on a box with no key and hangs in a login flow.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.cli.config import load_config
from lexi_research.tracking import lineage, wandb_run

PARAMS = """
tracking:
  project: lexi-research
  entity: ""
  group: ""
  mode: {mode}
  adapter_artifact: lexi-grader-adapter
train:
  lora_r: 32
"""


def _config(tmp_path, mode: str):
    path = tmp_path / "params.yaml"
    path.write_text(PARAMS.format(mode=mode), encoding="utf-8")
    return load_config(path)


def test_lineage_has_required_keys() -> None:
    collected = lineage.collect({"train": {"lora_r": 32}}, stage="sft")
    assert collected["stage"] == "sft"
    assert set(collected) >= {
        "git",
        "dvc_lock_sha256",
        "params_sha256",
        "config_sha256",
        "config",
        "libraries",
        "gpu",
    }
    assert set(collected["git"]) == {"sha", "branch", "dirty"}


def test_every_tracked_library_is_reported_even_when_absent() -> None:
    """A report always has the same shape; a missing library is `null`, not a gap."""
    versions = lineage.library_versions()
    assert set(versions) == set(lineage.TRACKED_DISTRIBUTIONS)


def test_the_config_hash_moves_with_an_override() -> None:
    """Two runs that differ only by a sweep arm must not share a config hash."""
    before = lineage.config_hash({"train": {"lora_r": 32}})
    after = lineage.config_hash({"train": {"lora_r": 64}})
    assert before != after
    assert before == lineage.config_hash({"train": {"lora_r": 32}})


def test_lineage_is_json_serialisable() -> None:
    """It is written into every report, so it has to survive `json.dumps`."""
    json.dumps(lineage.collect({}, stage="eval"))


def test_a_missing_dvc_lock_reports_null_rather_than_raising(tmp_path) -> None:
    collected = lineage.collect({}, stage="sft", root=tmp_path)
    assert collected["dvc_lock_sha256"] is None
    assert collected["git"]["sha"] is None or isinstance(collected["git"]["sha"], str)


def test_disabled_mode_records_nothing_and_imports_nothing(tmp_path, monkeypatch) -> None:
    """The mode CI runs. It must not need wandb to be installed."""
    monkeypatch.setattr(wandb_run, "resolve_mode", lambda config: "disabled", raising=True)
    run = wandb_run.start(_config(tmp_path, "disabled"), stage="sft", lineage={})
    assert not run.active
    assert run.url is None
    run.log({"loss": 1.0})
    run.summary({"examples": 50})
    run.log_artifact("adapter", ["/nonexistent"])
    run.finish()


def test_online_without_a_key_falls_back_to_disabled(tmp_path, monkeypatch) -> None:
    """Otherwise a headless box hangs inside W&B's login flow."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_run.resolve_mode(_config(tmp_path, "online")) == "disabled"


def test_online_with_a_key_stays_online(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    assert wandb_run.resolve_mode(_config(tmp_path, "online")) == "online"


def test_an_unknown_mode_raises(tmp_path) -> None:
    with pytest.raises(wandb_run.TrackingError, match="tracking.mode"):
        wandb_run.resolve_mode(_config(tmp_path, "sometimes"))


def test_the_repo_params_declare_tracking() -> None:
    config = load_config()
    assert config.get_str("tracking.mode") in wandb_run.MODES
    assert config.get_str("tracking.project")
    assert config.get_str("tracking.adapter_artifact")
