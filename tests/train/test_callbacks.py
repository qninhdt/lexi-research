"""Resume, and the in-loop evaluation that stops a session being wasted.

Loss falls smoothly while a model learns to emit prose instead of JSON. Only the
harness notices, so it runs during training — and the interval is honoured rather
than being a suggestion, because an in-loop eval that fires every step doubles the
wall clock and gets switched off.

Resume picks the latest checkpoint by *step*, not by modification time: a resumed
run rewrites files, so the most recently touched directory is not necessarily the
furthest along.
"""

from __future__ import annotations

import pytest

from lexi_research.train.callbacks import latest_checkpoint, resolve_resume


def _checkpoints(root, steps) -> None:
    for step in steps:
        (root / f"checkpoint-{step}").mkdir(parents=True)


def test_latest_checkpoint_is_by_step_not_by_mtime(tmp_path) -> None:
    _checkpoints(tmp_path, [200, 1000, 400])
    # Touch the earliest last, so mtime ordering would pick the wrong one.
    (tmp_path / "checkpoint-200").touch()
    assert latest_checkpoint(tmp_path).name == "checkpoint-1000"


def test_no_checkpoints_is_not_an_error(tmp_path) -> None:
    """The first arm of a sweep has nothing to resume from."""
    assert latest_checkpoint(tmp_path) is None
    assert resolve_resume(tmp_path, "auto") is None


def test_a_missing_directory_is_not_an_error(tmp_path) -> None:
    assert latest_checkpoint(tmp_path / "nope") is None


def test_unrelated_directories_are_ignored(tmp_path) -> None:
    (tmp_path / "adapter").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()
    _checkpoints(tmp_path, [7])
    assert latest_checkpoint(tmp_path).name == "checkpoint-7"


def test_resume_auto_finds_the_latest(tmp_path) -> None:
    _checkpoints(tmp_path, [100, 300])
    assert resolve_resume(tmp_path, "auto").endswith("checkpoint-300")


def test_resume_none_disables(tmp_path) -> None:
    _checkpoints(tmp_path, [100])
    assert resolve_resume(tmp_path, "none") is None
    assert resolve_resume(tmp_path, None) is None


def test_an_explicit_checkpoint_is_taken_as_given(tmp_path) -> None:
    _checkpoints(tmp_path, [100, 300])
    chosen = str(tmp_path / "checkpoint-100")
    assert resolve_resume(tmp_path, chosen) == chosen


def test_an_explicit_missing_checkpoint_raises(tmp_path) -> None:
    """Silently starting from zero would waste the session it was meant to save."""
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path, str(tmp_path / "checkpoint-999"))


def test_eval_interval(monkeypatch) -> None:
    """The harness fires on the interval, not every step."""
    pytest.importorskip("transformers")
    from lexi_research.train import callbacks

    calls: list[int] = []

    class Args:
        pass

    class State:
        def __init__(self, step: int) -> None:
            self.global_step = step

    callback = callbacks.build_eval_callback(
        config=None,
        run=None,
        tokenizer=None,
        rows=[{"text": "hi"}],
        band_config=None,
        ceiling={},
        every_steps=50,
    )
    monkeypatch.setattr(callback, "run_once", lambda model, step: calls.append(step))

    for step in range(0, 151):
        callback.on_step_end(Args(), State(step), None, model=object())

    assert calls == [50, 100, 150]


def test_eval_can_be_switched_off() -> None:
    pytest.importorskip("transformers")
    from lexi_research.train import callbacks

    calls: list[int] = []

    class State:
        global_step = 100

    callback = callbacks.build_eval_callback(
        config=None,
        run=None,
        tokenizer=None,
        rows=[],
        band_config=None,
        ceiling={},
        every_steps=0,
    )
    callback.run_once = lambda model, step: calls.append(step)  # type: ignore[method-assign]
    callback.on_step_end(None, State(), None, model=object())
    assert calls == []


def test_resolve_run_and_output_dir(tmp_path) -> None:
    from lexi_research.cli.config import load_config
    from lexi_research.train.callbacks import resolve_run_and_output_dir

    p = tmp_path / "params.yaml"
    p.write_text(
        """
train:
  task: corrector
  base_model: Qwen/Qwen3.5-0.8B
  lora_r: 64
tracking:
  mode: disabled
""",
        encoding="utf-8",
    )
    cfg = load_config(p)
    base_out = tmp_path / "lexi-runs"
    base_out.mkdir()

    # 1. Fresh run creates a subfolder with auto name inside base_out
    actual_dir, run_name, resume_chk = resolve_run_and_output_dir(base_out, "none", cfg, "sft")
    assert actual_dir.parent == base_out
    assert run_name.startswith("sft-corrector-qwen3.5-0.8b-r64-")
    assert resume_chk is None

    # Simulate saving checkpoints in that run directory
    actual_dir.mkdir(parents=True)
    (actual_dir / "checkpoint-100").mkdir()
    (actual_dir / "checkpoint-200").mkdir()

    # 2. Resuming by explicit run name
    resumed_dir, resumed_name, resumed_chk = resolve_run_and_output_dir(
        base_out, run_name, cfg, "sft"
    )
    assert resumed_dir == actual_dir
    assert resumed_name == run_name
    assert resumed_chk.endswith("checkpoint-200")

    # 3. Resuming by 'auto' on base_out finds the latest run folder
    auto_dir, auto_name, auto_chk = resolve_run_and_output_dir(base_out, "auto", cfg, "sft")
    assert auto_dir == actual_dir
    assert auto_name == run_name
    assert auto_chk.endswith("checkpoint-200")

