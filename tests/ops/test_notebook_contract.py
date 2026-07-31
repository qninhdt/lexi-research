"""The notebook contract: a launcher, not a program.

The rule the parent plan calls non-negotiable is that no Python is written inside
a notebook. A cell that defines a function is a piece of the pipeline living
outside version control — it cannot be tested, reviewed, or reproduced from a
commit. These tests are what make that a property rather than an intention.

The second failure they catch is drift: a flag renamed in the CLI leaves a cell
still passing the old one, and nobody finds out until a GPU is already rented.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "notebooks" / "lexi_colab.py"
NOTEBOOK = ROOT / "notebooks" / "lexi_colab.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "ops/build-notebook.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_no_definitions(notebook) -> None:
    """A `def` in a cell is pipeline code that no commit can reproduce."""
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            assert not line.startswith(("def ", "class ")), f"cell {index} defines {line!r}"


def test_committed_matches_source() -> None:
    """Regeneration is byte-for-byte, so a stale notebook cannot be committed."""
    rendered = _builder().render(SOURCE.read_text(encoding="utf-8"))
    assert NOTEBOOK.read_text(encoding="utf-8") == rendered


def test_no_outputs(notebook) -> None:
    """Committed outputs are stale the moment they land, and can leak data."""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_every_command_is_a_lexi_subcommand_or_setup(notebook) -> None:
    """No `python -c`, no `python -m`: the notebook types what a human types."""
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "python -c" not in source
        assert "python -m lexi" not in source


def test_the_notebook_actually_trains_and_verifies(notebook) -> None:
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "lexi smoke --gpu" in source
    assert "lexi train sft" in source
    assert "dvc pull" in source


def test_secrets_are_read_not_printed(notebook) -> None:
    """A printed key ends up in a committed output or a shared screen."""
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "userdata.get" in source
    assert "print(os.environ" not in source


def test_the_builder_is_idempotent() -> None:
    builder = _builder()
    text = SOURCE.read_text(encoding="utf-8")
    assert builder.render(text) == builder.render(text)


def test_the_builder_carries_no_clock_or_random_ids() -> None:
    """Otherwise `make notebook` produces a diff on every run."""
    rendered = _builder().render(SOURCE.read_text(encoding="utf-8"))
    payload = json.loads(rendered)
    assert "id" not in payload["cells"][0]
    assert set(payload["metadata"]) == {"kernelspec", "language_info"}
