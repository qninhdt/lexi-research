"""The hand-authored sample set must satisfy the format contract it teaches.

These rows are what local SFT checks train on, so an invalid `correction` here
does not fail loudly — it teaches the student to emit output the validator
rejects, and the damage shows up later as a low format-validity score that looks
like a training problem rather than a data problem.

The tags in particular are worth pinning: the taxonomy is a closed set of 16
short codes, and prose descriptions like `agreement` or `verb pattern` read
perfectly well to a human author while being rejected by the parser.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lexi_research.format import (
    BandConfig,
    ValidationError,
    default_config_path,
    validate_output,
)

ANSWER_FIELDS = ("correction", "meaning", "feedback")


def _rows() -> list[dict[str, object]]:
    """Import `data/make_sample.py` and return its static rows.

    Imported rather than read from the generated parquet so the test covers the
    source of truth, and so it does not depend on the artifact having been built.
    """
    source = Path(__file__).resolve().parents[2] / "data" / "make_sample.py"
    spec = importlib.util.spec_from_file_location("_make_sample", source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.fail(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = getattr(module, "ROWS", None)
    if not isinstance(rows, list) or not rows:
        pytest.fail("make_sample.py exposes no ROWS list")
    return rows


def test_every_sample_row_passes_output_validation() -> None:
    band_config = BandConfig.from_json(default_config_path())

    failures = []
    for index, row in enumerate(_rows()):
        answer = {field: row.get(field) for field in ANSWER_FIELDS}
        checked = validate_output(answer, str(row["text"]), band_config)
        if isinstance(checked, ValidationError):
            failures.append(
                f"row {index} ({row.get('target')!r}): {checked.code} — {checked.detail}"
            )

    assert not failures, "invalid sample rows:\n" + "\n".join(failures)


def test_sample_rows_carry_every_field_the_trainer_reads() -> None:
    required = {"target", "definition", "pos", "text", *ANSWER_FIELDS}

    for index, row in enumerate(_rows()):
        missing = required - set(row)
        assert not missing, f"row {index} is missing {sorted(missing)}"
        # `correction` is None for a row that carries no usable edit, which is
        # legal; the other fields are not optional.
        for field in required - {"correction"}:
            assert row[field] is not None, f"row {index} has {field}=None"
