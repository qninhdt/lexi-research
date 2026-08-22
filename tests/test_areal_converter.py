"""Tests for AReaL-tau2-data SFT conversion into prompt/completion records."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tau_research.data.load_areal_sft import (
    build_completion,
    convert_file,
    convert_row,
    normalize_tool_call,
)

FIXTURE = Path(__file__).parent / "fixtures" / "areal_sample.jsonl"


def _retail_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hi"},
        ],
        "answer": {
            "role": "assistant",
            "content": "",
            "thinking": "Need to look up the user first.",
            "tool_calls": [{"name": "find_user", "arguments": {"user_id": "u_1"}}],
        },
        "metadata": {
            "source_dialog_id": "retail_dialog_1",
            "turn_index": 0,
            "correct": 1,
            "reward": 1.0,
        },
    }
    for key, value in overrides.items():
        if key == "metadata":
            row["metadata"].update(value)
        else:
            row[key] = value
    return row


def test_normalize_tool_call_dict_shape() -> None:
    assert normalize_tool_call({"name": "t", "arguments": {"a": 1}}) == ("t", {"a": 1})


def test_normalize_tool_call_anthropic_json_string_shape() -> None:
    raw = {
        "id": "toolu_01",
        "type": "function",
        "function": {"name": "t", "arguments": '{"a": "x y"}'},
    }
    assert normalize_tool_call(raw) == ("t", {"a": "x y"})


def test_normalize_tool_call_null_args_dropped() -> None:
    assert normalize_tool_call({"name": "t", "arguments": None}) is None


def test_build_completion_single_call() -> None:
    stats: Counter[str] = Counter()
    text = build_completion(
        {
            "thinking": "Look up user.",
            "content": "",
            "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "e_1"}}],
        },
        stats,
    )
    assert text == "<think>\nLook up user.\n</think>\nget_user_details(user_id='e_1')"


def test_build_completion_message_turn_keeps_content() -> None:
    stats: Counter[str] = Counter()
    text = build_completion(
        {"thinking": "Done.", "content": "Your order is cancelled.", "tool_calls": []},
        stats,
    )
    assert text == "<think>\nDone.\n</think>\nYour order is cancelled."


def test_build_completion_drops_extra_calls_and_content() -> None:
    stats: Counter[str] = Counter()
    build_completion(
        {
            "thinking": "t",
            "content": "lead-in",
            "tool_calls": [
                {"name": "a", "arguments": {}},
                {"name": "b", "arguments": {}},
            ],
        },
        stats,
    )
    assert stats["calls_dropped_extra_actions"] == 1
    assert stats["contents_dropped_alongside_call"] == 1


def test_build_completion_requires_thinking() -> None:
    stats: Counter[str] = Counter()
    assert build_completion({"thinking": "", "content": "hi", "tool_calls": []}, stats) is None
    assert stats["dropped_empty_thinking"] == 1


def test_convert_row_filters_non_retail_and_failed_rows() -> None:
    stats: Counter[str] = Counter()
    assert (
        convert_row(_retail_row(metadata={"source_dialog_id": "airline_dialog_9"}), stats) is None
    )
    assert convert_row(_retail_row(metadata={"correct": 0}), stats) is None
    assert convert_row(_retail_row(metadata={"reward": 0.0}), stats) is None
    assert stats["dropped_not_retail"] == 1
    assert stats["dropped_not_correct"] == 1
    assert stats["dropped_reward_zero"] == 1


def test_convert_row_history_has_no_thinking_leak() -> None:
    row = _retail_row(
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "assistant", "content": "<think>old</think>\ncall:x(a=1)"},
            {"role": "tool", "content": "r"},
            {"role": "user", "content": "go on"},
        ]
    )
    example = convert_row(row, Counter[str]())
    assert example is not None
    contents = [m["content"] for m in example["prompt"]]
    assert not any("<think>" in c for c in contents)


def test_convert_file_splits_by_dialog_and_writes_manifest(tmp_path: Path) -> None:
    rows = [
        _retail_row(metadata={"source_dialog_id": f"retail_dialog_{i}", "correct": 1})
        for i in range(10)
    ]
    input_path = tmp_path / "in.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    manifest = convert_file(input_path, tmp_path / "out")
    assert manifest["train_dialogs"] + manifest["val_dialogs"] == 10
    assert manifest["train_examples"] + manifest["val_examples"] == 10

    train_lines = (tmp_path / "out" / "areal_sft_train.json").read_text().splitlines()
    val_lines = (tmp_path / "out" / "areal_sft_val.json").read_text().splitlines()
    train_dialogs = {json.loads(line)["dialog_id"] for line in train_lines}
    val_dialogs = {json.loads(line)["dialog_id"] for line in val_lines}
    assert train_dialogs.isdisjoint(val_dialogs)
