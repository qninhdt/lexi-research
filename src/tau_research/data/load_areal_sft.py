"""Streams AReaL-tau2-data SFT rows into per-turn prompt/completion records.

Dataset: https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data (Apache-2.0).
Each row is one assistant decision point: ``messages`` history (prior thinking
already stripped), an ``answer`` dict with thinking/tool_calls, and ``metadata``
with provenance plus correctness labels.

Only retail rows whose episode verified successful (correct=1, reward=1.0) and
that carry a non-empty thinking trace become training examples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tau_research.data.prepare_sft import (
    format_thinking_block,
    sanitize_history_for_turn,
)


def iter_jsonl(path: str | Path) -> Any:
    """Yields parsed JSON objects from a jsonl file line by line."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_tool_call(raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Normalizes observed AReaL tool-call shapes to ``(name, args-dict)``.

    Handles ``{name, arguments: dict}`` and the Anthropic-shaped
    ``{id, type, function: {name, arguments: JSON-string}}``. Returns None for
    null or unparseable arguments.
    """
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    args = raw.get("arguments")
    fn = raw.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        name = fn["name"]
        args = fn.get("arguments")
    if not isinstance(name, str) or not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return name, args


def build_completion(answer: dict[str, Any], stats: Counter[str]) -> str | None:
    """Renders one answer dict into canonical completion text, or None to drop.

    Contract taught to the model: one action per turn - either a single
    functional tool call or a plain message - preceded by a think block.
    """
    thinking = (answer.get("thinking") or "").strip()
    content = (answer.get("content") or "").strip()
    calls_raw = answer.get("tool_calls") or []

    normalized = []
    for call in calls_raw:
        parsed = normalize_tool_call(call)
        if parsed is None:
            stats["calls_dropped_null_args"] += 1
        else:
            normalized.append(parsed)

    if not thinking:
        stats["dropped_empty_thinking"] += 1
        return None

    parts = [format_thinking_block(thinking)]
    from tau_research.tau.action_parser import format_functional_tool_call

    if normalized:
        if len(normalized) > 1:
            stats["calls_dropped_extra_actions"] += 1
        if content:
            stats["contents_dropped_alongside_call"] += 1
        parts.append(format_functional_tool_call(*normalized[0]))
    elif content:
        parts.append(content)
    else:
        # Reasoning with no executable output teaches nothing at decode time.
        return None
    return "\n".join(parts)


def convert_row(row: dict[str, Any], stats: Counter[str]) -> dict[str, Any] | None:
    """Converts one AReaL row into a prompt/completion record, or None."""
    meta = row.get("metadata") or {}
    dialog_id = str(meta.get("source_dialog_id", ""))

    if not dialog_id.startswith("retail_"):
        stats["dropped_not_retail"] += 1
        return None
    if meta.get("correct") != 1:
        stats["dropped_not_correct"] += 1
        return None
    if meta.get("reward") != 1.0 and meta.get("reward") != 1:
        stats["dropped_reward_zero"] += 1
        return None

    completion_text = build_completion(row.get("answer") or {}, stats)
    if completion_text is None:
        return None

    prompt = sanitize_history_for_turn(row.get("messages") or [])
    stats["rows_kept"] += 1
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": completion_text}],
        "dialog_id": dialog_id,
    }


def split_dialog_ids(
    dialog_ids: list[str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Splits dialog IDs deterministically into train and val subsets."""
    unique = sorted(set(dialog_ids))
    rng = random.Random(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def convert_file(
    input_path: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Streams an AReaL SFT jsonl into train/val prompt-completion JSON files.

    Returns a manifest with counts, split sizes, and drop statistics.
    """
    stats: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    dialogs: list[str] = []

    for row in iter_jsonl(input_path):
        stats["rows_seen"] += 1
        example = convert_row(row, stats)
        if example is None:
            continue
        examples.append(example)
        dialogs.append(example["dialog_id"])

    train_dialogs, val_dialogs = split_dialog_ids(dialogs, train_ratio, seed)
    train_set, val_set = set(train_dialogs), set(val_dialogs)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    with open(out_path / "areal_sft_train.json", "w", encoding="utf-8") as f:
        for ex in (e for e in examples if e["dialog_id"] in train_set):
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
            counts["train"] += 1

    with open(out_path / "areal_sft_val.json", "w", encoding="utf-8") as f:
        for ex in (e for e in examples if e["dialog_id"] in val_set):
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
            counts["val"] += 1

    manifest = {
        "input": str(input_path),
        "train_examples": counts["train"],
        "val_examples": counts["val"],
        "train_dialogs": len(train_dialogs),
        "val_dialogs": len(val_dialogs),
        "stats": dict(stats),
    }
    with open(out_path / "conversion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert-areal",
        description="Converts AReaL-tau2-data SFT jsonl into prompt/completion records.",
    )
    parser.add_argument("--input", required=True, help="Path to tau2_sft_train.jsonl")
    parser.add_argument("--out-dir", default="artifacts/data")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = convert_file(args.input, args.out_dir, args.train_ratio, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
