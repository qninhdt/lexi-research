"""Audits n-gram overlap between AReaL SFT dialogs and official tau2 test tasks.

The AReaL dataset card does not commit to zero semantic overlap with the
official evaluation set. Before training on AReaL data, this audit quantifies
8-gram overlap between each AReaL retail dialog and each official retail test
task scenario so any suspicious pair can be reviewed or excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RETAIL_TASKS_PATH = "third_party/tau2-bench/data/tau2/domains/retail/tasks.json"
SPLIT_PATH = "third_party/tau2-bench/data/tau2/domains/retail/split_tasks.json"


def _norm_tokens(text: str) -> list[str]:
    return [t.strip().lower() for t in text.split() if t.strip()]


def ngram_set(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """Returns the set of n-grams of a text after lowercasing."""
    toks = _norm_tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def scenario_text(task: dict[str, Any]) -> str:
    """Extracts comparable text from an official task's user scenario."""
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions")
    if isinstance(instructions, dict):
        parts = [
            str(instructions.get(k) or "")
            for k in ("reason_for_call", "known_info", "task_instructions")
        ]
        return " ".join(p for p in parts if p)
    return str(instructions or "")


def dialog_texts_from_areal(path: str | Path) -> dict[str, str]:
    """Maps AReaL dialog id -> concatenated scenario text from its rows."""
    from tau_research.data.load_areal_sft import iter_jsonl

    texts: dict[str, str] = {}
    for row in iter_jsonl(path):
        meta = row.get("metadata") or {}
        dialog_id = str(meta.get("source_dialog_id", ""))
        if not dialog_id.startswith("retail_"):
            continue
        reason = str(meta.get("reason_for_call") or "")
        first_user = ""
        for m in row.get("messages") or []:
            if m.get("role") == "user":
                first_user = str(m.get("content") or "")
                break
        texts.setdefault(dialog_id, " ".join([reason, first_user]).strip())
    return texts


def official_test_texts() -> dict[str, str]:
    """Maps official retail test task id -> scenario text."""
    with open(SPLIT_PATH, encoding="utf-8") as f:
        splits = json.load(f)
    with open(RETAIL_TASKS_PATH, encoding="utf-8") as f:
        tasks = {t["id"]: t for t in json.load(f)}

    texts: dict[str, str] = {}
    for tid in splits["test"]:
        task = tasks.get(tid)
        if task is not None:
            texts[tid] = scenario_text(task)
    return texts


def run_audit(
    areal_path: str | Path,
    ngram_n: int = 8,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Computes pairwise n-gram overlap and returns a structured report."""
    areal = dialog_texts_from_areal(areal_path)
    official = official_test_texts()
    areal_grams = {k: ngram_set(v, ngram_n) for k, v in areal.items()}
    official_grams = {k: ngram_set(v, ngram_n) for k, v in official.items()}

    flagged = []
    best_for_dialog: dict[str, tuple[float, str]] = {}
    for dialog_id, dg in areal_grams.items():
        for task_id, tg in official_grams.items():
            score = jaccard(dg, tg)
            if not best_for_dialog.get(dialog_id) or score > best_for_dialog[dialog_id][0]:
                best_for_dialog[dialog_id] = (score, task_id)
            if score >= threshold:
                flagged.append(
                    {"areal_dialog": dialog_id, "test_task": task_id, "jaccard": round(score, 4)}
                )

    return {
        "ngram_n": ngram_n,
        "threshold": threshold,
        "areal_dialogs_audited": len(areal_grams),
        "official_test_tasks": len(official_grams),
        "flagged_pairs": sorted(flagged, key=lambda x: -x["jaccard"]),
        "max_overlap_per_dialog": {
            k: {"jaccard": round(v[0], 4), "test_task": v[1]}
            for k, v in sorted(best_for_dialog.items(), key=lambda kv: -kv[1][0])[:20]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audit-decontamination",
        description="Audits AReaL vs official test split n-gram overlap.",
    )
    parser.add_argument("--input", required=True, help="Path to tau2_sft_train.jsonl")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", default="artifacts/evaluation/decontamination_report.json")
    args = parser.parse_args()

    report = run_audit(args.input, threshold=args.threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "flagged_pairs"}, indent=2))
    print(f"flagged_pairs={len(report['flagged_pairs'])} -> {out}")


if __name__ == "__main__":
    main()
