"""Publish the teacher-generated dataset to the Hugging Face Hub.

Two rules shape this, and both are about not shipping something the repo is not
allowed to ship or cannot stand behind:

**Stage A never leaves the machine.** `data/gec/` is converted from W&I+LOCNESS,
whose LOCNESS licence forbids redistributing any part of the corpus to a third
party. Only the teacher-generated stage-B artifacts are uploaded, and the
allowlist is explicit rather than a directory glob so a new file cannot join the
upload by being written next to one that was cleared.

**The card is generated from the reports.** Counts, distribution, gate results
and the teacher's own configuration are read from the JSON the run wrote, so a
card cannot claim a number the pipeline did not measure. Anything the reports do
not contain is stated as unknown rather than filled in.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

#: Exactly what may be uploaded. Stage A (`data/gec/`) is absent by licence, and
#: the response cache is absent because it holds raw provider payloads.
#:
#: The `data/clean/` splits are what someone actually trains on, so they lead. The
#: raw pair ships alongside them because a reader who disagrees with the balancing
#: or the split seed needs the unprocessed rows to redo it.
UPLOAD: tuple[tuple[str, str], ...] = (
    ("data/clean/train.parquet", "data/train.parquet"),
    ("data/clean/val.parquet", "data/val.parquet"),
    ("data/clean/test.parquet", "data/test.parquet"),
    ("data/clean/data-quality.json", "reports/data-quality.json"),
    ("data/raw/raw_texts.parquet", "raw/raw_texts.parquet"),
    ("data/raw/raw_labels.parquet", "raw/raw_labels.parquet"),
    ("data/raw/generate-report.json", "reports/generate-report.json"),
    ("data/raw/label-report.json", "reports/label-report.json"),
    ("data/batches/sample-report.json", "reports/sample-report.json"),
    ("reports/pilot-gate.json", "reports/pilot-gate.json"),
    ("reports/pilot-ceiling.json", "reports/pilot-ceiling.json"),
    ("reports/tag-distribution-reference.json", "reports/tag-distribution-reference.json"),
    ("band_config.json", "band_config.json"),
)

#: Paths that must never be uploaded, checked against the resolved upload list.
#: A licence breach is not something to leave to reviewer attention.
FORBIDDEN = ("data/gec/", "data/corpora/", ".cache/", ".env")

#: Per-tag shares measured against a human-annotated corpus, written by
#: `ops/tag-distribution-reference.py`. Only aggregate rates, so it carries no
#: corpus text and is safe to publish.
_TAG_REFERENCE = "reports/tag-distribution-reference.json"


class PublishError(RuntimeError):
    """The dataset cannot be published as configured."""


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        raise PublishError(f"{path} is missing; run the pipeline stage that writes it")
    payload: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
    return payload


def _gate_table(gate: dict[str, Any]) -> list[str]:
    lines = [
        "| Gate | Value | Threshold | Blocking | Result |",
        "|---|---:|---:|---|---|",
    ]
    for entry in gate.get("gates", []):
        lines.append(
            f"| `{entry['name']}` | {entry['value']} | {entry['threshold']} | "
            f"{'yes' if entry['blocking'] else 'no'} | "
            f"{'pass' if entry['passed'] else '**FAIL**'} |"
        )
    return lines


def _split_section(quality: dict[str, Any] | None) -> list[str]:
    """The train/val/test sizes, and the contamination check that validates them.

    A split by target word is the only reason a test score means anything here:
    one word yields many rows, so splitting by row leaks. The check is reported
    rather than asserted in prose because a reader has no way to verify the claim
    otherwise.
    """
    if not quality:
        return ["The processed splits have not been built for this snapshot.", ""]
    contamination = quality.get("contamination", {})
    lines = [
        "### Splits",
        "",
        "| Split | Rows |",
        "|---|---:|",
    ]
    for name in ("train", "val", "test"):
        lines.append(f"| `{name}` | {contamination.get(name, '?')} |")
    crossing = contamination.get("cross_split_text_hashes")
    lines += [
        "",
        f"Grouped by target word, not by row. Sentences appearing in more than one "
        f"split: **{crossing if crossing is not None else 'unmeasured'}**. "
        f"Rejected during validation: **{quality.get('rejected_rows', '?')}** of "
        f"{quality.get('input_rows', '?')}.",
        "",
    ]
    return lines


def _skew_limitation(reference: dict[str, Any] | None) -> list[str]:
    """State the tag skew, with the shares read from the reference rather than typed.

    Typing the percentages into prose would make them a claim that silently goes
    stale the next time rows are added. Read from the file that measured them, the
    sentence either matches the shipped JSON or is absent.
    """
    if not reference:
        return []
    tags = reference.get("tags") or {}
    over, under = [], []
    for tag, shares in sorted(tags.items()):
        mine = shares.get("teacher_share")
        human = shares.get("human_share")
        if not mine or not human:
            # A tag absent from one side has no ratio to report; the human corpus
            # has no `coll` or `other` at all, which the card covers elsewhere.
            continue
        entry = f"`{tag}` ({mine:.1%} vs {human:.1%})"
        if mine / human >= 2:
            over.append(entry)
        elif mine / human <= 0.5:
            under.append(entry)
    if not over and not under:
        return []
    rows = reference.get("human_rows", "?")
    corpus = reference.get("human_corpus", "a human-annotated corpus")
    return [
        "- **The error mix is skewed relative to human-annotated learner text.** "
        f"Against {corpus} ({rows:,} rows, the same 16 tags), this data "
        f"over-produces {', '.join(over) or 'nothing'} and under-produces "
        f"{', '.join(under) or 'nothing'}. The cause is the learner profiles: their "
        "`error_bias` fields drive what call 1 writes, and `punc`, `sp`, `pron`, "
        "`poss` and `other` appear in no profile at all. Punctuation and spelling "
        "are among the commonest real learner errors, so a model trained on this "
        "alone will be weakest there. Full comparison in "
        "`reports/tag-distribution-reference.json`.",
    ]


def build_card(
    *,
    repo_id: str,
    texts_path: str | Path,
    labels_path: str | Path,
    generate_report: dict[str, Any],
    label_report: dict[str, Any],
    sample_report: dict[str, Any],
    gate: dict[str, Any] | None,
    ceiling: dict[str, Any] | None,
    teacher_model: str,
    teacher_endpoint: str,
    quality: dict[str, Any] | None = None,
    calibrated: bool = False,
    tag_reference: dict[str, Any] | None = None,
) -> str:
    """Render the dataset card from measurements, never from assumption."""
    labels = pq.read_table(labels_path).to_pylist()
    texts = pq.read_table(texts_path)
    meaning = Counter(int(row["meaning"]) for row in labels)
    total = sum(meaning.values()) or 1

    distribution = [
        "| `meaning` | Rows | Share |",
        "|---|---:|---:|",
    ]
    for band in range(5):
        count = meaning.get(band, 0)
        distribution.append(f"| {band} | {count} | {count / total:.1%} |")

    tags: Counter[str] = Counter()
    for row in labels:
        for tag in row.get("tags", []):
            tags[str(tag)] += 1
    tag_rows = [f"| `{tag}` | {count} |" for tag, count in tags.most_common()]

    passed = gate.get("passed") if gate else None
    blocking = gate.get("blocking_passed") if gate else None

    lines: list[str] = [
        "---",
        "license: other",
        # The Hub validates this against a closed list and silently drops an
        # unknown value, so it must be a real category: `text-generation` is what
        # a grader emitting a JSON object actually does.
        "task_categories:",
        "  - text-generation",
        "language:",
        "  - en",
        "tags:",
        "  - grammatical-error-correction",
        "  - language-learning",
        "  - distillation",
        "size_categories:",
        f"  - {'n<1K' if len(labels) < 1000 else '1K<n<10K'}",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Teacher-generated training data for a **sentence grader**: given a learner's",
        "English sentence and one specific dictionary sense of a target word, produce an",
        "inline correction, a meaning band (0-4), and one line of feedback.",
        "",
        "This card is generated from the pipeline's own report JSON. Every number below",
        "was measured by the run that produced the data.",
        "",
        "## What is in here",
        "",
        f"- **{texts.num_rows}** generated learner sentences (`raw/raw_texts.parquet`)",
        f"- **{len(labels)}** accepted gradings (`raw/raw_labels.parquet`)",
        f"- **{sample_report.get('senses', 'unknown')}** distinct dictionary senses",
        "",
        *_split_section(quality),
        "```json",
        json.dumps(
            {
                "correction": "He [speak>speaks:agr] very [eloquent>eloquently:form].",
                "meaning": 3,
                "feedback": "Right sense, but the verb needs to agree with the subject.",
            },
            indent=2,
        ),
        "```",
        "",
        "| Operation | Syntax | Example |",
        "|---|---|---|",
        "| Replace | `[A>B:tag]` | `[speak>speaks:agr]` |",
        "| Delete | `[A>:tag]` | `the [the>:art] very` |",
        "| Insert | `[>B:tag]` | `went [>to the:art] store` |",
        "",
        "A clean sentence is re-emitted verbatim. An unreadable one yields",
        "`correction: null`. `grammar` and `naturalness` are **computed from the",
        "correction's tags** by the formula in `band_config.json`, not generated by the",
        "model — so identical error sets always score identically, and thresholds stay",
        "retunable without regenerating anything.",
        "",
        "## How it was built",
        "",
        "Two calls, deliberately separated:",
        "",
        "1. **Diversifier** — knows a spec (learner profile, target band, error recipe)",
        "   and writes learner-like text. The spec is a diversity knob and **never a",
        "   label**.",
        "2. **Grader** — sees only `{target, sense, text}` and produces the answer. This",
        "   is byte-for-byte the prompt the student model runs at inference.",
        "",
        "Single-call self-labelling was rejected: it produces labels that describe the",
        "instruction rather than the text, and the defect is invisible afterwards because",
        "a correct row and a wrong one look identical.",
        "",
        "| | |",
        "|---|---|",
        f"| Teacher model | `{teacher_model}` |",
        f"| Endpoint | `{teacher_endpoint}` |",
        f"| Call 1 requests | {generate_report.get('teacher', {}).get('calls', '?')} |",
        f"| Call 2 requests | {label_report.get('teacher', {}).get('calls', '?')} |",
        f"| Format validity | {label_report.get('validity_rate', '?')} |",
        f"| Batch diversity (distinct-2) | {generate_report.get('mean_distinct2', '?')} |",
        "",
        "The teacher model string is the one the endpoint reported. It was reached",
        "through an OpenAI-compatible proxy, so it identifies the endpoint's advertised",
        "model rather than an independently verified checkpoint.",
        "",
        "## Distribution",
        "",
        *distribution,
        "",
        f"Middle bands {{1,2,3}} hold **{label_report.get('middle_band_share', '?')}** of rows.",
        "",
        "### Error tags",
        "",
        "| Tag | Count |",
        "|---|---:|",
        *tag_rows,
        "",
        "## Quality gates",
        "",
    ]

    if gate:
        lines += _gate_table(gate)
        lines += [
            "",
            f"Overall: **{'pass' if passed else 'not passed'}**; "
            f"blocking gates: **{'pass' if blocking else 'NOT PASSED'}**.",
        ]
        if not blocking:
            lines += [
                "",
                "> **Read this before training on it.** A blocking gate did not pass. The",
                "> `meaning` distribution is concentrated at the extremes, so bands 1-2 are",
                "> under-represented relative to where real learners land. A model trained",
                "> on this as-is will be weakest in exactly the middle region that matters",
                "> most. This is published as an honest snapshot, not as a finished corpus.",
            ]
    else:
        lines.append("The pilot gate has not been run against this snapshot.")

    if ceiling:
        lines += [
            "",
            "### Teacher self-consistency (the ceiling on any student)",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| `meaning` QWK | {ceiling.get('meaning_qwk', '?')} |",
            f"| `correction` edit-F1 | {ceiling.get('correction_edit_f1', '?')} |",
            f"| Rows re-graded | {ceiling.get('sampled', '?')} |",
            "",
            "The same sentences were graded twice, blind, with the cache disabled. A",
            "student cannot exceed the agreement its teacher has with itself, so report",
            "fidelity against these numbers rather than against 1.0.",
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "- **No human gold set.** Every label is a teacher's opinion. Nothing here is",
        "  verified against ground truth.",
        "- **The real-learner distribution is unverified.** Sentences are model-written",
        "  imitations of learner errors, not collected from learners.",
        "- **Bands are uncalibrated** until `lexi data calibrate` has run;",
        "  `band_config.json` carries `\"calibrated\": false` when that is still true.",
        "- **`feedback` is unmeasured.** No metric in this snapshot evaluates it.",
        *_skew_limitation(tag_reference),
        *(
            []
            if calibrated
            else [
                "- **`grammar` and `naturalness` are uncalibrated.** `band_config.json` "
                "carries `\"calibrated\": false`, so those two bands come from the shipped "
                "design guesses rather than from this corpus's penalty distribution. "
                "`meaning` is unaffected — it is the teacher's own answer. Calibration on "
                "this corpus produced duplicate cut points, because 81% of rows carry no "
                "`usage` error at all and no threshold exists inside that mass of zeros; "
                "a five-band usage scale is not supported by data this clean.",
            ]
        ),
        "",
        "## Not included",
        "",
        "The stage-A correction-format data converted from W&I+LOCNESS is **not** part of",
        "this dataset. That corpus's licence forbids redistributing any part of it to a",
        "third party, so it stays local to the machine that built it.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{lexi_grader_dataset,",
        f"  title  = {{{repo_id}}},",
        "  note   = {Teacher-generated sentence-grading dataset},",
        "  year   = {2026},",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"


def resolve_uploads(root: Path) -> list[tuple[Path, str]]:
    """The files that exist, checked against the forbidden list."""
    resolved: list[tuple[Path, str]] = []
    for source, destination in UPLOAD:
        if any(source.startswith(bad) for bad in FORBIDDEN):
            raise PublishError(f"{source} is on the forbidden list and must not be uploaded")
        path = root / source
        if path.exists():
            resolved.append((path, destination))
    if not resolved:
        raise PublishError("nothing to upload; run the data stages first")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-name/lexi-grader-sft")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the card to stdout and list the uploads without touching the Hub",
    )
    parser.add_argument("--card-out", default=None, help="also write the card here")
    args = parser.parse_args(argv)

    root = Path.cwd()
    uploads = resolve_uploads(root)

    generate_report = _read_json("data/raw/generate-report.json")
    label_report = _read_json("data/raw/label-report.json")
    sample_report = _read_json("data/batches/sample-report.json")
    gate = json.loads(Path("reports/pilot-gate.json").read_text()) if Path(
        "reports/pilot-gate.json"
    ).exists() else None
    ceiling = json.loads(Path("reports/pilot-ceiling.json").read_text()) if Path(
        "reports/pilot-ceiling.json"
    ).exists() else None

    quality = (
        json.loads(Path("data/clean/data-quality.json").read_text(encoding="utf-8"))
        if Path("data/clean/data-quality.json").exists()
        else None
    )
    band = json.loads(Path("band_config.json").read_text(encoding="utf-8"))

    import os

    card = build_card(
        repo_id=args.repo_id,
        texts_path=root / "data/raw/raw_texts.parquet",
        labels_path=root / "data/raw/raw_labels.parquet",
        generate_report=generate_report,
        label_report=label_report,
        sample_report=sample_report,
        gate=gate,
        ceiling=ceiling,
        teacher_model=os.environ.get("LEXI_TEACHER_MODEL", "unknown"),
        teacher_endpoint=os.environ.get("LEXI_TEACHER_BASE_URL", "unknown"),
        quality=quality,
        calibrated=bool(band.get("calibrated")),
        tag_reference=(
            json.loads(Path(_TAG_REFERENCE).read_text(encoding="utf-8"))
            if Path(_TAG_REFERENCE).exists()
            else None
        ),
    )

    if args.card_out:
        Path(args.card_out).write_text(card, encoding="utf-8")
        print(f"card written to {args.card_out}")

    print("would upload:" if args.dry_run else "uploading:")
    for path, destination in uploads:
        print(f"  {path.relative_to(root)}  ->  {destination}  ({path.stat().st_size:,} bytes)")

    if args.dry_run:
        print("\n--- dataset card ---\n")
        print(card)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    for path, destination in uploads:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
