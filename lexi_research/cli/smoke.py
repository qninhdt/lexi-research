"""`lexi smoke` — the acceptance gate.

One command that has to be green before any real experiment. On CPU it trains a
randomly-initialised model of a few thousand parameters for two optimiser steps
over a 50-row fixture: fast enough for CI, and it exercises the parts that
silently go wrong — chat templating, the completion-only mask, LoRA target
resolution, the collator, and the config path every hyperparameter travels.

It downloads nothing. The tokenizer is built from the fixture at run time, so CI
needs no network and never touches Cambridge-derived data.

`--gpu` repeats it against the checkpoint named in `train.base_model` to answer
the question a tiny stand-in cannot: whether PEFT and the quantiser actually
attach to that architecture's modules.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lexi_research.format import BandConfig, default_config_path, group_of
from lexi_research.format.parser import ParseOk, parse_correction
from lexi_research.format.tags import TagGroup
from lexi_research.format.validate import ValidationError, validate_output

from .config import Config, default_params_path

#: Matched the way `transformers` matches it, so stripping it from the template
#: below produces exactly the shape a published template has.
_GENERATION_BLOCK = re.compile(r"\{\%-?\s*(?:end)?generation\s*-?\%\}")

#: A Qwen-shaped template. The `{% generation %}` block is what lets the
#: tokenizer report which tokens the assistant produced, which is the primary
#: path the collator masks on.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n"
    "{% if enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% if enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)

SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


class SmokeFailure(RuntimeError):
    """The gate found something that would have wasted a real run."""


def check_fixture(rows: Sequence[Mapping[str, Any]]) -> str:
    """Every row must pass the six checks, and the set must span the contract.

    The fixture is what the gate trains on, so a defect in it reads as a defect
    in the trainer. Its coverage is asserted here rather than described in a
    comment that can drift.
    """
    if not rows:
        raise SmokeFailure("the fixture is empty")

    config = BandConfig.from_json(default_config_path())
    meanings: set[int] = set()
    groups: set[TagGroup] = set()
    tags: set[str] = set()
    unparseable = 0
    multiword = 0
    verbatim = 0

    for index, row in enumerate(rows):
        payload = {key: row[key] for key in ("correction", "meaning", "feedback")}
        result = validate_output(payload, str(row["text"]), config)
        if isinstance(result, ValidationError):
            raise SmokeFailure(f"fixture row {index}: {result.code}: {result.detail}")

        meanings.add(int(row["meaning"]))
        if row["correction"] is None:
            unparseable += 1
        else:
            parsed = parse_correction(str(row["correction"]))
            assert isinstance(parsed, ParseOk)  # validate_output already proved this
            if not parsed.edits:
                verbatim += 1
            for edit in parsed.edits:
                tags.add(edit.tag)
                groups.add(group_of(edit.tag))
        if " " in str(row["target"]):
            multiword += 1

    missing_bands = set(range(5)) - meanings
    if missing_bands:
        raise SmokeFailure(f"fixture covers no row at meaning band {sorted(missing_bands)}")
    missing_groups = set(TagGroup) - groups
    if missing_groups:
        raise SmokeFailure(f"fixture has no edit in tag group {sorted(missing_groups)}")
    if not unparseable:
        raise SmokeFailure("fixture has no row with a null correction")
    if not multiword:
        raise SmokeFailure("fixture has no multiword target")
    if not verbatim:
        raise SmokeFailure("fixture has no clean sentence re-emitted verbatim")

    return (
        f"{len(rows)} rows, bands {sorted(meanings)}, {len(tags)}/16 tags, "
        f"{unparseable} unparseable, {verbatim} verbatim, {multiword} multiword"
    )


def _corpus(rows: Sequence[Mapping[str, Any]]) -> Iterator[str]:
    """Text the throwaway tokenizer is fitted on: the prompts it will encode."""
    from lexi_research.train.collate import completion_text, training_messages

    for row in rows:
        for message in training_messages(row, nonce="smoke"):
            yield message["content"]
        yield completion_text(row)


def build_tiny_tokenizer(rows: Sequence[Mapping[str, Any]], *, model_max_length: int) -> Any:
    """A word-level tokenizer fitted on the fixture. No vocabulary is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        _corpus(rows),
        trainers.WordLevelTrainer(special_tokens=["<pad>", "<unk>", *SPECIAL_TOKENS]),
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        model_max_length=model_max_length,
    )
    # Registered as added tokens too, or the whitespace pre-tokenizer would tear
    # `<|im_start|>` into pieces before the vocabulary ever saw it.
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_tiny_model(config: Config, tokenizer: Any, *, max_seq_len: int) -> Any:
    """A randomly-initialised stack of the architecture named in `smoke.architecture`."""
    import transformers

    architecture = config.get_str("smoke.architecture")
    try:
        model_config = transformers.AutoConfig.for_model(
            architecture,
            vocab_size=len(tokenizer),
            hidden_size=config.get_int("smoke.hidden_size"),
            intermediate_size=config.get_int("smoke.intermediate_size"),
            num_hidden_layers=config.get_int("smoke.layers"),
            num_attention_heads=config.get_int("smoke.heads"),
            num_key_value_heads=config.get_int("smoke.kv_heads"),
            max_position_embeddings=max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
        )
    except (KeyError, ValueError) as exc:
        raise SmokeFailure(
            f"smoke.architecture={architecture!r} is not a model type this "
            f"transformers install knows: {exc}"
        ) from exc
    return transformers.AutoModelForCausalLM.from_config(model_config)


def check_masking_paths(rows: Sequence[Mapping[str, Any]]) -> str:
    """Both label paths, over a real tokenizer, before any training happens.

    The gate is worth little if it only exercises the path the fixture's own
    template happens to take. Most published chat templates carry no
    `{% generation %}` block, so the concatenation path is what a real run uses —
    and an earlier version of the collator raised on exactly that, invisibly,
    because the gate only ever ran the other one.
    """
    from lexi_research.train.collate import IGNORE_INDEX, build_example

    counts: dict[str, int] = {}
    for label, template in (
        ("template mask", CHAT_TEMPLATE),
        ("concatenation", _GENERATION_BLOCK.sub("", CHAT_TEMPLATE)),
    ):
        tokenizer = build_tiny_tokenizer(rows, model_max_length=100_000)
        tokenizer.chat_template = template
        example = build_example(tokenizer, rows[0], nonce="smoke")
        supervised = sum(1 for token in example.labels if token != IGNORE_INDEX)
        if not supervised:
            raise SmokeFailure(f"the {label} path supervised no tokens")
        if supervised >= len(example.labels):
            raise SmokeFailure(f"the {label} path supervised the prompt as well")
        counts[label] = supervised

    if len(set(counts.values())) != 1:
        raise SmokeFailure(f"the two label paths disagree on the answer length: {counts}")
    return ", ".join(f"{label} {count} tokens" for label, count in counts.items())


def check_data_stages(
    config: Config, rows: Sequence[Mapping[str, Any]], workdir: Path
) -> tuple[str, Path]:
    """Run validate/balance/split and calibration over the fixture.

    These stages are pure and cheap, and their failure modes are quiet: a
    validator that rejects everything, a split that leaks a target word across
    train and test, a calibration that collapses every row into one band. Running
    them here is what stops those from being discovered against real data on a
    rented GPU.

    Returns the report line and the calibrated band config the run then trains
    against.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lexi_research.data.stages import run_calibrate, run_process

    texts, labels = [], []
    for index, row in enumerate(rows):
        uid = f"smoke-{index:04d}"
        texts.append(
            {
                "req_uid": uid,
                "sense_uid": f"sense-{row['target']}",
                "target": str(row["target"]),
                # The split groups by target word, so it must be present and
                # normalised the way the export stage normalises it.
                "target_norm": str(row["target"]).lower(),
                "pos": str(row["pos"]),
                "definition": str(row["definition"]),
                "text": str(row["text"]),
                "error_spec": "none",
            }
        )
        labels.append(
            {
                "req_uid": uid,
                "correction": row["correction"],
                "meaning": int(row["meaning"]),
                "feedback": str(row["feedback"]),
            }
        )

    raw = workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(texts), raw / "raw_texts.parquet")
    pq.write_table(pa.Table.from_pylist(labels), raw / "raw_labels.parquet")

    clean = workdir / "clean"
    processed = run_process(
        config,
        texts=raw / "raw_texts.parquet",
        labels=raw / "raw_labels.parquet",
        out=clean,
    )
    if not processed["clean_rows"]:
        raise SmokeFailure("validation rejected every row of a fixture it should accept")

    band_config = clean / "band_config.json"
    calibrated = run_calibrate(config, rows_path=clean / "processed.parquet", out=band_config)
    if len(set(calibrated["thresholds"])) < 2:
        raise SmokeFailure(
            f"calibration collapsed to {calibrated['thresholds']}, so every penalty "
            "lands in one band"
        )

    return (
        f"{processed['clean_rows']} clean, {processed['rejected_rows']} rejected, "
        f"splits {processed['contamination']}, band config v{calibrated['version']}",
        band_config,
    )


def run_smoke(config: Config, *, gpu: bool = False) -> int:
    """Run the gate. Returns 0 only if every stage that exists succeeded."""
    from lexi_research.train.trainer import load_rows, train_sft

    fixture = Path(config.get_str("smoke.fixture"))
    if not fixture.is_absolute():
        # Repo-relative, like every other path in params.yaml, so the gate works
        # from a subdirectory rather than only from the root.
        fixture = default_params_path().parent / fixture
    rows = load_rows(fixture)
    print(f"fixture — {check_fixture(rows)}", flush=True)
    print(f"masking — {check_masking_paths(rows)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="lexi-smoke-") as tmp:
        workdir = Path(tmp)
        summary, _ = check_data_stages(config, rows, workdir)
        print(f"data — {summary}", flush=True)

        if gpu:
            run_config = config.with_overrides(["train.epochs=1"])
            model = tokenizer = None
        else:
            run_config = config.with_overrides(
                [
                    f"train.max_steps={config.get_int('smoke.steps')}",
                    "train.load_in_4bit=false",
                    "train.gradient_checkpointing=false",
                    f"train.per_device_batch_size={min(2, len(rows))}",
                    "train.grad_accum=1",
                ]
            )
            max_seq_len = run_config.get_int("train.max_seq_len")
            tokenizer = build_tiny_tokenizer(rows, model_max_length=max_seq_len)
            model = build_tiny_model(run_config, tokenizer, max_seq_len=max_seq_len)

        result = train_sft(
            run_config,
            train_path=fixture,
            output_dir=workdir / "adapter",
            model=model,
            tokenizer=tokenizer,
        )
        if result.steps < 1:
            raise SmokeFailure("training reported no optimiser steps")
        if not (workdir / "adapter").exists():
            raise SmokeFailure("training saved no adapter")
        print(f"train — {result.summary()}", flush=True)

        if not gpu:
            print(f"rl — {check_rl_tracks(run_config, model, tokenizer, workdir)}", flush=True)

    print("smoke — ok", flush=True)
    return 0


def check_rl_tracks(config: Config, model: Any, tokenizer: Any, workdir: Path) -> str:
    """Two steps of every RL track, on the same tiny model.

    All three share one loop and differ only in `compute_reward`, so running all
    three here is what keeps that true: a change that breaks the shared mask, the
    baseline, or the advantage normalisation fails on the CPU gate rather than
    after a GPU is rented.
    """
    from lexi_research.format import BandConfig, default_config_path
    from lexi_research.rl.base import ALGORITHMS
    from lexi_research.rl.trainer import train_rl

    band_config = BandConfig.from_json(default_config_path())
    summaries = []
    for algorithm in ALGORITHMS:
        run_config = config.with_overrides(
            [
                f"rl.algo={algorithm}",
                "rl.group_size=2",
                "rl.max_reasoning_tokens=8",
                "eval.max_new_tokens=8",
            ]
        )
        result = train_rl(
            run_config,
            train_path=config.get_str("smoke.fixture"),
            output_dir=workdir / f"rl-{algorithm}",
            band_config=band_config,
            model=model,
            tokenizer=tokenizer,
            max_steps=config.get_int("smoke.steps"),
        )
        if result.steps < 1:
            raise SmokeFailure(f"{algorithm} reported no optimiser steps")
        if not result.rollouts:
            raise SmokeFailure(f"{algorithm} sampled no rollouts")
        summaries.append(f"{algorithm} {result.steps} steps/{result.rollouts} rollouts")
    return ", ".join(summaries)


__all__ = [
    "CHAT_TEMPLATE",
    "SmokeFailure",
    "build_tiny_model",
    "build_tiny_tokenizer",
    "check_fixture",
    "check_masking_paths",
    "run_smoke",
]
