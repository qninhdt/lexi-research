"""SFT training. Every hyperparameter comes from config; nothing names a model.

The trainer this replaces had four defects that could not be diagnosed while the
others were present: the wrong auto class, hand-joined training text instead of
the chat template, loss over the prompt as well as the answer, and a hardcoded
LoRA target list. The first is fixed by asking the checkpoint which class it is,
the middle two by `collate`, the last by `modules`.

Heavy imports stay inside the functions that need them, so importing this module
— and therefore the CLI — costs nothing on a CPU-only machine with no training
stack installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lexi_research.cli.config import Config

from .callbacks import build_progress_callback, resolve_resume

from .collate import IGNORE_INDEX, Example, SequenceTooLong, build_example
from .collate_corrector import build_corrector_example
from .modules import Layout, TargetResolution, resolve_target_modules

#: The two supervised tasks. `grader` is stage B — the three-field answer behind
#: the prompt that is actually served. `corrector` is stage A, which learns the
#: markup and the tag set from a human-annotated learner corpus that has no
#: target sense and therefore no `meaning` and no `feedback`.
TASKS = ("grader", "corrector")


class TrainerSetupError(RuntimeError):
    """The environment, the checkpoint, or the data made training impossible."""


@dataclass(frozen=True)
class TrainResult:
    """What a run produced, for the caller to log and for the smoke gate to assert."""

    output_dir: Path
    examples: int
    dropped: int
    steps: int
    targets: TargetResolution

    def summary(self) -> str:
        dropped = f", {self.dropped} dropped over max_seq_len" if self.dropped else ""
        return (
            f"{self.examples} examples{dropped}; LoRA {self.targets.summary()}; "
            f"{self.steps} optimiser steps -> {self.output_dir}"
        )


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read training rows from parquet or JSONL."""
    resolved = Path(path)
    if not resolved.exists():
        raise TrainerSetupError(f"training data {resolved} does not exist")
    if resolved.suffix == ".jsonl":
        with resolved.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(resolved).to_pylist()
    return rows


def build_examples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_seq_len: int,
    thinking: str,
    completion_only: bool,
    task: str = "grader",
    max_drop_fraction: float = 0.02,
) -> tuple[list[Example], int]:
    """Tokenise every row, dropping and counting the ones that do not fit.

    Dropping is length-biased — short learner sentences and short feedback
    survive — so past a threshold the run trains on a different distribution than
    the one it claims to. That is a wrong answer rather than a slow one, so it
    raises instead of warning.

    `task` selects the collator. `grader` is the three-field answer behind the
    served prompt; `corrector` is stage A, where the corpus supplies a correction
    and nothing else. They are separate functions rather than a flag inside one
    because the grader path is the inference contract, and a training-only branch
    inside it is how that contract drifts.
    """
    if task not in TASKS:
        raise TrainerSetupError(f"train.task={task!r}; expected one of {list(TASKS)}")

    examples: list[Example] = []
    dropped = 0
    for row in rows:
        try:
            if task == "corrector":
                examples.append(
                    build_corrector_example(
                        tokenizer,
                        row,
                        thinking=thinking,
                        max_seq_len=max_seq_len,
                    )
                )
                continue
            examples.append(
                build_example(
                    tokenizer,
                    row,
                    thinking=thinking,
                    completion_only=completion_only,
                    max_seq_len=max_seq_len,
                )
            )
        except SequenceTooLong:
            dropped += 1
    if not examples:
        raise TrainerSetupError(f"every one of {len(rows)} rows exceeded max_seq_len={max_seq_len}")
    fraction = dropped / len(rows)
    if fraction > max_drop_fraction:
        raise TrainerSetupError(
            f"{dropped} of {len(rows)} rows ({fraction:.1%}) exceeded "
            f"max_seq_len={max_seq_len}, over the {max_drop_fraction:.1%} ceiling. "
            "Training on what is left would be training on the short rows."
        )
    return examples, dropped


def resolve_model_class(base_model: str) -> tuple[Any, Any]:
    """Ask the checkpoint which class it is, rather than assuming one.

    A config's `architectures` names the exact class, which is the only
    model-agnostic way to load a checkpoint whose head is not a plain causal LM.
    `AutoModelForCausalLM` is the fallback for checkpoints that omit it.
    """
    import transformers

    try:
        model_config = transformers.AutoConfig.from_pretrained(base_model)
    except Exception as exc:  # noqa: BLE001 - hub, network and parse errors all land here
        raise TrainerSetupError(f"could not read the config of {base_model!r}: {exc}") from exc

    for name in getattr(model_config, "architectures", None) or []:
        model_class = getattr(transformers, str(name), None)
        if model_class is not None and hasattr(model_class, "from_pretrained"):
            return model_class, model_config

    if not hasattr(transformers, "AutoModelForCausalLM"):  # pragma: no cover - defensive
        raise TrainerSetupError("this transformers install has no AutoModelForCausalLM")
    return transformers.AutoModelForCausalLM, model_config


def _drop_unused_multimodal_towers(model: Any) -> tuple[str, ...]:
    """Remove media towers from a text-only fine-tune after checkpoint loading.

    Gemma 4 E4B and Qwen3.5-4B are multimodal wrappers. Their text forward path
    never touches the vision/audio modules when the batch contains only text,
    but keeping those modules registered still consumes VRAM and makes PEFT's
    module scan needlessly large. The checkpoint must be loaded first because
    the media weights are part of the published state dict; removing them after
    loading preserves the language-model weights while making the hot path text
    only. A caller that needs images must leave ``text_only`` disabled.
    """
    wrapper = getattr(model, "model", None)
    modules = getattr(wrapper, "_modules", None)
    if not isinstance(modules, dict):
        return ()

    removed: list[str] = []
    for name in ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if name in modules and modules[name] is not None:
            # Assigning None unregisters the nn.Module while keeping the
            # attribute available to a defensive forward path.
            setattr(wrapper, name, None)
            removed.append(f"model.{name}")
    return tuple(removed)


def _configure_cuda_for_training(torch: Any, *, enable_tf32: bool) -> bool:
    """Enable L4-friendly math modes and return whether TF32 is active."""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    tf32 = bool(enable_tf32 and capability >= (8, 0))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32
    # This affects remaining fp32 matmuls (including optimizer-side work),
    # while bf16 model matmuls continue to use the GPU's native tensor cores.
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return tf32


def load_model_and_tokenizer(
    base_model: str,
    *,
    load_in_4bit: bool,
    attn_implementation: str = "sdpa",
    text_only: bool = False,
    bnb_4bit_use_double_quant: bool = True,
) -> tuple[Any, Any]:
    """Load a checkpoint with its declared class and fastest safe attention path."""
    import torch
    import transformers

    model_class, _ = resolve_model_class(base_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        # Avoid a second full copy of a 4B checkpoint in Colab host RAM while
        # Accelerate dispatches the modules to the L4.
        "low_cpu_mem_usage": True,
    }
    support_name = (
        "_supports_flash_attn"
        if attn_implementation == "flash_attention_2"
        else f"_supports_{attn_implementation}"
    )
    if attn_implementation not in {"", "auto", "eager"} and getattr(
        model_class, support_name, False
    ):
        # SDPA selects PyTorch's fused kernels on the L4 for full-attention
        # blocks. Models with a different attention implementation keep their
        # native path; the conditional avoids changing their numerics.
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    try:
        model = model_class.from_pretrained(base_model, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface the checkpoint, not the traceback
        raise TrainerSetupError(
            f"{model_class.__name__}.from_pretrained({base_model!r}) failed: {exc}"
        ) from exc
    if text_only:
        removed = _drop_unused_multimodal_towers(model)
        if removed:
            print(f"text-only load — removed {', '.join(removed)}", flush=True)
    return model, tokenizer


def collate_batch(batch: Sequence[Example], pad_token_id: int) -> dict[str, Any]:
    """Right-pad a batch; labels pad with the ignore index, never with the pad token."""
    import torch

    width = max(len(example.input_ids) for example in batch)
    return {
        "input_ids": torch.tensor(
            [e.input_ids + [pad_token_id] * (width - len(e.input_ids)) for e in batch]
        ),
        "attention_mask": torch.tensor(
            [e.attention_mask + [0] * (width - len(e.attention_mask)) for e in batch]
        ),
        "labels": torch.tensor(
            [e.labels + [IGNORE_INDEX] * (width - len(e.labels)) for e in batch]
        ),
    }


def _set_use_cache(model: Any, enabled: bool) -> None:
    """Set cache policy on both wrapper and nested text configs when present."""
    candidates = [getattr(model, "config", None)]
    base_model = getattr(model, "base_model", None)
    candidates.extend(
        [
            getattr(base_model, "config", None),
            getattr(getattr(base_model, "model", None), "config", None),
        ]
    )
    seen: set[int] = set()
    for config in candidates:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        if hasattr(config, "use_cache"):
            config.use_cache = enabled
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = enabled


class _ExampleDataset:
    """A list of examples with the two methods `Trainer` requires."""

    def __init__(self, examples: Sequence[Example]) -> None:
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Example:
        return self._examples[index]

    def __iter__(self) -> Iterator[Example]:
        return iter(self._examples)


def _trainer_class(transformers: Any) -> Any:
    """Build a Trainer that supports length-grouped sampling over Example dataclass."""

    class LexiTrainer(transformers.Trainer):  # type: ignore[misc]
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            if getattr(self.args, "train_sampling_strategy", None) == "group_by_length":
                dataset = train_dataset if train_dataset is not None else self.train_dataset
                lengths = [len(e.input_ids) for e in dataset]
                return transformers.trainer_pt_utils.LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=dataset,
                    lengths=lengths,
                )
            return super()._get_train_sampler(train_dataset)

    return LexiTrainer


def _check_liger_configuration(config: Config) -> None:
    """Verify Liger Kernel is installed when requested."""
    if not config.get_bool("train.use_liger_kernel"):
        return
    try:
        import liger_kernel  # noqa: F401
    except ImportError as exc:
        raise TrainerSetupError(
            "train.use_liger_kernel is set but liger-kernel is not installed"
        ) from exc


def attach_adapter(model: Any, config: Config) -> tuple[TargetResolution, Any]:
    """Resolve LoRA targets against this model and return the wrapped model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    spec: str | Sequence[str] = config.get("train.target_modules")
    layout = Layout.from_config(config.section("train").get("layout"))
    targets = resolve_target_modules(model, spec, layout)

    if getattr(model, "is_quantized", False):
        # Upcasts norms and the head to fp32. Without it, 4-bit LoRA trains but
        # is prone to diverging, which reads as a bad hyperparameter rather than
        # a missing setup step.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get_bool("train.gradient_checkpointing"),
            # Non-reentrant checkpointing is the current PyTorch/Transformers
            # path: it avoids the input-grad hook and can release activations
            # earlier during the backward recomputation.
            gradient_checkpointing_kwargs={
                "use_reentrant": config.get_bool("train.gradient_checkpointing_use_reentrant")
            },
        )

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=config.get_int("train.lora_r"),
            lora_alpha=config.get_int("train.lora_alpha"),
            lora_dropout=config.get_float("train.lora_dropout"),
            target_modules=list(targets.names),
            task_type="CAUSAL_LM",
        ),
    )
    return targets, peft_model


def train_sft(
    config: Config,
    *,
    train_path: str | Path,
    output_dir: str | Path,
    model: Any | None = None,
    tokenizer: Any | None = None,
    run: Any | None = None,
    resume: str | None = None,
    val_rows: Sequence[Mapping[str, Any]] | None = None,
    band_config: Any | None = None,
    ceiling: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Run supervised fine-tuning under `config`.

    `model` and `tokenizer` are injectable so the smoke gate can train a tiny
    randomly-initialised stack without a download; left unset, the checkpoint
    named by `train.base_model` is loaded. `run` is an open tracking handle —
    when it is recording, the Trainer's own metrics go to the same run as the
    lineage, rather than to a second one it opens for itself.

    `val_rows` turns on in-loop evaluation. Loss falls smoothly while a model
    learns to emit prose instead of JSON, so watching loss alone can burn an
    entire session before the failure is visible.
    """
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised off the training image
        raise TrainerSetupError(
            "the training stack is absent; install the `smoke` dependency group "
            "for CPU or requirements-colab.txt for GPU"
        ) from exc

    transformers.set_seed(config.get_int("train.seed"))
    _check_liger_configuration(config)

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            config.get_str("train.base_model"),
            load_in_4bit=config.get_bool("train.load_in_4bit"),
            attn_implementation=config.get_str("train.attn_implementation"),
            text_only=config.get_bool("train.text_only"),
            bnb_4bit_use_double_quant=config.get_bool("train.bnb_4bit_use_double_quant"),
        )

    rows = load_rows(train_path)
    examples, dropped = build_examples(
        tokenizer,
        rows,
        max_seq_len=config.get_int("train.max_seq_len"),
        thinking=config.get_str("train.thinking"),
        completion_only=config.get_bool("train.completion_only"),
        task=config.get_str("train.task"),
        max_drop_fraction=config.get_float("train.max_drop_fraction"),
    )
    supervised = sum(example.supervised_tokens for example in examples)
    total = sum(len(example.input_ids) for example in examples)
    print(
        f"examples — {len(examples)} built, {dropped} dropped, "
        f"{supervised / total:.1%} of tokens supervised",
        flush=True,
    )
    val_examples: list[Example] = []
    if val_rows:
        val_examples, val_dropped = build_examples(
            tokenizer,
            val_rows,
            max_seq_len=config.get_int("train.max_seq_len"),
            thinking=config.get_str("train.thinking"),
            completion_only=config.get_bool("train.completion_only"),
            task=config.get_str("train.task"),
            max_drop_fraction=1.0,
        )
        print(
            f"val examples — {len(val_examples)} built, {val_dropped} dropped",
            flush=True,
        )

    targets, model = attach_adapter(model, config)
    print(f"LoRA targets — {targets.summary()}", flush=True)
    # KV caches are useful for generation but waste memory and can conflict
    # with checkpointed training. This does not change the forward function.
    _set_use_cache(model, False)

    max_steps = config.get_int("train.max_steps")
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    tf32 = _configure_cuda_for_training(torch, enable_tf32=config.get_bool("train.tf32"))
    optimizer = config.get_str("train.optimizer")
    if optimizer == "auto":
        optimizer = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    dataloader_workers = config.get_int("train.dataloader_num_workers")
    dataloader_prefetch = config.get_int("train.dataloader_prefetch_factor")
    eval_steps = config.get_int("train.eval_steps")
    arguments = transformers.TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get_int("train.epochs"),
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=config.get_int("train.per_device_batch_size"),
        per_device_eval_batch_size=config.get_int("train.per_device_batch_size"),
        eval_strategy="steps" if val_examples else "no",
        eval_steps=eval_steps if val_examples else None,
        gradient_accumulation_steps=config.get_int("train.grad_accum"),
        learning_rate=config.get_float("train.learning_rate"),
        warmup_ratio=config.get_float("train.warmup_ratio"),
        logging_steps=config.get_int("train.logging_steps"),
        save_steps=config.get_int("train.save_steps"),
        gradient_checkpointing=config.get_bool("train.gradient_checkpointing"),
        gradient_checkpointing_kwargs={
            "use_reentrant": config.get_bool("train.gradient_checkpointing_use_reentrant")
        },
        optim=optimizer,
        torch_compile=config.get_bool("train.torch_compile"),
        torch_compile_backend=config.get("train.torch_compile_backend"),
        torch_compile_mode=config.get("train.torch_compile_mode"),
        use_liger_kernel=config.get_bool("train.use_liger_kernel"),
        remove_unused_columns=False,
        tf32=tf32,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=config.get_bool("train.dataloader_pin_memory"),
        dataloader_persistent_workers=(
            config.get_bool("train.dataloader_persistent_workers") and dataloader_workers > 0
        ),
        dataloader_prefetch_factor=(dataloader_prefetch if dataloader_workers > 0 else None),
        train_sampling_strategy=(
            "group_by_length" if config.get_bool("train.group_by_length") else "random"
        ),
        seed=config.get_int("train.seed"),
        bf16=bf16,
        report_to=["wandb"] if getattr(run, "active", False) else [],
    )
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    trainer_class = _trainer_class(transformers)
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=_ExampleDataset(examples),
        eval_dataset=_ExampleDataset(val_examples) if val_examples else None,
        data_collator=lambda batch: collate_batch(batch, pad_token_id),
    )
    if val_rows and config.get_str("train.task") == "grader":
        from .callbacks import build_eval_callback

        trainer.add_callback(
            build_eval_callback(
                config=config,
                run=run,
                tokenizer=tokenizer,
                rows=val_rows,
                band_config=band_config,
                ceiling=ceiling or {},
                every_steps=eval_steps,
            )
        )
    trainer.add_callback(build_progress_callback())

    outcome = trainer.train(resume_from_checkpoint=resolve_resume(output_dir, resume))

    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[VRAM Peak] Memory Allocated: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB", flush=True)

    trainer.save_model(str(output_dir))

    return TrainResult(
        output_dir=Path(output_dir),
        examples=len(examples),
        dropped=dropped,
        steps=int(outcome.global_step),
        targets=targets,
    )


__all__ = [
    "TrainResult",
    "TrainerSetupError",
    "attach_adapter",
    "build_examples",
    "collate_batch",
    "load_model_and_tokenizer",
    "load_rows",
    "resolve_model_class",
    "train_sft",
]
