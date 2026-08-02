"""The one RL loop the three tracks share.

Only `compute_reward` differs between them. Everything else — sampling a group of
reasonings, the empty-reasoning baseline, group-relative advantages, the KL to
reference, the combined loss, the health panel — is this file, shared literally
rather than by convention. That is what makes a difference between two tracks a
difference in reward definition.

Heavy imports stay inside the functions that use them, so importing this module
on a CPU box with no training stack costs nothing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lexi_research.cli.config import Config
from lexi_research.format import BandConfig
from lexi_research.train.collate import training_messages

from .base import (
    Group,
    RLError,
    Rollout,
    StepReport,
    advantages,
    baseline_clip,
    combined_loss,
    resolve_algorithm,
    step_report,
)
from .rewards import RewardWeights
from .segments import build_segments


@dataclass(frozen=True)
class RLResult:
    """What a run produced, for the caller to log and the gate to assert."""

    output_dir: Path
    algorithm: str
    steps: int
    rollouts: int
    last: StepReport

    def summary(self) -> str:
        panel = self.last.as_dict()
        return (
            f"{self.algorithm}: {self.steps} steps, {self.rollouts} rollouts; "
            f"reward {panel['rl/reward_mean']:.3f}, "
            f"empty gap {panel['rl/empty_gap']:.3f}, "
            f"KL {panel['rl/kl']:.4f}, "
            f"dead groups {panel['rl/zero_advantage_share']:.0%} -> {self.output_dir}"
        )


def _prompt_ids(tokenizer: Any, row: Mapping[str, Any], *, enable_thinking: bool) -> list[int]:
    """The served prompt, through the tokenizer's own template. Parity again."""
    rendered = tokenizer.apply_chat_template(
        training_messages(row),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    return [int(token) for token in rendered]


def _autocast_for(tensor: Any) -> Any:
    """Use the fastest reduced-precision path supported by this CUDA device."""
    import torch

    if not getattr(tensor, "is_cuda", False):
        return nullcontext()
    dtype = torch.bfloat16
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cuda_memory_snapshot() -> dict[str, float]:
    """Return synchronised current/peak CUDA memory in MiB for the run report."""
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    torch.cuda.synchronize()
    mib = 1024 * 1024
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / mib, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / mib, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / mib, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / mib, 2),
    }


def _reset_cuda_peak() -> None:
    """Reset only peak counters; cached/model memory remains resident."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _token_logprobs_batch(
    model: Any,
    sequences: Sequence[Sequence[int]],
    spans_per_sequence: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """Score span groups in one padded forward, retaining only needed logits."""
    import torch

    if len(sequences) != len(spans_per_sequence):
        raise ValueError("one span collection is required for each input sequence")

    requested_per_sequence = []
    active = []
    for spans in spans_per_sequence:
        requested = []
        for span in spans:
            start = max(span.start - 1, 0)
            end = max(span.end - 1, start)
            requested.append((span, start, end))
            if end > start:
                active.append((span, start, end))
        requested_per_sequence.append(requested)

    if not active:
        return [
            [torch.empty(0, device=model.device) for _ in requested]
            for requested in requested_per_sequence
        ]

    start = min(item[1] for item in active)
    sequence_end = max(item[0].end for item in active)
    rows = []
    masks = []
    for ids in sequences:
        sequence = list(ids[:sequence_end])
        mask = [1] * len(sequence)
        if len(sequence) < sequence_end:
            padding = sequence_end - len(sequence)
            sequence.extend([0] * padding)
            mask.extend([0] * padding)
        rows.append(sequence)
        masks.append(mask)
    tensor = torch.tensor(rows, device=model.device)
    attention_mask = torch.tensor(masks, device=model.device)
    keep = sequence_end - start
    forward = {
        "input_ids": tensor,
        "attention_mask": attention_mask,
        "use_cache": False,
    }

    compact = False
    with _autocast_for(tensor):
        try:
            output = model(**forward, logits_to_keep=keep)
            logits = output.logits
            compact = logits.shape[1] == keep
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            output = model(**forward)
            logits = output.logits
        del output

    # Position i predicts token i+1, so a token's own log-probability is read
    # from the row before it. Cross-entropy computes only the selected target
    # probabilities; materialising `log_softmax(logits)` for the whole compact
    # vocabulary projection needlessly holds another `[tokens, vocab]` tensor.
    result = []
    for row_index, requested in enumerate(requested_per_sequence):
        row_result = []
        for span, span_start, span_end in requested:
            if span_end <= span_start:
                row_result.append(torch.empty(0, device=model.device))
                continue
            if compact:
                relative_start = span_start - start
                relative_end = span_end - start
                selected = logits[row_index, relative_start:relative_end]
                target_start = max(span.start, 1)
                targets = tensor[row_index, target_start : span.end]
            else:
                selected = logits[row_index, span_start:span_end]
                targets = tensor[row_index, 1:][span_start:span_end]
            row_result.append(
                -torch.nn.functional.cross_entropy(selected.float(), targets, reduction="none")
            )
        result.append(row_result)
    return result


def token_logprobs_for_spans(model: Any, ids: Sequence[int], spans: Sequence[Any]) -> list[Any]:
    """Return differentiable token log-probabilities for one sequence."""
    return _token_logprobs_batch(model, (ids,), (spans,))[0]


def token_logprobs(model: Any, ids: Sequence[int], span: Any) -> Any:
    """Log-probabilities the model assigns to `ids` inside `span`."""
    return token_logprobs_for_spans(model, ids, (span,))[0]


def sample_reasonings(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    count: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """`count` reasonings for one prompt. The group advantages are relative to it."""
    import torch

    if count < 2:
        raise RLError(
            f"rl.group_size={count}: advantages are relative to the group, so a "
            "group of one carries no signal at all"
        )
    prompt = torch.tensor([list(prompt_ids)], device=model.device)
    attention_mask = torch.ones_like(prompt)
    with torch.inference_mode(), _autocast_for(prompt):
        generated = model.generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=count,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(sequence[prompt.shape[-1] :], skip_special_tokens=True)
        for sequence in generated
    ]


def _gold_probabilities(model: Any, segments: Any) -> list[float]:
    """`pi(y*_i | x, z, y*_<i)` over the answer, teacher-forced."""
    return _gold_probabilities_batch(model, (segments,))[0]


def _gold_probabilities_batch(model: Any, segments: Sequence[Any]) -> list[list[float]]:
    """Teacher-force several rollouts in one inference batch."""
    import torch

    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for offset in range(0, len(segments), 4):
            chunk = segments[offset : offset + 4]
            logprobs = _token_logprobs_batch(
                model,
                [segments_item.input_ids for segments_item in chunk],
                [(segments_item.answer,) for segments_item in chunk],
            )
            probabilities.extend(
                [float(value) for value in row[0].exp().tolist()] for row in logprobs
            )
    return probabilities


def compute_reward(
    algorithm: str,
    *,
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    segments: Any,
    reasoning: str,
    band_config: BandConfig,
    config: Config,
) -> float:
    """The one line that differs between the three tracks."""
    weights = RewardWeights.from_config(config.section("rl").get("reward_weights"))
    scope = config.get_str("rl.reward_scope")

    if algorithm == "grpo":
        # Exogenous: an answer is sampled and scored by code that never sees the
        # policy. This is why GRPO runs first.
        from .grpo import grpo_reward

        answer = _sample_answer(model, tokenizer, segments, config)
        return grpo_reward(answer, row, band_config, weights=weights, scope=scope).total

    return _reward_from_probabilities(algorithm, _gold_probabilities(model, segments), config)


def _sample_answer(model: Any, tokenizer: Any, segments: Any, config: Config) -> str:
    import torch

    prefix = torch.tensor([list(segments.input_ids[: segments.answer.start])], device=model.device)
    attention_mask = torch.ones_like(prefix)
    with torch.inference_mode(), _autocast_for(prefix):
        generated = model.generate(
            prefix,
            attention_mask=attention_mask,
            max_new_tokens=config.get_int("eval.max_new_tokens"),
            do_sample=True,
            temperature=config.get_float("rl.temperature"),
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(generated[0][prefix.shape[-1] :], skip_special_tokens=True))


def _sample_answers(
    model: Any,
    tokenizer: Any,
    segments: Sequence[Any],
    config: Config,
    *,
    batch_size: int = 4,
) -> list[str]:
    """Generate sampled answers in batches of equal-length prefixes.

    GRPO's reward is exogenous, so batching these inference-only generations
    changes no loss or label. A small cap keeps the KV cache below the L4 limit
    when a group also includes the empty-reasoning baseline. Equal lengths are
    intentional: Qwen3.5's linear-attention state makes left-padding a prefix
    observably different from running that prefix without padding.
    """
    import torch

    if not segments:
        return []
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    grouped: dict[int, list[tuple[int, Any]]] = {}
    for index, item in enumerate(segments):
        prefix = list(item.input_ids[: item.answer.start])
        grouped.setdefault(len(prefix), []).append((index, item))

    answers = [""] * len(segments)
    for same_length in grouped.values():
        for offset in range(0, len(same_length), batch_size):
            chunk = same_length[offset : offset + batch_size]
            prefixes = [list(item.input_ids[: item.answer.start]) for _, item in chunk]
            width = len(prefixes[0])
            rows = prefixes
            masks = [[1] * width for _ in prefixes]
            input_ids = torch.tensor(rows, device=model.device)
            attention_mask = torch.tensor(masks, device=model.device)
            with torch.inference_mode(), _autocast_for(input_ids):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.get_int("eval.max_new_tokens"),
                    do_sample=True,
                    temperature=config.get_float("rl.temperature"),
                    pad_token_id=pad_token_id,
                )
            for (index, _), sequence in zip(chunk, generated, strict=True):
                answers[index] = str(tokenizer.decode(sequence[width:], skip_special_tokens=True))
    return answers


def _reward_from_probabilities(
    algorithm: str, probabilities: Sequence[float], config: Config
) -> float:
    """Apply the JEPO/NRT scalar reward after a shared probability forward."""
    if algorithm == "jepo":
        import math

        from .jepo import jepo_reward

        return jepo_reward([math.log(max(probability, 1e-12)) for probability in probabilities])
    from .nrt import nrt_reward

    return nrt_reward(probabilities, aggregation=config.get_str("rl.nrt_aggregation"))


def train_rl(
    config: Config,
    *,
    train_path: str | Path,
    output_dir: str | Path,
    band_config: BandConfig,
    model: Any | None = None,
    tokenizer: Any | None = None,
    run: Any | None = None,
    max_steps: int | None = None,
) -> RLResult:
    """Run one RL track. `model`/`tokenizer` are injectable for the smoke gate."""
    import torch

    from lexi_research.train.trainer import attach_adapter, load_model_and_tokenizer, load_rows

    # RL sampling and LoRA initialisation both consume torch RNG state. Seed
    # before loading/attaching anything so a kernel ablation compares the same
    # rollout stream and adapter initialisation, rather than noise from two
    # unrelated experiments.
    torch.manual_seed(config.get_int("train.seed"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get_int("train.seed"))

    algorithm = resolve_algorithm(config.get_str("rl.algo"))
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            config.get_str("train.base_model"),
            load_in_4bit=config.get_bool("train.load_in_4bit"),
            attn_implementation=config.get_str("train.attn_implementation"),
            text_only=config.get_bool("train.text_only"),
            bnb_4bit_use_double_quant=config.get_bool("train.bnb_4bit_use_double_quant"),
        )
        targets, model = attach_adapter(model, config)
        print(f"LoRA targets — {targets.summary()}", flush=True)

    thinking = config.get_str("train.thinking")
    enable_thinking = thinking != "off"
    if not enable_thinking:
        raise RLError(
            "train.thinking='off' leaves no reasoning to reward; every track's "
            "policy gradient lands on the reasoning span alone"
        )

    rows = load_rows(train_path)
    steps = max_steps if max_steps is not None else config.get_int("train.max_steps")
    group_size = config.get_int("rl.group_size")
    lambda_rl = config.get_float("rl.lambda_rl")
    kl_coefficient = config.get_float("rl.kl_coefficient")
    use_baseline = config.get_bool("rl.empty_baseline")

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get_float("train.learning_rate"),
    )

    memory: dict[str, Any] = {
        "cuda": bool(torch.cuda.is_available()),
        "device": str(getattr(model, "device", "cpu")),
        "setup": _cuda_memory_snapshot(),
        "steps": [],
    }
    _reset_cuda_peak()

    groups: list[Group] = []
    rollouts: list[Rollout] = []
    report = step_report([Group(rewards=(0.0,))], [], kl=0.0)
    done = 0

    for row in rows:
        if steps > 0 and done >= steps:
            break
        # Rollouts and reward scoring are inference phases. The policy phase
        # must switch back to train mode so Transformers actually activates
        # gradient checkpointing; leaving a freshly loaded model in eval mode
        # defeats the L4 memory guard and makes the first backward OOM.
        model.eval()
        step_memory: dict[str, Any] = {"step": done + 1}
        _reset_cuda_peak()
        phase_started = time.perf_counter()
        prompt_ids = _prompt_ids(tokenizer, row, enable_thinking=enable_thinking)
        reasonings = sample_reasonings(
            model,
            tokenizer,
            prompt_ids,
            count=group_size,
            max_tokens=config.get_int("rl.max_reasoning_tokens"),
            temperature=config.get_float("rl.temperature"),
        )
        step_memory["sample_reasonings"] = _cuda_memory_snapshot()
        step_memory["sample_reasonings"]["seconds"] = round(time.perf_counter() - phase_started, 3)

        _reset_cuda_peak()
        phase_started = time.perf_counter()
        empty = build_segments(tokenizer, row, prompt_ids, thinking="forced-empty")
        built = []
        for reasoning in reasonings:
            segments = build_segments(
                tokenizer, row, prompt_ids, thinking="on", reasoning=reasoning or " "
            )
            built.append((segments, reasoning))

        reward_segments = ([empty] if use_baseline else []) + [segments for segments, _ in built]
        if algorithm == "grpo":
            from .grpo import grpo_reward

            sampled_answers = _sample_answers(model, tokenizer, reward_segments, config)
            reward_weights = RewardWeights.from_config(config.section("rl").get("reward_weights"))
            raw_rewards = [
                grpo_reward(
                    answer,
                    row,
                    band_config,
                    weights=reward_weights,
                    scope=config.get_str("rl.reward_scope"),
                ).total
                for answer in sampled_answers
            ]
        else:
            probability_rows = _gold_probabilities_batch(model, reward_segments)
            raw_rewards = [
                _reward_from_probabilities(algorithm, probabilities, config)
                for probabilities in probability_rows
            ]
        empty_reward = raw_rewards[0] if use_baseline else 0.0
        rewards = raw_rewards[1:] if use_baseline else raw_rewards
        step_memory["rewards"] = _cuda_memory_snapshot()
        step_memory["rewards"]["seconds"] = round(time.perf_counter() - phase_started, 3)

        group = Group(rewards=tuple(rewards), empty_reward=empty_reward)
        groups.append(group)
        advantage_values = advantages(
            baseline_clip(rewards, empty_reward) if use_baseline else rewards
        )

        model.train()
        optimiser.zero_grad()
        phase_started = time.perf_counter()
        ce_total = 0.0
        rl_total = 0.0
        kl_total = 0.0
        for (segments, reasoning), advantage, reward in zip(
            built, advantage_values, rewards, strict=True
        ):
            answer_logprobs, reasoning_logprobs = token_logprobs_for_spans(
                model, segments.input_ids, (segments.answer, segments.reasoning)
            )
            # Cross-entropy covers the whole answer, feedback included.
            ce_loss = -answer_logprobs.mean()
            ce_total += float(ce_loss.detach())
            rl_loss = torch.zeros_like(ce_loss)
            if len(reasoning_logprobs):
                # Policy gradient on the reasoning alone.
                rl_loss = -advantage * reasoning_logprobs.mean()
                rl_total += float(rl_loss.detach())
            loss = combined_loss(
                ce_loss,
                rl_loss,
                lambda_rl=lambda_rl,
                kl=0.0,
                kl_coefficient=kl_coefficient,
            ) / len(built)
            loss.backward()
            rollouts.append(
                Rollout(
                    reasoning=reasoning,
                    reward=reward,
                    parts={},
                    tokens=len(segments.reasoning),
                )
            )
        step_memory["policy_backward"] = _cuda_memory_snapshot()
        step_memory["policy_backward"]["seconds"] = round(time.perf_counter() - phase_started, 3)

        loss_value = combined_loss(
            ce_total / len(built),
            rl_total / len(built),
            lambda_rl=lambda_rl,
            kl=kl_total,
            kl_coefficient=kl_coefficient,
        )
        _reset_cuda_peak()
        phase_started = time.perf_counter()
        optimiser.step()
        step_memory["optimizer_step"] = _cuda_memory_snapshot()
        step_memory["optimizer_step"]["seconds"] = round(time.perf_counter() - phase_started, 3)
        phase_peaks = [
            values["peak_allocated_mb"]
            for name, values in step_memory.items()
            if name != "step" and isinstance(values, dict)
        ]
        phase_reserved_peaks = [
            values["peak_reserved_mb"]
            for name, values in step_memory.items()
            if name != "step" and isinstance(values, dict)
        ]
        step_memory["peak_allocated_mb"] = max(phase_peaks, default=0.0)
        step_memory["peak_reserved_mb"] = max(phase_reserved_peaks, default=0.0)
        memory["steps"].append(step_memory)
        done += 1

        report = step_report(groups, rollouts, kl=kl_total)
        if run is not None:
            run.log({**report.as_dict(), "rl/loss": float(loss_value)}, step=done)
        print(f"rl step {done} — {report.as_dict()}", flush=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    all_phase_names = ("sample_reasonings", "rewards", "policy_backward", "optimizer_step")
    memory["peak_allocated_mb"] = max(
        (step["peak_allocated_mb"] for step in memory["steps"]), default=0.0
    )
    memory["peak_reserved_mb"] = max(
        (step["peak_reserved_mb"] for step in memory["steps"]), default=0.0
    )
    memory["phase_peaks_mb"] = {
        name: max(
            (step[name]["peak_allocated_mb"] for step in memory["steps"] if name in step),
            default=0.0,
        )
        for name in all_phase_names
    }
    memory["phase_seconds"] = {
        name: round(
            sum(step[name]["seconds"] for step in memory["steps"] if name in step),
            3,
        )
        for name in all_phase_names
    }
    memory["train_seconds"] = round(sum(memory["phase_seconds"].values()), 3)
    report_payload: dict[str, Any] = report.as_dict()
    report_payload["memory"] = memory
    (destination / "rl-report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return RLResult(
        output_dir=destination,
        algorithm=algorithm,
        steps=done,
        rollouts=len(rollouts),
        last=report,
    )


__all__ = ["RLResult", "compute_reward", "sample_reasonings", "token_logprobs", "train_rl"]
