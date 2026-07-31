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
from collections.abc import Mapping, Sequence
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


def token_logprobs(model: Any, ids: Sequence[int], span: Any) -> Any:
    """Log-probabilities the model assigns to `ids` inside `span`.

    Returned as a tensor so the caller can back-propagate through it; the callers
    that only need numbers detach first.
    """
    import torch

    tensor = torch.tensor([list(ids)], device=model.device)
    logits = model(tensor).logits[0]
    # Position i predicts token i+1, so a token's own log-probability is read
    # from the row before it.
    logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
    targets = tensor[0][1:]
    picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    start = max(span.start - 1, 0)
    end = max(span.end - 1, start)
    return picked[start:end]


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
    with torch.no_grad():
        generated = model.generate(
            prompt,
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
    import torch

    with torch.no_grad():
        logprobs = token_logprobs(model, segments.input_ids, segments.answer)
    return [float(value) for value in torch.exp(logprobs).tolist()]


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

    probabilities = _gold_probabilities(model, segments)
    if algorithm == "jepo":
        import math

        from .jepo import jepo_reward

        return jepo_reward([math.log(max(p, 1e-12)) for p in probabilities])

    from .nrt import nrt_reward

    return nrt_reward(probabilities, aggregation=config.get_str("rl.nrt_aggregation"))


def _sample_answer(model: Any, tokenizer: Any, segments: Any, config: Config) -> str:
    import torch

    prefix = torch.tensor([list(segments.input_ids[: segments.answer.start])], device=model.device)
    with torch.no_grad():
        generated = model.generate(
            prefix,
            max_new_tokens=config.get_int("eval.max_new_tokens"),
            do_sample=True,
            temperature=config.get_float("rl.temperature"),
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(generated[0][prefix.shape[-1] :], skip_special_tokens=True))


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

    from lexi_research.train.trainer import load_model_and_tokenizer, load_rows

    algorithm = resolve_algorithm(config.get_str("rl.algo"))
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            config.get_str("train.base_model"),
            load_in_4bit=config.get_bool("train.load_in_4bit"),
        )

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

    groups: list[Group] = []
    rollouts: list[Rollout] = []
    report = step_report([Group(rewards=(0.0,))], [], kl=0.0)
    done = 0

    for row in rows:
        if steps > 0 and done >= steps:
            break
        prompt_ids = _prompt_ids(tokenizer, row, enable_thinking=enable_thinking)
        reasonings = sample_reasonings(
            model,
            tokenizer,
            prompt_ids,
            count=group_size,
            max_tokens=config.get_int("rl.max_reasoning_tokens"),
            temperature=config.get_float("rl.temperature"),
        )

        empty = build_segments(tokenizer, row, prompt_ids, thinking="forced-empty")
        empty_reward = (
            compute_reward(
                algorithm,
                model=model,
                tokenizer=tokenizer,
                row=row,
                segments=empty,
                reasoning="",
                band_config=band_config,
                config=config,
            )
            if use_baseline
            else 0.0
        )

        built = []
        rewards = []
        for reasoning in reasonings:
            segments = build_segments(
                tokenizer, row, prompt_ids, thinking="on", reasoning=reasoning or " "
            )
            reward = compute_reward(
                algorithm,
                model=model,
                tokenizer=tokenizer,
                row=row,
                segments=segments,
                reasoning=reasoning,
                band_config=band_config,
                config=config,
            )
            built.append((segments, reasoning))
            rewards.append(reward)

        group = Group(rewards=tuple(rewards), empty_reward=empty_reward)
        groups.append(group)
        weights = advantages(baseline_clip(rewards, empty_reward) if use_baseline else rewards)

        optimiser.zero_grad()
        ce_total = torch.zeros((), device=model.device)
        rl_total = torch.zeros((), device=model.device)
        kl_total = 0.0
        for (segments, reasoning), advantage, reward in zip(built, weights, rewards, strict=True):
            answer_logprobs = token_logprobs(model, segments.input_ids, segments.answer)
            # Cross-entropy covers the whole answer, feedback included.
            ce_total = ce_total - answer_logprobs.mean()
            reasoning_logprobs = token_logprobs(model, segments.input_ids, segments.reasoning)
            if len(reasoning_logprobs):
                # Policy gradient on the reasoning alone.
                rl_total = rl_total - advantage * reasoning_logprobs.mean()
            rollouts.append(
                Rollout(
                    reasoning=reasoning,
                    reward=reward,
                    parts={},
                    tokens=len(segments.reasoning),
                )
            )

        loss = combined_loss(
            ce_total / len(built),
            rl_total / len(built),
            lambda_rl=lambda_rl,
            kl=kl_total,
            kl_coefficient=kl_coefficient,
        )
        loss.backward()
        optimiser.step()
        done += 1

        report = step_report(groups, rollouts, kl=kl_total)
        if run is not None:
            run.log({**report.as_dict(), "rl/loss": float(loss.detach())}, step=done)
        print(f"rl step {done} — {report.as_dict()}", flush=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    (destination / "rl-report.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return RLResult(
        output_dir=destination,
        algorithm=algorithm,
        steps=done,
        rollouts=len(rollouts),
        last=report,
    )


__all__ = ["RLResult", "compute_reward", "sample_reasonings", "token_logprobs", "train_rl"]
