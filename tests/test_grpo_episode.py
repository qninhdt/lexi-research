"""Tests for GRPO episode token bookkeeping (env_mask spans, prompt boundary)."""

from typing import Any

from tau_research.tau.grpo_rollout import GenerationBackend, run_grpo_episode


class ScriptedBackend(GenerationBackend):
    """Returns pre-scripted turns; never touches a trainer."""

    def __init__(self, turns: list[str]) -> None:
        self.turns = turns
        self.calls = 0

    def complete(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        del prompt_ids
        text = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        ids = [len(text) + i for i in range(len(text.split()))]
        return ids, [0.5] * len(ids)


class FakeEnv:
    task_id = "0"

    def reset(self) -> tuple[str, dict[str, Any]]:
        return "user: I want to cancel order #100.", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if "cancel_order" in action:
            return '{"status": "cancelled"}', 0.0, False, False, {}
        return (
            "Thank you!",
            1.0,
            True,
            False,
            {
                "reward_info": (
                    '{"reward": 1.0, "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0}}'
                )
            },
        )


class FakeTokenizer:
    """Whitespace tokenizer with scripted decode output per generation call."""

    def __init__(self, turn_texts: list[str]) -> None:
        self.turn_texts = turn_texts
        self.decode_calls = 0

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False, **kw: Any
    ) -> str:
        del tokenize, kw
        return " ".join(f"{m['role']}:{m['content']}" for m in messages) + " GEN"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [len(text) + i for i in range(len(text.split()))]}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del ids, skip_special_tokens
        text = self.turn_texts[min(self.decode_calls, len(self.turn_texts) - 1)]
        self.decode_calls += 1
        return text


def test_episode_masks_env_tokens_and_extracts_reward() -> None:
    turn_texts = [
        "<think>Cancel it.</think>cancel_order(order_id='100')",
        "<think>Done.</think>Your order is cancelled.",
    ]
    tokenizer = FakeTokenizer(turn_texts)

    episode = run_grpo_episode(
        task_id="0",
        env=FakeEnv(),
        backend=ScriptedBackend(turn_texts),
        tokenizer=tokenizer,
        system_prompt="policy here",
        max_turns=4,
        max_completion_tokens=10_000,
    )

    assert episode["reward"] == 1.0
    assert episode["reward_info"] is not None
    assert episode["num_turns"] == 2

    completion, mask = episode["completion_ids"], episode["env_mask"]
    assert len(completion) == len(mask)
    assert mask[0] == 1, "first completion span is model-generated"
    assert set(mask) == {0, 1}, "env feedback tokens must be masked to 0"
    assert episode["prompt_ids"], "prompt span must be non-empty"


def test_episode_truncates_on_completion_budget() -> None:
    turn_texts = ["<think>Loop.</think>still thinking"] * 5

    class LongEnv(FakeEnv):
        def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
            # Never terminates; the budget must cut the episode off.
            del action
            return "user: ok", 0.0, False, False, {}

    episode = run_grpo_episode(
        task_id="0",
        env=LongEnv(),
        backend=ScriptedBackend(turn_texts),
        tokenizer=FakeTokenizer(turn_texts),
        system_prompt="policy",
        max_turns=50,
        max_completion_tokens=60,
    )

    assert len(episode["completion_ids"]) <= 60
    assert episode["truncated"]
    assert episode["reward"] == 0.0
