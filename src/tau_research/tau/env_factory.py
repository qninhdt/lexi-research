"""Factory building real tau2 AgentGymEnv instances from official task splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPLIT_TASKS_PATH = "third_party/tau2-bench/data/tau2/domains/{domain}/split_tasks.json"

# System prompt wrapper matching the AReaL/fuvty SFT data format exactly, so
# inference-time prompts stay in-distribution with training prompts.
SYSTEM_TEMPLATE = """<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
<policy>
{policy}
</policy>"""


def build_system_prompt(policy: str) -> str:
    """Wraps a domain policy into the training-format system prompt."""
    return SYSTEM_TEMPLATE.format(policy=policy.strip())


def load_split_task_ids(domain: str, split: str) -> list[str]:
    """Loads official train/test task IDs for a domain."""
    path = Path(SPLIT_TASKS_PATH.format(domain=domain))
    with open(path, encoding="utf-8") as handle:
        splits = json.load(handle)
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'; available: {sorted(splits)}")
    return [str(tid) for tid in splits[split]]


class TauEnvFactory:
    """Creates AgentGymEnv instances for official tau2 task splits.

    The user simulator is an external LLM API (frozen across all experiments);
    no local user model is hosted.
    """

    def __init__(
        self,
        domain: str = "retail",
        split: str = "train",
        user_model: str = "gpt-4.1-mini",
        user_temperature: float = 0.7,
        user_api_base: str | None = None,
        max_steps: int = 100,
    ) -> None:
        self.domain = domain
        self.split = split
        self.user_model = user_model
        self.user_temperature = user_temperature
        self.user_api_base = user_api_base
        self.max_steps = max_steps
        self.task_ids: list[str] = load_split_task_ids(domain, split)

    def create(self, task_id: str) -> Any:
        """Instantiates one AgentGymEnv for the given official task ID."""
        from tau2.gym.gym_agent import AgentGymEnv

        user_llm_args: dict[str, Any] = {"temperature": self.user_temperature}
        return AgentGymEnv(
            domain=self.domain,
            task_id=task_id,
            max_steps=self.max_steps,
            solo_mode=False,
            user_llm=self.user_model,
            user_llm_args=user_llm_args,
        )

    def iter_task_ids(self) -> list[str]:
        return list(self.task_ids)
