"""Wrapper for frozen user simulator configuration."""

from __future__ import annotations

import os
from typing import Any


def normalize_user_instructions(instructions: Any) -> str:
    """Coerce tau2 UserScenario.instructions (str | StructuredUserInstructions) to str."""
    if instructions is None:
        return ""
    if isinstance(instructions, str):
        return instructions
    # StructuredUserInstructions and similar pydantic models implement __str__.
    return str(instructions)


class UserSimulator:
    """Standardized user simulator preserving fixed parameters across evaluations and rollouts."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("TAU_USER_MODEL", "openai/grok-4.6")
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "")

    def get_config(self) -> dict[str, Any]:
        return {
            "user_model": self.model_name,
            "user_temperature": self.temperature,
            "api_base": self.api_base,
        }

    def build_live_simulator(self, task: Any) -> Any:
        """Builds an instantiated tau2 UserSimulator for the given task."""
        from tau2.user.user_simulator import UserSimulator as Tau2UserSim

        llm_args: dict[str, Any] = {"temperature": self.temperature}
        if self.api_key:
            llm_args["api_key"] = self.api_key
        if self.api_base:
            llm_args["api_base"] = self.api_base

        scenario = getattr(task, "user_scenario", None)
        raw_instructions = getattr(scenario, "instructions", None) if scenario else None
        instructions = normalize_user_instructions(raw_instructions)

        return Tau2UserSim(
            tools=getattr(task, "user_tools", None) or [],
            instructions=instructions,
            llm=self.model_name,
            llm_args=llm_args,
        )
