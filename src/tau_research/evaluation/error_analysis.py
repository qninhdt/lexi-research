"""Automated 11-category error taxonomy classification."""

from collections import Counter
from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    INVALID_TOOL_SYNTAX = "A. invalid tool syntax"
    NONEXISTENT_TOOL = "B. nonexistent tool"
    WRONG_TOOL = "C. wrong tool"
    WRONG_ARGUMENT = "D. wrong argument"
    POLICY_VIOLATION = "E. policy violation"
    MISSING_REQUIRED_COMMUNICATION = "F. missing required communication"
    INCORRECT_DB_MUTATION = "G. incorrect DB mutation"
    UNNECESSARY_REPEATED_READ = "H. unnecessary repeated read calls"
    PREMATURE_FINAL_ANSWER = "I. premature final answer"
    THINKING_LOOP_TRUNCATION = "J. thinking loop / truncation"
    USER_MISUNDERSTANDING = "K. user misunderstanding"


def classify_episode_error(trajectory: dict[str, Any]) -> ErrorCategory:
    """Classifies a failed episode into one of the 11 error taxonomy categories."""
    term_reason = trajectory.get("termination_reason", "")
    last_action = str(trajectory.get("last_action", ""))

    if term_reason == "truncation" or "<think>" in last_action:
        return ErrorCategory.THINKING_LOOP_TRUNCATION

    if term_reason == "invalid_syntax" or "bad json" in last_action:
        return ErrorCategory.INVALID_TOOL_SYNTAX

    if term_reason == "nonexistent_tool":
        return ErrorCategory.NONEXISTENT_TOOL

    if term_reason == "policy_violation":
        return ErrorCategory.POLICY_VIOLATION

    db_reward = trajectory.get("db_reward", 1.0)
    comm_reward = trajectory.get("communicate_reward", 1.0)

    if db_reward == 0.0 and comm_reward == 1.0:
        return ErrorCategory.INCORRECT_DB_MUTATION

    if db_reward == 1.0 and comm_reward == 0.0:
        return ErrorCategory.MISSING_REQUIRED_COMMUNICATION

    if trajectory.get("num_turns", 0) <= 1:
        return ErrorCategory.PREMATURE_FINAL_ANSWER

    return ErrorCategory.WRONG_TOOL


def summarize_error_distribution(
    errors: list[ErrorCategory],
) -> dict[ErrorCategory, int]:
    """Computes frequency counts of categorized errors."""
    return dict(Counter(errors))
