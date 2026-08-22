"""Automated 11-category error taxonomy classification.

Mapping uses the termination reasons actually emitted by the rollout loop
(agent_stop, max_turns, truncation, empty_output, empty_action, env_truncated)
plus the official reward breakdown; categories that need information the
harness does not capture (policy violations, wrong arguments) stay reachable
via manual review of eval_results.jsonl rather than fabricated signals.
"""

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
    term_reason = str(trajectory.get("termination_reason") or "")
    db_reward = float(trajectory.get("db_reward", 0.0))
    comm_reward = float(trajectory.get("communicate_reward", 0.0))
    num_turns = int(trajectory.get("num_turns", 0))

    # Truncation family: generation cut off mid-thought or empty action.
    if term_reason in {"truncation", "empty_output", "empty_action"}:
        return ErrorCategory.THINKING_LOOP_TRUNCATION

    # Environment-side truncation (e.g. user simulator stopped unexpectedly).
    if term_reason == "env_truncated":
        return ErrorCategory.USER_MISUNDERSTANDING

    # Ran out of turns: usually looping on reads without converging.
    if term_reason == "max_turns":
        return ErrorCategory.UNNECESSARY_REPEATED_READ

    if term_reason in {"invalid_syntax", "bad_json"}:
        return ErrorCategory.INVALID_TOOL_SYNTAX

    if term_reason == "nonexistent_tool":
        return ErrorCategory.NONEXISTENT_TOOL

    if term_reason == "policy_violation":
        return ErrorCategory.POLICY_VIOLATION

    # agent_stop failures: split by which reward component was lost.
    if db_reward < 1.0 and comm_reward >= 1.0:
        return ErrorCategory.INCORRECT_DB_MUTATION
    if db_reward >= 1.0 and comm_reward < 1.0:
        return ErrorCategory.MISSING_REQUIRED_COMMUNICATION
    if num_turns <= 1:
        return ErrorCategory.PREMATURE_FINAL_ANSWER

    return ErrorCategory.WRONG_TOOL


def summarize_error_distribution(
    errors: list[ErrorCategory],
) -> dict[ErrorCategory, int]:
    """Computes frequency counts of categorized errors."""
    return dict(Counter(errors))
