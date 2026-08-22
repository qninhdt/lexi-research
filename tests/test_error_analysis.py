"""Tests for the 11-category error taxonomy classification."""

from tau_research.evaluation.error_analysis import (
    ErrorCategory,
    classify_episode_error,
    summarize_error_distribution,
)


def test_classify_syntax_error() -> None:
    traj = {
        "is_success": False,
        "termination_reason": "invalid_syntax",
        "last_action": "call:modify_order({bad json",
    }
    cat = classify_episode_error(traj)
    assert cat == ErrorCategory.INVALID_TOOL_SYNTAX


def test_classify_truncation_error() -> None:
    traj = {
        "is_success": False,
        "termination_reason": "truncation",
        "last_action": "<think>thinking loop",
    }
    cat = classify_episode_error(traj)
    assert cat == ErrorCategory.THINKING_LOOP_TRUNCATION


def test_classify_incorrect_db_mutation() -> None:
    traj = {
        "is_success": False,
        "db_reward": 0.0,
        "communicate_reward": 1.0,
        "termination_reason": "agent_stop",
    }
    cat = classify_episode_error(traj)
    assert cat == ErrorCategory.INCORRECT_DB_MUTATION


def test_summarize_error_distribution() -> None:
    errors = [
        ErrorCategory.INVALID_TOOL_SYNTAX,
        ErrorCategory.INVALID_TOOL_SYNTAX,
        ErrorCategory.INCORRECT_DB_MUTATION,
    ]
    dist = summarize_error_distribution(errors)
    assert dist[ErrorCategory.INVALID_TOOL_SYNTAX] == 2
    assert dist[ErrorCategory.INCORRECT_DB_MUTATION] == 1
