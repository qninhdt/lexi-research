"""Tests for env factory, reward-info parsing, and difficulty profiling."""

from tau_research.tau.env_factory import build_system_prompt, load_split_task_ids
from tau_research.tau.rollout import parse_reward_info, strip_role_prefix


def test_strip_role_prefix_variants() -> None:
    assert strip_role_prefix("user: Hi there") == "Hi there"
    assert strip_role_prefix('tool: {"ok": 1}') == '{"ok": 1}'
    assert strip_role_prefix("plain content") == "plain content"


def test_build_system_prompt_matches_training_format() -> None:
    prompt = build_system_prompt("# Retail agent policy\n\nBe helpful.")
    assert prompt.startswith("<instructions>")
    assert "<policy>\n# Retail agent policy" in prompt
    assert prompt.rstrip().endswith("</policy>")


def test_load_split_task_ids_real_file() -> None:
    train_ids = load_split_task_ids("retail", "train")
    test_ids = load_split_task_ids("retail", "test")
    assert len(train_ids) == 74
    assert len(test_ids) == 40
    assert set(train_ids).isdisjoint(set(test_ids))


def test_parse_reward_info_full_breakdown() -> None:
    info = {"reward_info": ('{"reward": 1.0, "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0}}')}
    reward = parse_reward_info(info)
    assert reward is not None
    assert reward.reward == 1.0
    assert reward.db_reward == 1.0
    assert reward.communicate_reward == 1.0
    assert reward.is_success


def test_parse_reward_info_fallback_to_checks() -> None:
    info = {
        "reward_info": (
            '{"reward": 0.0, "db_check": {"passed": true},'
            ' "communicate_checks": [{"passed": false}]}'
        )
    }
    reward = parse_reward_info(info)
    assert reward is not None
    assert reward.db_reward == 1.0
    assert reward.communicate_reward == 0.0
    assert reward.reward == 0.0
    assert not reward.is_success


def test_parse_reward_info_missing_returns_none() -> None:
    assert parse_reward_info({"step_reward": 0.0}) is None
