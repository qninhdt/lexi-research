from pathlib import Path

from lexi_research.data.profiles import load_profiles, sample_profiles


def test_profiles_are_valid_and_include_near_native_examples() -> None:
    path = Path(__file__).parents[2] / "lexi_research/data/profiles.json"
    profiles = load_profiles(path)
    assert len(profiles.profiles) == 16
    assert len(profiles.near_native) >= 2
    assert all(profile.error_bias for profile in profiles.profiles)


def test_profile_sampling_is_deterministic() -> None:
    path = Path(__file__).parents[2] / "lexi_research/data/profiles.json"
    profiles = load_profiles(path)
    assert sample_profiles(profiles, 4, 42) == sample_profiles(profiles, 4, 42)
