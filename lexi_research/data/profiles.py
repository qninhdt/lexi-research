"""Validated learner profiles used only as diversifier metadata."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from lexi_research.format import Tag


@dataclass(frozen=True)
class Profile:
    id: str
    l1: str
    cefr: str
    error_bias: tuple[str, ...]
    length: str
    traits: str

    @property
    def is_near_native(self) -> bool:
        """Near-native profiles intentionally exercise usage rather than basics."""
        return self.l1 == "none"


# Kept as the public descriptive name used by the Phase 3 design doc.
LearnerProfile = Profile


@dataclass(frozen=True)
class ProfileRegistry:
    """Indexed immutable registry so generation never silently invents a profile."""

    profiles: tuple[Profile, ...]

    @property
    def near_native(self) -> tuple[Profile, ...]:
        return tuple(profile for profile in self.profiles if profile.is_near_native)

    def by_id(self, profile_id: str) -> Profile:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        raise KeyError(profile_id)

    def traits_map(self) -> dict[str, str]:
        return {profile.id: profile.traits for profile in self.profiles}


def load_profiles(path: str | Path | None = None) -> ProfileRegistry:
    """Load profiles and reject invalid/duplicate IDs before spending any tokens."""
    profile_path = Path(path) if path is not None else Path(__file__).with_name("profiles.json")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("profiles must be a non-empty JSON array")
    allowed = {tag.value for tag in Tag}
    profiles: list[Profile] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"profile {index} must be an object")
        required = ("id", "l1", "cefr", "error_bias", "length", "traits")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required if key != "error_bias"):
            raise ValueError(f"profile {index} has a missing textual field")
        bias = item.get("error_bias")
        if not isinstance(bias, list) or not bias or any(not isinstance(tag, str) for tag in bias):
            raise ValueError(f"profile {index} error_bias must be a non-empty string list")
        unknown = sorted(set(bias) - allowed)
        if unknown:
            raise ValueError(f"profile {item['id']!r} has unknown tags: {', '.join(unknown)}")
        profile_id = item["id"]
        if profile_id in ids:
            raise ValueError(f"duplicate profile id: {profile_id}")
        ids.add(profile_id)
        profiles.append(
            Profile(profile_id, item["l1"], item["cefr"], tuple(bias), item["length"], item["traits"])
        )
    return ProfileRegistry(tuple(profiles))


def sample_profiles(profiles: ProfileRegistry | tuple[Profile, ...], count: int, seed: int) -> tuple[Profile, ...]:
    """Sample reproducibly without mutating process-global random state."""
    if count < 0:
        raise ValueError("count must be non-negative")
    entries = profiles.profiles if isinstance(profiles, ProfileRegistry) else profiles
    if count > len(entries):
        raise ValueError("count cannot exceed available profiles")
    return tuple(random.Random(seed).sample(entries, count))
