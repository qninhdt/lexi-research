"""Load `params.yaml`, apply dotted `--override`s, and freeze the result.

Every stage reads its configuration through this module, so a run's identity is
exactly the file plus the overrides — nothing downstream can reach in and change
a value after the run config has been recorded. Hence the freeze.

Overrides are typed against the value already in the file rather than guessed
from the text: `--override train.enable_thinking=false` has to become `False`,
and `bool("false")` is `True`. An override naming a key path that does not exist
raises, because the alternative is a sweep arm that silently never changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


class ConfigError(ValueError):
    """An unknown key path, a malformed override, or a value of the wrong type."""


def default_params_path() -> Path:
    """The `params.yaml` at the repo root, which DVC hashes into every stage."""
    return Path(__file__).resolve().parents[2] / "params.yaml"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def parse_override(text: str) -> tuple[str, str]:
    """Split `key.path=value`. The value may itself contain `=`."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip() or not value.strip():
        raise ConfigError(f"override {text!r} is not of the form key.path=value")
    return key.strip(), value.strip()


def _coerce(text: str, current: Any, path: str) -> Any:
    """Parse `text` into the type the file already holds at `path`."""
    # `bool` is an `int` subclass, so it has to be tested first.
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigError(f"{path}={text!r} is not a boolean")
    if isinstance(current, int):
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(f"{path}={text!r} is not an integer") from exc
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError as exc:
            raise ConfigError(f"{path}={text!r} is not a number") from exc
    if isinstance(current, str):
        return text
    if isinstance(current, (list, tuple)):
        items = [part.strip() for part in text.split(",") if part.strip()]
        if not items:
            raise ConfigError(f"{path}={text!r} is an empty list")
        template = current[0] if current else ""
        return [_coerce(item, template, path) for item in items]
    if current is None:
        # No type to match, so fall back to YAML's own scalar rules.
        return yaml.safe_load(text)
    raise ConfigError(f"{path} holds {type(current).__name__}, which cannot be overridden")


_ALIASES = {
    "train.eval_step": "train.eval_steps",
    "train.save_step": "train.save_steps",
    "train.logging_step": "train.logging_steps",
    "train.epoch": "train.epochs",
}


def _apply(values: MutableMapping[str, Any], path: str, raw: str) -> None:
    path = _ALIASES.get(path, path)
    segments = path.split(".")
    cursor: Any = values
    for segment in segments[:-1]:
        if not isinstance(cursor, MutableMapping) or segment not in cursor:
            raise ConfigError(f"override key path {path!r} is unknown: no {segment!r}")
        cursor = cursor[segment]
    leaf = segments[-1]
    if not isinstance(cursor, MutableMapping) or leaf not in cursor:
        raise ConfigError(f"override key path {path!r} is unknown: no {leaf!r}")
    cursor[leaf] = _coerce(raw, cursor[leaf], path)


@dataclass(frozen=True)
class Config:
    """An immutable view of the resolved parameters."""

    values: Mapping[str, Any]

    def get(self, path: str) -> Any:
        """Value at a dotted key path. Raises rather than returning a default."""
        path = _ALIASES.get(path, path)
        cursor: Any = self.values
        for segment in path.split("."):
            if not isinstance(cursor, Mapping) or segment not in cursor:
                raise ConfigError(f"config key path {path!r} is unknown: no {segment!r}")
            cursor = cursor[segment]
        return cursor

    def get_int(self, path: str) -> int:
        value = self.get(path)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config {path} is {type(value).__name__}, expected an integer")
        return value

    def get_float(self, path: str) -> float:
        value = self.get(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config {path} is {type(value).__name__}, expected a number")
        return float(value)

    def get_str(self, path: str) -> str:
        value = self.get(path)
        if not isinstance(value, str):
            raise ConfigError(f"config {path} is {type(value).__name__}, expected a string")
        return value

    def get_bool(self, path: str) -> bool:
        value = self.get(path)
        if not isinstance(value, bool):
            raise ConfigError(f"config {path} is {type(value).__name__}, expected a boolean")
        return value

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"config {name} is {type(value).__name__}, expected a section")
        return value

    def as_dict(self) -> dict[str, Any]:
        """A detached, mutable copy — for logging the run config to W&B."""
        thawed: dict[str, Any] = _thaw(self.values)
        return thawed

    def with_overrides(self, overrides: Iterable[str]) -> Config:
        """A new config with further `key.path=value` overrides applied."""
        values = self.as_dict()
        for override in overrides:
            key, raw = parse_override(override)
            _apply(values, key, raw)
        return Config(values=_freeze(values))


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> Config:
    """Read `params.yaml`, apply `key.path=value` overrides, and freeze."""
    resolved = Path(path) if path is not None else default_params_path()
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"{resolved} does not hold a mapping")
    values: dict[str, Any] = loaded
    for override in overrides:
        key, raw = parse_override(override)
        _apply(values, key, raw)
    return Config(values=_freeze(values))


__all__ = [
    "Config",
    "ConfigError",
    "default_params_path",
    "load_config",
    "parse_override",
]
