"""Strict YAML configuration loading.

A run config is a *family* YAML (``config_sac.yaml``, ``config_wdro.yaml``, ...) that
names a shared ``base.yaml`` through a top-level ``base:`` key. The base owns everything
that must be identical across algorithms (environment, episode, data, reward, physics,
vehicle, station); a family file may not redefine those sections. Unknown keys anywhere
raise ``ConfigError`` instead of being silently ignored (audit F25).
"""

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .base import Config

BASE_SECTIONS = ("environment", "episode", "data", "reward", "physics", "vehicle", "station", "milp")


class ConfigError(ValueError):
    """Raised for unknown keys or a family file overriding a base-owned section."""


class ConfigLoader:

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _update_dataclass(obj: Any, data: Dict[str, Any], path: str = "") -> None:
        if not data:
            return
        if not is_dataclass(obj):
            raise ConfigError(f"'{path}' is not a config section")
        names = {f.name for f in fields(obj)}
        unknown = sorted(set(data) - names)
        if unknown:
            where = path or "<root>"
            raise ConfigError(f"Unknown config key(s) under '{where}': {', '.join(unknown)}")
        for name, val in data.items():
            current = getattr(obj, name)
            key = f"{path}.{name}" if path else name
            if is_dataclass(current):
                if not isinstance(val, dict):
                    raise ConfigError(f"'{key}' must be a mapping, got {type(val).__name__}")
                ConfigLoader._update_dataclass(current, val, key)
            else:
                setattr(obj, name, val)

    @staticmethod
    def resolve_yaml(path: str) -> Dict[str, Any]:
        """Return the merged dict of a family YAML and the base it names."""
        family = ConfigLoader.load_yaml(path)
        base_ref = family.pop("base", None)
        if base_ref is None:
            return family
        clash = [s for s in BASE_SECTIONS if s in family]
        if clash:
            raise ConfigError(
                f"{path}: section(s) {', '.join(clash)} belong to the shared base config "
                f"'{base_ref}' and may not be redefined in a family file"
            )
        base_path = Path(path).parent / base_ref
        base = ConfigLoader.load_yaml(str(base_path))
        if "base" in base:
            raise ConfigError(f"{base_path}: a base config may not itself name a base")
        foreign = [s for s in base if s not in BASE_SECTIONS]
        if foreign:
            raise ConfigError(f"{base_path}: only {', '.join(BASE_SECTIONS)} are allowed in a base config, found {', '.join(foreign)}")
        merged = copy.deepcopy(base)
        merged.update(family)
        return merged

    @staticmethod
    def from_yaml(path: str) -> Config:
        config = Config()
        ConfigLoader._update_dataclass(config, ConfigLoader.resolve_yaml(path))
        return config

    @staticmethod
    def to_dict(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: ConfigLoader.to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [ConfigLoader.to_dict(v) for v in obj]
        return obj

    @staticmethod
    def save_yaml(config: Config, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(ConfigLoader.to_dict(config), f, default_flow_style=False, sort_keys=False)
