"""Configuration loading, path resolution and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Small helper around the loaded YAML configuration."""

    def __init__(self, data: Dict[str, Any], path: Path):
        self.data = data
        self.path = path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, None)
        if value is None:
            raise ValueError(f"Missing required config key: {dotted_key}")
        return value

    def resolve(self, value: Optional[str]) -> Optional[Path]:
        """Resolve a configured path relative to cwd, then project root."""
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [Path.cwd() / path, PROJECT_ROOT / path, self.path.parent / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (PROJECT_ROOT / path).resolve()

    def data_path(self, key: str, filename: Optional[str] = None) -> Optional[Path]:
        if filename is None:
            filename = self.get(key)
        base = self.resolve(self.get("paths.data_dir"))
        if base is None:
            return self.resolve(filename)
        if not filename:
            return None
        candidate = base / filename
        if candidate.exists():
            return candidate.resolve()
        return self.resolve(filename)

    @property
    def company(self) -> str:
        return str(self.get("company", ""))

    @property
    def level_columns(self) -> list[str]:
        return list(self.require("classification.taxonomy.level_columns"))

    @property
    def max_levels(self) -> int:
        return len(self.level_columns)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        candidates = [Path.cwd() / config_path, PROJECT_ROOT / config_path]
        config_path = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config did not contain a YAML mapping: {config_path}")
    cfg = Config(data, config_path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if not cfg.get("company"):
        raise ValueError("Config must define 'company'")
    if not cfg.get("inputs"):
        raise ValueError("Config must define at least one input")
    if not cfg.get("classification.taxonomy.file"):
        raise ValueError("Config must define classification.taxonomy.file")
    if not cfg.get("classification.taxonomy.level_columns"):
        raise ValueError("Config must define classification.taxonomy.level_columns")
    for index, source in enumerate(cfg.get("inputs", [])):
        if not source.get("file"):
            raise ValueError(f"Input {index} has no file")
        if not source.get("record_id", {}).get("column"):
            raise ValueError(f"Input {index} has no record_id.column")
