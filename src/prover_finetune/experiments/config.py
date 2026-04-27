from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Config file must be a YAML mapping.")
        return ExperimentConfig(raw=data)

    def section(self, key: str) -> dict[str, Any]:
        value = self.raw.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{key}' must be a mapping.")
        return value

