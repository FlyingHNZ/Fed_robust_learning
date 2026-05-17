from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from src.utils.config import FedCDPConfig


def load_flat_yaml_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_dict: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if line == "":
            continue
        if ":" not in line:
            raise ValueError(f"Invalid YAML line {line_number}: `{raw_line}`")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        config_dict[key] = _parse_scalar(value)
    return config_dict


def build_config_from_dict(config_values: dict[str, Any]) -> FedCDPConfig:
    valid_fields = {field.name for field in fields(FedCDPConfig)}
    unknown_fields = sorted(set(config_values.keys()) - valid_fields)
    if len(unknown_fields) > 0:
        raise ValueError(f"Unknown config keys: {unknown_fields}")
    return FedCDPConfig(**config_values)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> FedCDPConfig:
    config_values: dict[str, Any] = {}
    if config_path is not None:
        config_values.update(load_flat_yaml_file(config_path))
    if overrides is not None:
        config_values.update(overrides)
    return build_config_from_dict(config_values)


def parse_override_items(items: list[str] | None) -> dict[str, Any]:
    if items is None:
        return {}

    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid override `{item}`. Expected format is `key=value`.",
            )
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_scalar(value.strip())
    return overrides


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
