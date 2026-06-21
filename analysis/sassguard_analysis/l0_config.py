"""Configuration loading for rolling-window L0 scheduling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .sass_tokens import CONTENT_TOKEN_BUDGET


DEFAULT_L0_CONFIG = Path("configs/analysis/l0_windows.json")


class L0ConfigError(RuntimeError):
    """Raised when L0 window configuration is invalid."""


@dataclass(frozen=True)
class L0GroupingConfig:
    emit_stream_windows: bool
    emit_process_aggregate_windows: bool
    include_tid_in_group: bool


@dataclass(frozen=True)
class L0RollingWindowConfig:
    content_token_budget: int


@dataclass(frozen=True)
class L0TriggerConfig:
    use_bitwise_gate: bool
    bitwise_integer_ratio_threshold: float


@dataclass(frozen=True)
class L0WindowConfig:
    enabled: bool
    grouping: L0GroupingConfig
    window: L0RollingWindowConfig
    trigger: L0TriggerConfig
    config_path: str

    def with_enabled(self, enabled: bool) -> "L0WindowConfig":
        return replace(self, enabled=enabled)

    def with_bitwise_gate(self, enabled: bool) -> "L0WindowConfig":
        return replace(self, trigger=replace(self.trigger, use_bitwise_gate=enabled))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_l0_config(path: str | Path = DEFAULT_L0_CONFIG) -> L0WindowConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise L0ConfigError(f"missing L0 config: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise L0ConfigError(f"{config_path}: invalid JSON: {exc}") from exc
    return parse_l0_config(raw, config_path)


def parse_l0_config(raw: dict[str, Any], config_path: Path | str = "<memory>") -> L0WindowConfig:
    required = ("enabled", "grouping", "window", "trigger")
    missing = [key for key in required if key not in raw]
    if missing:
        raise L0ConfigError(f"L0 config missing sections: {', '.join(missing)}")

    grouping = L0GroupingConfig(
        emit_stream_windows=_bool(raw["grouping"], "emit_stream_windows"),
        emit_process_aggregate_windows=_bool(raw["grouping"], "emit_process_aggregate_windows"),
        include_tid_in_group=_bool(raw["grouping"], "include_tid_in_group"),
    )
    if not grouping.emit_stream_windows:
        raise L0ConfigError("rolling L0 scheduler requires stream windows")
    if grouping.emit_process_aggregate_windows:
        raise L0ConfigError("rolling L0 scheduler does not support process aggregate windows")

    window = L0RollingWindowConfig(
        content_token_budget=_positive_int(raw["window"], "content_token_budget", section_name="window"),
    )
    if window.content_token_budget > CONTENT_TOKEN_BUDGET:
        raise L0ConfigError(
            f"window.content_token_budget must be <= {CONTENT_TOKEN_BUDGET} "
            "to leave room for ModernBERT special tokens"
        )
    trigger = L0TriggerConfig(
        use_bitwise_gate=_optional_bool(raw["trigger"], "use_bitwise_gate", default=True),
        bitwise_integer_ratio_threshold=_ratio(raw["trigger"], "bitwise_integer_ratio_threshold"),
    )
    return L0WindowConfig(
        enabled=_bool(raw, "enabled"),
        grouping=grouping,
        window=window,
        trigger=trigger,
        config_path=str(config_path),
    )


def _bool(section: dict[str, Any], key: str) -> bool:
    if key not in section:
        raise L0ConfigError(f"L0 config missing key: {key}")
    if not isinstance(section[key], bool):
        raise L0ConfigError(f"L0 config {key} must be boolean")
    return bool(section[key])


def _optional_bool(section: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in section:
        return default
    if not isinstance(section[key], bool):
        raise L0ConfigError(f"L0 config {key} must be boolean")
    return bool(section[key])


def _positive_int(section: dict[str, Any], key: str, section_name: str | None = None) -> int:
    value = _int(section, key, section_name=section_name)
    if value <= 0:
        raise L0ConfigError(_key_name(key, section_name) + " must be > 0")
    return value


def _int(section: dict[str, Any], key: str, section_name: str | None = None) -> int:
    if key not in section:
        raise L0ConfigError(f"L0 config missing key: {_key_name(key, section_name)}")
    if isinstance(section[key], bool):
        raise L0ConfigError(_key_name(key, section_name) + " must be an integer")
    try:
        return int(section[key])
    except (TypeError, ValueError) as exc:
        raise L0ConfigError(_key_name(key, section_name) + " must be an integer") from exc


def _ratio(section: dict[str, Any], key: str) -> float:
    if key not in section:
        raise L0ConfigError(f"L0 config missing key: {key}")
    try:
        value = float(section[key])
    except (TypeError, ValueError) as exc:
        raise L0ConfigError(f"L0 config {key} must be a number") from exc
    if value < 0.0 or value > 1.0:
        raise L0ConfigError(f"L0 config {key} must be between 0 and 1")
    return value


def _key_name(key: str, section_name: str | None) -> str:
    return f"{section_name}.{key}" if section_name else key
