"""Configuration loading for L0 launch-window scheduling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_L0_CONFIG = Path("configs/analysis/l0_windows.json")


class L0ConfigError(RuntimeError):
    """Raised when L0 window configuration is invalid."""


@dataclass(frozen=True)
class L0GroupingConfig:
    emit_stream_windows: bool
    emit_process_aggregate_windows: bool
    include_tid_in_group: bool


@dataclass(frozen=True)
class L0MaturityConfig:
    min_launches: int
    min_duration_ms: int
    long_min_duration_ms: int


@dataclass(frozen=True)
class L0WindowTypeConfig:
    duration_ms: int
    max_launches: int
    target_l1_chunks: int
    max_emitted_launches: int


@dataclass(frozen=True)
class L0RepetitionConfig:
    dominant_code_id_ratio: float
    top3_code_id_ratio: float
    normalized_entropy: float


@dataclass(frozen=True)
class L0CondensationConfig:
    enabled: bool
    preserve_first_last: bool
    min_per_code_id: int


@dataclass(frozen=True)
class L0TriggerConfig:
    stable_shape_min_launches: int
    grid_stability: float
    block_stability: float
    cooldown_ms: int
    major_change_min_interval_ms: int
    periodic_sample_ms: int
    entropy_shift: float
    top3_jaccard_threshold: float
    shape_change_min_stability: float


@dataclass(frozen=True)
class L0WindowConfig:
    enabled: bool
    grouping: L0GroupingConfig
    maturity: L0MaturityConfig
    short_window: L0WindowTypeConfig
    long_window: L0WindowTypeConfig
    repetition: L0RepetitionConfig
    trigger: L0TriggerConfig
    condensation: L0CondensationConfig
    config_path: str

    def with_enabled(self, enabled: bool) -> "L0WindowConfig":
        return replace(self, enabled=enabled)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


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
    required = (
        "enabled",
        "grouping",
        "maturity",
        "short_window",
        "long_window",
        "repetition",
        "trigger",
        "condensation",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise L0ConfigError(f"L0 config missing sections: {', '.join(missing)}")

    grouping = L0GroupingConfig(
        emit_stream_windows=_bool(raw["grouping"], "emit_stream_windows"),
        emit_process_aggregate_windows=_bool(raw["grouping"], "emit_process_aggregate_windows"),
        include_tid_in_group=_bool(raw["grouping"], "include_tid_in_group"),
    )
    if not grouping.emit_stream_windows and not grouping.emit_process_aggregate_windows:
        raise L0ConfigError("L0 config must emit at least one grouping type")

    maturity = L0MaturityConfig(
        min_launches=_positive_int(raw["maturity"], "min_launches"),
        min_duration_ms=_nonnegative_int(raw["maturity"], "min_duration_ms"),
        long_min_duration_ms=_nonnegative_int(raw["maturity"], "long_min_duration_ms"),
    )
    short_window = _window_type(raw["short_window"], "short_window")
    long_window = _window_type(raw["long_window"], "long_window")
    if long_window.duration_ms < short_window.duration_ms:
        raise L0ConfigError("long_window.duration_ms must be >= short_window.duration_ms")
    if long_window.max_launches < short_window.max_launches:
        raise L0ConfigError("long_window.max_launches must be >= short_window.max_launches")
    repetition = L0RepetitionConfig(
        dominant_code_id_ratio=_ratio(raw["repetition"], "dominant_code_id_ratio"),
        top3_code_id_ratio=_ratio(raw["repetition"], "top3_code_id_ratio"),
        normalized_entropy=_ratio(raw["repetition"], "normalized_entropy"),
    )
    trigger = L0TriggerConfig(
        stable_shape_min_launches=_positive_int(raw["trigger"], "stable_shape_min_launches", section_name="trigger"),
        grid_stability=_ratio(raw["trigger"], "grid_stability"),
        block_stability=_ratio(raw["trigger"], "block_stability"),
        cooldown_ms=_nonnegative_int(raw["trigger"], "cooldown_ms", section_name="trigger"),
        major_change_min_interval_ms=_nonnegative_int(
            raw["trigger"],
            "major_change_min_interval_ms",
            section_name="trigger",
        ),
        periodic_sample_ms=_positive_int(raw["trigger"], "periodic_sample_ms", section_name="trigger"),
        entropy_shift=_ratio(raw["trigger"], "entropy_shift"),
        top3_jaccard_threshold=_ratio(raw["trigger"], "top3_jaccard_threshold"),
        shape_change_min_stability=_ratio(raw["trigger"], "shape_change_min_stability"),
    )
    condensation = L0CondensationConfig(
        enabled=_bool(raw["condensation"], "enabled"),
        preserve_first_last=_bool(raw["condensation"], "preserve_first_last"),
        min_per_code_id=_positive_int(raw["condensation"], "min_per_code_id"),
    )

    return L0WindowConfig(
        enabled=_bool(raw, "enabled"),
        grouping=grouping,
        maturity=maturity,
        short_window=short_window,
        long_window=long_window,
        repetition=repetition,
        trigger=trigger,
        condensation=condensation,
        config_path=str(config_path),
    )


def _window_type(section: dict[str, Any], name: str) -> L0WindowTypeConfig:
    return L0WindowTypeConfig(
        duration_ms=_positive_int(section, "duration_ms", section_name=name),
        max_launches=_positive_int(section, "max_launches", section_name=name),
        target_l1_chunks=_positive_int(section, "target_l1_chunks", section_name=name),
        max_emitted_launches=_positive_int(
            section,
            "max_emitted_launches",
            section_name=name,
        )
        if "max_emitted_launches" in section
        else _positive_int(section, "max_launches", section_name=name),
    )


def _bool(section: dict[str, Any], key: str) -> bool:
    if key not in section:
        raise L0ConfigError(f"L0 config missing key: {key}")
    if not isinstance(section[key], bool):
        raise L0ConfigError(f"L0 config {key} must be boolean")
    return bool(section[key])


def _required(section: dict[str, Any], key: str, section_name: str | None = None) -> Any:
    if key not in section:
        raise L0ConfigError(f"L0 config missing key: {_key_name(key, section_name)}")
    return section[key]


def _positive_int(section: dict[str, Any], key: str, section_name: str | None = None) -> int:
    value = _int(section, key, section_name=section_name)
    if value <= 0:
        raise L0ConfigError(_key_name(key, section_name) + " must be > 0")
    return value


def _nonnegative_int(section: dict[str, Any], key: str, section_name: str | None = None) -> int:
    value = _int(section, key, section_name=section_name)
    if value < 0:
        raise L0ConfigError(_key_name(key, section_name) + " must be >= 0")
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
