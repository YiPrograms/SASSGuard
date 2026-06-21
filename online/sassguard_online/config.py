"""Online detection configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ONLINE_CONFIG = Path("configs/online/detection.json")


class OnlineConfigError(RuntimeError):
    """Raised when online detection configuration is invalid."""


def load_online_config(path: str | Path = DEFAULT_ONLINE_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OnlineConfigError(f"missing online config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise OnlineConfigError(f"{config_path}: invalid JSON: {exc}") from exc
    _validate(raw)
    raw["_config_path"] = str(config_path)
    return raw


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _validate(raw: dict[str, Any]) -> None:
    sections = (
        "enabled",
        "transport",
        "storage",
        "l0",
        "launch_batching",
        "kernel_analysis",
        "l1",
        "verdict",
        "enforcement",
    )
    missing = [section for section in sections if section not in raw]
    if missing:
        raise OnlineConfigError(f"online config missing sections: {', '.join(missing)}")
    if not isinstance(raw["enabled"], bool):
        raise OnlineConfigError("enabled must be boolean")
    for section, key in (
        ("transport", "processor_socket"),
        ("storage", "collector_output_dir"),
        ("storage", "processor_work_dir"),
        ("l0", "config_path"),
        ("l1", "training_config_path"),
        ("l1", "checkpoint_path"),
        ("verdict", "mining_label"),
    ):
        if not str(raw[section].get(key, "")).strip():
            raise OnlineConfigError(f"{section}.{key} must be set")
    for section, key in (
        ("transport", "connect_timeout_ms"),
        ("transport", "write_timeout_ms"),
        ("transport", "read_timeout_ms"),
        ("transport", "reconnect_backoff_ms"),
        ("transport", "frame_max_bytes"),
        ("launch_batching", "max_batch_count"),
        ("launch_batching", "flush_interval_ms"),
        ("launch_batching", "max_unsent_per_session"),
        ("kernel_analysis", "workers"),
        ("kernel_analysis", "short_kernel_threshold"),
        ("kernel_analysis", "max_code_queue_per_session"),
        ("kernel_analysis", "readiness_timeout_ms"),
        ("l1", "batch_size"),
        ("verdict", "top_k"),
    ):
        _positive_int(raw[section], key, f"{section}.{key}")
    policy = str(raw["verdict"].get("policy", "per_window"))
    if policy not in {"per_window", "rolling_mean_and_max"}:
        raise OnlineConfigError("verdict.policy must be one of: per_window, rolling_mean_and_max")
    if policy == "rolling_mean_and_max":
        _positive_int(raw["verdict"], "rolling_window_count", "verdict.rolling_window_count")
    for section, key in (
        ("verdict", "max_p_mining_threshold"),
        ("verdict", "topk_mean_p_mining_threshold"),
        ("verdict", "rolling_mean_mining_probability_threshold"),
        ("verdict", "rolling_max_mining_probability_threshold"),
    ):
        if key not in raw[section]:
            if key.startswith("rolling_") and policy != "rolling_mean_and_max":
                continue
            raise OnlineConfigError(f"{section}.{key} must be set")
        value = float(raw[section].get(key))
        if value < 0 or value > 1:
            raise OnlineConfigError(f"{section}.{key} must be in [0, 1]")


def _positive_int(section: dict[str, Any], key: str, name: str) -> None:
    try:
        value = int(section.get(key))
    except (TypeError, ValueError) as exc:
        raise OnlineConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise OnlineConfigError(f"{name} must be > 0")
