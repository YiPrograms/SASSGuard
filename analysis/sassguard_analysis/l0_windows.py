"""Dynamic L0 launch-window scheduling from CUDA kernel launch metadata."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from .ingest import event_sort_key
from .l0_config import L0WindowConfig


@dataclass
class StreamState:
    group_key: dict[str, Any]
    short_launches: list[dict[str, Any]] = field(default_factory=list)
    long_launches: list[dict[str, Any]] = field(default_factory=list)
    has_been_analyzed: bool = False
    last_l1_timestamp_ns: int | None = None
    last_signature: dict[str, Any] | None = None


@dataclass(frozen=True)
class L0Window:
    window_id: str
    window_type: str
    group_kind: str
    group_key: dict[str, Any]
    launches: list[dict[str, Any]]
    features: dict[str, Any]
    trigger_reason: list[str]
    packing_mode: str
    condensation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.group_kind
        payload["selected_launch_sequences"] = [
            launch.get("sequence") for launch in self.launches if launch.get("sequence") is not None
        ]
        return payload


def build_l0_windows(launches: list[dict[str, Any]], config: L0WindowConfig) -> list[L0Window]:
    if not config.enabled:
        return []

    states: dict[tuple[Any, ...], StreamState] = {}
    windows: list[L0Window] = []
    for launch in _sorted_launches(launches):
        key = _group_tuple(launch, include_tid=config.grouping.include_tid_in_group)
        state = states.setdefault(
            key,
            StreamState(_group_key_from_tuple(key, include_tid=config.grouping.include_tid_in_group)),
        )
        state.short_launches.append(launch)
        state.long_launches.append(launch)

        if should_evaluate_short_window(state.short_launches, state, config):
            short_features = launch_features(state.short_launches)
            trigger = trigger_reasons(state, short_features, config)
            if trigger:
                windows.append(
                    make_window(
                        state,
                        state.short_launches,
                        "short",
                        trigger,
                        "full_short_window",
                        config,
                    )
                )
                state.has_been_analyzed = True
                state.last_l1_timestamp_ns = _timestamp_ns(launch)
                state.last_signature = pattern_signature(short_features, state.short_launches)
                state.short_launches = []
                continue

        if should_roll_short_window(state.short_launches, config):
            state.short_launches = []

        long_window = maybe_emit_long_window(state, config)
        if long_window is not None:
            windows.append(long_window)
            state.last_l1_timestamp_ns = _timestamp_ns(launch)
            state.last_signature = pattern_signature(long_window.features, state.long_launches)
            state.long_launches = []

    return assign_window_ids(windows)


def should_evaluate_short_window(
    launches: list[dict[str, Any]],
    state: StreamState,
    config: L0WindowConfig,
) -> bool:
    if not launches:
        return False
    launch_count = len(launches)
    duration = window_duration_ms(launches)
    mature = launch_count >= config.maturity.min_launches or duration >= config.maturity.min_duration_ms
    if not mature:
        return False
    if not state.has_been_analyzed:
        return True
    return (
        launch_count % 16 == 0
        or duration >= config.short_window.duration_ms
        or launch_count >= config.short_window.max_launches
        or periodic_sample_due(state, launches, config)
    )


def trigger_reasons(
    state: StreamState,
    features: dict[str, Any],
    config: L0WindowConfig,
) -> list[str]:
    if not is_mature_features(features, config):
        return []
    major_change = is_major_pattern_change(state, features, state.short_launches, config)
    if in_cooldown(state, state.short_launches, config) and not (
        major_change and major_change_interval_elapsed(state, state.short_launches, config)
    ):
        return []

    reasons: list[str] = []
    if not state.has_been_analyzed:
        reasons.append("first_mature_window")
    if features["dominant_code_id_ratio"] >= config.repetition.dominant_code_id_ratio:
        reasons.append("dominant_code_id_ratio")
    if features["top3_code_id_ratio"] >= config.repetition.top3_code_id_ratio:
        reasons.append("top3_code_id_ratio")
    if features["normalized_entropy"] <= config.repetition.normalized_entropy:
        reasons.append("normalized_entropy")
    if (
        features["launch_count"] >= config.trigger.stable_shape_min_launches
        and features["grid_stability"] >= config.trigger.grid_stability
        and features["block_stability"] >= config.trigger.block_stability
    ):
        reasons.append("stable_launch_shape")
    if major_change:
        reasons.append("major_pattern_change")
    if periodic_sample_due(state, state.short_launches, config):
        reasons.append("periodic_safety_sample")
    return reasons


def should_roll_short_window(launches: list[dict[str, Any]], config: L0WindowConfig) -> bool:
    if not launches:
        return False
    return (
        window_duration_ms(launches) >= config.short_window.duration_ms
        or len(launches) >= config.short_window.max_launches
    )


def maybe_emit_long_window(state: StreamState, config: L0WindowConfig) -> L0Window | None:
    launches = state.long_launches
    if not launches:
        return None
    duration = window_duration_ms(launches)
    if duration < config.maturity.long_min_duration_ms:
        return None
    features = launch_features(launches)
    if not is_mature_features(features, config):
        return None
    selected, condensation = proportional_condense_launches(
        launches,
        config.long_window.max_emitted_launches,
    )
    return L0Window(
        window_id="",
        window_type="long",
        group_kind="stream",
        group_key=state.group_key,
        launches=selected,
        features=features,
        trigger_reason=["long_window_safety_sample"],
        packing_mode="proportional_long_window",
        condensation=condensation,
    )


def make_window(
    state: StreamState,
    launches: list[dict[str, Any]],
    window_type: str,
    trigger_reason: list[str],
    packing_mode: str,
    config: L0WindowConfig,
) -> L0Window:
    selected = list(launches)
    condensation = {
        "enabled": False,
        "applied": False,
        "original_launch_count": len(launches),
        "selected_launch_count": len(selected),
    }
    if window_type == "long":
        selected, condensation = proportional_condense_launches(
            launches,
            config.long_window.max_emitted_launches,
        )
    return L0Window(
        window_id="",
        window_type=window_type,
        group_kind="stream",
        group_key=state.group_key,
        launches=selected,
        features=launch_features(launches),
        trigger_reason=trigger_reason,
        packing_mode=packing_mode,
        condensation=condensation,
    )


def is_mature_features(features: dict[str, Any], config: L0WindowConfig) -> bool:
    return (
        features["launch_count"] >= config.maturity.min_launches
        or features["duration_ms"] >= config.maturity.min_duration_ms
    )


def in_cooldown(state: StreamState, launches: list[dict[str, Any]], config: L0WindowConfig) -> bool:
    if state.last_l1_timestamp_ns is None or not launches:
        return False
    current = _timestamp_ns(launches[-1])
    if current is None:
        return False
    return (current - state.last_l1_timestamp_ns) / 1_000_000.0 < config.trigger.cooldown_ms


def periodic_sample_due(state: StreamState, launches: list[dict[str, Any]], config: L0WindowConfig) -> bool:
    if state.last_l1_timestamp_ns is None or not launches:
        return False
    current = _timestamp_ns(launches[-1])
    if current is None:
        return False
    return (current - state.last_l1_timestamp_ns) / 1_000_000.0 >= config.trigger.periodic_sample_ms


def major_change_interval_elapsed(
    state: StreamState,
    launches: list[dict[str, Any]],
    config: L0WindowConfig,
) -> bool:
    if state.last_l1_timestamp_ns is None or not launches:
        return False
    current = _timestamp_ns(launches[-1])
    if current is None:
        return False
    return (current - state.last_l1_timestamp_ns) / 1_000_000.0 >= config.trigger.major_change_min_interval_ms


def is_major_pattern_change(
    state: StreamState,
    features: dict[str, Any],
    launches: list[dict[str, Any]],
    config: L0WindowConfig,
) -> bool:
    if not state.last_signature:
        return False
    signature = pattern_signature(features, launches)
    previous = state.last_signature
    if signature["dominant_code_id"] != previous.get("dominant_code_id"):
        return True
    previous_top3 = set(previous.get("top3_code_ids") or [])
    current_top3 = set(signature["top3_code_ids"])
    union = previous_top3 | current_top3
    jaccard = len(previous_top3 & current_top3) / len(union) if union else 1.0
    if jaccard <= config.trigger.top3_jaccard_threshold:
        return True
    if (
        signature["dominant_grid"] != previous.get("dominant_grid")
        and features["grid_stability"] >= config.trigger.shape_change_min_stability
        and float(previous.get("grid_stability", 0.0)) >= config.trigger.shape_change_min_stability
    ):
        return True
    if (
        signature["dominant_block"] != previous.get("dominant_block")
        and features["block_stability"] >= config.trigger.shape_change_min_stability
        and float(previous.get("block_stability", 0.0)) >= config.trigger.shape_change_min_stability
    ):
        return True
    return abs(signature["normalized_entropy"] - float(previous.get("normalized_entropy", 0.0))) >= config.trigger.entropy_shift


def pattern_signature(features: dict[str, Any], launches: list[dict[str, Any]]) -> dict[str, Any]:
    code_counts = Counter(str(launch.get("code_id")) for launch in launches)
    grid_counts = Counter(_shape_key(launch.get("grid_dim")) for launch in launches)
    block_counts = Counter(_shape_key(launch.get("block_dim")) for launch in launches)
    return {
        "dominant_code_id": code_counts.most_common(1)[0][0] if code_counts else None,
        "top3_code_ids": [code_id for code_id, _count in code_counts.most_common(3)],
        "dominant_grid": grid_counts.most_common(1)[0][0] if grid_counts else None,
        "dominant_block": block_counts.most_common(1)[0][0] if block_counts else None,
        "grid_stability": features["grid_stability"],
        "block_stability": features["block_stability"],
        "normalized_entropy": features["normalized_entropy"],
    }


def proportional_condense_launches(
    launches: list[dict[str, Any]],
    target_emitted_launches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(launches) <= target_emitted_launches:
        return list(launches), {
            "enabled": True,
            "applied": False,
            "mode": "proportional",
            "original_launch_count": len(launches),
            "selected_launch_count": len(launches),
            "target_emitted_launches": target_emitted_launches,
        }

    runs: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(launches)):
        if launches[idx].get("code_id") != launches[idx - 1].get("code_id"):
            runs.append((start, idx))
            start = idx
    runs.append((start, len(launches)))

    total = len(launches)
    raw_allocations = []
    for run_index, (start_idx, end_idx) in enumerate(runs):
        raw_count = end_idx - start_idx
        exact = raw_count * target_emitted_launches / total
        floor_count = max(1, math.floor(exact))
        raw_allocations.append([run_index, start_idx, end_idx, exact, floor_count])

    selected_total = sum(int(item[4]) for item in raw_allocations)
    while selected_total > target_emitted_launches:
        candidates = [item for item in raw_allocations if int(item[4]) > 1]
        if not candidates:
            break
        item = min(candidates, key=lambda row: float(row[3]) - math.floor(float(row[3])))
        item[4] = int(item[4]) - 1
        selected_total -= 1
    while selected_total < target_emitted_launches:
        item = max(raw_allocations, key=lambda row: float(row[3]) - math.floor(float(row[3])))
        item[4] = int(item[4]) + 1
        selected_total += 1

    selected_indices: list[int] = []
    for _run_index, start_idx, end_idx, _exact, emit_count in raw_allocations:
        count = int(emit_count)
        run_len = int(end_idx) - int(start_idx)
        if count >= run_len:
            selected_indices.extend(range(int(start_idx), int(end_idx)))
            continue
        if count == 1:
            selected_indices.append(int(start_idx))
            continue
        step = (run_len - 1) / (count - 1)
        selected_indices.extend(int(start_idx) + round(offset * step) for offset in range(count))

    selected = [launches[idx] for idx in sorted(set(selected_indices))[:target_emitted_launches]]
    return selected, {
        "enabled": True,
        "applied": True,
        "mode": "proportional",
        "original_launch_count": len(launches),
        "selected_launch_count": len(selected),
        "target_emitted_launches": target_emitted_launches,
        "compression_ratio": target_emitted_launches / len(launches),
    }


def launch_features(launches: list[dict[str, Any]]) -> dict[str, Any]:
    launch_count = len(launches)
    code_counts = Counter(str(launch.get("code_id")) for launch in launches)
    grid_counts = Counter(_shape_key(launch.get("grid_dim")) for launch in launches)
    block_counts = Counter(_shape_key(launch.get("block_dim")) for launch in launches)
    top_counts = [count for _value, count in code_counts.most_common()]
    gaps_us = launch_gaps_us(launches)
    avg_gap = sum(gaps_us) / len(gaps_us) if gaps_us else 0.0
    gap_std = _stddev(gaps_us, avg_gap) if gaps_us else 0.0
    return {
        "launch_count": launch_count,
        "duration_ms": window_duration_ms(launches),
        "unique_code_id_count": len(code_counts),
        "dominant_code_id_ratio": _ratio(top_counts[:1], launch_count),
        "top2_code_id_ratio": _ratio(top_counts[:2], launch_count),
        "top3_code_id_ratio": _ratio(top_counts[:3], launch_count),
        "normalized_entropy": normalized_entropy(code_counts),
        "grid_stability": _ratio([grid_counts.most_common(1)[0][1]] if grid_counts else [], launch_count),
        "block_stability": _ratio([block_counts.most_common(1)[0][1]] if block_counts else [], launch_count),
        "avg_launch_gap_us": avg_gap,
        "launch_gap_std": gap_std,
        "launch_gap_cv": gap_std / avg_gap if avg_gap > 0 else 0.0,
    }


def assign_window_ids(windows: list[L0Window]) -> list[L0Window]:
    assigned = []
    for idx, window in enumerate(windows):
        assigned.append(
            L0Window(
                window_id=f"w{idx:04d}_{window.window_type}_stream",
                window_type=window.window_type,
                group_kind=window.group_kind,
                group_key=window.group_key,
                launches=window.launches,
                features=window.features,
                trigger_reason=window.trigger_reason,
                packing_mode=window.packing_mode,
                condensation=window.condensation,
            )
        )
    return assigned


def grouped_launches(
    launches: list[dict[str, Any]],
    config: L0WindowConfig,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    by_stream: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for launch in launches:
        key = _group_tuple(launch, include_tid=config.grouping.include_tid_in_group)
        by_stream.setdefault(key, []).append(launch)
    return [
        ("stream", _group_key_from_tuple(key, include_tid=config.grouping.include_tid_in_group), _sorted_launches(rows))
        for key, rows in sorted(by_stream.items(), key=lambda item: str(item[0]))
    ]


def window_duration_ms(launches: list[dict[str, Any]]) -> float:
    if len(launches) < 2:
        return 0.0
    start = _timestamp_ns(launches[0])
    end = _timestamp_ns(launches[-1])
    if start is None or end is None or end < start:
        timestamps = [_timestamp_ns(launch) for launch in launches]
        timestamps = [ts for ts in timestamps if ts is not None]
        if len(timestamps) < 2:
            return 0.0
        return (max(timestamps) - min(timestamps)) / 1_000_000.0
    return (end - start) / 1_000_000.0


def launch_gaps_us(launches: list[dict[str, Any]]) -> list[float]:
    timestamps = [_timestamp_ns(launch) for launch in launches]
    timestamps = [ts for ts in timestamps if ts is not None]
    return [
        (right - left) / 1000.0
        for left, right in zip(timestamps, timestamps[1:])
        if right >= left
    ]


def _group_tuple(launch: dict[str, Any], include_tid: bool) -> tuple[Any, ...]:
    values: list[Any] = [launch.get("pid"), launch.get("device_pci_bus_id"), launch.get("stream")]
    if include_tid:
        values.append(launch.get("tid"))
    return tuple(values)


def _group_key_from_tuple(key: tuple[Any, ...], include_tid: bool) -> dict[str, Any]:
    result = {"pid": key[0], "device_pci_bus_id": key[1], "stream": key[2]}
    if include_tid:
        result["tid"] = key[3]
    return result


def _sorted_launches(launches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(launches, key=event_sort_key)


def _timestamp_ns(launch: dict[str, Any]) -> int | None:
    value = launch.get("timestamp_ns")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_key(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value)


def _ratio(counts: list[int], total: int) -> float:
    return float(sum(counts) / total) if total else 0.0


def normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    unique = len(counts)
    if total <= 0 or unique <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(unique)


def _stddev(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
