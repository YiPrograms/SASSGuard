"""Rolling-window L0 launch scheduling from READY CUDA kernel launches."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .ingest import event_sort_key
from .l0_config import L0WindowConfig


@dataclass
class StreamState:
    group_key: dict[str, Any]
    launches: deque[dict[str, Any]] = field(default_factory=deque)
    cost: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    ratios: dict[str, float] = field(default_factory=dict)
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=set)
    evicted_launches: int = 0
    dropped_unready_launches: int = 0


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


class L0WindowScheduler:
    """Per-stream rolling scheduler with token-budget eviction and composition dedup."""

    def __init__(self, config: L0WindowConfig):
        self.config = config
        self.states: dict[tuple[Any, ...], StreamState] = {}
        self.next_window_index = 0

    def add_launch(self, launch: dict[str, Any]) -> list[L0Window]:
        if not self.config.enabled:
            return []
        emitted = self._add_launch_unassigned(launch)
        return [self._assign_window_id(window) for window in emitted]

    def add_launches(self, launches: list[dict[str, Any]], *, sort: bool = False) -> list[L0Window]:
        rows = _sorted_launches(launches) if sort else launches
        windows: list[L0Window] = []
        for launch in rows:
            windows.extend(self.add_launch(launch))
        return windows

    def flush(self) -> list[L0Window]:
        if not self.config.enabled:
            return []
        emitted: list[L0Window] = []
        for state in self.states.values():
            window = self._candidate_window(state, trigger="offline_final_flush")
            if window is not None:
                emitted.append(self._assign_window_id(window))
        return emitted

    def _add_launch_unassigned(self, launch: dict[str, Any]) -> list[L0Window]:
        key = _group_tuple(launch, include_tid=self.config.grouping.include_tid_in_group)
        state = self.states.setdefault(
            key,
            StreamState(_group_key_from_tuple(key, include_tid=self.config.grouping.include_tid_in_group)),
        )
        ready = ready_launch_attributes(launch)
        if ready is None:
            state.dropped_unready_launches += 1
            return []

        row = dict(launch)
        kernel_id = ready["kernel_id"]
        token_cost = ready["token_cost"]
        ratio = ready["bitwise_integer_ratio"]
        row["l0_kernel_id"] = kernel_id
        row["l0_token_cost"] = token_cost
        row["l0_bitwise_integer_ratio"] = ratio
        state.launches.append(row)
        state.cost += token_cost
        state.counts[kernel_id] += 1
        state.ratios[kernel_id] = ratio

        budget = self.config.window.content_token_budget
        if state.cost <= budget:
            return []

        before_eviction_count = state.evicted_launches
        window = self._candidate_window(state, trigger="token_budget_overflow")
        self._evict_to_budget(state)
        post_emit_evicted = state.evicted_launches - before_eviction_count
        if window is not None:
            window.features["post_emit_evicted_launches"] = post_emit_evicted
            return [window]
        return []

    def _candidate_window(self, state: StreamState, trigger: str) -> L0Window | None:
        if not state.launches or not state.counts:
            return None

        max_ratio = max(state.ratios[kernel_id] for kernel_id in state.counts)
        gate_enabled = self.config.trigger.use_bitwise_gate
        if gate_enabled and max_ratio < self.config.trigger.bitwise_integer_ratio_threshold:
            return None

        signature = composition_signature(state.counts)
        if signature in state.seen_signatures:
            return None
        state.seen_signatures.add(signature)

        launches = list(state.launches)
        return L0Window(
            window_id="",
            window_type="dynamic",
            group_kind="stream",
            group_key=state.group_key,
            launches=launches,
            features={
                **launch_features(launches),
                "token_cost": state.cost,
                "pre_clip_token_cost": state.cost,
                "content_token_budget": self.config.window.content_token_budget,
                "kernel_set": list(signature[0]),
                "top3_kernels": list(signature[1]),
                "max_bitwise_integer_ratio": max_ratio,
                "bitwise_integer_gate_enabled": gate_enabled,
                "bitwise_integer_ratio_threshold": self.config.trigger.bitwise_integer_ratio_threshold,
                "distinct_kernel_count": len(state.counts),
                "evicted_launches": state.evicted_launches,
                "dropped_unready_launches": state.dropped_unready_launches,
                "post_emit_evicted_launches": 0,
                "front_clipped": False,
                "composition_signature": {
                    "kernel_set": list(signature[0]),
                    "top3": list(signature[1]),
                },
            },
            trigger_reason=[
                "int_bitwise_gate" if gate_enabled else "int_bitwise_gate_disabled",
                "composition_signature_new",
                trigger,
            ],
            packing_mode="rolling_token_window",
            condensation={
                "enabled": False,
                "applied": False,
                "original_launch_count": len(launches),
                "selected_launch_count": len(launches),
            },
        )

    def _evict_to_budget(self, state: StreamState) -> None:
        budget = self.config.window.content_token_budget
        while state.launches and state.cost > budget:
            evicted = state.launches.popleft()
            evicted_id = str(evicted["l0_kernel_id"])
            state.cost -= int(evicted["l0_token_cost"])
            state.counts[evicted_id] -= 1
            if state.counts[evicted_id] <= 0:
                del state.counts[evicted_id]
                state.ratios.pop(evicted_id, None)
            state.evicted_launches += 1

    def _assign_window_id(self, window: L0Window) -> L0Window:
        assigned = L0Window(
            window_id=f"w{self.next_window_index:04d}_{window.window_type}_stream",
            window_type=window.window_type,
            group_kind=window.group_kind,
            group_key=window.group_key,
            launches=window.launches,
            features=window.features,
            trigger_reason=window.trigger_reason,
            packing_mode=window.packing_mode,
            condensation=window.condensation,
        )
        self.next_window_index += 1
        return assigned


def build_l0_windows(launches: list[dict[str, Any]], config: L0WindowConfig) -> list[L0Window]:
    scheduler = L0WindowScheduler(config)
    windows = scheduler.add_launches(launches, sort=True)
    windows.extend(scheduler.flush())
    return windows


def ready_launch_attributes(launch: dict[str, Any]) -> dict[str, Any] | None:
    kernel_id = launch.get("l0_kernel_id")
    token_cost = launch.get("l0_token_cost")
    ratio = launch.get("l0_bitwise_integer_ratio")
    if kernel_id is None or token_cost is None or ratio is None:
        return None
    try:
        token_cost_int = int(token_cost)
        ratio_float = float(ratio)
    except (TypeError, ValueError):
        return None
    if token_cost_int <= 0:
        return None
    return {
        "kernel_id": str(kernel_id),
        "token_cost": token_cost_int,
        "bitwise_integer_ratio": max(0.0, min(1.0, ratio_float)),
    }


def composition_signature(counts: Counter[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kernel_set = tuple(sorted(str(kernel_id) for kernel_id in counts))
    return kernel_set, tuple(top3_kernels(counts))


def top3_kernels(counts: Counter[str]) -> list[str]:
    return [
        kernel_id
        for kernel_id, _count in sorted(
            ((str(kernel_id), int(count)) for kernel_id, count in counts.items() if count > 0),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    ]


def proportional_condense_launches(
    launches: list[dict[str, Any]],
    target_emitted_launches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Legacy helper retained for older reports/tests; not used by rolling L0."""
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
