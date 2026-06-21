from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json


POTT_METRICS = [
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "power_usage_watts",
    "temperature_celsius",
    "fan_speed_pct",
]


TANANA_METRICS = [
    "process_gpu_utilization_pct",
    "process_gpu_memory_pct",
    "host_ram_gb",
]

HYBRID_PROCESS_METRICS = [
    "process_gpu_utilization_pct",
    "process_gpu_memory_pct",
    "host_ram_gb",
]


def build_features(
    input_dir: Path,
    output_dir: Path,
    *,
    window_sec: float = 60.0,
    step_sec: float = 5.0,
    min_valid_sample_ratio: float = 0.8,
    sampling_interval_s: float = 1.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    tanana_rows: list[dict[str, Any]] = []
    pott_rows: list[dict[str, Any]] = []
    pott_point_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []
    labels = Counter()

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") in {"skipped_missing_executable"}:
            continue
        labels[str(metadata.get("binary_label"))] += 1
        device_samples = read_jsonl(run_dir / "device_metrics.jsonl") if (run_dir / "device_metrics.jsonl").exists() else []
        process_samples = read_jsonl(run_dir / "process_metrics.jsonl") if (run_dir / "process_metrics.jsonl").exists() else []
        if device_samples:
            pott_point_rows.extend(pott_points(device_samples, metadata))
            pott_rows.extend(
                pott_windows(
                    device_samples,
                    metadata,
                    window_sec,
                    step_sec,
                    min_valid_sample_ratio,
                    sampling_interval_s,
                )
            )
        if process_samples:
            tanana_rows.extend(
                tanana_windows(
                    process_samples,
                    metadata,
                    window_sec,
                    step_sec,
                    min_valid_sample_ratio,
                    sampling_interval_s,
                )
            )
        if device_samples and process_samples:
            hybrid_rows.extend(
                hybrid_windows(
                    process_samples,
                    device_samples,
                    metadata,
                    window_sec,
                    step_sec,
                    min_valid_sample_ratio,
                    sampling_interval_s,
                )
            )

    write_csv(output_dir / "tanana_windows.csv", tanana_rows)
    write_csv(output_dir / "pott_windows.csv", pott_rows)
    write_csv(output_dir / "pott_points.csv", pott_point_rows)
    write_csv(output_dir / "hybrid_windows.csv", hybrid_rows)
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "runs_scanned": len(run_dirs),
        "labels": dict(labels),
        "tanana_windows": len(tanana_rows),
        "pott_windows": len(pott_rows),
        "pott_points": len(pott_point_rows),
        "hybrid_windows": len(hybrid_rows),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "min_valid_sample_ratio": min_valid_sample_ratio,
        "sampling_interval_s": sampling_interval_s,
    }
    write_json(output_dir / "build_report.json", report)
    return report


def pott_points(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("run_id")), str(row.get("gpu_uuid")))].append(row)
    out: list[dict[str, Any]] = []
    for (run_id, gpu_uuid), samples in grouped.items():
        samples = sorted([row for row in samples if isinstance(row.get("timestamp_ns"), int)], key=lambda row: row["timestamp_ns"])
        for idx, sample in enumerate(samples):
            feature_row = base_feature_row(metadata, run_id, idx)
            feature_row.update(
                {
                    "gpu_uuid": gpu_uuid,
                    "baseline": "pott_point",
                    "sample_index": idx,
                    "timestamp_ns": sample.get("timestamp_ns"),
                }
            )
            for metric in POTT_METRICS:
                feature_row[metric] = numeric_or_none(sample.get(metric))
            out.append(feature_row)
    return out


def pott_windows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    window_sec: float,
    step_sec: float,
    min_valid_sample_ratio: float,
    sampling_interval_s: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("run_id")), str(row.get("gpu_uuid")))].append(row)
    out: list[dict[str, Any]] = []
    for (run_id, gpu_uuid), samples in grouped.items():
        for idx, window_samples in enumerate(iter_windows(samples, window_sec, step_sec, min_valid_sample_ratio, sampling_interval_s)):
            feature_row = base_feature_row(metadata, run_id, idx)
            feature_row.update({"gpu_uuid": gpu_uuid, "baseline": "pott"})
            for metric in POTT_METRICS:
                feature_row.update(prefixed_stats(metric, values(window_samples, metric)))
            out.append(feature_row)
    return out


def tanana_windows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    window_sec: float,
    step_sec: float,
    min_valid_sample_ratio: float,
    sampling_interval_s: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pid = row.get("pid")
        if pid is None:
            continue
        grouped[(str(row.get("run_id")), str(row.get("gpu_uuid")), int(pid))].append(row)
    out: list[dict[str, Any]] = []
    for (run_id, gpu_uuid, pid), samples in grouped.items():
        for idx, window_samples in enumerate(iter_windows(samples, window_sec, step_sec, min_valid_sample_ratio, sampling_interval_s)):
            util_values = values(window_samples, "process_gpu_utilization_pct")
            feature_row = base_feature_row(metadata, run_id, idx)
            feature_row.update(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "baseline": "tanana",
                    "tanana_status": "tanana_paper_exact" if util_values else "tanana_insufficient_process_gpu_util",
                    "avg_process_gpu_utilization_pct": mean(util_values),
                    "std_process_gpu_utilization_pct": stddev(util_values),
                    "avg_process_gpu_memory_pct": mean(values(window_samples, "process_gpu_memory_pct")),
                    "avg_host_ram_gb": mean(values(window_samples, "host_ram_gb")),
                    "sample_count": len(window_samples),
                }
            )
            out.append(feature_row)
    return out


def hybrid_windows(
    process_rows: list[dict[str, Any]],
    device_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    window_sec: float,
    step_sec: float,
    min_valid_sample_ratio: float,
    sampling_interval_s: float,
) -> list[dict[str, Any]]:
    devices_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in device_rows:
        devices_by_key[(str(row.get("run_id")), str(row.get("gpu_uuid")))].append(row)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in process_rows:
        pid = row.get("pid")
        if pid is None:
            continue
        grouped[(str(row.get("run_id")), str(row.get("gpu_uuid")), int(pid))].append(row)

    out: list[dict[str, Any]] = []
    for (run_id, gpu_uuid, pid), samples in grouped.items():
        matching_device_rows = sorted(
            devices_by_key.get((run_id, gpu_uuid), []),
            key=lambda row: row.get("timestamp_ns") or 0,
        )
        for idx, window_samples in enumerate(iter_windows(samples, window_sec, step_sec, min_valid_sample_ratio, sampling_interval_s)):
            timestamps = [row["timestamp_ns"] for row in window_samples if isinstance(row.get("timestamp_ns"), int)]
            if not timestamps:
                continue
            start_ns = min(timestamps)
            end_ns = max(timestamps)
            window_device_samples = [
                row
                for row in matching_device_rows
                if isinstance(row.get("timestamp_ns"), int) and start_ns <= row["timestamp_ns"] <= end_ns
            ]
            feature_row = base_feature_row(metadata, run_id, idx)
            feature_row.update(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "baseline": "hybrid_process_device_rf",
                    "window_start_ns": start_ns,
                    "window_end_ns": end_ns,
                    "sample_count_process": len(window_samples),
                    "sample_count_device": len(window_device_samples),
                }
            )
            for metric in HYBRID_PROCESS_METRICS:
                feature_row.update(prefixed_stats(hybrid_process_prefix(metric), values(window_samples, metric)))
            for metric in POTT_METRICS:
                feature_row.update(prefixed_stats(f"device_{metric}", values(window_device_samples, metric)))
            out.append(feature_row)
    return out


def hybrid_process_prefix(metric: str) -> str:
    if metric.startswith("process_"):
        return metric
    return f"process_{metric}"


def iter_windows(
    samples: list[dict[str, Any]],
    window_sec: float,
    step_sec: float,
    min_valid_sample_ratio: float,
    sampling_interval_s: float,
) -> list[list[dict[str, Any]]]:
    samples = sorted([row for row in samples if isinstance(row.get("timestamp_ns"), int)], key=lambda row: row["timestamp_ns"])
    if not samples:
        return []
    first = samples[0]["timestamp_ns"] / 1_000_000_000
    last = samples[-1]["timestamp_ns"] / 1_000_000_000
    duration = max(0.0, last - first)
    if duration <= window_sec:
        return [samples] if enough_samples(samples, duration, min_valid_sample_ratio, sampling_interval_s, adaptive=True) else []

    windows: list[list[dict[str, Any]]] = []
    start = first
    while start + window_sec <= last + 1e-9:
        end = start + window_sec
        window_samples = [row for row in samples if start <= row["timestamp_ns"] / 1_000_000_000 <= end]
        if enough_samples(window_samples, window_sec, min_valid_sample_ratio, sampling_interval_s, adaptive=False):
            windows.append(window_samples)
        start += step_sec
    return windows


def enough_samples(
    samples: list[dict[str, Any]],
    duration_s: float,
    min_valid_sample_ratio: float,
    sampling_interval_s: float,
    *,
    adaptive: bool,
) -> bool:
    if adaptive:
        return len(samples) >= 2 or (duration_s < sampling_interval_s and len(samples) >= 1)
    expected = max(1, int(math.floor(duration_s / sampling_interval_s)))
    return len(samples) >= math.ceil(expected * min_valid_sample_ratio)


def base_feature_row(metadata: dict[str, Any], run_id: str, window_index: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "window_index": window_index,
        "workload": metadata.get("workload"),
        "label": metadata.get("label"),
        "binary_label": metadata.get("binary_label"),
        "family": metadata.get("family"),
        "program": metadata.get("program"),
        "variant": metadata.get("variant"),
        "experiment": metadata.get("experiment"),
        "throttle_mode": metadata.get("throttle_mode"),
        "target_percent": metadata.get("target_percent"),
        "repeat": metadata.get("repeat"),
        "runtime_sec": metadata.get("runtime_sec"),
    }


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prefixed_stats(prefix: str, xs: list[float]) -> dict[str, float | None]:
    return {
        f"{prefix}_mean": mean(xs),
        f"{prefix}_std": stddev(xs),
        f"{prefix}_min": min(xs) if xs else None,
        f"{prefix}_max": max(xs) if xs else None,
        f"{prefix}_median": statistics.median(xs) if xs else None,
        f"{prefix}_p95": percentile(xs, 0.95),
    }


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def stddev(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return statistics.pstdev(xs)


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
