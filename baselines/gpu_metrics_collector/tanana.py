from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def evaluate(input_csv: Path, output_csv: Path, report_path: Path) -> dict[str, Any]:
    rows = read_csv(input_csv)
    evaluated: list[dict[str, Any]] = []
    counts = Counter()
    correct = 0
    comparable = 0
    for row in rows:
        verdict = tanana_verdict(row)
        row = dict(row)
        row["tanana_verdict"] = verdict
        counts[verdict] += 1
        label = row.get("binary_label")
        if label in {"benign", "mining"} and verdict in {"benign", "suspicious_mining"}:
            comparable += 1
            if (label == "mining" and verdict == "suspicious_mining") or (label == "benign" and verdict == "benign"):
                correct += 1
        evaluated.append(row)
    write_csv(output_csv, evaluated)
    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "windows": len(evaluated),
        "verdict_counts": dict(counts),
        "comparable_windows": comparable,
        "accuracy": (correct / comparable) if comparable else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def tanana_verdict(row: dict[str, Any]) -> str:
    if row.get("tanana_status") == "tanana_insufficient_process_gpu_util":
        return "tanana_insufficient_process_gpu_util"
    avg_gpu = as_float(row.get("avg_process_gpu_utilization_pct"))
    avg_mem = as_float(row.get("avg_process_gpu_memory_pct"))
    avg_ram = as_float(row.get("avg_host_ram_gb"))
    std_gpu = as_float(row.get("std_process_gpu_utilization_pct"))
    if None in {avg_gpu, avg_mem, avg_ram, std_gpu}:
        return "tanana_insufficient_metrics"
    if avg_gpu <= 80:
        return "benign"
    if avg_mem <= 90:
        return "benign"
    if avg_ram < 3 or avg_ram > 4.5:
        return "benign"
    if std_gpu >= 3.5:
        return "benign"
    return "suspicious_mining"


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
