"""Grouped train/validation/test split helpers for classification."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .manifest import write_json


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
OPT_SUFFIX_RE = re.compile(r"_(?P<opt>o[0-9]+)$", re.IGNORECASE)


class SplitError(RuntimeError):
    """Raised when dataset splits cannot be generated or validated."""


def load_workload_records(workloads_dir: Path, dataset_root: Path | None = None) -> list[dict[str, Any]]:
    if not workloads_dir.is_dir():
        raise SplitError(f"missing workloads directory: {workloads_dir}")
    dataset_root = dataset_root or workloads_dir.parent

    records: list[dict[str, Any]] = []
    for manifest_path in sorted(workloads_dir.glob("*/manifest.json")):
        workload_dir = manifest_path.parent
        workload_sass = workload_dir / "workload.sass"
        if not workload_sass.exists():
            raise SplitError(f"missing workload.sass for {workload_dir.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workload = str(manifest["workload"])
        records.append(
            {
                "workload": workload,
                "path": _display_path(workload_sass, dataset_root),
                "label": str(manifest["label"]),
                "binary_label": "mining" if manifest["label"] == "mining_like" else "benign",
                "family": str(manifest["family"]),
                "opt_level": str(manifest["opt_level"]),
                "group_id": group_id_for_workload(workload),
            }
        )
    if not records:
        raise SplitError(f"no workload manifests found in {workloads_dir}")
    return records


def group_id_for_workload(workload: str) -> str:
    return OPT_SUFFIX_RE.sub("", workload)


def make_grouped_stratified_split(
    records: list[dict[str, Any]],
    seed: int = 1337,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    ratios = ratios or DEFAULT_RATIOS
    validate_ratios(ratios)
    groups = group_records(records)
    grouped_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped_by_label[group["label"]].append(group)

    rng = random.Random(seed)
    split_groups: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(grouped_by_label):
        label_groups = grouped_by_label[label]
        rng.shuffle(label_groups)
        counts = split_group_counts(len(label_groups), ratios)
        cursor = 0
        for split_name in SPLIT_NAMES:
            take = counts[split_name]
            split_groups[split_name].extend(label_groups[cursor : cursor + take])
            cursor += take

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for group in sorted(split_groups[split_name], key=lambda item: item["group_id"]):
            splits[split_name].extend(sorted(group["records"], key=lambda item: item["workload"]))
    return splits


def group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)

    groups = []
    for group_id, group_records_ in sorted(by_group.items()):
        labels = {record["label"] for record in group_records_}
        if len(labels) != 1:
            raise SplitError(f"group {group_id} has multiple labels: {sorted(labels)}")
        groups.append(
            {
                "group_id": group_id,
                "label": group_records_[0]["label"],
                "records": sorted(group_records_, key=lambda item: item["workload"]),
            }
        )
    return groups


def split_group_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: n * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    remainder = n - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - counts[name], ratios[name]), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1

    if n >= len(SPLIT_NAMES):
        for name in SPLIT_NAMES:
            if counts[name] == 0:
                donor = max(SPLIT_NAMES, key=lambda item: counts[item])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] += 1
    return counts


def validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise SplitError(f"ratios must contain exactly: {', '.join(SPLIT_NAMES)}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"split ratios must sum to 1.0, got {total}")
    if any(value <= 0 for value in ratios.values()):
        raise SplitError("split ratios must be positive")


def validate_splits(splits: dict[str, list[dict[str, Any]]], all_records: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    seen_workloads: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    all_workloads = {record["workload"] for record in all_records}
    all_labels = {record["label"] for record in all_records}

    for split_name, records in splits.items():
        labels = Counter(record["label"] for record in records)
        opts = Counter(record["opt_level"] for record in records)
        for record in records:
            workload = record["workload"]
            if workload in seen_workloads:
                raise SplitError(f"workload {workload} appears in multiple splits")
            seen_workloads[workload] = split_name

            group_id = record["group_id"]
            previous = group_to_split.setdefault(group_id, split_name)
            if previous != split_name:
                raise SplitError(f"group {group_id} appears in {previous} and {split_name}")

        missing_labels = sorted(all_labels - set(labels))
        if missing_labels:
            warnings.append(f"{split_name} missing labels: {', '.join(missing_labels)}")
        if opts and abs(opts.get("O2", 0) - opts.get("O3", 0)) > 2:
            warnings.append(f"{split_name} has opt-level imbalance: {dict(opts)}")

    if set(seen_workloads) != all_workloads:
        missing = sorted(all_workloads - set(seen_workloads))
        raise SplitError(f"workloads missing from splits: {missing[:5]}")

    label_counts = Counter(record["label"] for record in all_records)
    for label, count in sorted(label_counts.items()):
        if count < 20:
            warnings.append(f"small class {label}: {count} workloads")
    return warnings


def write_splits(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    all_records: list[dict[str, Any]],
    seed: int,
    ratios: dict[str, float],
    warnings: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        write_jsonl(output_dir / f"{split_name}.jsonl", splits[split_name])

    manifest = {
        "seed": seed,
        "ratios": ratios,
        "total_examples": len(all_records),
        "total_groups": len({record["group_id"] for record in all_records}),
        "splits": {name: summarize_records(records) for name, records in splits.items()},
        "warnings": warnings,
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "groups": len({record["group_id"] for record in records}),
        "labels": dict(sorted(Counter(record["label"] for record in records).items())),
        "binary_labels": dict(sorted(Counter(record["binary_label"] for record in records).items())),
        "families": dict(sorted(Counter(record["family"] for record in records).items())),
        "opt_levels": dict(sorted(Counter(record["opt_level"] for record in records).items())),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, sort_keys=True)
            fh.write("\n")


def _display_path(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root.parent))
    except ValueError:
        return str(path)
