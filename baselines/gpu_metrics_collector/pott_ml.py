from __future__ import annotations

import csv
import json
import random
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POINT_FEATURE_COLUMNS = [
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "power_usage_watts",
    "temperature_celsius",
    "fan_speed_pct",
]


def train_random_forest(input_csv: Path, output_dir: Path, seed: int = 1337, *, baseline_name: str = "pott_rf") -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for train-pott-rf; install baselines/gpu_metrics_collector/requirements.txt") from exc

    rows = [row for row in read_csv(input_csv) if row.get("binary_label") in {"benign", "mining"}]
    candidate_features = [key for key in rows[0].keys() if is_feature_column(key)] if rows else []
    feature_columns = usable_feature_columns(rows, candidate_features)
    rows = [row for row in rows if complete(row, feature_columns)]
    labels = {row.get("binary_label") for row in rows}
    if len(labels) < 2:
        raise SystemExit("train-pott-rf requires at least one benign and one mining window with complete usable features")
    if not feature_columns:
        raise SystemExit("train-pott-rf found no complete numeric feature columns")
    splits = grouped_split(rows, seed)
    x_train = matrix(splits["train"], feature_columns)
    y_train = [row["binary_label"] for row in splits["train"]]
    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    model.fit(x_train, y_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "input_csv": str(input_csv),
        "feature_columns": feature_columns,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "seed": seed,
    }
    predictions: list[dict[str, Any]] = []
    for split_name, split_rows in splits.items():
        if not split_rows:
            continue
        y_true = [row["binary_label"] for row in split_rows]
        y_pred = list(model.predict(matrix(split_rows, feature_columns)))
        report[split_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["benign", "mining"]).tolist(),
            "labels": ["benign", "mining"],
        }
        prediction_column = "pott_prediction" if baseline_name == "pott_rf" else f"{baseline_name}_prediction"
        for row, pred in zip(split_rows, y_pred):
            predictions.append(
                {
                    "split": split_name,
                    "run_id": row.get("run_id"),
                    "workload": row.get("workload"),
                    "pid": row.get("pid"),
                    "binary_label": row.get("binary_label"),
                    prediction_column: pred,
                }
            )
    write_csv(output_dir / f"{baseline_name}_predictions.csv", predictions)
    (output_dir / f"{baseline_name}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import joblib

        joblib.dump({"model": model, "feature_columns": feature_columns}, output_dir / f"{baseline_name}_model.joblib")
    except ImportError:
        report["model_artifact"] = "skipped_joblib_unavailable"
    return report


def train_point_random_forest(
    input_csv: Path,
    output_dir: Path,
    seed: int = 1337,
    *,
    max_run_fpr: float = 0.01,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for train-pott-point-rf; install baselines/gpu_metrics_collector/requirements.txt") from exc

    rows = [row for row in read_csv(input_csv) if row.get("binary_label") in {"benign", "mining"}]
    feature_columns = usable_feature_columns(rows, POINT_FEATURE_COLUMNS)
    rows = [row for row in rows if complete(row, feature_columns)]
    labels = {row.get("binary_label") for row in rows}
    if len(labels) < 2:
        raise SystemExit("train-pott-point-rf requires at least one benign and one mining point with complete usable features")
    if not feature_columns:
        raise SystemExit("train-pott-point-rf found no complete numeric feature columns")

    splits = grouped_split(rows, seed)
    x_train = matrix(splits["train"], feature_columns)
    y_train = [row["binary_label"] for row in splits["train"]]
    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    model.fit(x_train, y_train)

    sample_predictions: list[dict[str, Any]] = []
    sample_reports: dict[str, Any] = {}
    run_predictions_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_rows in splits.items():
        if not split_rows:
            continue
        y_true = [row["binary_label"] for row in split_rows]
        y_pred = list(model.predict(matrix(split_rows, feature_columns)))
        sample_reports[split_name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["benign", "mining"]).tolist(),
            "labels": ["benign", "mining"],
        }
        split_sample_predictions = []
        mining_probabilities = mining_probability(model, matrix(split_rows, feature_columns))
        for row, pred, probability in zip(split_rows, y_pred, mining_probabilities):
            pred_row = {
                "split": split_name,
                "run_id": row.get("run_id"),
                "workload": row.get("workload"),
                "sample_index": row.get("sample_index"),
                "timestamp_ns": row.get("timestamp_ns"),
                "binary_label": row.get("binary_label"),
                "pott_point_prediction": pred,
                "pott_point_mining_probability": probability,
            }
            sample_predictions.append(pred_row)
            split_sample_predictions.append({**row, "pott_point_prediction": pred, "pott_point_mining_probability": probability})
        run_predictions_by_split[split_name] = run_alarm_rows(split_sample_predictions)

    alarm_policy = tune_alarm_policy(
        run_predictions_by_split.get("val") or run_predictions_by_split.get("train") or [],
        max_fpr=max_run_fpr,
    )
    run_reports: dict[str, Any] = {}
    run_predictions: list[dict[str, Any]] = []
    for split_name, run_rows in run_predictions_by_split.items():
        evaluated = [apply_alarm_policy(row, alarm_policy) for row in run_rows]
        y_true = [row["binary_label"] for row in evaluated]
        y_pred = [row["run_prediction"] for row in evaluated]
        run_reports[split_name] = {
            "accuracy": accuracy_score(y_true, y_pred) if y_true else None,
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0) if y_true else {},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["benign", "mining"]).tolist() if y_true else [],
            "labels": ["benign", "mining"],
        }
        for row in evaluated:
            run_predictions.append({"split": split_name, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "input_csv": str(input_csv),
        "feature_columns": feature_columns,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "run_split_sizes": {name: len(items) for name, items in run_predictions_by_split.items()},
        "seed": seed,
        "max_run_fpr": max_run_fpr,
        "alarm_policy": alarm_policy,
        "sample_level": sample_reports,
        "run_level": run_reports,
    }
    write_csv(output_dir / "pott_point_rf_sample_predictions.csv", sample_predictions)
    write_csv(output_dir / "pott_point_rf_run_predictions.csv", run_predictions)
    (output_dir / "pott_point_rf_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import joblib

        joblib.dump({"model": model, "feature_columns": feature_columns, "alarm_policy": alarm_policy}, output_dir / "pott_point_rf_model.joblib")
    except ImportError:
        report["model_artifact"] = "skipped_joblib_unavailable"
    return report


def is_feature_column(name: str) -> bool:
    suffixes = ("_mean", "_std", "_min", "_max", "_median", "_p95")
    return name.endswith(suffixes)


def run_alarm_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_id") or row.get("workload") or "")].append(row)
    out = []
    for run_id, run_rows in grouped.items():
        labels = Counter(row.get("binary_label") for row in run_rows)
        label = labels.most_common(1)[0][0]
        positives = sum(1 for row in run_rows if row.get("pott_point_prediction") == "mining")
        probabilities = [float(row.get("pott_point_mining_probability") or 0.0) for row in run_rows]
        total = len(run_rows)
        first_positive_index = None
        for idx, row in enumerate(sorted(run_rows, key=lambda item: int(float(item.get("timestamp_ns") or item.get("sample_index") or 0)))):
            if row.get("pott_point_prediction") == "mining":
                first_positive_index = idx
                break
        first = run_rows[0]
        out.append(
            {
                "run_id": run_id,
                "workload": first.get("workload"),
                "binary_label": label,
                "positive_count": positives,
                "sample_count": total,
                "positive_ratio": positives / total if total else 0.0,
                "mean_mining_probability": sum(probabilities) / len(probabilities) if probabilities else 0.0,
                "max_mining_probability": max(probabilities) if probabilities else 0.0,
                "first_positive_index": first_positive_index,
            }
        )
    return out


def tune_alarm_policy(run_rows: list[dict[str, Any]], *, max_fpr: float | None = None) -> dict[str, Any]:
    if not run_rows:
        return empty_alarm_policy(max_fpr)
    return tune_mean_probability_policy(run_rows, max_fpr=max_fpr)


def empty_alarm_policy(max_fpr: float | None) -> dict[str, Any]:
    return {
        "aggregation": "mean_probability",
        "min_mean_mining_probability": 0.5,
        "max_fpr": max_fpr,
        "selection_metric": "balanced_accuracy_with_fpr_constraint",
    }


def tune_mean_probability_policy(run_rows: list[dict[str, Any]], *, max_fpr: float | None = None) -> dict[str, Any]:
    thresholds = sorted(
        {
            0.0,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            *[round(float(row.get("mean_mining_probability") or 0.0), 4) for row in run_rows],
        }
    )
    feasible: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    fallback: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for threshold in thresholds:
        y_true = [row["binary_label"] for row in run_rows]
        y_pred = ["mining" if float(row.get("mean_mining_probability") or 0.0) >= threshold else "benign" for row in run_rows]
        correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
        accuracy = correct / len(y_true)
        balanced = balanced_accuracy(y_true, y_pred)
        mining_total = sum(1 for truth in y_true if truth == "mining")
        mining_detected = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "mining" and pred == "mining")
        benign_total = sum(1 for truth in y_true if truth == "benign")
        benign_false_alarms = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "benign" and pred == "mining")
        mining_recall = mining_detected / mining_total if mining_total else 0.0
        fpr = benign_false_alarms / benign_total if benign_total else 0.0
        fired = sum(1 for pred in y_pred if pred == "mining")
        candidate = {
            "aggregation": "mean_probability",
            "min_mean_mining_probability": threshold,
            "max_fpr": max_fpr,
            "tuning_accuracy": accuracy,
            "tuning_balanced_accuracy": balanced,
            "tuning_mining_recall": mining_recall,
            "tuning_benign_fpr": fpr,
            "tuning_runs": len(run_rows),
            "tuning_alarms": fired,
            "selection_metric": "balanced_accuracy_with_fpr_constraint",
        }
        key = (balanced, accuracy, mining_recall, -fpr, threshold)
        if max_fpr is None or fpr <= max_fpr + 1e-12:
            feasible.append((key, candidate))
        fallback.append(((-fpr, balanced, accuracy, mining_recall, threshold), candidate))
    if feasible:
        return max(feasible, key=lambda item: item[0])[1]
    if fallback:
        candidate = max(fallback, key=lambda item: item[0])[1]
        candidate["selection_metric"] = "fallback_lowest_fpr_then_balanced_accuracy"
        return candidate
    return empty_alarm_policy(max_fpr)


def apply_alarm_policy(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["alarm_aggregation"] = "mean_probability"
    out["alarm_min_mean_mining_probability"] = policy.get("min_mean_mining_probability", "")
    out["run_prediction"] = "mining" if alarm_fires(row, policy) else "benign"
    return out


def alarm_fires(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    return float(row.get("mean_mining_probability") or 0.0) >= float(policy.get("min_mean_mining_probability", 0.5))


def mining_probability(model: Any, rows: list[list[float]]) -> list[float]:
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if "mining" in classes:
            index = classes.index("mining")
            return [float(prob[index]) for prob in model.predict_proba(rows)]
    return [1.0 if pred == "mining" else 0.0 for pred in model.predict(rows)]


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    recalls = []
    for label in sorted(set(y_true)):
        total = sum(1 for truth in y_true if truth == label)
        if not total:
            continue
        correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        recalls.append(correct / total)
    return sum(recalls) / len(recalls) if recalls else math.nan


def complete(row: dict[str, str], features: list[str]) -> bool:
    for feature in features:
        if row.get(feature) in {None, ""}:
            return False
        try:
            float(row[feature])
        except ValueError:
            return False
    return True


def usable_feature_columns(rows: list[dict[str, str]], candidates: list[str]) -> list[str]:
    usable: list[str] = []
    for feature in candidates:
        if all(is_number(row.get(feature)) for row in rows):
            usable.append(feature)
    return usable


def is_number(value: str | None) -> bool:
    if value in {None, ""}:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def matrix(rows: list[dict[str, str]], features: list[str]) -> list[list[float]]:
    return [[float(row[feature]) for feature in features] for row in rows]


def grouped_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    groups = sorted({group_key(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    train_end = max(1, int(n * 0.70)) if n else 0
    val_end = train_end + max(1, int(n * 0.15)) if n - train_end > 1 else train_end
    split_for_group = {}
    for group in groups[:train_end]:
        split_for_group[group] = "train"
    for group in groups[train_end:val_end]:
        split_for_group[group] = "val"
    for group in groups[val_end:]:
        split_for_group[group] = "test"
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[split_for_group[group_key(row)]].append(row)
    return splits


def group_key(row: dict[str, str]) -> str:
    workload = row.get("workload") or row.get("run_id") or ""
    return re.sub(r"_(o2|o3)$", "", workload.lower())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
