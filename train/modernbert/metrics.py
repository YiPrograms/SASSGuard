"""Metrics and aggregation helpers for workload-level ModernBERT evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import ChunkRecord, WorkloadRecord


DEFAULT_SUSPICIOUS_TOP_K = 3
DEFAULT_SUSPICIOUS_MAX_THRESHOLD = 0.90
DEFAULT_SUSPICIOUS_TOPK_THRESHOLD = 0.75
MINING_LABEL_CANDIDATES = ("mining", "mining_like")


def aggregate_chunk_probabilities(
    chunks: list[ChunkRecord],
    probabilities: list[list[float]],
    top_k: int = DEFAULT_SUSPICIOUS_TOP_K,
) -> dict[str, dict[str, Any]]:
    if len(chunks) != len(probabilities):
        raise ValueError("chunks and probabilities must have the same length")

    grouped: dict[str, list[list[float]]] = defaultdict(list)
    metadata: dict[str, ChunkRecord] = {}
    for chunk, probs in zip(chunks, probabilities):
        grouped[chunk.workload].append([float(value) for value in probs])
        metadata.setdefault(chunk.workload, chunk)

    aggregated: dict[str, dict[str, Any]] = {}
    for workload, rows in grouped.items():
        num_labels = len(rows[0])
        mean_probs = [sum(row[idx] for row in rows) / len(rows) for idx in range(num_labels)]
        pred_id = max(range(num_labels), key=lambda idx: mean_probs[idx])
        probability_pooling = {}
        for idx in range(num_labels):
            values = sorted((row[idx] for row in rows), reverse=True)
            k = min(max(1, top_k), len(values))
            probability_pooling[str(idx)] = {
                "mean": mean_probs[idx],
                "max": max(values),
                f"top{k}_mean": sum(values[:k]) / k,
            }
        first = metadata[workload]
        aggregated[workload] = {
            "workload": workload,
            "label": first.label,
            "label_id": first.label_id,
            "pred_id": pred_id,
            "probabilities": mean_probs,
            "probability_pooling": probability_pooling,
            "num_chunks": len(rows),
            "source_path": first.source_path,
        }
    return aggregated


def softmax_rows(logits: Any) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to compute softmax probabilities") from exc

    arr = np.asarray(logits, dtype="float64")
    arr = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(arr)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.tolist()


def workload_metrics(
    records: list[WorkloadRecord],
    chunks: list[ChunkRecord],
    probabilities: list[list[float]],
    id2label: dict[int, str],
    aggregate_by_group: bool = False,
    group_policy: str = "any_window_suspicious",
    mean_mining_probability_threshold: float = 0.30,
) -> dict[str, Any]:
    aggregated = aggregate_chunk_probabilities(chunks, probabilities)
    add_no_l0_window_predictions(records, aggregated, id2label)
    if aggregate_by_group:
        return grouped_workload_metrics(
            records,
            aggregated,
            id2label,
            group_policy=group_policy,
            mean_mining_probability_threshold=mean_mining_probability_threshold,
        )
    return prediction_metrics(records, aggregated, id2label)


def add_no_l0_window_predictions(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
) -> None:
    labels = sorted(id2label)
    benign_id = default_benign_label_id(id2label)
    for record in records:
        if not record.row.get("no_l0_window") or record.workload in aggregated:
            continue
        pred_id = benign_id if benign_id is not None else record.label_id
        probabilities = [0.0 for _ in labels]
        if pred_id in labels:
            probabilities[labels.index(pred_id)] = 1.0
        aggregated[record.workload] = {
            "workload": record.workload,
            "label": record.label,
            "label_id": record.label_id,
            "pred_id": pred_id,
            "probabilities": probabilities,
            "probability_pooling": {
                str(label_id): {
                    "mean": probabilities[idx],
                    "max": probabilities[idx],
                    "top1_mean": probabilities[idx],
                }
                for idx, label_id in enumerate(labels)
            },
            "num_chunks": 0,
            "source_path": str(record.source_path),
            "no_l0_window": True,
            "default_prediction": id2label.get(pred_id, str(pred_id)),
            "default_prediction_reason": str(
                record.row.get("default_prediction_reason") or "no_l0_window_emitted"
            ),
        }


def default_benign_label_id(id2label: dict[int, str]) -> int | None:
    for label_id, label in id2label.items():
        if str(label).lower() == "benign":
            return label_id
    mining_id = mining_label_id_for(id2label)
    for label_id in sorted(id2label):
        if label_id != mining_id:
            return label_id
    return None


def prediction_metrics(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
) -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute classification metrics") from exc

    labels = sorted(id2label)
    y_true = [aggregated[record.workload]["label_id"] for record in records]
    y_pred = [aggregated[record.workload]["pred_id"] for record in records]
    y_score = [aggregated[record.workload]["probabilities"] for record in records]
    suspicious_report = suspicious_metrics(records, aggregated, id2label)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        id2label[label_id]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label_id in enumerate(labels)
    }

    report: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[id2label[idx] for idx in labels],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predictions": [
            {
                **aggregated[record.workload],
                "pred_label": id2label[aggregated[record.workload]["pred_id"]],
                "probabilities_by_label": {
                    id2label[idx]: prob
                    for idx, prob in enumerate(aggregated[record.workload]["probabilities"])
                },
                **suspicious_prediction_fields(aggregated[record.workload], id2label),
            }
            for record in records
        ],
        "per_family": per_family_breakdown(records, aggregated, id2label),
        "suspicious_detection": suspicious_report,
    }
    try:
        report["roc_auc_ovr"] = float(roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr"))
    except ValueError:
        report["roc_auc_ovr"] = None
    return report


def grouped_workload_metrics(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
    group_policy: str = "any_window_suspicious",
    mean_mining_probability_threshold: float = 0.30,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute classification metrics") from exc

    labels = sorted(id2label)
    window_predictions = prediction_rows(records, aggregated, id2label)
    group_predictions = aggregate_group_predictions(
        records,
        window_predictions,
        id2label,
        group_policy=group_policy,
        mean_mining_probability_threshold=mean_mining_probability_threshold,
    )
    y_true = [row["label_id"] for row in group_predictions]
    y_pred = [row["pred_id"] for row in group_predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        id2label[label_id]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label_id in enumerate(labels)
    }
    report: dict[str, Any] = {
        "aggregation": f"group_id_{group_policy}",
        "group_policy": {
            "name": group_policy,
            "mean_mining_probability_threshold": float(mean_mining_probability_threshold),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[id2label[idx] for idx in labels],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predictions": group_predictions,
        "window_predictions": window_predictions,
        "per_family": grouped_per_family_breakdown(group_predictions, id2label),
        "suspicious_detection": grouped_suspicious_metrics(group_predictions, id2label),
        "roc_auc_ovr": None,
    }
    return report


def aggregate_group_predictions(
    records: list[WorkloadRecord],
    window_predictions: list[dict[str, Any]],
    id2label: dict[int, str],
    group_policy: str = "any_window_suspicious",
    mean_mining_probability_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    if group_policy not in {"any_window_suspicious", "mean_mining_probability"}:
        raise ValueError(f"unsupported group policy: {group_policy}")
    labels = sorted(id2label)
    mining_label_id = mining_label_id_for(id2label)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_by_group: dict[str, WorkloadRecord] = {}
    for record, prediction in zip(records, window_predictions):
        group_id = str(record.row.get("group_id") or record.row.get("parent_workload") or record.workload)
        by_group[group_id].append(prediction)
        record_by_group.setdefault(group_id, record)

    group_predictions: list[dict[str, Any]] = []
    for group_id, rows in sorted(by_group.items()):
        record = record_by_group[group_id]
        mean_probs = [
            sum(float(row["probabilities"][idx]) for row in rows) / len(rows)
            for idx in labels
        ]
        if mining_label_id is not None:
            mining_pos = labels.index(mining_label_id)
            mean_mining = float(mean_probs[mining_pos])
            any_window_suspicious = any(bool(row.get("suspicious")) for row in rows)
            if group_policy == "mean_mining_probability":
                suspicious = mean_mining >= mean_mining_probability_threshold
                suspicious_reason = (
                    f"mean_mining_probability>={mean_mining_probability_threshold:g}"
                    if suspicious
                    else f"mean_mining_probability<{mean_mining_probability_threshold:g}"
                )
            else:
                suspicious = any_window_suspicious
                suspicious_reason = "any_window_suspicious" if suspicious else "all_windows_benign"
            if suspicious:
                pred_id = mining_label_id
            else:
                non_mining_labels = [idx for idx in labels if idx != mining_label_id] or labels
                pred_id = max(non_mining_labels, key=lambda idx: mean_probs[labels.index(idx)])
        else:
            pred_id = max(labels, key=lambda idx: mean_probs[labels.index(idx)])
            suspicious = False
            suspicious_reason = "no_mining_label"
            mean_mining = 0.0
        max_mining = (
            max(float(row.get("mining_probability_max", 0.0)) for row in rows)
            if mining_label_id is not None
            else 0.0
        )
        group_predictions.append(
            {
                "workload": group_id,
                "label": record.label,
                "label_id": record.label_id,
                "pred_id": pred_id,
                "pred_label": id2label[pred_id],
                "probabilities": mean_probs,
                "probabilities_by_label": {id2label[idx]: mean_probs[pos] for pos, idx in enumerate(labels)},
                "suspicious": suspicious,
                "suspicious_label": "suspicious" if suspicious else id2label[pred_id],
                "suspicious_reason": suspicious_reason,
                "mining_probability_max": max_mining,
                "mining_probability_mean": mean_mining,
                "group_policy": group_policy,
                "num_windows": len(rows),
                "windows": [row["workload"] for row in rows],
                "family": record.row.get("family"),
            }
        )
    return group_predictions


def prediction_rows(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        prediction = aggregated[record.workload]
        rows.append(
            {
                **prediction,
                "pred_label": id2label[prediction["pred_id"]],
                "probabilities_by_label": {
                    id2label[idx]: prob for idx, prob in enumerate(prediction["probabilities"])
                },
                **suspicious_prediction_fields(prediction, id2label),
                "group_id": record.row.get("group_id"),
                "parent_workload": record.row.get("parent_workload"),
                "no_l0_window": bool(record.row.get("no_l0_window")),
                "default_prediction_reason": prediction.get("default_prediction_reason"),
            }
        )
    return rows


def grouped_suspicious_metrics(group_predictions: list[dict[str, Any]], id2label: dict[int, str]) -> dict[str, Any]:
    mining_label_id = mining_label_id_for(id2label)
    if mining_label_id is None:
        return {
            "enabled": False,
            "reason": "no mining label present",
            "mining_label_candidates": list(MINING_LABEL_CANDIDATES),
        }
    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute suspicious metrics") from exc

    y_true = [row["label_id"] == mining_label_id for row in group_predictions]
    y_pred = [bool(row["suspicious"]) for row in group_predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[False, True], zero_division=0
    )
    return {
        "enabled": True,
        "mining_label": id2label[mining_label_id],
        "rule": {"any_window_suspicious": True},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[False, True]).tolist(),
        "not_suspicious": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "suspicious": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def grouped_per_family_breakdown(group_predictions: list[dict[str, Any]], id2label: dict[int, str]) -> dict[str, Any]:
    family_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for prediction in group_predictions:
        family = str(prediction.get("family") or "unknown")
        family_rows[family].append((int(prediction["label_id"]), int(prediction["pred_id"])))
    breakdown: dict[str, Any] = {}
    for family, rows in sorted(family_rows.items()):
        correct = sum(1 for truth, pred in rows if truth == pred)
        label_counts: dict[str, int] = defaultdict(int)
        pred_counts: dict[str, int] = defaultdict(int)
        for truth, pred in rows:
            label_counts[id2label[truth]] += 1
            pred_counts[id2label[pred]] += 1
        breakdown[family] = {
            "examples": len(rows),
            "accuracy": correct / len(rows) if rows else 0.0,
            "labels": dict(sorted(label_counts.items())),
            "predictions": dict(sorted(pred_counts.items())),
        }
    return breakdown


def suspicious_prediction_fields(
    prediction: dict[str, Any],
    id2label: dict[int, str],
    top_k: int = DEFAULT_SUSPICIOUS_TOP_K,
    max_threshold: float = DEFAULT_SUSPICIOUS_MAX_THRESHOLD,
    topk_threshold: float = DEFAULT_SUSPICIOUS_TOPK_THRESHOLD,
) -> dict[str, Any]:
    mining_label_id = mining_label_id_for(id2label)
    if mining_label_id is None:
        return {}

    pooling = prediction["probability_pooling"][str(mining_label_id)]
    k = min(max(1, top_k), max(1, int(prediction["num_chunks"])))
    topk_key = f"top{k}_mean"
    mining_mean = float(pooling["mean"])
    mining_max = float(pooling["max"])
    mining_topk = float(pooling[topk_key])
    mean_pooling_predicts_mining = int(prediction["pred_id"]) == mining_label_id
    suspicious = (
        mean_pooling_predicts_mining
        or mining_max >= max_threshold
        or mining_topk >= topk_threshold
    )
    if mean_pooling_predicts_mining:
        reason = "mean_pooling_decision"
    elif mining_max >= max_threshold:
        reason = f"max_p_mining>={max_threshold:g}"
    elif mining_topk >= topk_threshold:
        reason = f"top{k}_mean_p_mining>={topk_threshold:g}"
    else:
        reason = "below_threshold"
    return {
        "mining_probability_mean": mining_mean,
        "mining_probability_max": mining_max,
        f"mining_probability_top{k}_mean": mining_topk,
        "suspicious": suspicious,
        "suspicious_label": "suspicious" if suspicious else id2label[prediction["pred_id"]],
        "suspicious_reason": reason,
    }


def suspicious_metrics(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
) -> dict[str, Any]:
    mining_label_id = mining_label_id_for(id2label)
    if mining_label_id is None:
        return {
            "enabled": False,
            "reason": "no mining label present",
            "mining_label_candidates": list(MINING_LABEL_CANDIDATES),
        }

    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute suspicious metrics") from exc

    y_true = [aggregated[record.workload]["label_id"] == mining_label_id for record in records]
    y_pred = [
        bool(suspicious_prediction_fields(aggregated[record.workload], id2label)["suspicious"])
        for record in records
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[False, True], zero_division=0
    )
    return {
        "enabled": True,
        "mining_label": id2label[mining_label_id],
        "rule": {
            "mean_pooling_mining_prediction": True,
            "max_p_mining_threshold": DEFAULT_SUSPICIOUS_MAX_THRESHOLD,
            "top_k": DEFAULT_SUSPICIOUS_TOP_K,
            "topk_mean_p_mining_threshold": DEFAULT_SUSPICIOUS_TOPK_THRESHOLD,
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[False, True]).tolist(),
        "not_suspicious": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "suspicious": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def mining_label_id_for(id2label: dict[int, str]) -> int | None:
    label2id = {label: idx for idx, label in id2label.items()}
    for candidate in MINING_LABEL_CANDIDATES:
        if candidate in label2id:
            return label2id[candidate]
    return None


def per_family_breakdown(
    records: list[WorkloadRecord],
    aggregated: dict[str, dict[str, Any]],
    id2label: dict[int, str],
) -> dict[str, Any]:
    family_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        family = str(record.row.get("family", "unknown"))
        prediction = aggregated[record.workload]
        family_rows[family].append((prediction["label_id"], prediction["pred_id"]))

    breakdown: dict[str, Any] = {}
    for family, rows in sorted(family_rows.items()):
        correct = sum(1 for truth, pred in rows if truth == pred)
        label_counts: dict[str, int] = defaultdict(int)
        pred_counts: dict[str, int] = defaultdict(int)
        for truth, pred in rows:
            label_counts[id2label[truth]] += 1
            pred_counts[id2label[pred]] += 1
        breakdown[family] = {
            "examples": len(rows),
            "accuracy": correct / len(rows) if rows else 0.0,
            "labels": dict(sorted(label_counts.items())),
            "predictions": dict(sorted(pred_counts.items())),
        }
    return breakdown


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
