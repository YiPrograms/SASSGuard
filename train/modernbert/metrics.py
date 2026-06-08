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

    aggregated = aggregate_chunk_probabilities(chunks, probabilities)
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
    k = min(max(1, top_k), int(prediction["num_chunks"]))
    topk_key = f"top{k}_mean"
    mining_mean = float(pooling["mean"])
    mining_max = float(pooling["max"])
    mining_topk = float(pooling[topk_key])
    suspicious = mining_max >= max_threshold or mining_topk >= topk_threshold
    if mining_max >= max_threshold:
        reason = f"max_p_mining>={max_threshold:g}"
    elif mining_topk >= topk_threshold:
        reason = f"top{k}_mean_p_mining>={topk_threshold:g}"
    else:
        reason = "mean_pooling_decision"
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
