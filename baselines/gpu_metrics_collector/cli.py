from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .collector import RecordRequest, command_is_runnable, record_command, write_skip_metadata
from .io import append_jsonl
from .manifest import expand_manifest_patterns, load_manifest_rows, row_matches
from .pott_ml import train_point_random_forest, train_random_forest
from .tanana import evaluate as evaluate_tanana
from .windowize import build_features


DEFAULT_RAW_DIR = Path("baseline_dataset/gpu_metrics/raw")
DEFAULT_FEATURE_DIR = Path("baseline_dataset/gpu_metrics/features")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-metrics-collector")
    sub = parser.add_subparsers(required=True)

    record = sub.add_parser("record", help="Record one command with GPU metrics")
    add_common_record_args(record)
    record.add_argument("--label", required=True)
    record.add_argument("--binary-label")
    record.add_argument("--family")
    record.add_argument("--program")
    record.add_argument("--variant")
    record.add_argument("--workload", required=True)
    record.add_argument("--cwd", type=Path)
    record.add_argument("command", nargs=argparse.REMAINDER)
    record.set_defaults(func=cmd_record)

    run_manifest = sub.add_parser("run-manifest", help="Run workloads from existing capture manifests")
    add_common_record_args(run_manifest)
    run_manifest.add_argument("--manifest", action="append", help="Manifest path or glob; defaults to all known workload manifests")
    run_manifest.add_argument("--match", help="Only run rows whose workload/program/family/label contains this text")
    run_manifest.add_argument("--limit", type=int, help="Maximum number of matched rows to attempt")
    run_manifest.add_argument("--timeout-sec", type=float, default=120.0, help="Per-workload timeout for manifest runs")
    run_manifest.set_defaults(func=cmd_run_manifest)

    windowize = sub.add_parser("windowize", help="Build Tanana and Pott window features")
    windowize.add_argument("--input-dir", type=Path, default=DEFAULT_RAW_DIR)
    windowize.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    windowize.add_argument("--window-sec", type=float, default=60.0)
    windowize.add_argument("--step-sec", type=float, default=5.0)
    windowize.add_argument("--min-valid-sample-ratio", type=float, default=0.8)
    windowize.add_argument("--sampling-interval-sec", type=float, default=1.0)
    windowize.set_defaults(func=cmd_windowize)

    tanana = sub.add_parser("eval-tanana", help="Evaluate Tanana paper-exact decision tree")
    tanana.add_argument("--input-csv", type=Path, default=DEFAULT_FEATURE_DIR / "tanana_windows.csv")
    tanana.add_argument("--output-csv", type=Path, default=DEFAULT_FEATURE_DIR / "tanana_results.csv")
    tanana.add_argument("--report", type=Path, default=DEFAULT_FEATURE_DIR / "tanana_report.json")
    tanana.set_defaults(func=cmd_eval_tanana)

    pott = sub.add_parser("train-pott-rf", help="Train/evaluate Pott Random Forest baseline")
    pott.add_argument("--input-csv", type=Path, default=DEFAULT_FEATURE_DIR / "pott_windows.csv")
    pott.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR / "pott_rf")
    pott.add_argument("--seed", type=int, default=1337)
    pott.set_defaults(func=cmd_train_pott_rf)

    pott_point = sub.add_parser("train-pott-point-rf", help="Train/evaluate pointwise Pott RF baseline with temporal alarm threshold")
    pott_point.add_argument("--input-csv", type=Path, default=DEFAULT_FEATURE_DIR / "pott_points.csv")
    pott_point.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR / "pott_point_rf")
    pott_point.add_argument("--seed", type=int, default=1337)
    pott_point.add_argument("--max-run-fpr", type=float, default=0.01, help="Maximum benign run false-alarm rate allowed during threshold tuning")
    pott_point.set_defaults(func=cmd_train_pott_point_rf)

    hybrid = sub.add_parser("train-hybrid-rf", help="Train/evaluate process+device Random Forest baseline")
    hybrid.add_argument("--input-csv", type=Path, default=DEFAULT_FEATURE_DIR / "hybrid_windows.csv")
    hybrid.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR / "hybrid_rf")
    hybrid.add_argument("--seed", type=int, default=1337)
    hybrid.set_defaults(func=cmd_train_hybrid_rf)
    return parser


def add_common_record_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to expose as logical CUDA device 0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--sampling-interval-sec", type=float, default=1.0)


def cmd_record(args: argparse.Namespace) -> int:
    command = strip_command_separator(args.command)
    if not command:
        raise SystemExit("record requires a command after --")
    metadata = record_command(
        RecordRequest(
            command=command,
            output_dir=args.output_dir,
            gpu=args.gpu,
            workload=args.workload,
            label=args.label,
            binary_label=args.binary_label,
            family=args.family,
            program=args.program,
            variant=args.variant,
            cwd=args.cwd,
            sampling_interval_s=args.sampling_interval_sec,
        )
    )
    append_jsonl(args.output_dir / "manifest.jsonl", metadata)
    print(f"recorded {metadata['run_id']} status={metadata['status']}")
    return 0


def cmd_run_manifest(args: argparse.Namespace) -> int:
    manifests = expand_manifest_patterns(args.manifest)
    rows = [row for row in load_manifest_rows(manifests) if row_matches(row, args.match)]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempted = 0
    for row in rows:
        argv = list(row.get("argv") or [])
        cwd = Path(row["cwd"]) if row.get("cwd") else None
        attempted += 1
        if not command_is_runnable(argv, cwd):
            metadata = write_skip_metadata(args.output_dir, row, "skipped_missing_executable")
            append_jsonl(args.output_dir / "manifest.jsonl", metadata)
            print(f"skipped {row.get('workload')} missing executable")
            continue
        request = RecordRequest(
            command=argv,
            output_dir=args.output_dir,
            gpu=args.gpu,
            workload=str(row.get("workload") or row.get("program") or "workload"),
            label=str(row.get("label") or ""),
            binary_label=row.get("binary_label"),
            family=row.get("family"),
            program=row.get("program"),
            variant=row.get("variant"),
            cwd=cwd,
            sampling_interval_s=args.sampling_interval_sec,
            timeout_s=args.timeout_sec,
            extra_metadata={
                "source_manifest": row.get("_manifest_path"),
                "source_manifest_line": row.get("_manifest_line"),
            },
        )
        metadata = record_command(request)
        append_jsonl(args.output_dir / "manifest.jsonl", metadata)
        print(f"recorded {metadata['run_id']} status={metadata['status']}")
    print(f"attempted {attempted} manifest rows")
    return 0


def cmd_windowize(args: argparse.Namespace) -> int:
    report = build_features(
        args.input_dir,
        args.output_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        min_valid_sample_ratio=args.min_valid_sample_ratio,
        sampling_interval_s=args.sampling_interval_sec,
    )
    print(
        f"wrote features: tanana={report['tanana_windows']} "
        f"pott={report['pott_windows']} pott_points={report['pott_points']} "
        f"hybrid={report['hybrid_windows']}"
    )
    return 0


def cmd_eval_tanana(args: argparse.Namespace) -> int:
    report = evaluate_tanana(args.input_csv, args.output_csv, args.report)
    print(f"wrote Tanana results: windows={report['windows']} accuracy={report['accuracy']}")
    return 0


def cmd_train_pott_rf(args: argparse.Namespace) -> int:
    report = train_random_forest(args.input_csv, args.output_dir, args.seed, baseline_name="pott_rf")
    print(f"wrote Pott RF report to {args.output_dir / 'pott_rf_report.json'}")
    if "test" in report:
        print(f"test accuracy={report['test']['accuracy']}")
    return 0


def cmd_train_pott_point_rf(args: argparse.Namespace) -> int:
    report = train_point_random_forest(args.input_csv, args.output_dir, args.seed, max_run_fpr=args.max_run_fpr)
    print(f"wrote Pott point RF report to {args.output_dir / 'pott_point_rf_report.json'}")
    if "run_level" in report and "test" in report["run_level"]:
        print(f"test run accuracy={report['run_level']['test']['accuracy']}")
    print(f"alarm policy={report['alarm_policy']}")
    return 0


def cmd_train_hybrid_rf(args: argparse.Namespace) -> int:
    report = train_random_forest(args.input_csv, args.output_dir, args.seed, baseline_name="hybrid_rf")
    print(f"wrote Hybrid RF report to {args.output_dir / 'hybrid_rf_report.json'}")
    if "test" in report:
        print(f"test accuracy={report['test']['accuracy']}")
    return 0


def strip_command_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
