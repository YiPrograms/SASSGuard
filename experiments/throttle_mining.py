#!/usr/bin/env python3
"""Run throttle-mining captures and the final GPU-counter behavioral experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SYNTHETIC_ROOT = REPO_ROOT / "workloads" / "synthetic_kernels"
SASSGUARD_DATA = REPO_ROOT / "sassguard-data"
CAPTURE_DIR_BY_MODE = {
    "kernel-launch": SYNTHETIC_ROOT / "capture_throttle_kernel_launch",
    "duty-cycle": SYNTHETIC_ROOT / "capture_throttle_duty_cycle",
}
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_ALGORITHMS = ("ethash_split", "kawpow_split", "randomx_gpu_lite_mono", "sha256d_mono")
DEFAULT_BEHAVIORAL_ALGORITHMS = (
    "ethash_split",
    "kawpow_split",
    "randomx_gpu_lite_mono",
    "sha256d_mono",
    "autolykos2_split",
    "cryptonight_gpu_split",
    "cuckoo_cycle_split",
    "equihash144_5_split",
)
FAMILY_BY_PROGRAM = {
    "autolykos2_split": "memory_hard_table_hash",
    "cryptonight_gpu_split": "cryptonight_randomx_scratchpad",
    "cuckoo_cycle_split": "cuckoo_graph_cycle",
    "equihash144_5_split": "equihash_solver",
    "ethash_split": "ethash_dag_keccak",
    "kawpow_split": "progpow_kawpow_random_math",
    "randomx_gpu_lite_mono": "cryptonight_randomx_scratchpad",
    "sha256d_mono": "pure_hash_nonce_search",
}
CONDITIONS = (100, 75, 50, 25, 10, 5)
DEFAULT_SASS_CONDITIONS = (100, 75, 50, 10)
DEFAULT_BEHAVIORAL_CONDITIONS = CONDITIONS
DUTY_PERIODS = {
    100: (None, None),
    75: (45.0, 15.0),
    50: (5.0, 5.0),
    25: (5.0, 15.0),
    10: (5.0, 45.0),
    5: (3.0, 57.0),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("run-captures", help="Run throttle-mining captures")
    run.add_argument("--mode", choices=("kernel-launch", "duty-cycle", "all"), default="all")
    add_capture_args(run)
    run.set_defaults(func=cmd_run_captures)

    smoke = sub.add_parser("smoke", help="Run short one-condition capture checks")
    smoke.add_argument("--mode", choices=("kernel-launch", "duty-cycle", "all"), default="all")
    add_capture_args(smoke)
    smoke.set_defaults(runtime_sec=15, repeats=1, algorithms=["sha256d_mono"], conditions=[50], func=cmd_run_captures)

    eval_sass = sub.add_parser("evaluate-sassguard", help="Build SASSGuard datasets and optional L1 reports")
    eval_sass.add_argument("--mode", choices=("kernel-launch", "duty-cycle", "all"), default="all")
    eval_sass.add_argument("--jobs", type=int, default=8)
    eval_sass.add_argument("--l0-config", type=Path, default=REPO_ROOT / "configs" / "analysis" / "l0_windows.json")
    eval_sass.add_argument("--training-config", type=Path, default=REPO_ROOT / "configs" / "training" / "modernbert_sass_compact_binary_realworld.json")
    eval_sass.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models" / "modernbert" / "checkpoints" / "sass-modernbert-compact-binary" / "classifier" / "final")
    eval_sass.add_argument("--runtime-sec", type=int, help="Evaluate only manifest rows with this runtime_sec")
    eval_sass.add_argument("--repeats", type=int, help="Evaluate only manifest rows with repeat <= this value")
    eval_sass.set_defaults(func=cmd_evaluate_sassguard)

    behavioral = sub.add_parser("run-behavioral-throttle", help="Collect and score kernel-launch-throttled GPU-counter metrics")
    behavioral.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "throttle_behavioral_kernel_launch")
    behavioral.add_argument("--gpu-counter-model", type=Path, default=REPO_ROOT / "baseline_dataset" / "gpu_metrics" / "features" / "pott_point_rf" / "pott_point_rf_model.joblib")
    add_capture_args(
        behavioral,
        default_algorithms=DEFAULT_BEHAVIORAL_ALGORITHMS,
        default_conditions=DEFAULT_BEHAVIORAL_CONDITIONS,
        default_runtime_sec=60,
        default_repeats=1,
    )
    behavioral.set_defaults(func=cmd_run_behavioral_throttle)

    args = parser.parse_args(argv)
    return int(args.func(args))


def add_capture_args(
    parser: argparse.ArgumentParser,
    *,
    default_algorithms: tuple[str, ...] | list[str] = DEFAULT_ALGORITHMS,
    default_conditions: tuple[int, ...] | list[int] = DEFAULT_SASS_CONDITIONS,
    default_runtime_sec: int = 600,
    default_repeats: int = 3,
) -> None:
    parser.add_argument("--runtime-sec", type=int, default=default_runtime_sec)
    parser.add_argument("--repeats", type=int, default=default_repeats)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--opt-level", choices=("o2", "o3"), default="o3")
    parser.add_argument("--algorithms", nargs="+", choices=sorted(FAMILY_BY_PROGRAM), default=list(default_algorithms))
    parser.add_argument("--conditions", nargs="+", type=int, choices=CONDITIONS, default=list(default_conditions))
    parser.add_argument("--pilot-runtime-sec", type=int, default=15)
    parser.add_argument("--sassguard-data", type=Path, default=SASSGUARD_DATA)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")


def cmd_run_captures(args: argparse.Namespace) -> int:
    modes = ("kernel-launch", "duty-cycle") if args.mode == "all" else (args.mode,)
    ensure_capture_server_output(args.sassguard_data)
    for mode in modes:
        out_dir = CAPTURE_DIR_BY_MODE[mode]
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_rows = planned_rows(mode, args)
        existing = existing_keys(out_dir / "manifests.jsonl") if args.resume else set()
        sleep_by_program = {}
        if mode == "kernel-launch":
            sleep_by_program = dry_run_sleep_us(args) if args.dry_run else calibrate_sleep_us(args)
        for row in plan_rows:
            key = run_key(row)
            if key in existing:
                print(f"[SKIP] {mode} {row['program']} {row['target_percent']}% repeat={row['repeat']} already captured", flush=True)
                continue
            command = base_miner_command(row["program"], args.opt_level, args.runtime_sec)
            if mode == "kernel-launch":
                command.extend(["--sleep-between-launches-us", str(sleep_by_program[row["program"]][row["target_percent"]])])
            print(f"[RUN] {mode} {row['program']} {row['target_percent']}% repeat={row['repeat']}", flush=True)
            if args.dry_run:
                print("      " + " ".join(command), flush=True)
                continue
            metadata = run_capture(
                command,
                mode=mode,
                row=row,
                output_dir=out_dir,
                sassguard_data=args.sassguard_data,
                gpu=args.gpu,
                duty_period=DUTY_PERIODS[row["target_percent"]] if mode == "duty-cycle" else None,
            )
            append_jsonl(out_dir / "manifests.jsonl", metadata)
    return 0


def planned_rows(mode: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for program in args.algorithms:
        for pct in args.conditions:
            for repeat in range(1, args.repeats + 1):
                rows.append(
                    {
                        "experiment": "throttle_mining",
                        "throttle_mode": mode,
                        "target_percent": pct,
                        "repeat": repeat,
                        "runtime_sec": args.runtime_sec,
                        "program": program,
                        "family": FAMILY_BY_PROGRAM[program],
                        "variant": args.opt_level,
                        "workload": f"{program}_{args.opt_level}_{mode.replace('-', '_')}_{pct}pct_r{repeat}",
                    }
                )
    return rows


def existing_keys(manifest: Path) -> set[tuple[str, str, int, int, str]]:
    keys = set()
    if not manifest.exists():
        return keys
    for row in read_jsonl(manifest):
        keys.add(run_key(row))
    return keys


def run_key(row: dict[str, Any]) -> tuple[str, str, int, int, str, int]:
    return (
        str(row.get("throttle_mode")),
        str(row.get("program")),
        int(row.get("target_percent")),
        int(row.get("repeat")),
        str(row.get("variant")),
        int(row.get("runtime_sec") or 0),
    )


def dry_run_sleep_us(args: argparse.Namespace) -> dict[str, dict[int, int]]:
    return {program: {pct: 0 for pct in args.conditions} for program in args.algorithms}


def calibrate_sleep_us(args: argparse.Namespace) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for program in args.algorithms:
        command = base_miner_command(program, args.opt_level, args.pilot_runtime_sec)
        env = workload_env(args.gpu, capture=False)
        print(f"[PILOT] {program} {args.pilot_runtime_sec}s", flush=True)
        proc = subprocess.run(command, cwd=SYNTHETIC_ROOT / "scripts", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        launches = parse_total_launches(proc.stdout)
        if launches <= 0:
            raise RuntimeError(f"could not parse total_launches from pilot for {program}")
        base_lps = launches / max(1, args.pilot_runtime_sec)
        out[program] = {}
        for pct in args.conditions:
            if pct >= 100:
                out[program][pct] = 0
            else:
                out[program][pct] = max(1, round(1_000_000.0 * ((100.0 / pct) - 1.0) / base_lps))
        print(f"[PILOT] {program} base_loop_rate={base_lps:.3f}/s sleep_us={out[program]}", flush=True)
    return out


def parse_total_launches(stdout: str) -> int:
    match = re.search(r"^total_launches=(\d+)$", stdout, re.MULTILINE)
    return int(match.group(1)) if match else 0


def base_miner_command(program: str, opt_level: str, runtime_sec: int) -> list[str]:
    binary = SYNTHETIC_ROOT / "binaries" / f"{program}_{opt_level}"
    if not binary.exists():
        raise FileNotFoundError(f"missing synthetic binary: {binary}")
    return [str(binary), str(runtime_sec)]


def run_capture(
    command: list[str],
    *,
    mode: str,
    row: dict[str, Any],
    output_dir: Path,
    sassguard_data: Path,
    gpu: int,
    duty_period: tuple[float | None, float | None] | None,
) -> dict[str, Any]:
    before = list_capture_dirs(sassguard_data)
    env = workload_env(gpu, capture=True)
    stdout_dir = output_dir / "run_logs"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    log_stem = row["workload"]
    with (stdout_dir / f"{log_stem}.stdout.log").open("wb") as stdout, (stdout_dir / f"{log_stem}.stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=SYNTHETIC_ROOT / "scripts", env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        started_ns = time.time_ns()
        if duty_period and duty_period[0] is not None:
            drive_duty_cycle(proc, active_s=float(duty_period[0]), idle_s=float(duty_period[1]), runtime_s=float(row["runtime_sec"]))
        exit_code = proc.wait()
    finished_ns = time.time_ns()
    new_dirs = wait_for_new_captures(sassguard_data, before, expected_pid=proc.pid, expected_exe=Path(command[0]))
    copied = []
    manifest_rows = []
    for capture_dir in new_dirs:
        dest = output_dir / capture_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(capture_dir, dest)
        copied.append(dest)
        manifest_rows.append(capture_manifest_row(dest, output_dir, row, command, gpu, exit_code, started_ns, finished_ns))
    if not manifest_rows:
        raise RuntimeError(f"no new capture appeared for {row['workload']}")
    if len(manifest_rows) > 1:
        for idx, item in enumerate(manifest_rows, 1):
            item["capture_part"] = idx
            item["capture_parts"] = len(manifest_rows)
    return manifest_rows[0] if len(manifest_rows) == 1 else combined_capture_row(manifest_rows, row, command, gpu, exit_code, started_ns, finished_ns)


def drive_duty_cycle(proc: subprocess.Popen[bytes], *, active_s: float, idle_s: float, runtime_s: float) -> None:
    running = True
    start = time.monotonic()
    deadline = start + runtime_s
    next_switch = start + active_s
    while proc.poll() is None:
        now = time.monotonic()
        if now >= next_switch:
            try:
                if running:
                    if now + idle_s >= deadline:
                        next_switch = deadline
                        time.sleep(max(0.05, min(0.5, deadline - now)))
                        continue
                    os.killpg(proc.pid, signal.SIGSTOP)
                    running = False
                    next_switch = time.monotonic() + idle_s
                else:
                    os.killpg(proc.pid, signal.SIGCONT)
                    running = True
                    next_switch = time.monotonic() + active_s
            except ProcessLookupError:
                break
        time.sleep(0.1)
    if not running:
        try:
            os.killpg(proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def capture_manifest_row(
    capture_dir: Path,
    output_dir: Path,
    row: dict[str, Any],
    command: list[str],
    gpu: int,
    exit_code: int,
    started_ns: int,
    finished_ns: int,
) -> dict[str, Any]:
    process = read_json(capture_dir / "process.json")
    event_counts = event_type_counts(capture_dir / "events.jsonl")
    rel_path = capture_dir.relative_to(REPO_ROOT)
    metadata = {
        **row,
        "label": "mining_like",
        "binary_label": "mining",
        "capture_id": capture_dir.name,
        "capture_path": str(rel_path),
        "argv": command,
        "cwd": str(SYNTHETIC_ROOT / "scripts"),
        "gpu_selection": [{"env": "CUDA_VISIBLE_DEVICES", "source": "throttle_mining", "value": str(gpu)}],
        "single_gpu_intended": True,
        "exit_code": exit_code,
        "started_at_ns": started_ns,
        "finished_at_ns": finished_ns,
        "observed_duration_s": round((finished_ns - started_ns) / 1_000_000_000, 6),
        "event_type_counts": dict(event_counts),
        "event_count": sum(event_counts.values()),
        "pid": process.get("pid"),
        "exe_path": process.get("exe_path"),
    }
    return metadata


def combined_capture_row(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    command: list[str],
    gpu: int,
    exit_code: int,
    started_ns: int,
    finished_ns: int,
) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row.get("event_type_counts") or {})
    return {
        **source,
        "label": "mining_like",
        "binary_label": "mining",
        "capture_id": ",".join(row["capture_id"] for row in rows),
        "capture_path": rows[0]["capture_path"],
        "capture_paths": [row["capture_path"] for row in rows],
        "argv": command,
        "cwd": str(SYNTHETIC_ROOT / "scripts"),
        "gpu_selection": [{"env": "CUDA_VISIBLE_DEVICES", "source": "throttle_mining", "value": str(gpu)}],
        "single_gpu_intended": True,
        "exit_code": exit_code,
        "started_at_ns": started_ns,
        "finished_at_ns": finished_ns,
        "observed_duration_s": round((finished_ns - started_ns) / 1_000_000_000, 6),
        "event_type_counts": dict(counts),
        "event_count": sum(counts.values()),
    }


def cmd_evaluate_sassguard(args: argparse.Namespace) -> int:
    modes = ("kernel-launch", "duty-cycle") if args.mode == "all" else (args.mode,)
    for mode in modes:
        capture_dir = CAPTURE_DIR_BY_MODE[mode]
        manifest = capture_dir / "manifests.jsonl"
        result_dir = RESULTS_ROOT / "throttle_sassguard" / mode.replace("-", "_")
        dataset_dir = result_dir / "dataset"
        result_dir.mkdir(parents=True, exist_ok=True)
        if not manifest.exists():
            print(f"[SKIP] no manifest for {mode}: {manifest}")
            continue
        build_manifest = filtered_manifest(manifest, result_dir, args.runtime_sec, args.repeats)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        run_checked(
            [
                sys.executable,
                str(REPO_ROOT / "analysis" / "build_dataset.py"),
                "--capture-manifest",
                str(build_manifest),
                "--capture-root",
                str(REPO_ROOT),
                "--output-dir",
                str(dataset_dir),
                "--l0-config",
                str(args.l0_config),
                "--jobs",
                str(args.jobs),
                "--overwrite-existing",
            ],
            cwd=REPO_ROOT,
        )
        report = summarize_sassguard_dataset(dataset_dir, build_manifest)
        l1_status = maybe_run_l1_eval(args, dataset_dir, result_dir)
        report["l1_evaluation"] = l1_status
        write_json(result_dir / "throttle_sassguard_report.json", report)
        print(f"[REPORT] {result_dir / 'throttle_sassguard_report.json'}")
    return 0


def filtered_manifest(manifest: Path, result_dir: Path, runtime_sec: int | None, repeats: int | None = None) -> Path:
    if runtime_sec is None and repeats is None:
        return manifest
    rows = list(read_jsonl(manifest))
    suffix_parts: list[str] = []
    if runtime_sec is not None:
        rows = [row for row in rows if int(row.get("runtime_sec") or -1) == runtime_sec]
        suffix_parts.append(f"runtime{runtime_sec}")
    if repeats is not None:
        rows = [row for row in rows if int(row.get("repeat") or -1) <= repeats]
        suffix_parts.append(f"repeat1-{repeats}")
    if not rows:
        raise RuntimeError(f"no manifest rows in {manifest} match runtime_sec={runtime_sec} repeats={repeats}")
    filtered = result_dir / f"manifests_{'_'.join(suffix_parts)}.jsonl"
    write_jsonl(filtered, rows)
    return filtered


def summarize_sassguard_dataset(dataset_dir: Path, manifest: Path) -> dict[str, Any]:
    by_capture = {row.get("capture_id"): row for row in read_jsonl(manifest)}
    windows_by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    total_windows = 0
    for window_manifest in (dataset_dir / "workloads").glob("*/windows/manifests.jsonl"):
        for row in read_jsonl(window_manifest):
            total_windows += 1
            capture = by_capture.get(row.get("capture_id"), {})
            condition = f"{capture.get('program', 'unknown')}:{capture.get('target_percent', 'unknown')}%"
            windows_by_condition[condition][str(row.get("window_type") or "unknown")] += 1
    build_report = read_json(dataset_dir / "build_report.json") if (dataset_dir / "build_report.json").exists() else {}
    return {
        "dataset_dir": str(dataset_dir),
        "capture_manifest": str(manifest),
        "captures": len(by_capture),
        "total_l0_windows": total_windows,
        "windows_by_condition": {key: dict(value) for key, value in sorted(windows_by_condition.items())},
        "build_report": build_report,
    }


def maybe_run_l1_eval(args: argparse.Namespace, dataset_dir: Path, result_dir: Path) -> dict[str, Any]:
    checkpoint = args.checkpoint
    if not (checkpoint / "config.json").exists():
        return {"status": "skipped_missing_checkpoint", "checkpoint": str(checkpoint)}
    split_dir = result_dir / "splits_all_test"
    reports_dir = result_dir / "modernbert_l1"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    test_rows = collect_window_rows(dataset_dir)
    write_jsonl(split_dir / "train.jsonl", [])
    write_jsonl(split_dir / "val.jsonl", [])
    write_jsonl(split_dir / "test.jsonl", test_rows)
    write_json(split_dir / "split_manifest.json", {"mode": "all_windows_test", "test_rows": len(test_rows)})
    raw_config = read_json(args.training_config)
    raw_config["splits_dir"] = str(split_dir.relative_to(REPO_ROOT))
    raw_config["reports_dir"] = str(reports_dir.relative_to(REPO_ROOT))
    eval_config = result_dir / "modernbert_eval_config.json"
    write_json(eval_config, raw_config)
    command = [preferred_python(), "-m", "train.modernbert.evaluate", "--config", str(eval_config), "--checkpoint", str(checkpoint), "--split", "test"]
    try:
        run_checked(command, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        return {
            "status": "failed_l1_evaluator",
            "checkpoint": str(checkpoint),
            "test_rows": len(test_rows),
            "reports_dir": raw_config["reports_dir"],
            "exit_code": exc.returncode,
            "command": command,
        }
    return {"status": "ok", "checkpoint": str(checkpoint), "test_rows": len(test_rows), "reports_dir": raw_config["reports_dir"]}


def cmd_run_behavioral_throttle(args: argparse.Namespace) -> int:
    from baselines.gpu_metrics_collector.collector import RecordRequest, record_command
    from baselines.gpu_metrics_collector.io import append_jsonl as append_metrics_jsonl
    from baselines.gpu_metrics_collector.windowize import build_features

    raw_dir = args.output_dir / "raw"
    features_dir = args.output_dir / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_behavioral_keys(raw_dir / "manifest.jsonl") if args.resume else set()
    sleep_by_program = dry_run_sleep_us(args) if args.dry_run else calibrate_sleep_us(args)

    for row in planned_rows("kernel-launch", args):
        key = run_key(row)
        if key in existing:
            print(f"[SKIP] behavioral kernel-launch {row['program']} {row['target_percent']}% repeat={row['repeat']} already collected", flush=True)
            continue
        command = behavioral_command(row, args, sleep_by_program)
        print(f"[RUN] behavioral kernel-launch {row['program']} {row['target_percent']}% repeat={row['repeat']}", flush=True)
        if args.dry_run:
            print("      " + " ".join(command), flush=True)
            continue
        metadata = record_command(
            RecordRequest(
                command=command,
                output_dir=raw_dir,
                gpu=args.gpu,
                workload=f"{row['workload']}_behavioral",
                label="mining_like",
                binary_label="mining",
                family=row["family"],
                program=row["program"],
                variant=row["variant"],
                cwd=SYNTHETIC_ROOT / "scripts",
                timeout_s=args.runtime_sec + 90,
                extra_metadata={
                    "experiment": "throttle_mining_behavioral",
                    "throttle_mode": "kernel-launch",
                    "target_percent": row["target_percent"],
                    "repeat": row["repeat"],
                    "runtime_sec": row["runtime_sec"],
                },
            )
        )
        append_metrics_jsonl(raw_dir / "manifest.jsonl", metadata)

    if args.dry_run:
        return 0
    build_features(raw_dir, features_dir)
    report = score_behavioral_throttle_predictions(features_dir / "pott_points.csv", args.gpu_counter_model)
    write_json(args.output_dir / "throttle_behavioral_report.json", report)
    print(f"[REPORT] {args.output_dir / 'throttle_behavioral_report.json'}", flush=True)
    return 0


def behavioral_command(row: dict[str, Any], args: argparse.Namespace, sleep_by_program: dict[str, dict[int, int]]) -> list[str]:
    command = base_miner_command(row["program"], args.opt_level, args.runtime_sec)
    command.extend(["--sleep-between-launches-us", str(sleep_by_program[row["program"]][row["target_percent"]])])
    return command


def existing_behavioral_keys(manifest: Path) -> set[tuple[str, str, int, int, str, int]]:
    keys = set()
    for row in read_jsonl(manifest):
        if row.get("experiment") == "throttle_mining_behavioral":
            keys.add(run_key(row))
    return keys


def score_behavioral_throttle_predictions(csv_path: Path, model_path: Path) -> dict[str, Any]:
    return score_gpu_counter_point_predictions(
        csv_path,
        model_path,
        point_predictions_path=csv_path.parent / "gpu_counter_throttle_point_predictions.csv",
        run_predictions_path=csv_path.parent / "gpu_counter_throttle_alarm_predictions.csv",
        truth_label="mining",
        condition_key=lambda row: f"{row.get('program', 'unknown')}:{row.get('throttle_mode', 'unknown')}:{row.get('target_percent', 'unknown')}%",
    )


def collect_window_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest in sorted((dataset_dir / "workloads").glob("*/windows/manifests.jsonl")):
        workload_dir = manifest.parents[1]
        for row in read_jsonl(manifest):
            source_path = Path(str(row.get("path") or ""))
            if not source_path.is_absolute():
                row["path"] = str((workload_dir / source_path).relative_to(REPO_ROOT))
            rows.append(row)
    return rows


def score_gpu_counter_point_predictions(
    csv_path: Path,
    model_path: Path,
    *,
    point_predictions_path: Path,
    run_predictions_path: Path,
    truth_label: str,
    condition_key,
) -> dict[str, Any]:
    rows = read_csv(csv_path)
    if not model_path.exists():
        return {"status": "skipped_missing_gpu_counter_model", "model": str(model_path), "points": len(rows)}
    try:
        import joblib
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        return {"status": "skipped_missing_gpu_counter_dependencies", "model": str(model_path), "points": len(rows), "error": str(exc)}

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["feature_columns"]
    alarm_policy = bundle.get("alarm_policy") or {"aggregation": "mean_probability", "min_mean_mining_probability": 0.5}
    complete_rows = [row for row in rows if all(is_number(row.get(feature)) for feature in features)]
    point_y_true = []
    point_y_pred = []
    point_predictions = []
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    mining_probabilities = predict_mining_probabilities(model, complete_rows, features)
    for row, probability in zip(complete_rows, mining_probabilities):
        truth = truth_label
        pred = str(model.predict([[float(row[feature]) for feature in features]])[0])
        point_y_true.append(truth)
        point_y_pred.append(pred)
        by_condition[condition_key(row)][pred] += 1
        point_predictions.append({**row, "point_ground_truth": truth, "gpu_counter_rf_prediction": pred, "gpu_counter_rf_mining_probability": probability})
    write_csv(point_predictions_path, point_predictions)

    run_predictions = alarm_rows(point_predictions, alarm_policy, truth_label)
    write_csv(run_predictions_path, run_predictions)
    run_y_true = [row["run_ground_truth"] for row in run_predictions]
    run_y_pred = [row["run_prediction"] for row in run_predictions]
    return {
        "status": "ok",
        "model": str(model_path),
        "alarm_policy": alarm_policy,
        "points": len(complete_rows),
        "runs": len(run_predictions),
        "point_accuracy": accuracy_score(point_y_true, point_y_pred) if point_y_true else None,
        "point_classification_report": classification_report(point_y_true, point_y_pred, output_dict=True, zero_division=0) if point_y_true else {},
        "point_confusion_matrix": confusion_matrix(point_y_true, point_y_pred, labels=["benign", "mining"]).tolist() if point_y_true else [],
        "run_accuracy": accuracy_score(run_y_true, run_y_pred) if run_y_true else None,
        "run_classification_report": classification_report(run_y_true, run_y_pred, output_dict=True, zero_division=0) if run_y_true else {},
        "run_confusion_matrix": confusion_matrix(run_y_true, run_y_pred, labels=["benign", "mining"]).tolist() if run_y_true else [],
        "accuracy": accuracy_score(run_y_true, run_y_pred) if run_y_true else None,
        "confusion_matrix": confusion_matrix(run_y_true, run_y_pred, labels=["benign", "mining"]).tolist() if run_y_true else [],
        "labels": ["benign", "mining"],
        "point_predictions_by_condition": {key: dict(value) for key, value in sorted(by_condition.items())},
    }


def alarm_rows(point_rows: list[dict[str, Any]], alarm_policy: dict[str, Any], truth_label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        grouped[str(row.get("run_id") or row.get("workload") or "")].append(row)
    min_mean_probability = float(alarm_policy.get("min_mean_mining_probability", 0.5))
    out = []
    for run_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(float(row.get("timestamp_ns") or row.get("sample_index") or 0)))
        positive_count = sum(1 for row in rows if row.get("gpu_counter_rf_prediction") == "mining")
        probabilities = [float(row.get("gpu_counter_rf_mining_probability") or 0.0) for row in rows]
        sample_count = len(rows)
        positive_ratio = positive_count / sample_count if sample_count else 0.0
        mean_probability = sum(probabilities) / len(probabilities) if probabilities else 0.0
        run_prediction = "mining" if mean_probability >= min_mean_probability else "benign"
        first_positive_index = None
        for idx, row in enumerate(rows):
            if row.get("gpu_counter_rf_prediction") == "mining":
                first_positive_index = idx
                break
        first = rows[0] if rows else {}
        out.append(
            {
                "run_id": run_id,
                "workload": first.get("workload"),
                "program": first.get("program"),
                "variant": first.get("variant"),
                "throttle_mode": first.get("throttle_mode"),
                "target_percent": first.get("target_percent"),
                "sample_count": sample_count,
                "positive_count": positive_count,
                "positive_ratio": positive_ratio,
                "mean_mining_probability": mean_probability,
                "max_mining_probability": max(probabilities) if probabilities else 0.0,
                "first_positive_index": first_positive_index,
                "alarm_aggregation": "mean_probability",
                "alarm_min_mean_mining_probability": min_mean_probability,
                "run_ground_truth": truth_label,
                "run_prediction": run_prediction,
            }
        )
    return out


def predict_mining_probabilities(model: Any, rows: list[dict[str, str]], features: list[str]) -> list[float]:
    matrix = [[float(row[feature]) for feature in features] for row in rows]
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if "mining" in classes:
            index = classes.index("mining")
            return [float(prob[index]) for prob in model.predict_proba(matrix)]
    return [1.0 if pred == "mining" else 0.0 for pred in model.predict(matrix)]


def ensure_capture_server_output(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"capture server output directory does not exist: {path}")


def workload_env(gpu: int, *, capture: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("SASSGUARD_SERVER_ADDR", "127.0.0.1:59400")
    if capture:
        env.pop("SASSGUARD_CAPTURE_DISABLE", None)
        prepend_env_path(env, "LD_LIBRARY_PATH", REPO_ROOT / "collector" / "build")
    else:
        env["SASSGUARD_CAPTURE_DISABLE"] = "1"
    return env


def prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    current = env.get(key)
    env[key] = str(path) if not current else f"{path}:{current}"


def list_capture_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.iterdir() if path.is_dir()}


def wait_for_new_captures(
    root: Path,
    before: set[Path],
    *,
    expected_pid: int | None = None,
    expected_exe: Path | None = None,
    timeout_s: float = 60.0,
    settle_s: float = 2.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    seen: list[Path] = []
    while time.monotonic() < deadline:
        seen = sorted(list_capture_dirs(root) - before)
        ready = [path for path in seen if (path / "process.json").exists() and (path / "events.jsonl").exists()]
        ready = [path for path in ready if capture_matches(path, expected_pid, expected_exe)]
        if ready and all(file_stable(path / "events.jsonl", settle_s) for path in ready):
            return ready
        time.sleep(0.5)
    return [path for path in seen if capture_matches(path, expected_pid, expected_exe)]


def capture_matches(path: Path, expected_pid: int | None, expected_exe: Path | None) -> bool:
    if expected_pid is None and expected_exe is None:
        return True
    try:
        process = read_json(path / "process.json")
    except Exception:
        return False
    if expected_pid is not None and int(process.get("pid") or -1) != int(expected_pid):
        return False
    if expected_exe is not None:
        captured_exe = process.get("exe_path")
        if captured_exe and Path(str(captured_exe)).resolve() != expected_exe.resolve():
            return False
    return True


def file_stable(path: Path, settle_s: float) -> bool:
    try:
        first = path.stat().st_size
        time.sleep(settle_s)
        second = path.stat().st_size
        return first == second
    except OSError:
        return False


def event_type_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = row.get("type") or row.get("event_type")
            counts[str(event_type or "unknown")] += 1
    return counts


def run_checked(command: list[str], *, cwd: Path) -> None:
    print("[CMD] " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def preferred_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def is_number(value: str | None) -> bool:
    if value in (None, ""):
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
