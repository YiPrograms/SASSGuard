#!/usr/bin/env python3
"""Build a SASSGuard dataset from CUDA capture directories."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sassguard_analysis.cfg import build_cfg_for_kernel
from sassguard_analysis.disassemble import (
    DisassemblyError,
    disassemble_code_objects,
    find_cuda_tools,
    write_extraction_report,
)
from sassguard_analysis.ingest import (
    IngestError,
    copy_code_objects,
    read_events,
    read_jsonl,
    split_events,
    write_launches,
)
from sassguard_analysis.l0_config import DEFAULT_L0_CONFIG, L0ConfigError, load_l0_config
from sassguard_analysis.l0_windows import L0Window, build_l0_windows
from sassguard_analysis.loop_extract import extract_main_loop_for_kernel
from sassguard_analysis.manifest import (
    ManifestError,
    write_json,
    write_workload_manifest,
)
from sassguard_analysis.normalize import normalize_kernel_files
from sassguard_analysis.split_kernels import split_launched_kernels
from sassguard_analysis.validate import ValidationError, validate_workload
from sassguard_analysis.workload_sass import (
    WorkloadSassError,
    build_workload_sass,
    render_workload_sass,
)


class CaptureBuildError(RuntimeError):
    """Raised when one capture cannot produce a workload."""


OPT_LEVEL_CAPTURE = "capture"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-manifest",
        action="append",
        default=None,
        help=(
            "Build from one or more capture manifests.jsonl files. "
            "May be repeated and may contain shell-style globs. "
            "Example: --capture-manifest 'workloads/*_samples/*/captures/manifests.jsonl'."
        ),
    )
    parser.add_argument(
        "--capture-root",
        dest="capture_root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative capture_path entries from capture manifests.",
    )
    parser.add_argument(
        "--skip-empty-captures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip manifest captures that contain no kernel_launch events instead of failing the build.",
    )
    parser.add_argument(
        "--skip-unmapped-captures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip captures whose launched kernels cannot be mapped to disassembled SASS.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--l0-config", type=Path, default=DEFAULT_L0_CONFIG)
    parser.add_argument("--l0-windowing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-launches", type=int, default=16)
    parser.add_argument("--short-kernel-threshold", type=int, default=256)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of capture worker processes to run in parallel.",
    )
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--overwrite-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--keep-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    capture_manifest_patterns = list(
        args.capture_manifest or ["workloads/synthetic_kernels/captures/manifests.jsonl"]
    )

    try:
        capture_specs = load_capture_manifest_specs(
            capture_manifest_patterns,
            captures_root=args.capture_root,
        )
        l0_config = load_l0_config(args.l0_config)
        if args.l0_windowing is not None:
            l0_config = l0_config.with_enabled(bool(args.l0_windowing))
        args.l0_config_obj = l0_config
    except ManifestError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    except L0ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    tools = find_cuda_tools()
    if args.verbose:
        for name in ("nvdisasm", "cuobjdump"):
            print(f"[INFO] {name}: {tools.get(name, 'not found')}")

    report = new_build_report()
    report["l0_config"] = args.l0_config_obj.to_dict()
    if args.jobs < 1:
        print("[FATAL] --jobs must be >= 1", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[INFO] jobs: {args.jobs}")
        print(f"[INFO] captures: {len(capture_specs)}")

    total_captures = len(capture_specs)
    results = iter_capture_results(capture_specs, args, tools)
    for result in results:
        report["captures_scanned"] += 1
        progress = f"[{report['captures_scanned']}/{total_captures}]"
        if result["status"] == "failed":
            report["failed_captures"] += 1
            report["failures"].append({"capture": result["capture"], "reason": result["reason"]})
            print(f"{progress} [ERROR] {result['capture']}: {result['reason']}", flush=True)
            continue

        if result["status"] == "skipped":
            report["duplicates_skipped"] += 1
            if result.get("label"):
                report["labels"][result["label"]] += 1
            if result.get("opt_level"):
                report["opt_levels"][result["opt_level"]] += 1
            print(f"{progress} [SKIP] duplicate workload already exists: {result['workload']}", flush=True)
            continue
        if result["status"] == "skipped_empty":
            report["empty_captures_skipped"] += 1
            print(f"{progress} [SKIP] {result['capture']}: {result['reason']}", flush=True)
            continue
        if result["status"] == "skipped_unmapped":
            report["unmapped_captures_skipped"] += 1
            print(f"{progress} [SKIP] {result['capture']}: {result['reason']}", flush=True)
            continue
        if result["status"] == "dry_run":
            report["dry_run_ok"] += 1
            if args.verbose:
                print(f"{progress} [DRY-RUN] {result['workload']}", flush=True)
            continue

        created = int(result.get("workloads_created", 1))
        report["workloads_created"] += created
        report["windows_created"] += int(result.get("windows_created", 0))
        report["labels"][result["label"]] += created
        report["opt_levels"][result["opt_level"]] += created
        windows_created = int(result.get("windows_created", 0))
        suffix = f" ({windows_created} windows)" if windows_created else ""
        print(f"{progress} [OK] {result['workload']}{suffix}", flush=True)

    write_report(args.output_dir, report, dry_run=args.dry_run)
    print_summary(report)
    return 0 if report["failed_captures"] == 0 else 1


def iter_capture_results(
    capture_specs: list[dict[str, Any]],
    args: argparse.Namespace,
    tools: dict[str, Path],
):
    if args.jobs == 1 or len(capture_specs) <= 1:
        for capture_spec in capture_specs:
            yield process_capture_worker(capture_spec, args, tools)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_worker_star, (capture_spec, args, tools))
            for capture_spec in capture_specs
        ]
        for future in as_completed(futures):
            yield future.result()


def _worker_star(payload: tuple[dict[str, Any], argparse.Namespace, dict[str, Path]]):
    capture_spec, args, tools = payload
    return process_capture_worker(capture_spec, args, tools)


def process_capture_worker(
    capture_spec: dict[str, Any],
    args: argparse.Namespace,
    tools: dict[str, Path],
) -> dict[str, Any]:
    capture_dir = Path(capture_spec["capture_dir"])
    try:
        result = process_capture(capture_spec, args, tools)
        result["capture"] = capture_dir.name
        return result
    except CaptureBuildError as exc:
        return {"status": "failed", "capture": capture_dir.name, "reason": str(exc)}
    except Exception as exc:
        return {
            "status": "failed",
            "capture": capture_dir.name,
            "reason": f"unexpected {type(exc).__name__}: {exc}",
        }


def process_capture(
    capture_spec: dict[str, Any],
    args: argparse.Namespace,
    tools: dict[str, Path],
) -> dict[str, Any]:
    capture_dir = Path(capture_spec["capture_dir"])
    try:
        if args.skip_empty_captures and capture_spec_has_no_kernel_launch(capture_spec):
            return {
                "status": "skipped_empty",
                "workload": str(capture_spec["workload"]),
                "reason": "no kernel_launch event",
            }

        workload = str(capture_spec["workload"])
        manifest_entry = dict(capture_spec["manifest_entry"])
        l0_config = args.l0_config_obj

        workloads_root = args.output_dir / "workloads"
        final_dir = workloads_root / workload
        if final_dir.exists() and args.skip_existing:
            result = {"status": "skipped", "workload": workload}
            existing_manifest_path = final_dir / "manifest.json"
            if existing_manifest_path.exists():
                existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                result["label"] = existing_manifest.get("label")
                result["opt_level"] = existing_manifest.get("opt_level")
            return result
        if args.dry_run:
            events = read_events(capture_dir)
            split_events(events)
            return {"status": "dry_run", "workload": workload}

        workloads_root.mkdir(parents=True, exist_ok=True)
        temp_dir = workloads_root / f".{workload}.tmp.{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        try:
            if l0_config.enabled:
                created = _build_l0_windows_from_capture(
                    capture_dir, temp_dir, workload, args, manifest_entry, tools
                )
                if created == 0:
                    return {
                        "status": "skipped",
                        "workload": workload,
                        "label": manifest_entry["label"],
                        "opt_level": manifest_entry["opt_level"],
                    }
                if final_dir.exists() and not args.skip_existing:
                    shutil.rmtree(final_dir)
                temp_dir.replace(final_dir)
            else:
                _build_into_temp_dir(capture_dir, temp_dir, workload, args, manifest_entry, tools)
                if final_dir.exists() and args.skip_existing:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return {
                        "status": "skipped",
                        "workload": workload,
                        "label": manifest_entry["label"],
                        "opt_level": manifest_entry["opt_level"],
                    }
                if final_dir.exists() and not args.skip_existing:
                    shutil.rmtree(final_dir)
                temp_dir.replace(final_dir)
        except Exception:
            if args.keep_partial:
                print(f"[PARTIAL] kept partial workload at {temp_dir}")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            if l0_config.enabled and temp_dir.exists() and not args.keep_partial:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "status": "created",
            "workload": workload,
            "label": manifest_entry["label"],
            "opt_level": manifest_entry["opt_level"],
            "workloads_created": 1,
            "windows_created": created if l0_config.enabled else 0,
        }
    except IngestError as exc:
        if args.skip_empty_captures and str(exc) == "no kernel_launch event":
            return {
                "status": "skipped_empty",
                "workload": str(capture_spec["workload"]),
                "reason": str(exc),
            }
        raise CaptureBuildError(str(exc)) from exc
    except WorkloadSassError as exc:
        if args.skip_unmapped_captures and str(exc) == "no launched kernel could be mapped to disassembled SASS":
            return {
                "status": "skipped_unmapped",
                "workload": str(capture_spec["workload"]),
                "reason": str(exc),
            }
        raise CaptureBuildError(str(exc)) from exc
    except (DisassemblyError, ValidationError) as exc:
        raise CaptureBuildError(str(exc)) from exc


def _build_into_temp_dir(
    capture_dir: Path,
    workload_dir: Path,
    workload: str,
    args: argparse.Namespace,
    manifest_entry: dict[str, str],
    tools: dict[str, Path],
) -> None:
    (workload_dir / "dumps").mkdir(parents=True, exist_ok=True)
    (workload_dir / "kernels").mkdir(parents=True, exist_ok=True)
    write_workload_manifest(workload_dir / "manifest.json", workload, manifest_entry)

    launches, extraction_report = _build_capture_base(capture_dir, workload_dir, tools)
    workload_result = build_workload_sass(
        workload_dir,
        max_launches=args.max_launches,
        short_kernel_threshold=args.short_kernel_threshold,
    )
    extraction_report["missing_launches"] = workload_result["missing_launches"]
    extraction_report["included_launches"] = workload_result["included_launches"]
    write_extraction_report(workload_dir, extraction_report)
    validate_workload(workload_dir, max_launches=args.max_launches)


def _build_capture_base(
    capture_dir: Path,
    workload_dir: Path,
    tools: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_events(capture_dir)
    code_events, launch_events = split_events(events)
    code_map = copy_code_objects(capture_dir, code_events, workload_dir)
    launches = write_launches(workload_dir, launch_events)
    extraction_report = disassemble_code_objects(workload_dir, code_map, tools)

    kernel_dirs, missing_kernels = split_launched_kernels(
        workload_dir, launches, code_map, extraction_report
    )
    extraction_report["missing_kernels"] = missing_kernels
    write_extraction_report(workload_dir, extraction_report)
    if not kernel_dirs:
        raise WorkloadSassError("no launched kernel could be mapped to disassembled SASS")

    for kernel_dir in kernel_dirs.values():
        cfg = build_cfg_for_kernel(kernel_dir)
        extract_main_loop_for_kernel(kernel_dir, cfg)
        normalize_kernel_files(kernel_dir)

    return launches, extraction_report


def _build_l0_windows_from_capture(
    capture_dir: Path,
    workload_dir: Path,
    parent_workload: str,
    args: argparse.Namespace,
    manifest_entry: dict[str, str],
    tools: dict[str, Path],
) -> int:
    (workload_dir / "dumps").mkdir(parents=True, exist_ok=True)
    (workload_dir / "kernels").mkdir(parents=True, exist_ok=True)
    write_workload_manifest(workload_dir / "manifest.json", parent_workload, manifest_entry)
    launches, extraction_report = _build_capture_base(capture_dir, workload_dir, tools)
    windows = build_l0_windows(launches, args.l0_config_obj)
    if not windows:
        raise WorkloadSassError("no mature L0 windows could be built from launched kernels")

    windows_dir = workload_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    created = 0
    for window in windows:
        manifest_rows.append(
            _materialize_l0_window(
                workload_dir,
                windows_dir,
                safe_workload_name(f"{parent_workload}__{window.window_id}"),
                parent_workload,
                manifest_entry,
                window,
                args,
            )
        )
        created += 1
    write_jsonl(windows_dir / "manifests.jsonl", manifest_rows)

    extraction_report["l0_windows"] = {
        "count": created,
        "config_path": args.l0_config_obj.config_path,
        "resolved_config": args.l0_config_obj.to_dict(),
    }
    write_extraction_report(workload_dir, extraction_report)
    validate_l0_workload(workload_dir)
    return created


def _materialize_l0_window(
    workload_dir: Path,
    windows_dir: Path,
    window_workload: str,
    parent_workload: str,
    manifest_entry: dict[str, str],
    window: L0Window,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rendered = render_workload_sass(
        workload_dir,
        window.launches,
        short_kernel_threshold=args.short_kernel_threshold,
    )
    sass_name = f"{window.window_id}.sass"
    launch_name = f"{window.window_id}.launches.jsonl"
    meta_name = f"{window.window_id}.json"
    (windows_dir / sass_name).write_text(rendered["text"], encoding="utf-8")
    write_jsonl(windows_dir / launch_name, window.launches)

    row = {
        "workload": window_workload,
        "path": f"windows/{sass_name}",
        "label": manifest_entry["label"],
        "binary_label": manifest_entry.get(
            "binary_label",
            "mining" if manifest_entry["label"] == "mining_like" else "benign",
        ),
        "family": manifest_entry["family"],
        "opt_level": manifest_entry["opt_level"],
        "program": manifest_entry.get("program"),
        "variant": manifest_entry.get("variant"),
        "capture_id": manifest_entry.get("capture_id"),
        "parent_workload": parent_workload,
        "window_id": window.window_id,
        "window_type": window.window_type,
        "l0_group_kind": window.group_kind,
        "l0_group_key": window.group_key,
        "group_id": parent_workload,
        "source_capture_path": manifest_entry.get("source_capture_path"),
        "trigger_reason": window.trigger_reason,
        "packing_mode": window.packing_mode,
    }
    row = {key: value for key, value in row.items() if value is not None}

    window_report = {
        **window.to_dict(),
        "config_path": args.l0_config_obj.config_path,
        "resolved_config": args.l0_config_obj.to_dict(),
        "parent_workload": parent_workload,
        "workload": window_workload,
        "sass_path": f"windows/{sass_name}",
        "launches_path": f"windows/{launch_name}",
        "missing_launches": rendered["missing_launches"],
        "included_launches": rendered["included_launches"],
    }
    write_json(windows_dir / meta_name, window_report)
    return row


def validate_l0_workload(workload_dir: Path) -> None:
    required = [
        "manifest.json",
        "launches.jsonl",
        "dumps/code_map.json",
        "dumps/extraction_report.json",
        "windows/manifests.jsonl",
    ]
    missing = [path for path in required if not (workload_dir / path).exists()]
    if missing:
        raise ValidationError(f"missing required files: {', '.join(missing)}")
    rows = read_jsonl(workload_dir / "windows" / "manifests.jsonl")
    if not rows:
        raise ValidationError("windows/manifests.jsonl has no window rows")
    for row in rows:
        path = workload_dir / str(row["path"])
        if not path.exists():
            raise ValidationError(f"missing window SASS: {path.relative_to(workload_dir)}")
        if not path.read_text(encoding="utf-8").strip():
            raise ValidationError(f"empty window SASS: {path.relative_to(workload_dir)}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, sort_keys=True)
            fh.write("\n")


def load_capture_manifest_specs(
    manifest_patterns: list[str],
    captures_root: Path,
) -> list[dict[str, Any]]:
    manifest_paths = sorted(
        {
            Path(match)
            for pattern in manifest_patterns
            for match in (glob.glob(pattern) or [pattern])
        }
    )
    if not manifest_paths:
        raise ManifestError(f"no capture manifests matched: {', '.join(manifest_patterns)}")

    specs: list[dict[str, Any]] = []
    used_workloads: Counter[str] = Counter()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise ManifestError(f"missing capture manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"{manifest_path}:{line_no}: invalid JSON: {exc}") from exc
                specs.append(capture_spec_from_manifest_row(row, manifest_path, line_no, captures_root, used_workloads))
    if not specs:
        raise ManifestError(f"no capture rows found in: {', '.join(manifest_patterns)}")
    return specs


def capture_spec_from_manifest_row(
    row: dict[str, Any],
    manifest_path: Path,
    line_no: int,
    captures_root: Path,
    used_workloads: Counter[str],
) -> dict[str, Any]:
    missing = [key for key in ("label", "family", "workload", "program", "variant") if key not in row]
    if "capture_path" not in row and "capture_dir" not in row:
        missing.append("capture_path")
    if missing:
        raise ManifestError(f"{manifest_path}:{line_no}: missing fields: {', '.join(missing)}")

    capture_path = str(row.get("capture_path") or row["capture_dir"])
    capture_dir = captures_root / capture_path
    if not capture_dir.is_dir():
        raise ManifestError(f"{manifest_path}:{line_no}: missing capture_path: {capture_dir}")

    base_workload = safe_workload_name(str(row["workload"]))
    capture_id = safe_workload_name(str(row.get("capture_id") or capture_dir.name))
    used_workloads[base_workload] += 1
    workload = base_workload
    if used_workloads[base_workload] > 1:
        workload = f"{base_workload}_{capture_id[:12]}"

    entry = {
        "family": str(row["family"]),
        "label": str(row["label"]),
        "opt_level": str(row.get("opt_level") or OPT_LEVEL_CAPTURE),
        "program": str(row["program"]),
        "variant": str(row["variant"]),
        "capture_id": str(row.get("capture_id") or capture_dir.name),
        "source_capture_path": capture_path,
    }
    if "binary_label" in row:
        entry["binary_label"] = str(row["binary_label"])
    return {
        "capture_dir": capture_dir,
        "workload": workload,
        "manifest_entry": entry,
        "event_type_counts": row.get("event_type_counts"),
    }


def capture_spec_has_no_kernel_launch(capture_spec: dict[str, Any]) -> bool:
    counts = capture_spec.get("event_type_counts")
    if not isinstance(counts, dict):
        return False
    return int(counts.get("kernel_launch") or 0) == 0


def safe_workload_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "unknown"


def new_build_report() -> dict[str, Any]:
    return {
        "captures_scanned": 0,
        "workloads_created": 0,
        "duplicates_skipped": 0,
        "empty_captures_skipped": 0,
        "unmapped_captures_skipped": 0,
        "failed_captures": 0,
        "dry_run_ok": 0,
        "windows_created": 0,
        "labels": Counter(),
        "opt_levels": Counter(),
        "failures": [],
    }


def write_report(output_dir: Path, report: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    serializable = {
        **report,
        "labels": dict(report["labels"]),
        "opt_levels": dict(report["opt_levels"]),
    }
    write_json(output_dir / "build_report.json", serializable)


def print_summary(report: dict[str, Any]) -> None:
    print("\nDataset build complete.\n")
    print(f"captures scanned: {report['captures_scanned']}")
    print(f"workloads created: {report['workloads_created']}")
    if report.get("windows_created"):
        print(f"windows created: {report['windows_created']}")
    print(f"duplicates skipped: {report['duplicates_skipped']}")
    print(f"empty captures skipped: {report['empty_captures_skipped']}")
    print(f"unmapped captures skipped: {report['unmapped_captures_skipped']}")
    print(f"failed captures: {report['failed_captures']}")
    if report["dry_run_ok"]:
        print(f"dry-run valid captures: {report['dry_run_ok']}")
    print("\nlabels:")
    for label, count in sorted(report["labels"].items()):
        print(f"  {label}: {count}")
    print("\nopt levels:")
    for opt_level, count in sorted(report["opt_levels"].items()):
        print(f"  {opt_level}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
