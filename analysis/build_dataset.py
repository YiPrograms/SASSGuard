#!/usr/bin/env python3
"""Build the SASSGuard synthetic kernel dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
    read_process,
    split_events,
    workload_name_from_process,
    write_launches,
)
from sassguard_analysis.loop_extract import extract_main_loop_for_kernel
from sassguard_analysis.manifest import (
    ManifestError,
    load_binary_manifest,
    write_json,
    write_workload_manifest,
)
from sassguard_analysis.normalize import normalize_kernel_files
from sassguard_analysis.split_kernels import split_launched_kernels
from sassguard_analysis.validate import ValidationError, validate_workload
from sassguard_analysis.workload_sass import WorkloadSassError, build_workload_sass


class CaptureBuildError(RuntimeError):
    """Raised when one capture cannot produce a workload."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, default=Path("workloads/synthetic_kernels/captures"))
    parser.add_argument(
        "--binary-manifest",
        type=Path,
        default=Path("workloads/synthetic_kernels/binaries/manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
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
    try:
        binary_manifest = load_binary_manifest(args.binary_manifest)
    except ManifestError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    if not args.captures_dir.is_dir():
        print(f"[FATAL] missing captures directory: {args.captures_dir}", file=sys.stderr)
        return 2

    tools = find_cuda_tools()
    if args.verbose:
        for name in ("nvdisasm", "cuobjdump"):
            print(f"[INFO] {name}: {tools.get(name, 'not found')}")

    report = new_build_report()
    captures = sorted(path for path in args.captures_dir.iterdir() if path.is_dir())
    if args.jobs < 1:
        print("[FATAL] --jobs must be >= 1", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[INFO] jobs: {args.jobs}")

    results = iter_capture_results(captures, args, binary_manifest, tools)
    for result in results:
        report["captures_scanned"] += 1
        if result["status"] == "failed":
            report["failed_captures"] += 1
            report["failures"].append({"capture": result["capture"], "reason": result["reason"]})
            print(f"[ERROR] {result['capture']}: {result['reason']}")
            continue

        if result["status"] == "skipped":
            report["duplicates_skipped"] += 1
            print(f"[SKIP] duplicate workload already exists: {result['workload']}")
            continue
        if result["status"] == "dry_run":
            report["dry_run_ok"] += 1
            if args.verbose:
                print(f"[DRY-RUN] {result['workload']}")
            continue

        report["workloads_created"] += 1
        report["labels"][result["label"]] += 1
        report["opt_levels"][result["opt_level"]] += 1
        print(f"[OK] {result['workload']}")

    write_report(args.output_dir, report, dry_run=args.dry_run)
    print_summary(report)
    return 0 if report["failed_captures"] == 0 else 1


def iter_capture_results(
    captures: list[Path],
    args: argparse.Namespace,
    binary_manifest: dict[str, dict[str, str]],
    tools: dict[str, Path],
):
    if args.jobs == 1 or len(captures) <= 1:
        for capture_dir in captures:
            yield process_capture_worker(capture_dir, args, binary_manifest, tools)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        yield from executor.map(
            _worker_star,
            [(capture_dir, args, binary_manifest, tools) for capture_dir in captures],
            chunksize=1,
        )


def _worker_star(payload: tuple[Path, argparse.Namespace, dict[str, dict[str, str]], dict[str, Path]]):
    capture_dir, args, binary_manifest, tools = payload
    return process_capture_worker(capture_dir, args, binary_manifest, tools)


def process_capture_worker(
    capture_dir: Path,
    args: argparse.Namespace,
    binary_manifest: dict[str, dict[str, str]],
    tools: dict[str, Path],
) -> dict[str, Any]:
    try:
        result = process_capture(capture_dir, args, binary_manifest, tools)
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
    capture_dir: Path,
    args: argparse.Namespace,
    binary_manifest: dict[str, dict[str, str]],
    tools: dict[str, Path],
) -> dict[str, Any]:
    try:
        process = read_process(capture_dir)
        workload = workload_name_from_process(process)
        if workload not in binary_manifest:
            raise CaptureBuildError(f"missing manifest entry for workload: {workload}")

        workloads_root = args.output_dir / "workloads"
        final_dir = workloads_root / workload
        if final_dir.exists() and args.skip_existing:
            return {"status": "skipped", "workload": workload}
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
            _build_into_temp_dir(capture_dir, temp_dir, workload, args, binary_manifest[workload], tools)
            if final_dir.exists() and args.skip_existing:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {"status": "skipped", "workload": workload}
            if final_dir.exists() and not args.skip_existing:
                shutil.rmtree(final_dir)
            temp_dir.replace(final_dir)
        except Exception:
            if args.keep_partial:
                print(f"[PARTIAL] kept partial workload at {temp_dir}")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return {
            "status": "created",
            "workload": workload,
            "label": binary_manifest[workload]["label"],
            "opt_level": binary_manifest[workload]["opt_level"],
        }
    except (IngestError, DisassemblyError, ValidationError, WorkloadSassError) as exc:
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

    workload_result = build_workload_sass(
        workload_dir,
        max_launches=args.max_launches,
        short_kernel_threshold=args.short_kernel_threshold,
    )
    extraction_report["missing_launches"] = workload_result["missing_launches"]
    extraction_report["included_launches"] = workload_result["included_launches"]
    write_extraction_report(workload_dir, extraction_report)
    validate_workload(workload_dir, max_launches=args.max_launches)


def new_build_report() -> dict[str, Any]:
    return {
        "captures_scanned": 0,
        "workloads_created": 0,
        "duplicates_skipped": 0,
        "failed_captures": 0,
        "dry_run_ok": 0,
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
    print(f"duplicates skipped: {report['duplicates_skipped']}")
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
