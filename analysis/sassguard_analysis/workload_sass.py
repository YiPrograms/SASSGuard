"""Create workload.sass in runtime launch order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ingest import read_jsonl
from .split_kernels import count_instruction_lines, load_kernel_metadata


class WorkloadSassError(RuntimeError):
    """Raised when workload.sass cannot be produced."""


def build_workload_sass(
    workload_dir: Path,
    max_launches: int = 16,
    short_kernel_threshold: int = 256,
) -> dict[str, Any]:
    launches = read_jsonl(workload_dir / "launches.jsonl", limit=max_launches)
    result = render_workload_sass(
        workload_dir,
        launches,
        short_kernel_threshold=short_kernel_threshold,
    )
    (workload_dir / "workload.sass").write_text(result["text"], encoding="utf-8")
    return {"included_launches": result["included_launches"], "missing_launches": result["missing_launches"]}


def render_workload_sass(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    short_kernel_threshold: int = 256,
) -> dict[str, Any]:
    fragments = render_workload_sass_fragments(
        workload_dir,
        launches,
        short_kernel_threshold=short_kernel_threshold,
    )
    output_lines: list[str] = []
    for fragment in fragments["fragments"]:
        output_lines.extend(fragment["lines"])
    if not output_lines:
        raise WorkloadSassError("no launched kernel could be mapped to normalized SASS")

    return {
        "text": "\n".join(output_lines).rstrip() + "\n",
        "included_launches": len(fragments["fragments"]),
        "missing_launches": fragments["missing_launches"],
    }


def render_workload_sass_fragments(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    short_kernel_threshold: int = 256,
) -> dict[str, Any]:
    kernel_dirs = load_kernel_metadata(workload_dir / "kernels")
    by_name: dict[str, Path] = {}
    for (kernel_name, _code_id), path in kernel_dirs.items():
        by_name.setdefault(kernel_name, path)

    fragments: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for launch in launches:
        key = (launch.get("kernel_name"), launch.get("code_id"))
        kernel_dir = kernel_dirs.get(key) or by_name.get(str(launch.get("kernel_name")))
        if not kernel_dir:
            missing.append(
                {
                    "kernel_name": launch.get("kernel_name"),
                    "code_id": launch.get("code_id"),
                    "reason": "normalized SASS missing",
                }
            )
            continue
        kernel_path = kernel_dir / "kernel.normalized.sass"
        main_loop_path = kernel_dir / "main_loop.normalized.sass"
        kernel_lines = _read_nonempty(kernel_path)
        main_loop_lines = _read_nonempty(main_loop_path)
        if not kernel_lines or not main_loop_lines:
            missing.append(
                {
                    "kernel_name": launch.get("kernel_name"),
                    "code_id": launch.get("code_id"),
                    "reason": "empty normalized SASS",
                }
            )
            continue
        chosen = kernel_lines if count_instruction_lines(kernel_lines) <= short_kernel_threshold else main_loop_lines
        fragments.append({"launch": launch, "lines": [*chosen, "KERNEL_BOUNDARY"]})

    if not fragments:
        raise WorkloadSassError("no launched kernel could be mapped to normalized SASS")

    return {"fragments": fragments, "missing_launches": missing}


def _read_nonempty(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
