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
    kernel_dirs = load_kernel_metadata(workload_dir / "kernels")
    by_name: dict[str, Path] = {}
    for (kernel_name, _code_id), path in kernel_dirs.items():
        by_name.setdefault(kernel_name, path)

    output_lines: list[str] = []
    missing: list[dict[str, Any]] = []
    included = 0
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
        output_lines.extend(chosen)
        output_lines.append("KERNEL_BOUNDARY")
        included += 1

    if included == 0:
        raise WorkloadSassError("no launched kernel could be mapped to normalized SASS")

    (workload_dir / "workload.sass").write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    return {"included_launches": included, "missing_launches": missing}


def _read_nonempty(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
