"""Dataset artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ingest import read_jsonl
from .split_kernels import load_kernel_metadata


class ValidationError(RuntimeError):
    """Raised when generated dataset artifacts are incomplete."""


def validate_workload(workload_dir: Path, max_launches: int) -> dict[str, Any]:
    required = [
        "manifest.json",
        "launches.jsonl",
        "workload.sass",
        "dumps/code_map.json",
        "dumps/extraction_report.json",
    ]
    missing = [path for path in required if not (workload_dir / path).exists()]
    if missing:
        raise ValidationError(f"missing required files: {', '.join(missing)}")

    manifest = json.loads((workload_dir / "manifest.json").read_text(encoding="utf-8"))
    if sorted(manifest) != ["family", "label", "opt_level", "workload"]:
        raise ValidationError("manifest.json contains unexpected fields")

    launches = read_jsonl(workload_dir / "launches.jsonl", limit=max_launches + 1)
    capped_launch_count = min(len(launches), max_launches)
    kernel_dirs = load_kernel_metadata(workload_dir / "kernels")
    warnings: list[str] = []
    for launch in launches[:max_launches]:
        if (launch.get("kernel_name"), launch.get("code_id")) not in kernel_dirs:
            warnings.append(f"missing kernel directory for {launch.get('kernel_name')}")

    workload_text = (workload_dir / "workload.sass").read_text(encoding="utf-8")
    if not workload_text.strip():
        raise ValidationError("workload.sass is empty")
    boundaries = workload_text.splitlines().count("KERNEL_BOUNDARY")
    if boundaries < 1:
        raise ValidationError("workload.sass has no KERNEL_BOUNDARY")
    if boundaries > capped_launch_count:
        raise ValidationError("workload.sass has too many KERNEL_BOUNDARY markers")

    for kernel_dir in sorted((workload_dir / "kernels").iterdir()):
        if not kernel_dir.is_dir():
            continue
        for name in (
            "metadata.json",
            "kernel.sass",
            "kernel.normalized.sass",
            "main_loop.sass",
            "main_loop.normalized.sass",
            "cfg.json",
            "loop_summary.json",
        ):
            path = kernel_dir / name
            if not path.exists():
                raise ValidationError(f"missing {path.relative_to(workload_dir)}")
            if name.endswith(".sass") and not path.read_text(encoding="utf-8").strip():
                raise ValidationError(f"empty {path.relative_to(workload_dir)}")

    return {"warnings": warnings, "boundaries": boundaries}
