"""Split disassembly output into per-kernel SASS files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import write_json


FUNCTION_RE = re.compile(r"\bFunction\s*:\s*(?P<name>\S+)")
INSTR_RE = re.compile(r"/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?P<instr>.*?)\s*;")
BRANCH_TARGET_RE = re.compile(r"\b(?P<op>BRA|JMP|JMX|BRX)\b[^;]*\b(?P<target>0x[0-9a-fA-F]+)\b")


@dataclass(frozen=True)
class SASSInstruction:
    address: str
    text: str


def split_launched_kernels(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    code_map: dict[str, dict[str, Any]],
    extraction_report: dict[str, Any],
) -> tuple[dict[tuple[str, Any], Path], list[dict[str, Any]]]:
    kernels_dir = workload_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    functions_by_code = _parse_ok_disassemblies(workload_dir, extraction_report)
    launch_counts = Counter((launch.get("kernel_name"), launch.get("code_id")) for launch in launches)
    first_seen = OrderedDict()
    for launch in launches:
        key = (launch.get("kernel_name"), launch.get("code_id"))
        first_seen.setdefault(key, launch)

    kernel_dirs: dict[tuple[str, Any], Path] = {}
    missing: list[dict[str, Any]] = []
    used_dirs: set[str] = set()

    for (kernel_name, code_id), launch in first_seen.items():
        if not kernel_name:
            missing.append({"kernel_name": kernel_name, "code_id": code_id, "reason": "missing name"})
            continue
        functions = functions_by_code.get(str(code_id), {})
        instructions = functions.get(str(kernel_name))
        if not instructions:
            missing.append(
                {
                    "kernel_name": kernel_name,
                    "code_id": code_id,
                    "reason": "launched kernel not found in disassembly",
                }
            )
            continue

        safe_dir = safe_kernel_dir(str(kernel_name))
        if safe_dir in used_dirs:
            safe_dir = f"{safe_dir}__code_{code_id}"
        used_dirs.add(safe_dir)
        out_dir = kernels_dir / safe_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        kernel_sass = render_kernel_sass(instructions)
        (out_dir / "kernel.sass").write_text(kernel_sass, encoding="utf-8")
        source = code_map[str(code_id)]["dump_path"] if str(code_id) in code_map else None
        metadata = {
            "kernel_name": kernel_name,
            "safe_kernel_dir": safe_dir,
            "code_id": code_id,
            "source_code_file": source,
            "instruction_count": count_instruction_lines(kernel_sass.splitlines()),
            "launch_count": launch_counts[(kernel_name, code_id)],
            "launched": True,
        }
        write_json(out_dir / "metadata.json", metadata)
        kernel_dirs[(str(kernel_name), code_id)] = out_dir

    return kernel_dirs, missing


def parse_disassembly(text: str) -> dict[str, list[SASSInstruction]]:
    functions: dict[str, list[SASSInstruction]] = {}
    current_name: str | None = None
    current: list[SASSInstruction] = []

    for line in text.splitlines():
        match = FUNCTION_RE.search(line)
        if match:
            if current_name is not None:
                functions[current_name] = current
            current_name = match.group("name")
            current = []
            continue
        if current_name is None:
            continue
        instr_match = INSTR_RE.search(line)
        if not instr_match:
            continue
        instr = _clean_instruction(instr_match.group("instr"))
        if instr:
            current.append(SASSInstruction(instr_match.group("addr").lower(), instr))

    if current_name is not None:
        functions[current_name] = current
    return functions


def render_kernel_sass(instructions: list[SASSInstruction]) -> str:
    targets = _branch_targets(instructions)
    rendered: list[str] = []
    for instr in instructions:
        label = _label_for_addr(instr.address)
        if instr.address in targets:
            rendered.append(f"{label}:")
        rendered.append(_rewrite_branch_target(instr.text))
    return "\n".join(rendered).rstrip() + "\n"


def safe_kernel_dir(kernel_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", kernel_name).strip(".-")
    if not safe:
        safe = "kernel"
    if len(safe) > 120:
        digest = hashlib.sha1(kernel_name.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[:100]}_{digest}"
    return safe


def count_instruction_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if line.strip() and not line.strip().endswith(":"))


def _parse_ok_disassemblies(
    workload_dir: Path,
    extraction_report: dict[str, Any],
) -> dict[str, dict[str, list[SASSInstruction]]]:
    parsed: dict[str, dict[str, list[SASSInstruction]]] = {}
    for item in extraction_report.get("code_objects", []):
        if item.get("status") != "ok":
            continue
        path = workload_dir / item["disassembly_output"]
        if path.exists():
            parsed[str(item["code_id"])] = parse_disassembly(path.read_text(encoding="utf-8"))
    return parsed


def _branch_targets(instructions: list[SASSInstruction]) -> set[str]:
    addrs = {instr.address for instr in instructions}
    targets: set[str] = set()
    for instr in instructions:
        match = BRANCH_TARGET_RE.search(instr.text)
        if not match:
            continue
        target = match.group("target")[2:].lower()
        if target in addrs:
            targets.add(target)
    return targets


def _rewrite_branch_target(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return match.group(0).replace(match.group("target"), _label_for_addr(match.group("target")[2:]))

    return BRANCH_TARGET_RE.sub(repl, text)


def _label_for_addr(addr: str) -> str:
    return f"L_{addr.lower().lstrip('0') or '0'}"


def _clean_instruction(instr: str) -> str:
    instr = re.sub(r"\s+", " ", instr.strip())
    return instr


def load_kernel_metadata(kernels_root: Path) -> dict[tuple[str, Any], Path]:
    mapping: dict[tuple[str, Any], Path] = {}
    for path in sorted(kernels_root.glob("*/metadata.json")):
        with path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        mapping[(metadata["kernel_name"], metadata.get("code_id"))] = path.parent
    return mapping
