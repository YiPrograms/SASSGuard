#!/usr/bin/env python3
"""Collect top launched-kernel metrics for selected captures."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from sassguard_analysis.split_kernels import (  # noqa: E402
    SASSInstruction,
    parse_disassembly,
    render_kernel_sass,
    select_fallback_function,
)
from sassguard_analysis.static_features import (  # noqa: E402
    BITWISE_OPS,
    INTEGER_OPS,
    _is_memory_op,
    _opcode,
)


CAPTURES = {
    "vLLM": REPO_ROOT
    / "workloads/benign_samples/vllm_benchmark/captures/24c6ca9c4b15e7c80e12ed1c78d3bd87",
    "HPL": REPO_ROOT
    / "workloads/benign_samples/nv_hpl/captures/8113ca03e06b3f0a1baddc8f5b794dfe",
    "T-Rex": REPO_ROOT
    / "workloads/mining_samples/trex/captures/032e92b4fabe6f55a78b9a851b3b2c34",
}

HPL_SOURCES = [
    REPO_ROOT / "workloads/benign_samples/nv_hpl/xhpl",
    REPO_ROOT / "workloads/benign_samples/nv_hpl/Library/nvidia_cuda/libcublas.so.11",
    REPO_ROOT / "workloads/benign_samples/nv_hpl/Library/nvidia_cuda/libcublasLt.so.11",
]

CUDA_ROOTS = [
    Path("/usr/local/cuda-11.8"),
    Path("/usr/local/cuda-12.8"),
    Path("/usr/local/cuda"),
    Path("/usr/local/cuda-13.0"),
]

INSTR_RE = re.compile(r"/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?P<instr>.*?)\s*;")

FLOAT_OPS = {
    "DADD",
    "DFMA",
    "DMUL",
    "DSETP",
    "F2F",
    "F2I",
    "FADD",
    "FCHK",
    "FFMA",
    "FMNMX",
    "FMUL",
    "FSEL",
    "FSETP",
    "HADD2",
    "HFMA2",
    "HMUL2",
    "MUFU",
}
TENSOR_OP_PREFIXES = ("HMMA", "IMMA", "MMA", "BMMA", "DMMA")
COMPUTE_OPS = BITWISE_OPS | INTEGER_OPS | FLOAT_OPS


@dataclass(frozen=True)
class CudaTools:
    cuobjdump: Path
    nvdisasm: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/results/top_kernel_metrics")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    tools = find_cuda_tools()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "method": {
            "top_n": args.top_n,
            "launch_rate_window": "first named/unnamed kernel_launch timestamp to last kernel_launch timestamp",
            "top_kernel_filter": "named kernel_launch events only",
            "instruction_count": "SASS instruction lines parsed from cuobjdump/nvdisasm output",
            "compute_instruction_ratio": (
                "compute-class SASS opcodes divided by all parsed SASS instructions; "
                "compute includes integer/bitwise, floating-point/SFU, and tensor op families"
            ),
            "cuda_tool_order": [str(t.cuobjdump.parent.parent) for t in tools],
        },
        "workloads": OrderedDict(),
    }

    for workload, capture_dir in CAPTURES.items():
        workload_report = process_workload(workload, capture_dir, tools, args.top_n, args.output_dir)
        report["workloads"][workload] = workload_report

    json_path = args.output_dir / "top_kernel_metrics.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = args.output_dir / "top_kernel_metrics.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(markdown_path)
    print(json_path)
    return 0


def find_cuda_tools() -> list[CudaTools]:
    tools: list[CudaTools] = []
    seen: set[tuple[Path, Path]] = set()
    for root in CUDA_ROOTS:
        cuobjdump = root / "bin/cuobjdump"
        nvdisasm = root / "bin/nvdisasm"
        if cuobjdump.is_file() and nvdisasm.is_file():
            pair = (cuobjdump, nvdisasm)
            if pair not in seen:
                tools.append(CudaTools(cuobjdump=cuobjdump, nvdisasm=nvdisasm))
                seen.add(pair)
    fallback_cuobjdump = shutil.which("cuobjdump")
    fallback_nvdisasm = shutil.which("nvdisasm")
    if fallback_cuobjdump and fallback_nvdisasm:
        pair = (Path(fallback_cuobjdump), Path(fallback_nvdisasm))
        if pair not in seen:
            tools.append(CudaTools(cuobjdump=pair[0], nvdisasm=pair[1]))
    if not tools:
        raise RuntimeError("no CUDA cuobjdump/nvdisasm pairs found")
    return tools


def process_workload(
    workload: str,
    capture_dir: Path,
    tools: list[CudaTools],
    top_n: int,
    output_dir: Path,
) -> dict[str, Any]:
    events = read_jsonl(capture_dir / "events.jsonl")
    launches = [event for event in events if event.get("type") == "kernel_launch"]
    code_map = {
        event["code_id"]: capture_dir / str(event["path"])
        for event in events
        if event.get("type") == "code" and "code_id" in event and event.get("path")
    }

    if not launches:
        raise RuntimeError(f"{workload}: no launches")
    first_ts = min(int(launch["timestamp_ns"]) for launch in launches)
    last_ts = max(int(launch["timestamp_ns"]) for launch in launches)
    duration_s = max((last_ts - first_ts) / 1e9, 1e-9)
    named_launches = [launch for launch in launches if str(launch.get("kernel_name") or "").strip()]
    unnamed_launches = len(launches) - len(named_launches)

    counts = Counter((str(launch["kernel_name"]), launch.get("code_id")) for launch in named_launches)
    first_seen: OrderedDict[tuple[str, Any], dict[str, Any]] = OrderedDict()
    for launch in named_launches:
        first_seen.setdefault((str(launch["kernel_name"]), launch.get("code_id")), launch)

    rows = []
    for rank, ((kernel_name, code_id), launch_count) in enumerate(counts.most_common(top_n), 1):
        metric = collect_static_metric(
            workload=workload,
            kernel_name=kernel_name,
            code_id=code_id,
            code_map=code_map,
            tools=tools,
            output_dir=output_dir,
        )
        rows.append(
            {
                "rank": rank,
                "kernel_name": kernel_name,
                "demangled_kernel_name": demangle_kernel_name(kernel_name),
                "code_id": code_id,
                "launch_count": launch_count,
                "launches_per_second": launch_count / duration_s,
                "first_grid_dim": first_seen[(kernel_name, code_id)].get("grid_dim"),
                "first_block_dim": first_seen[(kernel_name, code_id)].get("block_dim"),
                **metric,
            }
        )

    return {
        "capture_dir": str(capture_dir.relative_to(REPO_ROOT)),
        "total_kernel_launches": len(launches),
        "named_kernel_launches": len(named_launches),
        "unnamed_or_empty_kernel_launches": unnamed_launches,
        "duration_seconds": duration_s,
        "top_kernels": rows,
    }


def collect_static_metric(
    workload: str,
    kernel_name: str,
    code_id: Any,
    code_map: dict[Any, Path],
    tools: list[CudaTools],
    output_dir: Path,
) -> dict[str, Any]:
    candidates: list[Path] = []
    if code_id in code_map:
        candidates.append(code_map[code_id])
    if workload == "HPL":
        candidates.extend(path for path in HPL_SOURCES if path.exists())

    errors: list[str] = []
    for source in candidates:
        metric, error = disassemble_kernel(kernel_name, source, tools, output_dir)
        if metric is not None:
            return metric
        if error:
            errors.append(f"{source.relative_to(REPO_ROOT)}: {error}")

    return {
        "instruction_count": None,
        "compute_instruction_count": None,
        "compute_instruction_ratio": None,
        "memory_instruction_count": None,
        "disassembly_status": "unresolved",
        "disassembly_source": None,
        "disassembly_function": None,
        "disassembly_error": " | ".join(errors[-4:]) if errors else "no candidate code object",
    }


def disassemble_kernel(
    kernel_name: str,
    source: Path,
    tools: list[CudaTools],
    output_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    for tool in tools:
        text, tool_label, error = dump_function_with_cuobjdump(tool, source, kernel_name)
        if text:
            metric = metric_from_disassembly(text, kernel_name)
            if metric:
                return finalize_metric(metric, source, tool_label, output_dir), None
        if error:
            last_error = error
        else:
            last_error = "function not found"

        raw_text, raw_label, raw_error = dump_extracted_text_with_nvdisasm(tool, source, kernel_name)
        if raw_text:
            metric = metric_from_raw_disassembly(raw_text, kernel_name)
            if metric:
                return finalize_metric(metric, source, raw_label, output_dir), None
        if raw_error:
            last_error = raw_error
    return None, last_error


def dump_function_with_cuobjdump(
    tool: CudaTools,
    source: Path,
    kernel_name: str,
) -> tuple[str | None, str, str | None]:
    command = [str(tool.cuobjdump), "--dump-sass", "--function", kernel_name, str(source)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = result.stdout
    if result.returncode == 0 and "/*" in text and "Function :" in text:
        return text, f"{tool.cuobjdump} --dump-sass --function", None

    # Some captures have launch aliases. Parse a full small code object and apply
    # the same fallback name matching used by the dataset pipeline.
    if source.stat().st_size <= 8 * 1024 * 1024:
        full = subprocess.run(
            [str(tool.cuobjdump), "--dump-sass", str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if full.returncode == 0 and "/*" in full.stdout:
            functions = parse_disassembly(full.stdout)
            selected = select_fallback_function(kernel_name, functions)
            if selected is not None:
                function_name, instructions, match_reason = selected
                if match_reason == "largest_function_fallback":
                    return None, f"{tool.cuobjdump} full-dump fallback", "no exact/canonical/substring function match"
                rendered = render_kernel_sass(instructions)
                return (
                    rendered,
                    f"{tool.cuobjdump} full-dump fallback:{match_reason}:{function_name}",
                    None,
                )
    error = (result.stderr or result.stdout).strip()
    return None, f"{tool.cuobjdump} --dump-sass --function", error or "no SASS output"


def dump_extracted_text_with_nvdisasm(
    tool: CudaTools,
    source: Path,
    kernel_name: str,
) -> tuple[str | None, str, str | None]:
    with tempfile.TemporaryDirectory(prefix="sassguard_extract_") as tmp:
        tmp_path = Path(tmp)
        result = subprocess.run(
            [str(tool.cuobjdump), "--extract-text", kernel_name, str(source)],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        extracted = sorted(tmp_path.glob("*.bin"))
        sm70 = [path for path in extracted if ".sm_70." in path.name]
        selected = sm70[0] if sm70 else (extracted[0] if extracted else None)
        if result.returncode != 0 or selected is None:
            error = (result.stderr or result.stdout).strip()
            return None, f"{tool.cuobjdump} --extract-text", error or "no extracted text section"

        sm_match = re.search(r"\.sm_(\d+)\.", selected.name)
        sm = f"SM{sm_match.group(1)}" if sm_match else "SM70"
        disasm = subprocess.run(
            [str(tool.nvdisasm), "--binary", sm, str(selected)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if disasm.returncode == 0 and "/*" in disasm.stdout:
            return (
                disasm.stdout,
                f"{tool.cuobjdump} --extract-text + {tool.nvdisasm} --binary {sm}",
                None,
            )
        error = (disasm.stderr or disasm.stdout).strip()
        return None, f"{tool.nvdisasm} --binary {sm}", error or "no raw disassembly"


def metric_from_disassembly(text: str, kernel_name: str) -> dict[str, Any] | None:
    if "Function :" in text:
        functions = parse_disassembly(text)
        if kernel_name in functions:
            selected_name = kernel_name
            instructions = functions[kernel_name]
            match_reason = "exact"
        else:
            selected = select_fallback_function(kernel_name, functions)
            if selected is None:
                return None
            selected_name, instructions, match_reason = selected
        sass = render_kernel_sass(instructions)
    else:
        selected_name = kernel_name
        match_reason = "rendered"
        sass = text
    return metric_from_sass_lines(sass.splitlines(), selected_name, match_reason)


def metric_from_raw_disassembly(text: str, kernel_name: str) -> dict[str, Any] | None:
    instructions = []
    for instr_match in INSTR_RE.finditer(text):
        instr = re.sub(r"\s+", " ", instr_match.group("instr").strip())
        if instr:
            instructions.append(SASSInstruction(instr_match.group("addr").lower(), instr))
    if not instructions:
        return None
    sass = render_kernel_sass(instructions)
    return metric_from_sass_lines(sass.splitlines(), kernel_name, "raw_text_section")


def metric_from_sass_lines(lines: list[str], function_name: str, match_reason: str) -> dict[str, Any]:
    opcodes = [_opcode(line) for line in lines if _opcode(line)]
    instruction_count = len(opcodes)
    compute_count = sum(1 for opcode in opcodes if is_compute_opcode(opcode))
    memory_count = sum(1 for opcode in opcodes if _is_memory_op(opcode))
    return {
        "instruction_count": instruction_count,
        "compute_instruction_count": compute_count,
        "compute_instruction_ratio": compute_count / instruction_count if instruction_count else None,
        "memory_instruction_count": memory_count,
        "disassembly_function": function_name,
        "kernel_match": match_reason,
    }


def finalize_metric(
    metric: dict[str, Any],
    source: Path,
    tool_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    metric["disassembly_status"] = "ok"
    metric["disassembly_source"] = str(source.relative_to(REPO_ROOT))
    metric["disassembly_tool"] = tool_label
    return metric


def is_compute_opcode(opcode: str) -> bool:
    return opcode in COMPUTE_OPS or any(opcode.startswith(prefix) for prefix in TENSOR_OP_PREFIXES)


def demangle_kernel_name(name: str) -> str:
    if not name.startswith("_Z"):
        return name
    cxxfilt = shutil.which("c++filt")
    if not cxxfilt:
        return name
    result = subprocess.run([cxxfilt, name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return name
    demangled = result.stdout.strip()
    return demangled or name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Top Launched Kernel Metrics",
        "",
        "Launches per second are computed over each capture's first-to-last kernel-launch timestamp. "
        "The top-3 ranking excludes empty kernel names because they cannot be mapped to kernel SASS.",
        "",
    ]
    for workload, data in report["workloads"].items():
        lines.extend(
            [
                f"## {workload}",
                "",
                f"- Capture duration: {data['duration_seconds']:.3f} s",
                f"- Kernel launches: {data['total_kernel_launches']} total, "
                f"{data['named_kernel_launches']} named, "
                f"{data['unnamed_or_empty_kernel_launches']} unnamed/empty",
                "",
                "| Rank | Kernel | Code ID | Launches | Launches/s | Instruction count | Compute instr. ratio | Notes |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in data["top_kernels"]:
            ratio = row["compute_instruction_ratio"]
            ratio_text = "" if ratio is None else f"{ratio:.3f}"
            inst = "" if row["instruction_count"] is None else str(row["instruction_count"])
            kernel_display = row.get("demangled_kernel_name") or row["kernel_name"]
            notes = row["disassembly_status"]
            if row["disassembly_status"] == "ok":
                notes = f"{row['kernel_match']}; {row['disassembly_source']}"
            elif row.get("disassembly_error"):
                notes = f"unresolved: {row['disassembly_error']}"
            lines.append(
                "| {rank} | `{kernel}` | {code_id} | {launches} | {lps:.3f} | {inst} | {ratio} | {notes} |".format(
                    rank=row["rank"],
                    kernel=kernel_display,
                    code_id=row["code_id"],
                    launches=row["launch_count"],
                    lps=row["launches_per_second"],
                    inst=inst,
                    ratio=ratio_text,
                    notes=notes.replace("|", "\\|"),
                )
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
