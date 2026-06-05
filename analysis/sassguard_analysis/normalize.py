"""SASS normalization for model-ready text."""

from __future__ import annotations

import re
from pathlib import Path


def normalize_kernel_files(kernel_dir: Path) -> None:
    for raw_name, normalized_name in (
        ("kernel.sass", "kernel.normalized.sass"),
        ("main_loop.sass", "main_loop.normalized.sass"),
    ):
        raw_path = kernel_dir / raw_name
        if raw_path.exists():
            normalized = normalize_sass(raw_path.read_text(encoding="utf-8").splitlines())
            (kernel_dir / normalized_name).write_text(normalized, encoding="utf-8")


def normalize_sass(lines: list[str]) -> str:
    normalized = []
    for line in lines:
        instr = normalize_instruction(line)
        if instr:
            normalized.append(instr)
    return "\n".join(normalized).rstrip() + ("\n" if normalized else "")


def normalize_instruction(line: str) -> str | None:
    line = _strip_comments(line).strip().rstrip(";")
    if not line or line.endswith(":"):
        return None

    predicate = ""
    pred_match = re.match(r"^@(?P<neg>!)?(?P<pred>[A-Za-z0-9_]+)\s+(?P<body>.*)$", line)
    if pred_match:
        predicate = "@!PRED " if pred_match.group("neg") else "@PRED "
        line = pred_match.group("body").strip()

    if not line:
        return None
    opcode, _, operands = line.partition(" ")
    opcode = opcode.split(".")[0].upper()
    operand_list = [_normalize_operand(op.strip()) for op in _split_operands(operands)]
    operand_list = [op for op in operand_list if op]
    if operand_list:
        return f"{predicate}{opcode} {', '.join(operand_list)}"
    return f"{predicate}{opcode}"


def _normalize_operand(operand: str) -> str:
    operand = operand.strip()
    if not operand:
        return ""
    if "[" in operand and "]" in operand and not operand.lower().startswith("c["):
        return "MEM"
    if re.fullmatch(r"-?c\[[^\]]+\]\[[^\]]+\]", operand, flags=re.IGNORECASE):
        return "CONST"
    if re.fullmatch(r"L_[A-Za-z0-9_]+|\.?L[A-Za-z0-9_.$]+|TARGET\d+", operand):
        return "LABEL"

    operand = re.sub(r"\.reuse\b", "", operand)
    operand = re.sub(r"\bURZ\b", "ZERO", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bRZ\b", "ZERO", operand, flags=re.IGNORECASE)
    operand = re.sub(r"!?\bPT\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"!?\bP\d+\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bUP\d+\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"[-+]?\bUR\d+(?:\.[A-Za-z0-9_]+)?\b", "UREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"[-+]?\bR\d+(?:\.[A-Za-z0-9_]+)?\b", "REG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bSR_[A-Za-z0-9_.]+\b", "SREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bB\d+\b", "BREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"-?0x[0-9a-fA-F]+", "IMM", operand)
    operand = re.sub(r"(?<![A-Za-z_])-?\d+\.\d+\b", "IMM", operand)
    operand = re.sub(r"(?<![A-Za-z_])-?\d+\b", "IMM", operand)
    operand = re.sub(r"\s+", " ", operand).strip()
    return operand


def _split_operands(operands: str) -> list[str]:
    if not operands.strip():
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(operands):
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(operands[start:idx])
            start = idx + 1
    parts.append(operands[start:])
    return parts


def _strip_comments(line: str) -> str:
    return re.sub(r"/\*.*?\*/", "", line)
