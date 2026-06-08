"""Configuration loading for the ModernBERT SASS training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/training/modernbert_sass_compact.json")


@dataclass(frozen=True)
class ModernBertPaths:
    repo_root: Path
    config_path: Path
    splits_dir: Path
    tokenizer_dir: Path
    checkpoint_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ModernBertRunConfig:
    raw: dict[str, Any]
    paths: ModernBertPaths

    @property
    def label_column(self) -> str:
        return str(self.raw["label_column"])

    @property
    def label2id(self) -> dict[str, int]:
        return {str(label): int(idx) for label, idx in self.raw["label2id"].items()}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: label for label, idx in self.label2id.items()}

    @property
    def max_seq_length(self) -> int:
        return int(self.raw["max_seq_length"])

    @property
    def stride(self) -> int:
        return int(self.raw["stride"])


def load_run_config(config_path: str | Path = DEFAULT_CONFIG) -> ModernBertRunConfig:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    repo_root = find_repo_root(path.parent)
    paths = ModernBertPaths(
        repo_root=repo_root,
        config_path=path,
        splits_dir=_resolve_repo_path(repo_root, raw["splits_dir"]),
        tokenizer_dir=_resolve_repo_path(repo_root, raw["tokenizer_dir"]),
        checkpoint_dir=_resolve_repo_path(repo_root, raw["checkpoint_dir"]),
        reports_dir=_resolve_repo_path(repo_root, raw["reports_dir"]),
    )
    _validate_config(raw)
    return ModernBertRunConfig(raw=raw, paths=paths)


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "dataset" / "splits").exists():
            return candidate
    return current


def ensure_output_dirs(config: ModernBertRunConfig) -> None:
    config.paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _validate_config(raw: dict[str, Any]) -> None:
    required = (
        "splits_dir",
        "tokenizer_dir",
        "checkpoint_dir",
        "reports_dir",
        "label_column",
        "label2id",
        "max_seq_length",
        "stride",
        "tokenizer",
        "modernbert",
        "mlm",
        "classifier",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if int(raw["max_seq_length"]) < 4:
        raise ValueError("max_seq_length must leave room for special tokens")
    content_window = int(raw["max_seq_length"]) - 2
    if int(raw["stride"]) >= content_window:
        raise ValueError("stride must be smaller than max_seq_length - 2")
    label_ids = list(raw["label2id"].values())
    if sorted(label_ids) != list(range(len(label_ids))):
        raise ValueError("label2id values must be contiguous zero-based ids")
