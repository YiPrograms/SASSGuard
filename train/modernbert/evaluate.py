"""Evaluate a saved ModernBERT SASS classifier checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import ChunkDataset, load_all_splits, make_chunks
from .deps import hard_exit_success
from .metrics import softmax_rows, workload_metrics, write_json_report
from .modeling import classifier_output_dir, final_dir
from .tokenization import load_sass_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    return parser.parse_args()


def is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is required for classifier evaluation") from exc

    run_config = load_run_config(args.config)
    ensure_output_dirs(run_config)
    checkpoint = args.checkpoint or final_dir(classifier_output_dir(run_config))
    tokenizer = load_sass_tokenizer(checkpoint if (checkpoint / "tokenizer.json").exists() else run_config.paths.tokenizer_dir)
    records_by_split = load_all_splits(
        run_config.paths.splits_dir,
        run_config.paths.repo_root,
        run_config.label_column,
        run_config.label2id,
    )
    chunks = make_chunks(records_by_split[args.split], tokenizer, run_config.max_seq_length, run_config.stride)
    model_config = AutoConfig.from_pretrained(checkpoint)
    if hasattr(model_config, "reference_compile"):
        model_config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=model_config)
    trainer_args = TrainingArguments(
        output_dir=str(run_config.paths.reports_dir / ".evaluate_trainer"),
        per_device_eval_batch_size=int(run_config.raw["classifier"].get("per_device_eval_batch_size", 1)),
        fp16=bool(run_config.raw["classifier"].get("fp16", False)) and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    predictions = trainer.predict(ChunkDataset(chunks), metric_key_prefix=args.split)
    report = workload_metrics(
        records_by_split[args.split],
        chunks,
        softmax_rows(predictions.predictions),
        run_config.id2label,
    )
    report_path = run_config.paths.reports_dir / f"evaluate_{args.split}_report.json"
    if not is_distributed_launch() or trainer.is_world_process_zero():
        write_json_report(report_path, report)
        print(f"Saved evaluation report to {report_path}")
    hard_exit_success()


if __name__ == "__main__":
    main()
