"""MLM warm-up for the compact scratch ModernBERT SASS model."""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import ChunkDataset, load_all_splits, make_chunks
from .deps import hard_exit_success
from .metrics import write_json_report
from .modeling import final_dir, make_mlm_model, mlm_output_dir, training_args
from .tokenization import load_sass_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import DataCollatorForLanguageModeling, Trainer
    except ImportError as exc:
        raise RuntimeError("transformers is required for MLM warm-up") from exc

    run_config = load_run_config(args.config)
    ensure_output_dirs(run_config)
    tokenizer = load_sass_tokenizer(run_config.paths.tokenizer_dir)
    records_by_split = load_all_splits(
        run_config.paths.splits_dir,
        run_config.paths.repo_root,
        run_config.label_column,
        run_config.label2id,
    )

    train_chunks = make_chunks(
        records_by_split["train"], tokenizer, run_config.max_seq_length, run_config.stride
    )
    val_chunks = make_chunks(
        records_by_split["val"], tokenizer, run_config.max_seq_length, run_config.stride
    )
    output_dir = mlm_output_dir(run_config)
    model = make_mlm_model(run_config, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(run_config.raw["mlm"].get("mlm_probability", 0.15)),
        pad_to_multiple_of=8,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args(output_dir, run_config.raw["mlm"], "eval_loss", False),
        "train_dataset": ChunkDataset(train_chunks, include_labels=False),
        "eval_dataset": ChunkDataset(val_chunks, include_labels=False),
        "data_collator": collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_output = final_dir(output_dir)
    if trainer.is_world_process_zero():
        trainer.save_model(final_output)
        tokenizer.save_pretrained(final_output)
        write_json_report(
            run_config.paths.reports_dir / "mlm_report.json",
            {
                "train_chunks": len(train_chunks),
                "val_chunks": len(val_chunks),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
            },
        )
        print(f"Saved MLM checkpoint to {final_output}")
    hard_exit_success()


if __name__ == "__main__":
    main()
