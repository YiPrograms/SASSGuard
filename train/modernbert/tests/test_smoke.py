from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train.modernbert.data import ChunkDataset, WorkloadRecord, make_chunks
from train.modernbert.modeling import make_classifier_model, make_mlm_model
from train.modernbert.tokenization import train_wordlevel_tokenizer


@unittest.skipUnless(os.getenv("RUN_ML_SMOKE") == "1", "set RUN_ML_SMOKE=1 to run ML smoke test")
class ModernBertMlSmokeTests(unittest.TestCase):
    def test_tiny_tokenizer_mlm_and_classifier_steps(self) -> None:
        try:
            from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
        except ImportError as exc:
            self.skipTest(f"ML dependencies are not installed: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = self._records(root)
            tokenizer = train_wordlevel_tokenizer(
                records,
                {
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]",
                    "min_frequency": 1,
                },
                32,
                root / "tokenizer",
            )
            chunks = make_chunks(records, tokenizer, max_seq_length=32, stride=4)
            run_config = self._tiny_config()

            mlm_model = make_mlm_model(run_config, tokenizer)
            mlm_batch = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
            )([ChunkDataset(chunks, include_labels=False)[0]])
            mlm_loss = mlm_model(**mlm_batch).loss
            mlm_loss.backward()

            clf_model = make_classifier_model(run_config, tokenizer)
            clf_batch = DataCollatorWithPadding(tokenizer=tokenizer)([ChunkDataset(chunks)[0]])
            clf_loss = clf_model(**clf_batch).loss
            clf_loss.backward()

    def _records(self, root: Path) -> list[WorkloadRecord]:
        labels = [
            "benign_compute_like",
            "benign_crypto_hash_like",
            "benign_memory_like",
            "mining_like",
        ]
        records: list[WorkloadRecord] = []
        for idx, label in enumerate(labels):
            source = root / f"workload_{idx}.sass"
            source.write_text(
                "IMAD REG, ZERO, ZERO, CONST\nLOP3 REG, REG, IMM, ZERO, IMM, PRED\n",
                encoding="utf-8",
            )
            records.append(
                WorkloadRecord(
                    split="train",
                    workload=f"workload_{idx}",
                    source_path=source,
                    label=label,
                    label_id=idx,
                    row={"workload": f"workload_{idx}", "label": label},
                )
            )
        return records

    def _tiny_config(self):
        label2id = {
            "benign_compute_like": 0,
            "benign_crypto_hash_like": 1,
            "benign_memory_like": 2,
            "mining_like": 3,
        }
        return SimpleNamespace(
            raw={
                "modernbert": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 96,
                    "local_attention": 16,
                    "classifier_pooling": "cls",
                }
            },
            max_seq_length=32,
            label2id=label2id,
            id2label={idx: label for label, idx in label2id.items()},
        )


if __name__ == "__main__":
    unittest.main()
