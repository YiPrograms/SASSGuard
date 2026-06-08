from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train.modernbert.config import load_run_config
from train.modernbert.data import (
    ChunkRecord,
    WorkloadRecord,
    chunk_token_ids,
    class_weights,
    load_all_splits,
    make_chunks_for_record,
)
from train.modernbert.metrics import aggregate_chunk_probabilities, suspicious_prediction_fields


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(1, len(text.replace(",", " , ").split()) + 1))


class ModernBertDataTests(unittest.TestCase):
    def test_loads_existing_splits_and_counts(self) -> None:
        config = load_run_config()
        records = load_all_splits(
            config.paths.splits_dir,
            config.paths.repo_root,
            config.label_column,
            config.label2id,
        )
        self.assertEqual(
            {split: len(rows) for split, rows in records.items()},
            {"train": 176, "val": 38, "test": 38},
        )
        labels = sorted({record.label for rows in records.values() for record in rows})
        self.assertEqual(labels, sorted(config.label2id))

    def test_binary_label_config_loads_existing_splits(self) -> None:
        config = load_run_config("configs/training/modernbert_sass_compact_binary.json")
        records = load_all_splits(
            config.paths.splits_dir,
            config.paths.repo_root,
            config.label_column,
            config.label2id,
        )
        self.assertEqual(config.label_column, "binary_label")
        self.assertEqual(config.label2id, {"benign": 0, "mining": 1})
        labels = sorted({record.label for rows in records.values() for record in rows})
        self.assertEqual(labels, ["benign", "mining"])

    def test_chunk_token_ids_uses_stride_overlap(self) -> None:
        windows = chunk_token_ids(list(range(10)), content_window=4, stride=1)
        self.assertEqual(windows, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]])

    def test_make_chunks_adds_specials_and_keeps_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "workload.sass"
            source.write_text("A, B C D E F G H I J\n", encoding="utf-8")
            record = WorkloadRecord(
                split="train",
                workload="toy",
                source_path=source,
                label="mining_like",
                label_id=3,
                row={"workload": "toy", "path": str(source), "label": "mining_like"},
            )
            chunks = make_chunks_for_record(record, FakeTokenizer(), max_seq_length=6, stride=1)
        self.assertEqual(len(chunks), 4)
        for idx, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk.input_ids), 6)
            self.assertEqual(chunk.input_ids[0], FakeTokenizer.cls_token_id)
            self.assertEqual(chunk.input_ids[-1], FakeTokenizer.sep_token_id)
            self.assertEqual(chunk.chunk_index, idx)
            self.assertEqual(chunk.num_chunks, 4)
            self.assertEqual(chunk.label_id, 3)

    def test_class_weights_use_workload_counts(self) -> None:
        records = [
            self._record("a", 0),
            self._record("b", 1),
            self._record("c", 1),
            self._record("d", 2),
            self._record("e", 2),
            self._record("f", 2),
            self._record("g", 3),
            self._record("h", 3),
            self._record("i", 3),
            self._record("j", 3),
        ]
        weights = class_weights(records, {"a": 0, "b": 1, "c": 2, "d": 3})
        self.assertEqual(weights, [2.5, 1.25, 10 / 12, 0.625])

    def test_aggregate_chunk_probabilities_by_workload_mean(self) -> None:
        chunks = [
            self._chunk("w0", 0, 0),
            self._chunk("w0", 1, 0),
            self._chunk("w1", 0, 3),
        ]
        aggregated = aggregate_chunk_probabilities(
            chunks,
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.1, 0.2, 0.6],
            ],
        )
        self.assertEqual(aggregated["w0"]["pred_id"], 0)
        self.assertEqual(aggregated["w0"]["num_chunks"], 2)
        self.assertAlmostEqual(aggregated["w0"]["probabilities"][0], 0.7)
        self.assertEqual(aggregated["w1"]["pred_id"], 3)

    def test_suspicious_fields_can_fire_from_max_chunk_probability(self) -> None:
        chunks = [
            self._chunk("w0", 0, 0),
            self._chunk("w0", 1, 0),
            self._chunk("w0", 2, 0),
        ]
        aggregated = aggregate_chunk_probabilities(
            chunks,
            [
                [0.95, 0.05],
                [0.95, 0.05],
                [0.05, 0.95],
            ],
        )
        self.assertEqual(aggregated["w0"]["pred_id"], 0)
        fields = suspicious_prediction_fields(aggregated["w0"], {0: "benign", 1: "mining"})
        self.assertTrue(fields["suspicious"])
        self.assertEqual(fields["suspicious_reason"], "max_p_mining>=0.9")
        self.assertAlmostEqual(fields["mining_probability_mean"], 0.35)
        self.assertAlmostEqual(fields["mining_probability_max"], 0.95)

    def test_suspicious_fields_fire_from_mean_pooling_mining_prediction(self) -> None:
        chunks = [
            self._chunk("w0", 0, 1),
            self._chunk("w0", 1, 1),
            self._chunk("w0", 2, 1),
        ]
        aggregated = aggregate_chunk_probabilities(
            chunks,
            [
                [0.42, 0.58],
                [0.43, 0.57],
                [0.44, 0.56],
            ],
        )
        self.assertEqual(aggregated["w0"]["pred_id"], 1)
        fields = suspicious_prediction_fields(aggregated["w0"], {0: "benign", 1: "mining"})
        self.assertTrue(fields["suspicious"])
        self.assertEqual(fields["suspicious_reason"], "mean_pooling_decision")
        self.assertEqual(fields["suspicious_label"], "suspicious")

    def test_suspicious_fields_report_below_threshold_for_benign_prediction(self) -> None:
        chunks = [
            self._chunk("w0", 0, 0),
            self._chunk("w0", 1, 0),
            self._chunk("w0", 2, 0),
        ]
        aggregated = aggregate_chunk_probabilities(
            chunks,
            [
                [0.65, 0.35],
                [0.62, 0.38],
                [0.6, 0.4],
            ],
        )
        self.assertEqual(aggregated["w0"]["pred_id"], 0)
        fields = suspicious_prediction_fields(aggregated["w0"], {0: "benign", 1: "mining"})
        self.assertFalse(fields["suspicious"])
        self.assertEqual(fields["suspicious_reason"], "below_threshold")
        self.assertEqual(fields["suspicious_label"], "benign")

    def _record(self, workload: str, label_id: int) -> WorkloadRecord:
        return WorkloadRecord(
            split="train",
            workload=workload,
            source_path=Path("unused"),
            label=str(label_id),
            label_id=label_id,
            row={"workload": workload},
        )

    def _chunk(self, workload: str, chunk_index: int, label_id: int) -> ChunkRecord:
        return ChunkRecord(
            workload=workload,
            split="val",
            source_path="unused",
            label=str(label_id),
            label_id=label_id,
            chunk_index=chunk_index,
            num_chunks=1,
            input_ids=[101, 102],
            attention_mask=[1, 1],
        )


if __name__ == "__main__":
    unittest.main()
