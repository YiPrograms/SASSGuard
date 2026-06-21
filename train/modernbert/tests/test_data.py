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
    iter_sass_texts,
    load_all_splits,
    make_chunks_for_record,
)
from train.modernbert.metrics import (
    add_no_l0_window_predictions,
    aggregate_chunk_probabilities,
    aggregate_group_predictions,
    prediction_rows,
    suspicious_prediction_fields,
)


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
        counts = {split: len(rows) for split, rows in records.items()}
        self.assertEqual(set(counts), {"train", "val", "test"})
        self.assertTrue(all(count > 0 for count in counts.values()))
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

    def test_no_l0_window_records_do_not_make_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "no_l0_window.sass"
            source.write_text("NO_L0_WINDOW\n", encoding="utf-8")
            record = WorkloadRecord(
                split="test",
                workload="benign__no_l0_window",
                source_path=source,
                label="benign",
                label_id=0,
                row={"workload": "benign__no_l0_window", "path": str(source), "no_l0_window": True},
            )
            chunks = make_chunks_for_record(record, FakeTokenizer(), max_seq_length=6, stride=1)
        self.assertEqual(chunks, [])

    def test_no_l0_window_records_are_not_tokenizer_or_weight_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_source = root / "real.sass"
            no_window_source = root / "no_l0_window.sass"
            real_source.write_text("IADD3 REG, REG, IMM, ZERO\n", encoding="utf-8")
            no_window_source.write_text("NO_L0_WINDOW\n", encoding="utf-8")
            records = [
                WorkloadRecord("train", "benign_real", real_source, "benign", 0, {"workload": "benign_real"}),
                WorkloadRecord("train", "mining_real", real_source, "mining", 1, {"workload": "mining_real"}),
                WorkloadRecord(
                    "train",
                    "benign__no_l0_window",
                    no_window_source,
                    "benign",
                    0,
                    {"workload": "benign__no_l0_window", "no_l0_window": True},
                ),
            ]
            texts = list(iter_sass_texts(records))
        self.assertEqual(texts, ["IADD3 REG, REG, IMM, ZERO\n", "IADD3 REG, REG, IMM, ZERO\n"])
        self.assertEqual(class_weights(records, {"benign": 0, "mining": 1}), [1.0, 1.0])

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

    def test_class_weights_count_parent_workloads_not_windows(self) -> None:
        records = [
            WorkloadRecord("train", "benign_a__w0", Path("unused"), "benign", 0, {"group_id": "benign_a"}),
            WorkloadRecord("train", "benign_b__w0", Path("unused"), "benign", 0, {"group_id": "benign_b"}),
            WorkloadRecord("train", "mining_a__w0", Path("unused"), "mining", 1, {"group_id": "mining_a"}),
            WorkloadRecord("train", "mining_a__w1", Path("unused"), "mining", 1, {"group_id": "mining_a"}),
            WorkloadRecord("train", "mining_a__w2", Path("unused"), "mining", 1, {"group_id": "mining_a"}),
            WorkloadRecord("train", "mining_b__w0", Path("unused"), "mining", 1, {"group_id": "mining_b"}),
        ]
        self.assertEqual(class_weights(records, {"benign": 0, "mining": 1}), [1.0, 1.0])

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

    def test_grouped_workload_metrics_fire_if_any_window_is_suspicious(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for name in ("miner__w0", "miner__w1", "benign__w0"):
                path = root / f"{name}.sass"
                path.write_text("IADD3 REG, REG, IMM, ZERO\n", encoding="utf-8")
                paths.append(path)
            records = [
                WorkloadRecord("test", "miner__w0", paths[0], "mining", 1, {"group_id": "miner", "family": "mining"}),
                WorkloadRecord("test", "miner__w1", paths[1], "mining", 1, {"group_id": "miner", "family": "mining"}),
                WorkloadRecord("test", "benign__w0", paths[2], "benign", 0, {"group_id": "benign", "family": "benign"}),
            ]
        chunks = [
            self._chunk("miner__w0", 0, 1),
            self._chunk("miner__w1", 0, 1),
            self._chunk("benign__w0", 0, 0),
        ]
        window_predictions = prediction_rows(
            records,
            aggregate_chunk_probabilities(
                chunks,
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                    [0.95, 0.05],
                ],
            ),
            {0: "benign", 1: "mining"},
        )
        group_predictions = aggregate_group_predictions(records, window_predictions, {0: "benign", 1: "mining"})
        predictions = {row["workload"]: row for row in group_predictions}
        self.assertEqual(predictions["miner"]["pred_label"], "mining")
        self.assertTrue(predictions["miner"]["suspicious"])
        self.assertEqual(predictions["benign"]["pred_label"], "benign")
        self.assertEqual(predictions["miner"]["windows"], ["miner__w0", "miner__w1"])
        self.assertEqual(predictions["benign"]["windows"], ["benign__w0"])

    def test_grouped_workload_metrics_can_use_mean_mining_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for name in ("miner__w0", "miner__w1", "benign__w0", "benign__w1", "benign__w2", "benign__w3"):
                path = root / f"{name}.sass"
                path.write_text("IADD3 REG, REG, IMM, ZERO\n", encoding="utf-8")
                paths.append(path)
            records = [
                WorkloadRecord("test", "miner__w0", paths[0], "mining", 1, {"group_id": "miner"}),
                WorkloadRecord("test", "miner__w1", paths[1], "mining", 1, {"group_id": "miner"}),
                WorkloadRecord("test", "benign__w0", paths[2], "benign", 0, {"group_id": "benign"}),
                WorkloadRecord("test", "benign__w1", paths[3], "benign", 0, {"group_id": "benign"}),
                WorkloadRecord("test", "benign__w2", paths[4], "benign", 0, {"group_id": "benign"}),
                WorkloadRecord("test", "benign__w3", paths[5], "benign", 0, {"group_id": "benign"}),
            ]
        chunks = [
            self._chunk("miner__w0", 0, 1),
            self._chunk("miner__w1", 0, 1),
            self._chunk("benign__w0", 0, 0),
            self._chunk("benign__w1", 0, 0),
            self._chunk("benign__w2", 0, 0),
            self._chunk("benign__w3", 0, 0),
        ]
        window_predictions = prediction_rows(
            records,
            aggregate_chunk_probabilities(
                chunks,
                [
                    [0.72, 0.28],
                    [0.68, 0.32],
                    [0.01, 0.99],
                    [0.99, 0.01],
                    [0.99, 0.01],
                    [0.99, 0.01],
                ],
            ),
            {0: "benign", 1: "mining"},
        )
        group_predictions = aggregate_group_predictions(
            records,
            window_predictions,
            {0: "benign", 1: "mining"},
            group_policy="mean_mining_probability",
            mean_mining_probability_threshold=0.30,
        )
        predictions = {row["workload"]: row for row in group_predictions}
        self.assertEqual(predictions["miner"]["pred_label"], "mining")
        self.assertEqual(predictions["miner"]["suspicious_reason"], "mean_mining_probability>=0.3")
        self.assertAlmostEqual(predictions["miner"]["mining_probability_mean"], 0.30)
        self.assertEqual(predictions["benign"]["pred_label"], "benign")
        self.assertEqual(predictions["benign"]["suspicious_reason"], "mean_mining_probability<0.3")
        self.assertAlmostEqual(predictions["benign"]["mining_probability_mean"], 0.255)

    def test_no_l0_window_group_aggregation_defaults_to_benign(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "no_l0_window.sass"
            source.write_text("NO_L0_WINDOW\n", encoding="utf-8")
            records = [
                WorkloadRecord(
                    "test",
                    "benign__no_l0_window",
                    source,
                    "benign",
                    0,
                    {
                        "group_id": "benign",
                        "family": "benign",
                        "no_l0_window": True,
                        "default_prediction_reason": "no_l0_window_emitted",
                    },
                )
            ]
        aggregated = {}
        add_no_l0_window_predictions(records, aggregated, {0: "benign", 1: "mining"})
        window_predictions = prediction_rows(records, aggregated, {0: "benign", 1: "mining"})
        group_predictions = aggregate_group_predictions(records, window_predictions, {0: "benign", 1: "mining"})
        self.assertEqual(group_predictions[0]["pred_label"], "benign")
        self.assertFalse(group_predictions[0]["suspicious"])
        self.assertEqual(window_predictions[0]["default_prediction_reason"], "no_l0_window_emitted")
        self.assertTrue(window_predictions[0]["no_l0_window"])

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
