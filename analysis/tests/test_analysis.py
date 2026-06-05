from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sassguard_analysis.cfg import build_cfg
from sassguard_analysis.disassemble import detect_code_format
from sassguard_analysis.ingest import event_sort_key, write_launches
from sassguard_analysis.loop_extract import extract_main_loop
from sassguard_analysis.manifest import load_binary_manifest, workload_manifest
from sassguard_analysis.normalize import normalize_instruction, normalize_sass
from sassguard_analysis.splits import (
    group_id_for_workload,
    make_grouped_stratified_split,
    validate_splits,
)
from sassguard_analysis.split_kernels import parse_disassembly, render_kernel_sass, safe_kernel_dir
from sassguard_analysis.workload_sass import build_workload_sass


class ManifestTests(unittest.TestCase):
    def test_manifest_load_and_exact_workload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text(
                '{"binary_name":"demo_o2","family":"hpc","label":"benign","opt_level":"O2"}\n',
                encoding="utf-8",
            )
            manifest = load_binary_manifest(path)
            self.assertEqual(
                workload_manifest("demo_o2", manifest["demo_o2"]),
                {
                    "workload": "demo_o2",
                    "family": "hpc",
                    "label": "benign",
                    "opt_level": "O2",
                },
            )


class IngestTests(unittest.TestCase):
    def test_event_sort_uses_sequence_before_timestamp(self) -> None:
        events = [
            {"timestamp_ns": 2, "_line_index": 2},
            {"sequence": 3, "timestamp_ns": 10, "_line_index": 3},
            {"sequence": 1, "timestamp_ns": 20, "_line_index": 1},
        ]
        self.assertEqual(
            [e.get("sequence", e.get("timestamp_ns")) for e in sorted(events, key=event_sort_key)],
            [1, 3, 2],
        )

    def test_launch_normalization_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workload_dir = Path(tmp)
            launches = write_launches(
                workload_dir,
                [
                    {
                        "sequence": 1,
                        "timestamp_ns": 2,
                        "pid": 3,
                        "tid": 4,
                        "code_id": 0,
                        "kernel_name": "k",
                        "grid_dim": [1, 1, 1],
                        "block_dim": [2, 1, 1],
                        "shared_mem_bytes": 0,
                        "stream": 0,
                        "device_pci_bus_id": "bus",
                        "session_id": "drop",
                    }
                ],
            )
            self.assertNotIn("session_id", launches[0])
            row = json.loads((workload_dir / "launches.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["kernel_name"], "k")


class DisassemblyTests(unittest.TestCase):
    def test_detect_fatbin_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.bin"
            path.write_bytes(bytes.fromhex("50ed55ba01001000"))
            self.assertEqual(detect_code_format(path), "fatbin")

    def test_parse_and_render_cuobjdump_function(self) -> None:
        text = """
Fatbin elf code:
        Function : _Z6kernelv
        /*0000*/                   IADD3 R1, R2, 0x1, RZ ;                  /* 0x0 */
        /*0010*/              @!P0 BRA 0x0000 ;                             /* 0x0 */
"""
        parsed = parse_disassembly(text)
        rendered = render_kernel_sass(parsed["_Z6kernelv"])
        self.assertIn("L_0:", rendered)
        self.assertIn("@!P0 BRA L_0", rendered)

    def test_safe_kernel_dir_sanitizes(self) -> None:
        self.assertEqual(safe_kernel_dir("kernel/name:with space"), "kernel_name_with_space")
        self.assertEqual(safe_kernel_dir("_Z6kernelv"), "_Z6kernelv")


class SassAnalysisTests(unittest.TestCase):
    def test_normalize_examples(self) -> None:
        self.assertEqual(
            normalize_instruction("IADD3 R12, R13, c[0x0][0x20], RZ"),
            "IADD3 REG, REG, CONST, ZERO",
        )
        self.assertEqual(normalize_instruction("@P0 BRA L_3"), "@PRED BRA LABEL")
        self.assertEqual(normalize_instruction("LDG.E R8, [R2+0x10]"), "LDG REG, MEM")

    def test_normalize_skips_labels(self) -> None:
        self.assertEqual(normalize_sass(["L_0:", "BRA L_0"]), "BRA LABEL\n")

    def test_cfg_predicated_branch_has_fallthrough(self) -> None:
        cfg = build_cfg(["L_0:", "IADD3 R1, R1, 0x1, RZ", "@P0 BRA L_0", "EXIT"])
        edge_types = sorted(edge["type"] for edge in cfg["edges"])
        self.assertEqual(edge_types, ["branch", "fallthrough"])

    def test_loop_extraction_selects_backward_branch(self) -> None:
        body, summary = extract_main_loop(
            ["IADD3 R0, R0, 0x1, RZ", "L_10:", "LOP3 R1, R1, R2, R3, 0xff, PT", "BRA L_10"]
        )
        self.assertEqual(summary["num_loops"], 1)
        self.assertIsNone(summary["fallback"])
        self.assertEqual(body[-1], "BRA L_10")

    def test_terminal_self_loop_is_not_selected_as_loop(self) -> None:
        lines = ["LOP3 R1, R1, R2, R3, 0xff, PT" for _ in range(300)]
        lines += ["RET", "L_trap:", "BRA L_trap"]
        cfg = build_cfg(lines)
        body, summary = extract_main_loop(lines, cfg)
        self.assertEqual(summary["num_loops"], 0)
        self.assertEqual(summary["fallback"], "compute_region")
        self.assertNotEqual(body, ["BRA L_trap"])
        self.assertNotIn("RET", body)

    def test_large_unrolled_kernel_selects_compute_region_across_blocks(self) -> None:
        lines = ["NOP" for _ in range(24)]
        lines += ["L_compute_a:"]
        lines += ["LOP3 R1, R1, R2, R3, 0xff, PT" for _ in range(80)]
        lines += ["L_compute_b:"]
        lines += ["IMAD R4, R5, R6, R7" for _ in range(90)]
        lines += ["L_compute_c:"]
        lines += ["SHF.R.U32.HI R8, RZ, 0x10, R9" for _ in range(90)]
        lines += ["L_tail:", "NOP", "EXIT"]
        cfg = build_cfg(lines)
        body, summary = extract_main_loop(lines, cfg)
        self.assertEqual(summary["fallback"], "compute_region")
        self.assertGreaterEqual(len(body), 128)
        self.assertGreater(summary["selected_region_num_blocks"], 1)
        self.assertNotIn("EXIT", body)

    def test_small_no_loop_kernel_keeps_full_kernel_fallback(self) -> None:
        lines = ["IADD3 R1, R1, 0x1, RZ", "LOP3 R1, R1, R2, R3, 0xff, PT"]
        body, summary = extract_main_loop(lines, build_cfg(lines))
        self.assertEqual(summary["fallback"], "full_kernel_small")
        self.assertEqual(body, lines)

    def test_compute_region_prefers_compute_over_control(self) -> None:
        lines = ["CALL.REL.NOINC 0x100" for _ in range(140)]
        lines += ["L_compute:"]
        lines += ["LOP3 R1, R1, R2, R3, 0xff, PT" for _ in range(160)]
        lines += ["L_tail:", "EXIT"]
        body, summary = extract_main_loop(lines, build_cfg(lines))
        self.assertEqual(summary["fallback"], "compute_region")
        self.assertTrue(all("LOP3" in line for line in body))
        self.assertGreater(summary["selected_region_start_instruction"], 100)

    def test_compute_region_caps_oversized_straight_line_block(self) -> None:
        lines = ["LOP3 R1, R1, R2, R3, 0xff, PT" for _ in range(1200)]
        body, summary = extract_main_loop(lines, build_cfg(lines))
        self.assertEqual(summary["fallback"], "compute_region")
        self.assertLessEqual(len(body), 768)
        self.assertGreaterEqual(len(body), 512)


class WorkloadSassTests(unittest.TestCase):
    def test_workload_sass_caps_launches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload = root / "demo"
            kernel = workload / "kernels" / "k"
            kernel.mkdir(parents=True)
            (kernel / "metadata.json").write_text(
                json.dumps(
                    {
                        "kernel_name": "k",
                        "safe_kernel_dir": "k",
                        "code_id": 0,
                        "instruction_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (kernel / "kernel.normalized.sass").write_text("IADD3 REG, REG, IMM, ZERO\n", encoding="utf-8")
            (kernel / "main_loop.normalized.sass").write_text("LOP3 REG, REG, REG, IMM\n", encoding="utf-8")
            with (workload / "launches.jsonl").open("w", encoding="utf-8") as fh:
                for seq in range(3):
                    json.dump({"sequence": seq, "kernel_name": "k", "code_id": 0}, fh)
                    fh.write("\n")
            result = build_workload_sass(workload, max_launches=2, short_kernel_threshold=256)
            text = (workload / "workload.sass").read_text(encoding="utf-8")
            self.assertEqual(result["included_launches"], 2)
            self.assertEqual(text.splitlines().count("KERNEL_BOUNDARY"), 2)


class SplitTests(unittest.TestCase):
    def test_group_id_strips_final_opt_suffix(self) -> None:
        self.assertEqual(group_id_for_workload("sha256d_mono_o3"), "sha256d_mono")
        self.assertEqual(group_id_for_workload("attention_score_o2"), "attention_score")

    def test_grouped_split_keeps_optimization_pairs_together(self) -> None:
        records = []
        labels = ["mining_like", "benign_compute_like", "benign_crypto_hash_like", "benign_memory_like"]
        for label in labels:
            for group_index in range(6):
                group = f"{label}_{group_index}"
                for opt in ("O2", "O3"):
                    workload = f"{group}_{opt.lower()}"
                    records.append(
                        {
                            "workload": workload,
                            "path": f"dataset/workloads/{workload}/workload.sass",
                            "label": label,
                            "binary_label": "mining" if label == "mining_like" else "benign",
                            "family": f"family_{label}",
                            "opt_level": opt,
                            "group_id": group,
                        }
                    )
        splits = make_grouped_stratified_split(records, seed=1337)
        warnings = validate_splits(splits, records)
        self.assertFalse([warning for warning in warnings if "missing labels" in warning])

        group_locations = {}
        for split_name, split_records in splits.items():
            for record in split_records:
                previous = group_locations.setdefault(record["group_id"], split_name)
                self.assertEqual(previous, split_name)

        for split_records in splits.values():
            labels_in_split = {record["label"] for record in split_records}
            self.assertEqual(labels_in_split, set(labels))
            opts = [record["opt_level"] for record in split_records]
            self.assertEqual(opts.count("O2"), opts.count("O3"))


if __name__ == "__main__":
    unittest.main()
