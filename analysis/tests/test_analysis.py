from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sassguard_analysis.cfg import build_cfg
from sassguard_analysis.disassemble import detect_code_format, disassemble_code_objects
from sassguard_analysis.ingest import copy_code_objects, event_sort_key, write_launches
from sassguard_analysis.l0_config import L0ConfigError, load_l0_config, parse_l0_config
from sassguard_analysis.l0_windows import build_l0_windows, launch_features, proportional_condense_launches
from sassguard_analysis.loop_extract import extract_main_loop
from build_dataset import (
    capture_spec_has_no_kernel_launch,
    load_capture_manifest_specs,
    parse_args,
)
from sassguard_analysis.manifest import load_binary_manifest, workload_manifest
from sassguard_analysis.normalize import normalize_instruction, normalize_sass
from sassguard_analysis.splits import (
    group_id_for_workload,
    make_grouped_stratified_split,
    validate_splits,
)
from sassguard_analysis.split_kernels import (
    SASSInstruction,
    _parse_ok_disassemblies,
    parse_disassembly,
    render_kernel_sass,
    safe_kernel_dir,
    select_fallback_function,
    split_launched_kernels,
    unique_safe_kernel_dir,
)
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

    def test_workload_manifest_keeps_capture_provenance(self) -> None:
        self.assertEqual(
            workload_manifest(
                "miner_algo",
                {
                    "family": "equihash_solver",
                    "label": "mining_like",
                    "opt_level": "capture",
                    "program": "miner",
                    "variant": "algo",
                    "capture_id": "abc",
                    "source_capture_path": "captures/abc",
                    "binary_label": "mining",
                },
            ),
            {
                "workload": "miner_algo",
                "family": "equihash_solver",
                "label": "mining_like",
                "opt_level": "capture",
                "program": "miner",
                "variant": "algo",
                "capture_id": "abc",
                "source_capture_path": "captures/abc",
                "binary_label": "mining",
            },
        )

    def test_capture_manifest_rows_build_general_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_a = root / "captures" / "abc123456789"
            capture_b = root / "captures" / "def123456789"
            capture_a.mkdir(parents=True)
            capture_b.mkdir(parents=True)
            manifest_path = root / "manifests.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "capture_path": "captures/abc123456789",
                                "label": "mining_like",
                                "family": "ethash_dag_keccak",
                                "workload": "miner_ethash",
                                "program": "miner",
                                "variant": "ethash",
                                "capture_id": "abc123456789",
                                "event_type_counts": {"code": 1, "kernel_launch": 2},
                            }
                        ),
                        json.dumps(
                            {
                                "capture_path": "captures/def123456789",
                                "label": "mining_like",
                                "family": "ethash_dag_keccak",
                                "workload": "miner_ethash",
                                "program": "miner",
                                "variant": "ethash",
                                "capture_id": "def123456789",
                                "event_type_counts": {"code": 1},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            specs = load_capture_manifest_specs([str(manifest_path)], captures_root=root)
            self.assertEqual(specs[0]["workload"], "miner_ethash")
            self.assertEqual(specs[1]["workload"], "miner_ethash_def123456789")
            self.assertEqual(specs[0]["manifest_entry"]["opt_level"], "capture")
            self.assertEqual(specs[0]["manifest_entry"]["program"], "miner")
            self.assertFalse(capture_spec_has_no_kernel_launch(specs[0]))
            self.assertTrue(capture_spec_has_no_kernel_launch(specs[1]))


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

    def test_copy_code_objects_reuses_duplicate_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            code_dir = capture / "code"
            workload = root / "workload"
            code_dir.mkdir(parents=True)
            payload = b"duplicate-code-object"
            (code_dir / "a.bin").write_bytes(payload)
            (code_dir / "b.bin").write_bytes(payload)
            import hashlib

            digest = hashlib.sha256(payload).hexdigest()
            code_map = copy_code_objects(
                capture,
                [
                    {"code_id": 0, "path": "code/a.bin", "sha256": digest, "size": len(payload)},
                    {"code_id": 1, "path": "code/b.bin", "sha256": digest, "size": len(payload)},
                ],
                workload,
            )
            self.assertEqual(code_map["1"]["deduplicated_dump_from"], "0")
            self.assertEqual(code_map["0"]["sha256"], code_map["1"]["sha256"])
            self.assertEqual(code_map["0"]["dump_path"], code_map["1"]["dump_path"])
            self.assertEqual(len(list((workload / "dumps").glob("*.bin"))), 1)


class L0ConfigTests(unittest.TestCase):
    def test_default_l0_config_loads(self) -> None:
        config = load_l0_config()
        self.assertTrue(config.enabled)
        self.assertFalse(config.grouping.emit_process_aggregate_windows)
        self.assertEqual(config.short_window.max_launches, 256)
        self.assertEqual(config.long_window.target_l1_chunks, 12)
        self.assertEqual(config.long_window.max_emitted_launches, 256)
        self.assertEqual(config.trigger.cooldown_ms, 10000)

    def test_l0_config_requires_sections(self) -> None:
        with self.assertRaisesRegex(L0ConfigError, "missing sections"):
            parse_l0_config({"enabled": True})

    def test_l0_config_rejects_invalid_thresholds(self) -> None:
        raw = {
            "enabled": True,
            "grouping": {
                "emit_stream_windows": True,
                "emit_process_aggregate_windows": True,
                "include_tid_in_group": False,
            },
            "maturity": {"min_launches": 0, "min_duration_ms": 1000, "long_min_duration_ms": 30000},
            "short_window": {"duration_ms": 5000, "max_launches": 256, "target_l1_chunks": 4},
            "long_window": {"duration_ms": 60000, "max_launches": 2048, "target_l1_chunks": 8, "max_emitted_launches": 64},
            "repetition": {
                "dominant_code_id_ratio": 0.7,
                "top3_code_id_ratio": 0.85,
                "normalized_entropy": 0.45,
            },
            "trigger": {
                "stable_shape_min_launches": 64,
                "grid_stability": 0.8,
                "block_stability": 0.8,
                "cooldown_ms": 10000,
                "major_change_min_interval_ms": 5000,
                "periodic_sample_ms": 30000,
                "entropy_shift": 0.35,
                "top3_jaccard_threshold": 0.34,
                "shape_change_min_stability": 0.8,
            },
            "condensation": {"enabled": True, "preserve_first_last": True, "min_per_code_id": 1},
        }
        with self.assertRaisesRegex(L0ConfigError, "min_launches"):
            parse_l0_config(raw)

    def test_no_l0_windowing_cli_override_is_available(self) -> None:
        args = parse_args(["--no-l0-windowing"])
        self.assertFalse(args.l0_windowing)


class L0WindowTests(unittest.TestCase):
    def test_l0_groups_by_stream_without_tid(self) -> None:
        config = parse_l0_config(_l0_test_config(min_launches=2, short_duration_ms=1000, long_duration_ms=2000))
        launches = [
            _launch(seq=0, tid=10, stream=1, code_id=0),
            _launch(seq=1, tid=11, stream=1, code_id=0),
            _launch(seq=2, tid=12, stream=2, code_id=1),
            _launch(seq=3, tid=13, stream=2, code_id=1),
        ]
        windows = build_l0_windows(launches, config)
        stream_windows = [window for window in windows if window.group_kind == "stream" and window.window_type == "short"]
        process_windows = [window for window in windows if window.group_kind == "process" and window.window_type == "short"]
        self.assertEqual(len(stream_windows), 2)
        self.assertEqual(len(process_windows), 0)
        self.assertNotIn("tid", stream_windows[0].group_key)

    def test_l0_features_capture_repetition(self) -> None:
        features = launch_features(
            [
                _launch(seq=0, code_id=0, grid=[1, 1, 1], block=[128, 1, 1]),
                _launch(seq=1, code_id=0, grid=[1, 1, 1], block=[128, 1, 1]),
                _launch(seq=2, code_id=1, grid=[2, 1, 1], block=[128, 1, 1]),
                _launch(seq=3, code_id=0, grid=[1, 1, 1], block=[128, 1, 1]),
            ]
        )
        self.assertEqual(features["launch_count"], 4)
        self.assertEqual(features["unique_code_id_count"], 2)
        self.assertAlmostEqual(features["dominant_code_id_ratio"], 0.75)
        self.assertAlmostEqual(features["grid_stability"], 0.75)
        self.assertAlmostEqual(features["block_stability"], 1.0)

    def test_l0_long_condensation_is_proportional(self) -> None:
        launches = [_launch(seq=seq, code_id=0 if seq < 12 else 1) for seq in range(18)]
        selected, report = proportional_condense_launches(launches, 6)
        sequences = [launch["sequence"] for launch in selected]
        self.assertTrue(report["applied"])
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), 6)
        self.assertEqual([launch["code_id"] for launch in selected].count(0), 4)
        self.assertEqual([launch["code_id"] for launch in selected].count(1), 2)

    def test_l0_max_launches_rolls_without_trigger(self) -> None:
        raw = _l0_test_config(min_launches=100, short_duration_ms=100000, long_duration_ms=200000)
        raw["short_window"]["max_launches"] = 4  # type: ignore[index]
        config = parse_l0_config(raw)
        launches = [_launch(seq=seq, code_id=seq % 2) for seq in range(10)]
        windows = [window for window in build_l0_windows(launches, config) if window.window_type == "short"]
        self.assertEqual(windows, [])


class DisassemblyTests(unittest.TestCase):
    def test_detect_fatbin_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.bin"
            path.write_bytes(bytes.fromhex("50ed55ba01001000"))
            self.assertEqual(detect_code_format(path), "fatbin")

    def test_disassembly_reuses_duplicate_code_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dumps = root / "dumps"
            dumps.mkdir()
            cubin = dumps / "code_0.bin"
            cubin.write_bytes(b"\x7fELF" + b"\x00" * 32)
            duplicate = dumps / "code_1.bin"
            duplicate.write_bytes(cubin.read_bytes())
            tool = root / "fake_nvdisasm.py"
            tool.write_text(
                "#!/usr/bin/env python3\n"
                "print('.type kernel,@function')\n"
                "print('kernel:')\n"
                "print('        /*0000*/ MOV R1, R2 ;')\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)
            report = disassemble_code_objects(
                root,
                {
                    "0": {"code_id": 0, "dump_path": "dumps/code_0.bin", "sha256": "same"},
                    "1": {"code_id": 1, "dump_path": "dumps/code_1.bin", "sha256": "same"},
                },
                tools={"nvdisasm": tool},
            )
            self.assertEqual(report["deduplicated_code_objects"], 1)
            self.assertEqual(report["code_objects"][1]["deduplicated_from_code_id"], 0)
            self.assertEqual(
                report["code_objects"][1]["disassembly_output"],
                report["code_objects"][0]["disassembly_output"],
            )

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

    def test_parse_ok_disassemblies_reuses_shared_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dumps = root / "dumps"
            dumps.mkdir()
            (dumps / "code_0.nvdisasm.txt").write_text(
                ".type kernel,@function\n"
                "kernel:\n"
                "        /*0000*/ MOV R1, R2 ;\n",
                encoding="utf-8",
            )
            parsed = _parse_ok_disassemblies(
                root,
                {
                    "code_objects": [
                        {"code_id": 0, "status": "ok", "disassembly_output": "dumps/code_0.nvdisasm.txt"},
                        {"code_id": 1, "status": "ok", "disassembly_output": "dumps/code_0.nvdisasm.txt"},
                    ]
                },
            )
            self.assertIs(parsed["0"], parsed["1"])

    def test_split_launched_kernels_reuses_duplicate_code_dump_kernel_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dumps = root / "dumps"
            dumps.mkdir()
            (dumps / "code_0.nvdisasm.txt").write_text(
                ".type kernel,@function\n"
                "kernel:\n"
                "        /*0000*/ MOV R1, R2 ;\n",
                encoding="utf-8",
            )
            code_map = {
                "0": {"dump_path": "dumps/code.bin"},
                "1": {"dump_path": "dumps/code.bin", "deduplicated_dump_from": "0"},
            }
            report = {
                "code_objects": [
                    {"code_id": 0, "status": "ok", "disassembly_output": "dumps/code_0.nvdisasm.txt"},
                    {"code_id": 1, "status": "ok", "disassembly_output": "dumps/code_0.nvdisasm.txt"},
                ]
            }
            kernel_dirs, missing = split_launched_kernels(
                root,
                [
                    {"kernel_name": "kernel", "code_id": 0},
                    {"kernel_name": "kernel", "code_id": 1},
                ],
                code_map,
                report,
            )
            self.assertEqual(missing, [])
            self.assertEqual(kernel_dirs[("kernel", 0)], kernel_dirs[("kernel", 1)])
            self.assertEqual(len(list((root / "kernels").glob("*/metadata.json"))), 1)

    def test_parse_nvdisasm_type_function(self) -> None:
        text = """
        .type           cuda_crc32_tweaked,@function
cuda_crc32_tweaked:
        /*0000*/                   MOV R1, R2 ;
        /*0010*/                   EXIT ;
"""
        parsed = parse_disassembly(text)
        self.assertEqual([instr.text for instr in parsed["cuda_crc32_tweaked"]], ["MOV R1, R2", "EXIT"])

    def test_parse_multiline_cuobjdump_function_name(self) -> None:
        text = "Function : $\t \n \t\n\t.headerflags\t@\"EF_CUDA_SM50\"\n        /*0000*/                   MOV R1, R2 ;\n"
        parsed = parse_disassembly(text)
        self.assertEqual(len(parsed), 1)
        name = next(iter(parsed))
        self.assertEqual(name, "$\t \n \t")
        self.assertEqual(parsed[name][0].text, "MOV R1, R2")
        fallback = select_fallback_function("$\t \r \t", parsed)
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback[2], "canonical_name")

    def test_safe_kernel_dir_sanitizes(self) -> None:
        self.assertEqual(safe_kernel_dir("kernel/name:with space"), "kernel_name_with_space")
        self.assertEqual(safe_kernel_dir("_Z6kernelv"), "_Z6kernelv")

    def test_fallback_function_prefers_substring_then_largest(self) -> None:
        functions = {
            "_Z18cuda_ethash_searchv": [SASSInstruction("0", "NOP")],
            "_Z5otherv": [SASSInstruction("0", "NOP"), SASSInstruction("10", "EXIT")],
        }
        self.assertEqual(
            select_fallback_function("cuda_ethash_search", functions)[2],
            "substring_name",
        )
        self.assertEqual(
            select_fallback_function("missing_name", functions)[0],
            "_Z5otherv",
        )

    def test_unique_safe_kernel_dir_handles_repeated_sanitized_names(self) -> None:
        used: set[str] = set()
        counts = Counter()
        first = unique_safe_kernel_dir("$ \n", 0, used, counts)
        used.add(first)
        second = unique_safe_kernel_dir("$\t", 0, used, counts)
        self.assertNotEqual(first, second)
        self.assertEqual(second, "___code_0")


class SassAnalysisTests(unittest.TestCase):
    def test_normalize_examples(self) -> None:
        self.assertEqual(
            normalize_instruction("IADD3 R12, R13, c[0x0][0x20], RZ"),
            "IADD3 REG, REG, CONST, ZERO",
        )
        self.assertEqual(normalize_instruction("@P0 BRA L_3"), "@PRED BRA LABEL")
        self.assertEqual(
            normalize_instruction("CALL.REL.NOINC `($cuda_crc32_tweaked$__cuda_sm20_bfe_u64_)"),
            "CALL LABEL",
        )
        self.assertEqual(normalize_instruction("CALL cuda_crc32_tweaked"), "CALL LABEL")
        self.assertEqual(normalize_instruction("BRA `(.L_x_0)"), "BRA LABEL")
        self.assertEqual(normalize_instruction("MOV R2, 32@lo(CRCTable)"), "MOV REG, IMM")
        self.assertEqual(
            normalize_instruction("LDC R8, c[0x0][0x20] &wr=0 ?trans1"),
            "LDC REG, CONST",
        )
        self.assertEqual(
            normalize_instruction("S2R R8, SR_TID.X &wr=0 ?trans1"),
            "S2R REG, SREG",
        )
        self.assertEqual(
            normalize_instruction("FMUL R8, R9, 5.5511151231257827021e-17 &req={0, 1} ?trans1"),
            "FMUL REG, REG, IMM",
        )
        self.assertEqual(normalize_instruction("BSSY B0, `(.L_x_70)"), "BSSY BREG, LABEL")
        self.assertEqual(normalize_instruction("@P0 BRA L_1ac0 ?trans6"), "@PRED BRA LABEL")
        self.assertEqual(
            normalize_instruction('BRX R18 -L_3090 (*"BRANCH_TARGETS .L_x_64634,.L_x_64635"*)'),
            "BRX REG, LABEL",
        )
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

    def test_load_workload_records_reads_l0_window_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workload_dir = root / "dataset" / "workloads" / "demo"
            windows_dir = workload_dir / "windows"
            windows_dir.mkdir(parents=True)
            (workload_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "workload": "demo",
                        "family": "hpc",
                        "label": "benign_compute_like",
                        "opt_level": "capture",
                        "capture_id": "cap",
                    }
                ),
                encoding="utf-8",
            )
            (windows_dir / "w0000_short_stream.sass").write_text("IADD3 REG, REG, IMM, ZERO\n", encoding="utf-8")
            (windows_dir / "manifests.jsonl").write_text(
                json.dumps(
                    {
                        "workload": "demo__w0000_short_stream",
                        "path": "windows/w0000_short_stream.sass",
                        "label": "benign_compute_like",
                        "binary_label": "benign",
                        "family": "hpc",
                        "opt_level": "capture",
                        "parent_workload": "demo",
                        "capture_id": "cap",
                        "window_id": "w0000_short_stream",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            from sassguard_analysis.splits import load_workload_records

            records = load_workload_records(root / "dataset" / "workloads", root / "dataset")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["workload"], "demo__w0000_short_stream")
            self.assertEqual(records[0]["group_id"], "demo")
            self.assertEqual(records[0]["path"], "dataset/workloads/demo/windows/w0000_short_stream.sass")


def _l0_test_config(
    min_launches: int = 32,
    short_duration_ms: int = 5000,
    long_duration_ms: int = 60000,
) -> dict[str, object]:
    return {
        "enabled": True,
        "grouping": {
            "emit_stream_windows": True,
            "emit_process_aggregate_windows": False,
            "include_tid_in_group": False,
        },
        "maturity": {
            "min_launches": min_launches,
            "min_duration_ms": 1000,
            "long_min_duration_ms": 30000,
        },
        "short_window": {
            "duration_ms": short_duration_ms,
            "max_launches": 256,
            "target_l1_chunks": 4,
        },
        "long_window": {
            "duration_ms": long_duration_ms,
            "max_launches": 2048,
            "target_l1_chunks": 8,
            "max_emitted_launches": 64,
        },
        "repetition": {
            "dominant_code_id_ratio": 0.70,
            "top3_code_id_ratio": 0.85,
            "normalized_entropy": 0.45,
        },
        "trigger": {
            "stable_shape_min_launches": 64,
            "grid_stability": 0.80,
            "block_stability": 0.80,
            "cooldown_ms": 10000,
            "major_change_min_interval_ms": 5000,
            "periodic_sample_ms": 30000,
            "entropy_shift": 0.35,
            "top3_jaccard_threshold": 0.34,
            "shape_change_min_stability": 0.8,
        },
        "condensation": {
            "enabled": True,
            "preserve_first_last": True,
            "min_per_code_id": 1,
        },
    }


def _launch(
    seq: int,
    tid: int = 1,
    stream: int = 0,
    code_id: int = 0,
    grid: list[int] | None = None,
    block: list[int] | None = None,
) -> dict[str, object]:
    return {
        "sequence": seq,
        "timestamp_ns": seq * 1_000_000,
        "pid": 123,
        "tid": tid,
        "code_id": code_id,
        "kernel_name": f"k{code_id}",
        "grid_dim": grid or [1, 1, 1],
        "block_dim": block or [128, 1, 1],
        "shared_mem_bytes": 0,
        "stream": stream,
        "device_pci_bus_id": "0000:52:00.0",
    }


if __name__ == "__main__":
    unittest.main()
