from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baselines.gpu_metrics_collector.backends.nvidia_smi import parse_csv, parse_pmon
from baselines.gpu_metrics_collector.command_rewrite import rewrite_gpu_args
from baselines.gpu_metrics_collector.windowize import build_features
from baselines.gpu_metrics_collector.io import write_json, write_jsonl
from baselines.gpu_metrics_collector.tanana import tanana_verdict


class NvidiaSmiParserTests(unittest.TestCase):
    def test_parse_gpu_csv(self) -> None:
        text = "0, GPU-abc, 00000000:52:00.0, Tesla V100, 580.65.06, 91, 42, 150.5, 250.0, 70, N/A, 1024, 32768, 1200, 877\n"
        rows = parse_csv(
            text,
            [
                "index",
                "uuid",
                "pci.bus_id",
                "name",
                "driver_version",
                "utilization.gpu",
                "utilization.memory",
                "power.draw",
                "power.limit",
                "temperature.gpu",
                "fan.speed",
                "memory.used",
                "memory.total",
                "clocks.gr",
                "clocks.mem",
            ],
        )
        self.assertEqual(rows[0]["uuid"], "GPU-abc")
        self.assertEqual(rows[0]["fan.speed"], "N/A")

    def test_parse_pmon(self) -> None:
        text = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0       1234     C     98      7      -      -      -      -    python
    1          -     -      -      -      -      -      -      -    -
"""
        rows = parse_pmon(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gpu_index"], 0)
        self.assertEqual(rows[0]["pid"], 1234)
        self.assertEqual(rows[0]["process_gpu_utilization_pct"], 98.0)


class CommandRewriteTests(unittest.TestCase):
    def test_rewrite_zero_based_flags(self) -> None:
        result = rewrite_gpu_args(["miner", "--devices", "3", "--cuda-devices=5", "-d", "7"])
        self.assertEqual(result.argv, ["miner", "--devices", "0", "--cuda-devices=0", "-d", "0"])
        self.assertEqual(len(result.changes), 3)

    def test_rewrite_claymore_gpus_as_one_based_visible_device(self) -> None:
        result = rewrite_gpu_args(["ethdcrminer64", "-gpus", "4"])
        self.assertEqual(result.argv, ["ethdcrminer64", "-gpus", "1"])


class WindowizeTests(unittest.TestCase):
    def test_adaptive_windows_and_tanana_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "raw" / "workload-1"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "metadata.json",
                {
                    "run_id": "workload-1",
                    "workload": "workload_o2",
                    "label": "mining_like",
                    "binary_label": "mining",
                    "family": "test",
                    "program": "workload",
                    "variant": "o2",
                },
            )
            device_rows = []
            process_rows = []
            for i in range(5):
                ts = 1_000_000_000 + i * 1_000_000_000
                device_rows.append(
                    {
                        "timestamp_ns": ts,
                        "run_id": "workload-1",
                        "gpu_uuid": "GPU-abc",
                        "gpu_utilization_pct": 99,
                        "memory_utilization_pct": 95,
                        "power_usage_watts": 150,
                        "temperature_celsius": 70,
                        "fan_speed_pct": None,
                    }
                )
                process_rows.append(
                    {
                        "timestamp_ns": ts,
                        "run_id": "workload-1",
                        "gpu_uuid": "GPU-abc",
                        "pid": 123,
                        "process_gpu_utilization_pct": 95,
                        "process_gpu_memory_pct": 95,
                        "host_ram_gb": 4,
                    }
                )
            write_jsonl(run_dir / "device_metrics.jsonl", device_rows)
            write_jsonl(run_dir / "process_metrics.jsonl", process_rows)
            report = build_features(root / "raw", root / "features")
            self.assertEqual(report["pott_windows"], 1)
            self.assertEqual(report["pott_points"], 5)
            self.assertEqual(report["tanana_windows"], 1)
            self.assertEqual(report["hybrid_windows"], 1)

    def test_tanana_decision_tree(self) -> None:
        row = {
            "tanana_status": "tanana_paper_exact",
            "avg_process_gpu_utilization_pct": "95",
            "avg_process_gpu_memory_pct": "95",
            "avg_host_ram_gb": "4",
            "std_process_gpu_utilization_pct": "1.5",
        }
        self.assertEqual(tanana_verdict(row), "suspicious_mining")


if __name__ == "__main__":
    unittest.main()
