from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from experiments import throttle_mining


class ThrottleMiningTests(unittest.TestCase):
    def test_parse_total_launches(self) -> None:
        self.assertEqual(throttle_mining.parse_total_launches("total_launches=123\nstatus=ok\n"), 123)
        self.assertEqual(throttle_mining.parse_total_launches("status=ok\n"), 0)

    def test_planned_rows_cover_conditions_and_repeats(self) -> None:
        args = argparse.Namespace(
            algorithms=["sha256d_mono"],
            conditions=[100, 75, 50, 10],
            repeats=2,
            runtime_sec=600,
            opt_level="o3",
        )
        rows = throttle_mining.planned_rows("kernel-launch", args)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["target_percent"] for row in rows}, {100, 75, 50, 10})
        self.assertEqual({row["repeat"] for row in rows}, {1, 2})
        self.assertTrue(all(row["variant"] == "o3" for row in rows))

    def test_dry_run_sleep_matrix_is_non_running_placeholder(self) -> None:
        args = argparse.Namespace(algorithms=["sha256d_mono"], conditions=[100, 75, 50, 10])
        self.assertEqual(
            throttle_mining.dry_run_sleep_us(args),
            {"sha256d_mono": {100: 0, 75: 0, 50: 0, 10: 0}},
        )

    def test_all_mining_sources_have_one_throttle_call(self) -> None:
        root = Path("workloads/synthetic_kernels/src/mining")
        sources = sorted(root.glob("*/*.cu"))
        self.assertGreater(len(sources), 0)
        bad = [str(path) for path in sources if path.read_text(encoding="utf-8").count("sleep_after_launch_if_requested(args);") != 1]
        self.assertEqual(bad, [])

    def test_shared_cli_exposes_ms_and_us_flags(self) -> None:
        text = Path("workloads/synthetic_kernels/include/common/cli_args.hpp").read_text(encoding="utf-8")
        self.assertIn("--sleep-between-launches-ms", text)
        self.assertIn("--sleep-between-launches-us", text)
        self.assertIn("std::chrono::microseconds", text)


if __name__ == "__main__":
    unittest.main()
