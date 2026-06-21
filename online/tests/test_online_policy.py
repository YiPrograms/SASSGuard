import unittest
import tempfile
from pathlib import Path

from online.sassguard_online.processor import KernelArtifact, launch_for_artifact, rolling_mean_and_max_decision
from sassguard_analysis.manifest import write_json
from sassguard_analysis.workload_sass import render_workload_sass


class RollingMeanAndMaxPolicyTest(unittest.TestCase):
    def test_requires_mean_and_max_thresholds(self) -> None:
        decision = rolling_mean_and_max_decision(
            [0.40, 0.40, 0.40],
            mean_threshold=0.30,
            max_threshold=0.50,
        )
        self.assertFalse(decision["suspicious"])
        self.assertAlmostEqual(decision["mean"], 0.40)
        self.assertAlmostEqual(decision["max"], 0.40)

    def test_inclusive_thresholds(self) -> None:
        decision = rolling_mean_and_max_decision(
            [0.20, 0.20, 0.50],
            mean_threshold=0.30,
            max_threshold=0.50,
        )
        self.assertTrue(decision["suspicious"])
        self.assertAlmostEqual(decision["mean"], 0.30)
        self.assertAlmostEqual(decision["max"], 0.50)


class LaunchArtifactMappingTest(unittest.TestCase):
    def test_fallback_artifact_name_is_used_for_l1_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workload_dir = Path(tmp)
            kernel_dir = workload_dir / "kernels" / "resolved_kernel"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "kernel.normalized.sass").write_text("LOP3 REG, REG, REG, REG, IMM\n", encoding="utf-8")
            (kernel_dir / "main_loop.normalized.sass").write_text("", encoding="utf-8")
            write_json(
                kernel_dir / "metadata.json",
                {
                    "kernel_name": "resolved_kernel",
                    "code_id": 7,
                    "safe_kernel_dir": "resolved_kernel",
                },
            )
            artifact = KernelArtifact(
                code_id=7,
                kernel_name="resolved_kernel",
                kernel_dir=kernel_dir,
                features={},
                kernel_id="resolved_kernel::7",
                token_cost=2,
                bitwise_integer_ratio=1.0,
                rendered_instruction_count=1,
                render_mode="full_kernel",
            )
            launch = {"code_id": 7, "kernel_name": "$\n\t\r ", "stream": 0}

            ready = launch_for_artifact(launch, artifact)
            rendered = render_workload_sass(workload_dir, [ready])

            self.assertEqual(ready["kernel_name"], "resolved_kernel")
            self.assertEqual(ready["captured_kernel_name"], "$\n\t\r ")
            self.assertIn("LOP3", rendered["text"])

    def test_l1_renderer_can_resolve_raw_name_by_l0_kernel_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workload_dir = Path(tmp)
            kernel_dir = workload_dir / "kernels" / "resolved_kernel"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "kernel.normalized.sass").write_text("LOP3 REG, REG, REG, REG, IMM\n", encoding="utf-8")
            (kernel_dir / "main_loop.normalized.sass").write_text("", encoding="utf-8")
            write_json(
                kernel_dir / "metadata.json",
                {
                    "kernel_name": "resolved_kernel",
                    "code_id": 7,
                    "safe_kernel_dir": "resolved_kernel",
                },
            )
            launch = {
                "code_id": 7,
                "kernel_name": "$\n\t\r ",
                "l0_kernel_id": "7:resolved_kernel",
                "stream": 0,
            }

            rendered = render_workload_sass(workload_dir, [launch])

            self.assertIn("LOP3", rendered["text"])


if __name__ == "__main__":
    unittest.main()
