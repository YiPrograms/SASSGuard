# SASSGuard Analysis

This directory contains the CUDA-capture dataset builder.

Default run:

```bash
python3 analysis/generate_synthetic_capture_manifest.py
python3 analysis/build_dataset.py \
  --capture-manifest workloads/synthetic_kernels/captures/manifests.jsonl \
  --capture-root . \
  --output-dir dataset \
  --l0-config configs/analysis/l0_windows.json \
  --jobs 8
```

Build from capture manifests:

```bash
python3 analysis/build_dataset.py \
  --capture-manifest 'workloads/*_samples/*/captures/manifests.jsonl' \
  --capture-root . \
  --output-dir dataset_realworld \
  --l0-config configs/analysis/l0_windows.json \
  --jobs 8
```

The builder writes workload artifacts to:

```text
dataset/workloads/<workload_name>/
```

By default, the builder applies dynamic L0 launch-window scheduling and writes emitted window SASS files under each workload:

```text
dataset/workloads/<workload_name>/windows/<window_id>.sass
```

Each workload directory also contains `windows/manifests.jsonl`, which is what split generation uses to create one dataset row per emitted L0 window. Use `--no-l0-windowing --max-launches 16` to produce the legacy single `workload.sass` per capture.

L0 is a window scheduler, not a detector. It uses only kernel launch metadata, groups launches by stream, and emits L1 jobs only when a mature stream-local window hits a trigger rule. Maintenance bounds roll/reset windows without automatically invoking L1. Long windows use proportional launch condensation and are meant for low-duty or persistent streams. The resolved window policy from `configs/analysis/l0_windows.json` is recorded in `build_report.json` and each `windows/<window_id>.json`.

Useful checks:

```bash
python3 -m unittest discover -s analysis/tests
python3 analysis/build_dataset.py \
  --capture-manifest workloads/synthetic_kernels/captures/manifests.jsonl \
  --dry-run \
  --verbose
```

Create ModernBERT classification splits:

```bash
python3 analysis/split_dataset.py \
  --dataset-dir dataset \
  --output-dir dataset/splits \
  --seed 1337
```

Notes:

- Existing `dataset/workloads/<workload_name>/` directories are skipped by default.
- L0 windowing is controlled by `--l0-config`; use `--no-l0-windowing` for legacy whole-capture preprocessing.
- Each workload is built in a temporary directory and moved into place only after validation succeeds.
- Captures are processed in parallel by default with `--jobs min(8, os.cpu_count())`; use `--jobs 1` for serial debugging.
- CUDA tools are discovered from `PATH` and common locations such as `/usr/local/cuda*/bin`.
- Capture-manifest rows must include `label`, `family`, `workload`, `program`, `variant`, and `capture_path` or `capture_dir`.
- `analysis/generate_synthetic_capture_manifest.py` writes the synthetic capture manifest from the existing synthetic binary-label metadata.
- Manifest-driven builds skip captures with no `kernel_launch` events by default; use `--no-skip-empty-captures` to make those hard failures.
- Captures whose launches cannot be mapped to disassembled SASS are skipped by default; use `--no-skip-unmapped-captures` to make those hard failures.
- When a capture is skipped or fails during build, partial output is removed unless `--keep-partial` is set.
- Classification splits are workload-level and grouped by optimization-pair stem, so `_o2` and `_o3` variants stay in the same split.
