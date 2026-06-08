# SASSGuard Analysis

This directory contains the CUDA-capture dataset builder.

Default run:

```bash
python3 analysis/generate_synthetic_capture_manifest.py
python3 analysis/build_dataset.py \
  --capture-manifest workloads/synthetic_kernels/captures/manifests.jsonl \
  --capture-root . \
  --output-dir dataset \
  --max-launches 16 \
  --jobs 8
```

Build from capture manifests:

```bash
python3 analysis/build_dataset.py \
  --capture-manifest 'workloads/*_samples/*/captures/manifests.jsonl' \
  --capture-root . \
  --output-dir dataset_realworld \
  --max-launches 16 \
  --jobs 8
```

The builder writes workload artifacts to:

```text
dataset/workloads/<workload_name>/
```

The builder uses each manifest row's `workload` field and appends a capture-id suffix when multiple rows would otherwise produce the same workload directory. The synthetic manifest generator sets `workload` to the captured binary name, so optimization suffixes such as `_o2` and `_o3` are preserved.

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
- Each workload is built in a temporary directory and moved into place only after validation succeeds.
- Captures are processed in parallel by default with `--jobs min(8, os.cpu_count())`; use `--jobs 1` for serial debugging.
- CUDA tools are discovered from `PATH` and common locations such as `/usr/local/cuda*/bin`.
- Capture-manifest rows must include `label`, `family`, `workload`, `program`, `variant`, and `capture_path` or `capture_dir`.
- `analysis/generate_synthetic_capture_manifest.py` writes the synthetic capture manifest from the existing synthetic binary-label metadata.
- Manifest-driven builds skip captures with no `kernel_launch` events by default; use `--no-skip-empty-captures` to make those hard failures.
- Captures whose launches cannot be mapped to disassembled SASS are skipped by default; use `--no-skip-unmapped-captures` to make those hard failures.
- When a capture is skipped or fails during build, partial output is removed unless `--keep-partial` is set.
- Classification splits are workload-level and grouped by optimization-pair stem, so `_o2` and `_o3` variants stay in the same split.
