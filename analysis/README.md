# SASSGuard Analysis

This directory contains the synthetic-kernel dataset builder.

Default run:

```bash
python3 analysis/build_dataset.py \
  --captures-dir workloads/synthetic_kernels/captures \
  --binary-manifest workloads/synthetic_kernels/binaries/manifest.jsonl \
  --output-dir dataset \
  --max-launches 16 \
  --jobs 8
```

The builder writes workload artifacts to:

```text
dataset/workloads/<workload_name>/
```

It derives `<workload_name>` from `basename(process.json["exe_path"])`, so optimization suffixes such as `_o2` and `_o3` are preserved.

Useful checks:

```bash
python3 -m unittest discover -s analysis/tests
python3 analysis/build_dataset.py --dry-run --verbose
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
- If no launched kernel can be mapped to SASS, the capture is marked failed and partial output is removed unless `--keep-partial` is set.
- Classification splits are workload-level and grouped by optimization-pair stem, so `_o2` and `_o3` variants stay in the same split.
