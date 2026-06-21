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

## Online detection

Online detection runs as a streaming path alongside the existing collector output:

```text
hooklib -> sassguard-collector -> online processor -> sassguard-collector -> hooklib
```

Start the Python processor first:

```bash
SASSGUARD_CAPTURE_DISABLE=1 \
python3 -m online.sassguard_online.processor \
  --config configs/online/detection.json
```

Set `SASSGUARD_CAPTURE_DISABLE=1` for the processor and any other trusted
SASSGuard-side process that may load CUDA. The hook checks this environment
variable before starting telemetry, which prevents the processor's own
ModernBERT CUDA inference from being captured and fed back into the collector.
The value must be present before the process loads the hook library.

Then start the collector:

```bash
collector/build/sassguard-collector \
  --config configs/online/detection.json
```

All online runtime knobs live in `configs/online/detection.json`, including the collector listen address, Unix processor socket, launch batching limits, L0 config path, ModernBERT config/checkpoint, verdict thresholds, and enforcement behavior. The processor feeds L1 only single rolling-window inputs that fit the ModernBERT sequence budget. Set `l1.devices` to a list such as `["cuda:0", "cuda:1"]` to load one inference worker per GPU, or leave it empty with `l1.device: "auto"` to use all visible CUDA devices.

The collector still writes the same capture artifacts (`process.json`, `events.jsonl`, and `code/*.bin`). Online forwarding is fail-open: if the processor socket is unavailable, code analysis fails, inference fails, or queues overflow, the CUDA client keeps running and the collector continues storing captures.

When a hook client disconnects or exits, the collector sends a `session_end` frame to the processor. The processor cancels queued code-analysis and inference jobs for that session, drops unsent verdicts, and discards results from any already-running inference job that finishes after the session ended.

L0 launch-window rules are shared with offline dataset generation. A launch is schedulable only after its kernel has been disassembled, normalized, token-costed, and assigned an int/bitwise ratio; launches that arrive before their kernel is ready are dropped rather than deferred.

When enforcement is enabled and L1 returns a mining verdict, the collector sends a control command back to hooklib. Hooklib prints the configured message and verdict details to stderr, then terminates the client process.
