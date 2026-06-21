# Behavioral Baseline Experiments

This document records the GPU-counter behavioral baseline experiments. These
experiments do not use SASSGuard captures or CUDA-hook telemetry. They use only
whole-device `nvidia-smi` measurements.

The experiments currently documented here are:

- basic training and evaluation on `baseline_dataset/gpu_metrics/`
- kernel-launch throttle mining
- mixed benign + mining workloads
- benign controls for the mixed-workload setting

The kernel-launch throttle experiment is the only behavioral experiment with an
active high-level runner in `experiments/throttle_mining.py`. The mixed and
benign-control experiments are retained as archived evaluations; their raw
metric directories can still be re-windowized and re-scored.

## Baseline Model

The behavioral baseline is a Pott-style Random Forest trained on pointwise GPU
device measurements:

```text
baseline_dataset/gpu_metrics/features/pott_points.csv
```

Each row is one `nvidia-smi` measuring point. The feature columns are:

- `gpu_utilization_pct`
- `memory_utilization_pct`
- `power_usage_watts`
- `temperature_celsius`
- `fan_speed_pct`, when available

The model predicts a mining probability for every measuring point. A run-level
alarm is raised by averaging mining probabilities over time:

```text
alarm if mean_mining_probability >= tuned_threshold
```

Current tuned policy:

```text
aggregation = mean_probability
min_mean_mining_probability = 0.8
max validation benign FPR = 0.01
```

## Basic Training And Evaluation

This experiment trains the pointwise GPU-counter Random Forest and tunes the
run-level mean-probability alarm threshold on the validation split.

Collect the baseline raw metric dataset from the existing workload manifests:

```bash
python3 -m baselines.gpu_metrics_collector.cli run-manifest \
  --gpu 0 \
  --manifest workloads/synthetic_kernels/captures/manifests.jsonl \
  --manifest 'workloads/benign_samples/*/captures/manifests.jsonl' \
  --manifest 'workloads/mining_samples/*/captures/manifests.jsonl' \
  --output-dir baseline_dataset/gpu_metrics/raw
```

Build point features from the baseline raw GPU-metric dataset:

```bash
python3 -m baselines.gpu_metrics_collector.cli windowize \
  --input-dir baseline_dataset/gpu_metrics/raw \
  --output-dir baseline_dataset/gpu_metrics/features
```

Train and evaluate the pointwise Pott RF:

```bash
python3 -m baselines.gpu_metrics_collector.cli train-pott-point-rf \
  --input-csv baseline_dataset/gpu_metrics/features/pott_points.csv \
  --output-dir baseline_dataset/gpu_metrics/features/pott_point_rf \
  --max-run-fpr 0.01
```

Model outputs:

```text
baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_model.joblib
baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_report.json
baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_sample_predictions.csv
baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_run_predictions.csv
```

Current baseline training/evaluation summary:

| Split | Point Accuracy | Run Accuracy |
|---|---:|---:|
| train | 0.994 | 1.000 |
| val | 0.941 | 1.000 |
| test | 0.902 | 0.980 |

Run split sizes:

| Split | Runs |
|---|---:|
| train | 231 |
| val | 48 |
| test | 50 |

The tuned threshold is selected on the validation run split with a benign
false-alarm constraint:

| Policy Field | Value |
|---|---:|
| `aggregation` | `mean_probability` |
| `min_mean_mining_probability` | 0.8 |
| `max_fpr` | 0.01 |
| validation benign FPR | 0.0 |
| validation mining recall | 1.0 |

## Kernel-Launch Throttle Experiment

This experiment tests whether the GPU-counter baseline detects miners when
their kernel-launch loops are slowed down.

Scope:

```text
kernel-launch throttling only
no SASS capture
no process/PID features
no duty-cycle throttling
no mixed workloads
```

Throttle percentages:

```text
5%, 10%, 25%, 50%, 75%, 100%
```

Mining programs:

```text
ethash_split
kawpow_split
randomx_gpu_lite_mono
sha256d_mono
autolykos2_split
cryptonight_gpu_split
cuckoo_cycle_split
equihash144_5_split
```

Default run settings:

```text
GPU: physical GPU 0 only
runtime: 60 seconds
repeat: 1
optimization: o3
```

Total:

```text
8 programs x 6 percentages x 1 repeat = 48 runs
```

Collect metrics, build point features, and score the experiment:

```bash
SASSGUARD_CAPTURE_DISABLE=1 \
python3 experiments/throttle_mining.py run-behavioral-throttle \
  --output-dir experiments/results/throttle_behavioral_kernel_launch \
  --runtime-sec 60 \
  --repeats 1 \
  --gpu 0
```

Useful variants:

```bash
# Dry-run the exact commands without launching workloads.
SASSGUARD_CAPTURE_DISABLE=1 \
python3 experiments/throttle_mining.py run-behavioral-throttle \
  --output-dir experiments/results/throttle_behavioral_kernel_launch \
  --runtime-sec 60 \
  --repeats 1 \
  --gpu 0 \
  --dry-run

# Resume an interrupted run, skipping completed workload/condition pairs.
SASSGUARD_CAPTURE_DISABLE=1 \
python3 experiments/throttle_mining.py run-behavioral-throttle \
  --output-dir experiments/results/throttle_behavioral_kernel_launch \
  --runtime-sec 60 \
  --repeats 1 \
  --gpu 0 \
  --resume
```

The runner performs a short unthrottled pilot for each mining program, estimates
the base launch-loop rate, converts each target launch-rate percentage into
`--sleep-between-launches-us`, records GPU metrics, builds point features, and
scores run-level alarms.

Outputs:

```text
experiments/results/throttle_behavioral_kernel_launch/raw/manifest.jsonl
experiments/results/throttle_behavioral_kernel_launch/features/pott_points.csv
experiments/results/throttle_behavioral_kernel_launch/features/gpu_counter_throttle_point_predictions.csv
experiments/results/throttle_behavioral_kernel_launch/features/gpu_counter_throttle_alarm_predictions.csv
experiments/results/throttle_behavioral_kernel_launch/throttle_behavioral_report.json
```

Current result:

| Metric | Value |
|---|---:|
| Runs | 48 |
| Points | 2928 |
| Detected mining runs | 23 |
| Missed mining runs | 25 |
| Run recall | 0.479 |
| Confusion matrix | `[[0, 0], [25, 23]]` |

Recall by throttle percentage:

| Launch rate | Detected / total | Recall |
|---:|---:|---:|
| 5% | 8 / 8 | 1.000 |
| 10% | 7 / 8 | 0.875 |
| 25% | 0 / 8 | 0.000 |
| 50% | 0 / 8 | 0.000 |
| 75% | 0 / 8 | 0.000 |
| 100% | 8 / 8 | 1.000 |

Recall by mining program:

| Program | Detected / total | Recall |
|---|---:|---:|
| `autolykos2_split` | 3 / 6 | 0.500 |
| `cryptonight_gpu_split` | 3 / 6 | 0.500 |
| `cuckoo_cycle_split` | 3 / 6 | 0.500 |
| `equihash144_5_split` | 3 / 6 | 0.500 |
| `ethash_split` | 3 / 6 | 0.500 |
| `kawpow_split` | 3 / 6 | 0.500 |
| `randomx_gpu_lite_mono` | 3 / 6 | 0.500 |
| `sha256d_mono` | 2 / 6 | 0.333 |

Interpretation: the behavioral baseline detects the extremes, `5%` and
`100%`, but misses the middle launch-rate throttling regimes. Whole-device GPU
counters are weak when the miner lands in ordinary-looking mid-utilization
regimes.

## Mixed Workload Evaluation

The mixed workload experiment is an archived behavioral evaluation. It is not
part of the active high-level runner anymore, but the data and reports remain
under:

```text
experiments/results/mixed_behavioral/
```

The mixed experiment ran a benign GPU workload and a throttled miner
concurrently on one GPU. Because the behavioral baseline is device-only, every
mixed run is labeled as `mining` at run level if the miner is present. The
baseline does not perform PID attribution.

The original mixed collection used an older `mixed-wrapper` helper that is no
longer exposed by the current `experiments/throttle_mining.py` CLI. The
commands below reproduce the feature build and scoring from the archived raw
metrics.

Rebuild mixed-workload point features from the archived raw metrics:

```bash
python3 -m baselines.gpu_metrics_collector.cli windowize \
  --input-dir experiments/results/mixed_behavioral/raw \
  --output-dir experiments/results/mixed_behavioral/features
```

Re-score the mixed-workload features with the trained pointwise RF:

```bash
python3 - <<'PY'
import json
from pathlib import Path

from experiments.throttle_mining import score_gpu_counter_point_predictions

root = Path("experiments/results/mixed_behavioral")
report = score_gpu_counter_point_predictions(
    root / "features" / "pott_points.csv",
    Path("baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_model.joblib"),
    point_predictions_path=root / "features" / "gpu_counter_mixed_point_predictions.csv",
    run_predictions_path=root / "features" / "gpu_counter_mixed_alarm_predictions.csv",
    truth_label="mining",
    condition_key=lambda row: str(row.get("workload") or row.get("program") or "mixed"),
)
(root / "mixed_behavioral_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
```

Important files:

```text
experiments/results/mixed_behavioral/raw/manifest.jsonl
experiments/results/mixed_behavioral/features/pott_points.csv
experiments/results/mixed_behavioral/features/gpu_counter_mixed_point_predictions.csv
experiments/results/mixed_behavioral/features/gpu_counter_mixed_alarm_predictions.csv
experiments/results/mixed_behavioral/mixed_behavioral_report.json
```

Current mixed result:

| Metric | Value |
|---|---:|
| Runs | 7 |
| Points | 1187 |
| Detected mining-present runs | 2 |
| Missed mining-present runs | 5 |
| Mining-present recall | 0.286 |
| Confusion matrix | `[[0, 0], [5, 2]]` |

Per-run mixed alarms:

| Workload | Mean Mining Probability | Alarm |
|---|---:|---|
| `mixed_aes_ethash_split_50pct` | 0.092 | benign |
| `mixed_cublas_gemm_randomx_gpu_lite_mono_10pct` | 0.524 | benign |
| `mixed_cublas_gemm_randomx_gpu_lite_mono_10pct` | 0.540 | benign |
| `mixed_cudnn_convolution_kawpow_split_10pct` | 0.124 | benign |
| `mixed_hpl_sha256d_mono_10pct` | 0.152 | benign |
| `mixed_pytorch_training_ethash_split_10pct` | 0.812 | mining |
| `mixed_vllm_inference_fallback_randomx_gpu_lite_mono_50pct` | 0.918 | mining |
