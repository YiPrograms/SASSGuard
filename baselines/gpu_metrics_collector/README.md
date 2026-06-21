# GPU Metrics Collector

Standalone telemetry collector for the Tanana and Pott GPU cryptojacking baselines.

The collector does not use the SASSGuard CUDA hook. It records raw metrics only:

- `device_metrics.jsonl` for Pott device-level telemetry.
- `process_metrics.jsonl` for Tanana process-level telemetry.
- `metadata.json`, `stdout.log`, and `stderr.log` for each workload run.

All workload commands are launched with one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=<gpu>
```

Inside that restricted environment, known GPU-selection flags are rewritten to logical device `0`.

## Usage

Record one command:

```bash
python3 -m baselines.gpu_metrics_collector.cli record \
  --gpu 0 \
  --output-dir baseline_dataset/gpu_metrics/raw \
  --label benign_compute_like \
  --binary-label benign \
  --workload example \
  -- ./workload 10
```

Run existing workload manifests:

```bash
python3 -m baselines.gpu_metrics_collector.cli run-manifest \
  --gpu 0 \
  --output-dir baseline_dataset/gpu_metrics/raw
```

Build features and baseline outputs:

```bash
python3 -m baselines.gpu_metrics_collector.cli windowize
python3 -m baselines.gpu_metrics_collector.cli eval-tanana
python3 -m baselines.gpu_metrics_collector.cli train-pott-point-rf
```

`train-pott-point-rf` trains the GPU-counter behavioral baseline on
`pott_points.csv`, where each row is one `nvidia-smi` device measurement. The
saved model predicts every measuring point, counts positive predictions over a
run, and raises an alarm once the tuned temporal threshold is met. The default
policy averages RF mining probabilities over time and alarms when the run mean
exceeds the tuned threshold. By default, threshold tuning enforces
`--max-run-fpr 0.01` on the validation split, making the detector favor low
benign false alarms over mining recall. It requires `scikit-learn`; recording
and feature building use only the Python standard library plus `nvidia-smi`.

`train-pott-rf` remains available as the old 60s-window Pott comparison.
`train-hybrid-rf` remains available as a process+device ablation on
`hybrid_windows.csv`. Neither is the behavioral baseline used by the final
kernel-launch throttle experiment.
