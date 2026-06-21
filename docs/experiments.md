# Experiment Results

This document consolidates the available SASSGuard ModernBERT and behavioral
baseline experiment results. Unless stated otherwise, confusion matrices use
rows as ground truth and columns as predictions, in the label order printed
above each matrix.

## Result Sources

The results below are taken from the checked-in experiment artifacts:

- SASSGuard synthetic SASS dataset: `dataset/build_report.json`,
  `dataset/splits/split_manifest.json`
- SASSGuard real-world SASS dataset: `dataset_realworld/build_report.json`,
  `dataset_realworld/splits/split_manifest.json`
- ModernBERT synthetic binary reports:
  `experiments/results/reports/modernbert_binary_gpu3/classifier_report.json`,
  `experiments/results/reports/modernbert_binary_gpu3/classifier_val_report.json`,
  `experiments/results/reports/modernbert_binary_gpu3/classifier_test_report.json`
- ModernBERT real-world binary report:
  `experiments/results/reports/modernbert_binary_gpu3_realworld/evaluate_test_report.json`
- SASSGuard throttle report:
  `experiments/results/throttle_sassguard/kernel_launch/throttle_sassguard_report.json`
- Behavioral baseline normal report:
  `baseline_dataset/gpu_metrics/features/pott_point_rf/pott_point_rf_report.json`
- Behavioral throttle report:
  `experiments/results/throttle_behavioral_kernel_launch/throttle_behavioral_report.json`
- Behavioral mixed-workload reports:
  `experiments/results/mixed_behavioral/mixed_behavioral_report.json`,
  `experiments/results/behavioral_normal_mixed_with_benign_controls_report.json`
- Top launched-kernel metrics:
  `experiments/results/top_kernel_metrics/top_kernel_metrics.json`,
  `experiments/results/top_kernel_metrics/top_kernel_metrics.md`

## Top Launched Kernel Metrics

The table below summarizes the three most frequently launched named kernels in
the selected vLLM, HPL, and T-Rex captures. Launches per second are computed
over each capture's first-to-last kernel-launch timestamp. Instruction count
is the parsed SASS instruction count for the matched kernel. Compute
instruction ratio is the fraction of parsed SASS instructions whose opcode is
in the integer/bitwise, floating-point/SFU, or tensor-compute families.

vLLM has 17,611 launches with an empty kernel name and unknown code ID in this
capture. Those launches are excluded from the top-kernel ranking because they
cannot be mapped to kernel SASS.

| Workload | Rank | Kernel | Launches/s | Instruction count | Compute instruction ratio |
|---|---:|---|---:|---:|---:|
| vLLM | 1 | `reshape_and_cache_kernel_flash` | 132.758 | 50 | 0.680 |
| vLLM | 2 | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` | 127.371 | 158 | 0.753 |
| vLLM | 3 | `triton_poi_fused_mul_silu_slice_1` | 127.320 | 143 | 0.881 |
| HPL | 1 | `volta_dgemm_128x64_nt` | 42.471 | 1054 | 0.793 |
| HPL | 2 | `void trsm_right_kernel<double, 256, 4, true, false, true, false, true>(cublasTrsmParams<double>, double, double const*, int)` | 41.683 | 198 | 0.576 |
| HPL | 3 | `void trsm_right_kernel<double, 256, 4, false, false, true, false, true>(cublasTrsmParams<double>, double, double const*, int)` | 41.683 | 184 | 0.516 |
| T-Rex | 1 | `cuda_ethash_search` | 57.796 | 1512 | 0.854 |
| T-Rex | 2 | `cuda_ethash_search_1` | 29.190 | 1555 | 0.849 |
| T-Rex | 3 | `cuda_ethash_search_2` | 29.044 | 2088 | 0.800 |

For HPL, the CUDA hook reported unknown code IDs, so the SASS metrics were
recovered directly from the workload binary and bundled CUDA libraries. The
top DGEMM kernel was resolved from `libcublasLt.so.11`, while the two TRSM
kernels were resolved from `libcublas.so.11`.

## Dataset Summary

SASSGuard and the behavioral baseline observe different signals. SASSGuard
uses normalized SASS windows built from CUDA-hook captures and classifies code
structure. The behavioral baseline uses only whole-device `nvidia-smi` point
measurements and raises a run-level alarm from the mean mining probability.

### SASSGuard Synthetic SASS Dataset

The synthetic SASS dataset contains 252 captured workloads, evenly split
between O2 and O3 builds. The four-class labels are:

| Label | Workloads |
|---|---:|
| `benign_compute_like` | 80 |
| `benign_crypto_hash_like` | 28 |
| `benign_memory_like` | 14 |
| `mining_like` | 130 |

The binary view groups the first three labels as `benign`, giving 122 benign
and 130 mining workloads. Under the current overflow-window L0 scheduler,
each emitted `.sass` file is clipped to one ModernBERT content window
(`8190` tokens, reserving `[CLS]` and `[SEP]`). Split mining workloads can
therefore contribute multiple training examples, while each example still maps
to exactly one ModernBERT input. The split manifest uses seed 1337 and a
70/15/15 grouped train/validation/test split:

| Split | Examples | Workload groups | Benign examples | Mining examples | Four-class example counts |
|---|---:|---:|---:|---:|---|
| train | 253 | 177 | 86 | 167 | compute 56, crypto 20, memory 10, mining 167 |
| val | 56 | 38 | 18 | 38 | compute 12, crypto 4, memory 2, mining 38 |
| test | 52 | 37 | 18 | 34 | compute 12, crypto 4, memory 2, mining 34 |

The main dataset imbalance is the small `benign_memory_like` class. It has
only 14 workloads total, so any memory-subclass result is high variance.

### SASSGuard Real-World SASS Dataset

The real-world dataset contains captured windows from benign applications and
real miners. The build scanned 77 captures, skipped 17 empty captures and one
unmapped capture, and produced 59 workload groups with 392 emitted L0 windows
plus 6 no-window/default-benign rows.

| Workload-level label | Workload groups |
|---|---:|
| `benign_compute_like` | 6 |
| `benign_crypto_hash_like` | 3 |
| `benign_memory_like` | 1 |
| `mining_like` | 49 |

The real-world split is evaluation-only: train and validation are empty, and
all rows are placed in `test.jsonl`.

| Split | Examples | Groups | Benign examples | Mining examples |
|---|---:|---:|---:|---:|
| train | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 |
| test | 398 | 59 | 66 | 332 |

### Behavioral GPU-Counter Dataset

The behavioral baseline dataset is device-level rather than code-level. It
contains 329 runs and 8,921 point samples from `nvidia-smi`.

| Binary label | Runs |
|---|---:|
| `benign` | 132 |
| `mining` | 197 |

The pointwise Random Forest uses:

- `gpu_utilization_pct`
- `memory_utilization_pct`
- `power_usage_watts`
- `temperature_celsius`

The trained point model predicts a mining probability at each sample. A
run-level alarm is raised when the mean mining probability is at least 0.8.
That threshold was tuned on the validation split with a maximum benign false
positive rate of 0.01.

| Split | Runs |
|---|---:|
| train | 231 |
| val | 48 |
| test | 50 |

### Throttle And Mixed Stress Sets

The SASSGuard throttle set contains kernel-launch throttled SASS captures for
four programs: `ethash_split`, `kawpow_split`, `randomx_gpu_lite_mono`, and
`sha256d_mono`. Each program was evaluated at 10%, 50%, 75%, and 100% launch
rate. The 16 captures produced 288 L0 windows, all labeled `mining`.

The behavioral throttle set is larger and device-counter only. It contains
48 mining runs: eight mining programs, six launch-rate percentages
(`5`, `10`, `25`, `50`, `75`, `100`), and one repeat. These runs produced
2,928 point samples.

The behavioral mixed-workload set contains seven mining-present mixed runs
where a benign GPU workload and a throttled miner run concurrently on one GPU.
It produced 1,187 point samples. A separate combined report adds six benign
control runs, giving 13 total runs for the mixed-with-controls setting.

## Normal Experiments

### SASSGuard ModernBERT On Synthetic Binary Windows

The deployable ModernBERT detector uses binary labels:

```text
benign, mining
```

The current GPU3 binary classifier was trained on the synthetic overflow-window
dataset. Because L0 owns the model budget, every emitted `.sass` file maps to
one ModernBERT input. The run used 253 training chunks, 56 validation chunks,
and 52 test chunks. The best classifier checkpoint is
`checkpoint-48`; the classifier uses class weights `[1.029, 0.973]` for
`benign` and `mining`.

Window-level validation accuracy and macro F1 are both 1.000:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 18 | 0 |
| mining | 0 | 38 |

Window-level test accuracy and macro F1 are also both 1.000:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 18 | 0 |
| mining | 0 | 34 |

For synthetic diagnostics, we also aggregate windows to workloads by predicting
a workload as mining if any emitted window is predicted mining. This produces
perfect validation, test, and combined validation+test workload-level results:

| Split | True benign predicted benign | True benign predicted mining | True mining predicted benign | True mining predicted mining | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| val | 18 | 0 | 0 | 20 | 1.000 | 1.000 |
| test | 18 | 0 | 0 | 19 | 1.000 | 1.000 |
| val+test | 36 | 0 | 0 | 39 | 1.000 | 1.000 |

The synthetic result should be read as a controlled transfer precondition, not
as the final deployment claim: the model sees generated kernels from the same
pipeline used for training. The harder deployment question is whether those
instruction patterns transfer to real captured miners and benign GPU programs.

### SASSGuard ModernBERT On Real-World Workloads

The real-world dataset is evaluation-only. It has 59 workload groups:
10 benign groups and 49 mining groups. The split files place all 398 rows in
`test.jsonl`; train and validation are empty.

Real-world reporting uses a workload policy that matches the online decision
logic. A workload is predicted mining only when both conditions hold over its
emitted windows:

```text
mean mining probability >= 0.30
max mining probability  >= 0.50
```

This two-part rule is intentionally stricter than a mean-only trigger. The
mean threshold requires sustained mining-like evidence across the workload,
while the max threshold prevents a benign workload with many steady
low-confidence windows, for example all near 0.40, from becoming a mining
alarm. No-window benign workloads remain default benign rows.

With the GPU3 binary model and this workload policy, the real-world confusion
matrix is:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 10 | 0 |
| mining | 2 | 47 |

Real-world workload-level accuracy is 0.966 and macro F1 is 0.944. Mining
precision is 1.000 and mining recall is 0.959; benign recall is 1.000.

The two missed real-world mining workloads are:

| Workload | Windows | Mean mining p | Max mining p | Explanation |
|---|---:|---:|---:|---|
| `lolminer_pyrin` | 1 | 0.027 | 0.027 | The emitted window does not contain enough mining-like SASS under the current L0 gate and model. |
| `lolminer_pyrinv2` | 1 | 0.100 | 0.100 | Same failure mode: one low-confidence emitted window, below both workload-policy thresholds. |

Six benign real-world workloads produce no L0 window under the real-world gate
and are treated as default benign rows: Blender OptiX rendering, CRC32,
cuBLAS GEMM, cuDNN convolution, HPL, and benign SHA-256. This behavior is
intentional for deployment: below-gate benign workloads should not become
model queries or alerts merely because they executed CUDA kernels.

### Behavioral Baseline On Normal GPU-Counter Runs

The behavioral baseline is strong on the normal run-level test split:

| Split | Point accuracy | Run accuracy |
|---|---:|---:|
| train | 0.994 | 1.000 |
| val | 0.941 | 1.000 |
| test | 0.902 | 0.980 |

Point-level test confusion matrix, labels `benign, mining`:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 231 | 11 |
| mining | 103 | 823 |

Run-level test confusion matrix, labels `benign, mining`:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 22 | 0 |
| mining | 1 | 27 |

The only run-level miss is:

| Workload | True label | Predicted label | Mean mining p | Max mining p | Positive samples | Explanation |
|---|---|---|---:|---:|---:|---|
| `xmrig_rx_0` | mining | benign | 0.449 | 1.000 | 35 / 121 | The run contains mining-like spikes, but only 28.9% of samples are pointwise mining. The 0.8 mean-probability alarm suppresses intermittent or low-duty mining signatures. |

The 11 benign point false positives do not become run false positives. Several
benign runs have one high-probability sample, for example `jacobi_2d_o3` has a
max mining probability of 1.0, but its mean mining probability is only 0.309.
The aggregation policy is therefore good at suppressing isolated benign
counter spikes, but it can miss miners whose mining-like samples are sparse.

## Throttle Experiments

### SASSGuard ModernBERT Under Kernel-Launch Throttling

The SASSGuard throttle experiment evaluates kernel-launch throttled miners
after L0 windowization. It uses labels:

```text
benign, mining
```

All 288 evaluated windows are mining windows. Accuracy and recall are 1.000:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 0 | 0 |
| mining | 0 | 288 |

There are no misclassified windows. The per-family breakdown is also perfect:

| Family | Windows | Correct |
|---|---:|---:|
| `ethash_dag_keccak` | 72 | 72 |
| `progpow_kawpow_random_math` | 72 | 72 |
| `cryptonight_randomx_scratchpad` | 72 | 72 |
| `pure_hash_nonce_search` | 72 | 72 |

The reason this stress test is easy for SASSGuard is that kernel-launch
throttling changes timing, not the normalized SASS body. The model still sees
the mining kernel instructions in each L0 window. The limitation is that this
throttle set has no benign rows, so it measures mining recall under throttling
but not false-positive behavior.

### Behavioral Baseline Under Kernel-Launch Throttling

The behavioral throttle experiment uses labels:

```text
benign, mining
```

All 48 runs are mining runs. Run-level accuracy is therefore mining recall:
0.479.

Run-level confusion matrix:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 0 | 0 |
| mining | 25 | 23 |

Point-level confusion matrix:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 0 | 0 |
| mining | 452 | 2476 |

Recall by launch-rate percentage:

| Launch rate | Detected / total | Recall | Mean-probability pattern |
|---:|---:|---:|---|
| 5% | 8 / 8 | 1.000 | All runs have mean mining p >= 0.944. |
| 10% | 7 / 8 | 0.875 | Most runs exceed 0.8; `sha256d_mono` is below threshold. |
| 25% | 0 / 8 | 0.000 | Mean mining p is 0.660-0.743, below the 0.8 alarm threshold. |
| 50% | 0 / 8 | 0.000 | Mean mining p is 0.475-0.773. |
| 75% | 0 / 8 | 0.000 | Mean mining p is 0.278-0.784. |
| 100% | 8 / 8 | 1.000 | All runs have mean mining p >= 0.974. |

The missed runs are:

| Program | Missed launch rates | Mean mining p for missed rates | Positive sample pattern | Explanation |
|---|---|---|---|---|
| `autolykos2_split` | 25%, 50%, 75% | 0.670, 0.553, 0.391 | 61/61, 59/61, 0/61 | At 25% and 50%, many samples are classified mining but with low confidence; at 75%, the device counters look benign to the RF. |
| `cryptonight_gpu_split` | 25%, 50%, 75% | 0.660, 0.475, 0.278 | 60/61, 10/61, 0/61 | Middle launch rates shift utilization and power into ordinary-looking bands. |
| `cuckoo_cycle_split` | 25%, 50%, 75% | 0.664, 0.531, 0.331 | 61/61, 46/61, 0/61 | The graph-cycle miner is visible at the point level in some regimes, but the mean probability stays below 0.8. |
| `equihash144_5_split` | 25%, 50%, 75% | 0.672, 0.553, 0.382 | 61/61, 60/61, 0/61 | The alarm threshold is too strict for low-confidence point predictions. |
| `ethash_split` | 25%, 50%, 75% | 0.670, 0.535, 0.317 | 61/61, 50/61, 0/61 | Throttling hides the sustained high-utilization profile expected by the device-counter model. |
| `kawpow_split` | 25%, 50%, 75% | 0.669, 0.549, 0.365 | 61/61, 60/61, 0/61 | Same failure mode as ethash/progpow-like kernels: mid-rate counter values are not distinctive enough. |
| `randomx_gpu_lite_mono` | 25%, 50%, 75% | 0.736, 0.566, 0.512 | 61/61 at all three rates | Every point is classified mining, but the average probability never reaches 0.8. This is a calibration and aggregation failure. |
| `sha256d_mono` | 10%, 25%, 50%, 75% | 0.637, 0.743, 0.773, 0.784 | 60/61, 61/61, 60/61, 60/61 | Pure hash nonce search has near-zero memory utilization in these runs. It is repeatedly just below the alarm threshold, including 0.784 at 75%. |

This is the clearest contrast between SASSGuard and the behavioral baseline.
SASSGuard is invariant to launch-rate throttling because it reads the kernel
body. The behavioral baseline relies on device-level utilization and power, so
throttling can move a miner into counter regimes that either resemble benign
workloads or produce low-confidence mining probabilities.

## Mixed-Workload Experiments

### SASSGuard ModernBERT Mixed-Workload Status

There is no ModernBERT mixed-workload report in the checked-in artifacts. The
available mixed artifacts under `experiments/results/mixed_behavioral/` are
GPU-counter behavioral outputs only, collected with an older mixed-wrapper
helper and without SASSGuard L1 evaluation. Therefore this document does not
report a SASSGuard mixed confusion matrix.

The missing SASSGuard mixed experiment is important future coverage. Because
SASSGuard windows are PID/stream based, the expected evaluation should report
whether mining kernels remain detectable when benign kernels are concurrent on
the same device, and whether benign-process windows stay benign.

### Behavioral Baseline On Mixed Mining-Present Runs

The archived mixed workload experiment labels every mixed run as mining-present
because a miner is running somewhere on the GPU. The behavioral baseline has no
PID attribution in this experiment.

Run-level confusion matrix, labels `benign, mining`:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 0 | 0 |
| mining | 5 | 2 |

Point-level confusion matrix:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 0 | 0 |
| mining | 686 | 501 |

Per-run mixed alarms:

| Mixed workload | Mean mining p | Max mining p | Positive samples | Prediction | Explanation |
|---|---:|---:|---:|---|---|
| `mixed_aes_ethash_split_50pct` | 0.092 | 0.520 | 1 / 183 | benign | The device is busy, but the counter profile is dominated by the benign AES/memory-heavy workload. The miner does not produce enough distinctive device-level evidence. |
| `mixed_cublas_gemm_randomx_gpu_lite_mono_10pct` | 0.524 | 1.000 | 51 / 73 | benign | Many samples look mining-like, but the mean stays below 0.8 because the benign GEMM workload dominates utilization. |
| `mixed_cublas_gemm_randomx_gpu_lite_mono_10pct` | 0.540 | 0.985 | 102 / 193 | benign | Repeat run with the same failure mode: intermittent mining evidence is diluted by sustained benign compute. |
| `mixed_cudnn_convolution_kawpow_split_10pct` | 0.124 | 0.945 | 5 / 189 | benign | cuDNN convolution keeps the device near full utilization and high power, masking the low-rate kawpow miner. |
| `mixed_hpl_sha256d_mono_10pct` | 0.152 | 0.900 | 2 / 182 | benign | HPL dominates the counter stream. The sha256d miner has little memory-utilization signal and appears only as rare spikes. |
| `mixed_pytorch_training_ethash_split_10pct` | 0.812 | 1.000 | 159 / 184 | mining | The ethash component contributes enough sustained mining-like evidence to barely exceed the 0.8 threshold. |
| `mixed_vllm_inference_fallback_randomx_gpu_lite_mono_50pct` | 0.918 | 0.992 | 181 / 183 | mining | The 50% miner is strong enough that most samples remain mining-like despite the concurrent benign workload. |

The mixed result is a major failure mode for device-only detection: run recall
is 2/7. The baseline can detect mixed runs only when the miner's contribution
dominates the device-level counters for most samples.

### Mixed Workload With Benign Controls

The combined mixed-with-controls report adds six benign control runs to the
seven mining-present mixed runs. Its labels are:

```text
benign, mining
```

Run-level confusion matrix:

| True \ Predicted | benign | mining |
|---|---:|---:|
| benign | 6 | 0 |
| mining | 5 | 2 |

Accuracy is 0.615 and balanced accuracy is 0.643. The result is asymmetric:
the behavioral baseline raises no false alarms on the benign controls, but it
misses most mining-present mixed workloads.

## Cross-Experiment Takeaways

SASSGuard ModernBERT is strongest when the question is whether a captured SASS
window contains mining-like code. The current GPU3 binary model is perfect on
the synthetic validation and test windows and also detects all mining windows
in the kernel-launch throttle stress set. On real-world captures, the
workload-level mean+max policy reaches 47/49 mining recall with no benign
false positives over 59 workload groups.

The behavioral baseline is strong on ordinary isolated runs but brittle under
adversarial timing and concurrency. Its 0.8 mean-probability alarm suppresses
isolated benign spikes, which is good for false-positive control, but the same
aggregation causes misses for throttled miners and mixed workloads where mining
evidence is intermittent or masked by benign work.

The remaining evaluation gap is a full SASSGuard mixed-workload experiment.
The current repository supports behavioral mixed analysis, but it does not yet
contain a ModernBERT mixed report with PID/stream-attributed SASS windows.
The two current real-world mining misses are both low-confidence one-window
lolMiner Pyrin-family captures, so the next data task is to inspect whether L0
is selecting the right steady-state kernels for those algorithms.


# Capture Experiment

## API Overhead

vector add program with CUDA Driver API over 100 runs
cuDeviceGetCount is for tranpoline overhead.

With capture (with asynchronous buffering):
cuDeviceGetCount median: 2706.0 ns (2.706 us)
cuModuleLoad median: 202484.0 ns (202.484 us)
cuLaunchKernel median: 35081.0 ns (31.081 us)

With capture (without asynchronous buffering, cuPCAP):
cuDeviceGetCount overhead 3.97us
cuModuleLoad overhead 269.02us
cuLaunchKernel overhead 15.99us

Without capture:
cuDeviceGetCount median: 2574.5 ns (2.575 us)
cuModuleLoad median: 133819.0 ns (133.819 us)
cuLaunchKernel median: 25301.5 ns (25.302 us)

## Application Overhead
- Without capture
  - HPL: 3608 GFLOPS
  - T-Rex: 90.75 MH/s
  - vLLM: 1827.68 tokens/s
- With capture
  - HPL: 3721 GFLOPS
  - T-Rex: 90.75 MH/s
  - vLLM: 1827.91 tokens/s
