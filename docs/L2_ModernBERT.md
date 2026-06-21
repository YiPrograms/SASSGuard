# L2 ModernBERT Training

This stage trains a compact scratch ModernBERT classifier over normalized SASS
workloads. It does not rewrite the dataset: every command reads the existing
split files in `dataset/splits/` and follows each row's `path` to load
`workload.sass`.

## Target

The primary classifier is a binary model using `binary_label` from each split
row:

- `benign`
- `mining`

The previous 4-class target is still available through
`configs/training/modernbert_sass_compact.json`, which uses `label`:
`benign_compute_like`, `benign_crypto_hash_like`, `benign_memory_like`, and
`mining_like`.

Long workloads are handled by chunking tokenized SASS into 8,192-token model
windows with overlap, then aggregating chunk probabilities back to one
workload-level prediction.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-modernbert.txt
```

The training host's Tesla V100 GPUs are Volta (`sm_70`). Use the pinned
CUDA 12.6 PyTorch wheel in `requirements-modernbert.txt`; the default PyPI
CUDA 13.x wheel does not include V100 kernels.

If an earlier install pulled an incompatible CUDA 13.x PyTorch wheel, reinstall
Torch in the existing venv:

```bash
python -m pip install --upgrade --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.12.0+cu126'
```

If an earlier install pulled `transformers 5.x`, downgrade the existing venv:

```bash
python -m pip install --upgrade --force-reinstall \
  'transformers>=4.48.0,<5.0.0' \
  'tokenizers>=0.21.0,<0.22.0' \
  'fsspec<=2026.2.0'
```

The binary training config is:

```text
configs/training/modernbert_sass_compact_binary.json
```

The matching real-world binary evaluation config is:

```text
configs/training/modernbert_sass_compact_binary_realworld.json
```

## Train

Train the tokenizer from train-split SASS only:

```bash
python -m train.modernbert.train_tokenizer \
  --config configs/training/modernbert_sass_compact_binary.json
```

Run MLM warm-up from scratch:

```bash
torchrun --nproc_per_node=8 -m train.modernbert.pretrain_mlm \
  --config configs/training/modernbert_sass_compact_binary.json
```

Train the binary classifier:

```bash
torchrun --nproc_per_node=8 -m train.modernbert.train_classifier \
  --config configs/training/modernbert_sass_compact_binary.json
```

Evaluate a saved checkpoint:

```bash
python -m train.modernbert.evaluate \
  --config configs/training/modernbert_sass_compact_binary.json \
  --split test
```

The evaluation script disables ModernBERT reference compilation when loading
the saved checkpoint. This avoids a `torch.compile`/DataParallel interaction on
multi-GPU hosts that can fail with `FX to symbolically trace a
dynamo-optimized function`.

For distributed evaluation, launch it explicitly:

```bash
torchrun --nproc_per_node=8 -m train.modernbert.evaluate \
  --config configs/training/modernbert_sass_compact_binary.json \
  --split test
```

Evaluate the same binary checkpoint on the real-world dataset:

```bash
python -m train.modernbert.evaluate \
  --config configs/training/modernbert_sass_compact_binary_realworld.json \
  --split test
```

## Outputs

- Tokenizer: `models/modernbert/tokenizer/sass-wordlevel-v1`
- MLM checkpoint: `models/modernbert/checkpoints/sass-modernbert-compact-binary/mlm/final`
- Classifier checkpoint: `models/modernbert/checkpoints/sass-modernbert-compact-binary/classifier/final`
- Reports: `experiments/results/reports/modernbert_binary`
- Real-world evaluation reports:
  `experiments/results/reports/modernbert_binary_realworld`

Reports include workload-level accuracy, macro/micro/weighted F1, per-class
precision/recall/F1, confusion matrix, per-workload predictions, and per-family
breakdowns.

Evaluation keeps the primary `pred_label` based on mean probability across all
chunks for comparability. Reports also include a high-recall mining detector
under `suspicious_detection` and per-prediction suspicious fields:

- `mining_probability_mean`
- `mining_probability_max`
- `mining_probability_top3_mean`
- `suspicious`
- `suspicious_label`
- `suspicious_reason`

The default suspicious rule is:

```python
if max_p_mining >= 0.90:
    suspicious = True
elif top3_mean_p_mining >= 0.75:
    suspicious = True
else:
    suspicious = False
```

This catches workloads where only a few chunks contain strongly mining-like
SASS while setup/helper chunks dilute the mean probability.

## Tests

Core tests avoid importing the heavy ML stack:

```bash
python3 -m unittest discover -s train/modernbert/tests
```
