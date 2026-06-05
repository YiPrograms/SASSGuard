#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python3 analysis/build_dataset.py \
  --captures-dir workloads/synthetic_kernels/captures \
  --binary-manifest workloads/synthetic_kernels/binaries/manifest.jsonl \
  --output-dir dataset \
  --max-launches 16 \
  "$@"
