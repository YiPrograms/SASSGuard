#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python3 analysis/build_dataset.py \
  --capture-root . \
  --output-dir dataset \
  --max-launches 16 \
  "$@"
