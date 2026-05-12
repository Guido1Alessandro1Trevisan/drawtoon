#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-datasets/_training/training_multifranchise_jpg_non_mixed}"
MODEL_ID="${MODEL_ID:-multifranchise-klein9b-fullft-jpg-non-mixed-v1}"
EPOCHS="${EPOCHS:-3}"
CONFIG_PATH="/root/training/configs/mngrm12_klein9b_multifranchise_full_ft_768x1024_h200_v1.yaml"

uv run --with modal --with pyyaml python -m modal run --detach lora-klein/training/run_modal.py \
  --config-path "$CONFIG_PATH" \
  --dataset-subpath "$DATASETS" \
  --model-id "$MODEL_ID" \
  --target-epochs "$EPOCHS"
