#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${1:-configs/model_350m.yaml}
TRAIN_CONFIG=${2:-configs/train_base.yaml}
MAX_SAMPLES=${3:-}

if [[ -n "$MAX_SAMPLES" ]]; then
  python -m xenutron.train --model-config "$MODEL_CONFIG" --train-config "$TRAIN_CONFIG" --max-samples "$MAX_SAMPLES"
else
  python -m xenutron.train --model-config "$MODEL_CONFIG" --train-config "$TRAIN_CONFIG"
fi
