#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/mesh8-8000-training.log 2>&1

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_PATH="${DATASET_PATH:-/home/user/global-ml-weather-joint-20261012/outputs/joint-20261012/dataset}"
NORM_PATH="${NORM_PATH:-docs/reports/r8/graphcast-norms.json}"
CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:-/home/user/global-ml-weather-r10-c4f1476-20261014/outputs/r10-fixed-20261014T053240Z/climatology-mesh4}"
STATIC_PATH="${STATIC_PATH:-/home/user/global-ml-weather-r10-c4f1476-20261014/outputs/r10-fixed-20261014T053240Z/static-mesh8}"
CONFIG_PATH="${CONFIG_PATH:-configs/joint_observation_training_mesh8_8000.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/mesh8-500-run/training}"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Mesh-8 72-hour rollout training (up to epoch 8000) on GPU ${CUDA_VISIBLE_DEVICES}..."
echo "$$" > outputs/mesh8-8000-training.pid
exec .venv/bin/python -u -m global_weather.joint_observation_training \
  --dataset "$DATASET_PATH" \
  --norm "$NORM_PATH" \
  --climatology "$CLIMATOLOGY_PATH" \
  --static "$STATIC_PATH" \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_PATH"
