#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/mesh8_2000_eval_and_inference.log 2>&1

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
CURRENT_SHA=$(git rev-parse HEAD)

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Mesh-8 Epoch 2000 evaluation and inference pipeline on GPU ${CUDA_VISIBLE_DEVICES} (commit ${CURRENT_SHA})..."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === STEP 1: PHYSICAL METRICS EVALUATION ==="
.venv/bin/python -u scripts/evaluate_mesh8_2000.py \
  --dataset /home/user/global-ml-weather-joint-20261012/outputs/joint-20261012/dataset \
  --training outputs/mesh8-500-run/training \
  --output outputs/mesh8-500-run/evaluation_metrics_epoch2000.json \
  --baseline-epoch500 outputs/mesh8-500-run/evaluation_metrics.json \
  --device cuda:0

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Evaluation completed successfully."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === STEP 2: RESEARCH FORECAST BATCH INFERENCE ==="
rm -rf outputs/mesh8-2000-forecast
.venv/bin/python -u -m global_weather.joint_observation_inference \
  --dataset /home/user/global-ml-weather-joint-20261012/outputs/joint-20261012/dataset \
  --training outputs/mesh8-500-run/training \
  --static /home/user/global-ml-weather-r10-c4f1476-20261014/outputs/r10-fixed-20261014T053240Z/static-mesh8 \
  --output outputs/mesh8-2000-forecast \
  --split test \
  --limit 2 \
  --horizon-hours 3 \
  --device cuda:0 \
  --warmups 1 \
  --repeats 3 \
  --threads 8 \
  --spacing-degrees 5 \
  --source-revision "${CURRENT_SHA}"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === ALL STEPS COMPLETED CLEANLY ==="
