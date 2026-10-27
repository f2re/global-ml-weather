#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/mesh8-500-training.log 2>&1

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Continuing Mesh-8 training to 4000 epochs on GPU ${CUDA_VISIBLE_DEVICES}..."
echo "$$" > outputs/mesh8-4000-training.pid
exec .venv/bin/python -u -m global_weather.joint_observation_training \
  --dataset /home/user/global-ml-weather-joint-20261012/outputs/joint-20261012/dataset \
  --norm /home/user/global-ml-weather-r7/outputs/remote-r7-stations/graphcast-norms.json \
  --climatology /home/user/global-ml-weather-r10-c4f1476-20261014/outputs/r10-fixed-20261014T053240Z/climatology-mesh4 \
  --static /home/user/global-ml-weather-r10-c4f1476-20261014/outputs/r10-fixed-20261014T053240Z/static-mesh8 \
  --config configs/joint_observation_training_mesh8_4000.json \
  --output outputs/mesh8-500-run/training
