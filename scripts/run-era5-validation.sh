#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/era5_validation_experiment.log 2>&1

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting ERA5 background validation experiment..."
echo "$$" > outputs/era5_validation_experiment.pid

exec .venv/bin/python -u scripts/run_era5_validation_experiment.py \
  --era5-dataset /home/user/graphcast_project/dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc \
  --forecast-nc /home/user/graphcast_project/forecast_real_latest.nc \
  --output outputs/era5_validation_experiment
