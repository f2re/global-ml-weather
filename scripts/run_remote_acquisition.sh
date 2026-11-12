#!/usr/bin/env bash
# Фоновый скрипт загрузки и формирования расширенной выборки 2023-2026 на удаленном узле.
# Запускается независимо от обучения GPU и не конкурирует за видеопамять.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

LOG="outputs/remote_acquisition_runner.log"
mkdir -p outputs data/extended_sample_2023_2026

ADMISSION="${ADMISSION_PATH:-assets/admission_851_stations.json}"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === Старт удаленного сбора данных 2023-2026 ===" >> "$LOG"

exec .venv/bin/python -u scripts/run_dataset_pipeline_background.py \
    --output-dir data/extended_sample_2023_2026 \
    --log-file outputs/dataset_pipeline_2023_2026.log \
    --stride-hours 6 \
    --acquire-satellites \
    --max-satellite-scenes 100 \
    --acquire-ghcnh \
    --admission "$ADMISSION" \
    --ghcnh-years 2023 2024 2025 2026 \
    >> "$LOG" 2>&1
