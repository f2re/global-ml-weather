#!/usr/bin/env bash
# Фоновый запуск подготовки датасета 2023-2026 и обучения модели на GPU 1 (RTX 4090).
# Вариант А: Обучение с нуля на расширенной выборке 18 ГБ.
set -euo pipefail
trap '' HUP

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p outputs outputs/mesh8-extended-2023-2026/training
LOG="outputs/extended_training_2023_2026.log"
PID_FILE="outputs/extended_training_2023_2026.pid"

exec >> "$LOG" 2>&1

echo "$$" > "$PID_FILE"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === Запуск Option A: Обучение на расширенной выборке 2023–2026 (PID: $$) ==="

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] GPU устройство: ${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" || true
fi

DATASET_DIR="data/extended_sample_2023_2026/surface_dataset"
if [ ! -f "${DATASET_DIR}/dataset.json" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ЭТАП 1: Формирование surface_dataset из ghcnh_cache..."
    .venv/bin/python -u -m global_weather.observation_data prepare \
        --source-cache data/extended_sample_2023_2026/ghcnh_cache \
        --admission assets/admission_851_stations.json \
        --output "${DATASET_DIR}" \
        --campaign extended_2023_2026
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ЭТАП 1 завершен: ${DATASET_DIR} готов."
else
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Датасет ${DATASET_DIR} уже сформирован."
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ЭТАП 2: Запуск обучения модели station_anomaly_training..."
exec .venv/bin/python -u -m global_weather.station_anomaly_training \
    --dataset "${DATASET_DIR}" \
    --output outputs/mesh8-extended-2023-2026/training \
    --config configs/joint_observation_training_2023_2026.json
