#!/usr/bin/env bash
# Фоновый запуск обучения спутниковых модулей ассимиляции (Этап S1: Radiometric Warmup)
# Строго на GPU 1 (NVIDIA GeForce RTX 4090). GPU 0 не затрагивается.
set -euo pipefail
trap '' HUP

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p outputs outputs/s1_satellite_radiometric_warmup/training
LOG="outputs/s1_satellite_radiometric_warmup.log"
PID_FILE="outputs/s1_satellite_radiometric_warmup.pid"

exec >> "$LOG" 2>&1

echo "$$" > "$PID_FILE"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === Запуск Stage S1: Спутниковая радиометрическая калибровка и ассимиляция (PID: $$) ==="

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Изолированное GPU устройство: ${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" || true
fi

DATASET_DIR="data/extended_sample_2023_2026/surface_dataset"
CLIMATOLOGY="outputs/extended_2023_2026/training/station_climatology.npz"
BASE_CHECKPOINT="outputs/extended_2023_2026/training/epoch-0081/state.pt"
CONFIG="configs/s1_satellite_radiometric_warmup.json"

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Базовая контрольная точка ${BASE_CHECKPOINT} не найдена!"
    exit 1
fi

if [ ! -f "${CLIMATOLOGY}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Файл климатологии ${CLIMATOLOGY} не найден!"
    exit 1
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Запуск обучения Stage S1 (satellite_radiometric_training)..."
exec .venv/bin/python -u -m global_weather.satellite_radiometric_training \
    --dataset "${DATASET_DIR}" \
    --output outputs/s1_satellite_radiometric_warmup/training \
    --config "${CONFIG}" \
    --climatology "${CLIMATOLOGY}"
