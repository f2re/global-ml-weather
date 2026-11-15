#!/usr/bin/env bash
# Фоновый запуск совместного обучения ядра прогноза погоды и спутниковых операторов (Этап S2: Joint Clearsky & AMV)
# Строго на GPU 1 (NVIDIA GeForce RTX 4090). GPU 0 не затрагивается.
set -euo pipefail
trap '' HUP

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p outputs outputs/s2_joint_clearsky_and_amv/training
LOG="outputs/s2_joint_clearsky_and_amv.log"
PID_FILE="outputs/s2_joint_clearsky_and_amv.pid"

exec >> "$LOG" 2>&1

echo "$$" > "$PID_FILE"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === Запуск Stage S2: Совместное обучение (Joint Clearsky & AMV) (PID: $$) ==="

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
SATELLITE_CHECKPOINT="outputs/s1_satellite_radiometric_warmup/training/epoch-0007/state.pt"
CONFIG="configs/s2_joint_clearsky_and_amv.json"

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Базовая контрольная точка ${BASE_CHECKPOINT} не найдена!"
    exit 1
fi

if [ ! -f "${SATELLITE_CHECKPOINT}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Спутниковая контрольная точка этапа S1 ${SATELLITE_CHECKPOINT} не найдена!"
    exit 1
fi

if [ ! -f "${CLIMATOLOGY}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Файл климатологии ${CLIMATOLOGY} не найден!"
    exit 1
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Запуск обучения Stage S2 (satellite_joint_stage2_training)..."
exec .venv/bin/python -u -m global_weather.satellite_joint_stage2_training \
    --dataset "${DATASET_DIR}" \
    --output outputs/s2_joint_clearsky_and_amv/training \
    --config "${CONFIG}" \
    --climatology "${CLIMATOLOGY}"
