#!/usr/bin/env bash
# Фоновый запуск этапа S3: Всепогодная СВЧ-ассимиляция (МТВЗА-ГЯ 29 каналов, C_sym) и кондиционирование ледового покрова Арктики (БРЛК-Северянин)
# Строго на GPU 1 (NVIDIA GeForce RTX 4090). GPU 0 не затрагивается.
set -euo pipefail
trap '' HUP

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p outputs outputs/s3_allsky_microwave_and_arctic_ice/training
LOG="outputs/s3_allsky_microwave_and_arctic_ice.log"
PID_FILE="outputs/s3_allsky_microwave_and_arctic_ice.pid"

exec >> "$LOG" 2>&1

echo "$$" > "$PID_FILE"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === Запуск Stage S3: All-sky microwave & Arctic ice (PID: $$) ==="

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Изолированное GPU устройство: ${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" || true
fi

DATASET_DIR="data/extended_sample_2023_2026/surface_dataset"
CLIMATOLOGY="outputs/extended_2023_2026/training/station_climatology.npz"
BASE_CHECKPOINT="outputs/s2_joint_clearsky_and_amv/training/epoch-0001/state.pt"
CONFIG="configs/s3_allsky_microwave_and_arctic_ice.json"

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Контрольная точка этапа S2 ${BASE_CHECKPOINT} не найдена!"
    exit 1
fi

if [ ! -f "${CLIMATOLOGY}" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ОШИБКА: Файл климатологии ${CLIMATOLOGY} не найден!"
    exit 1
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Запуск обучения Stage S3 (satellite_allsky_stage3_training)..."
exec .venv/bin/python -u -m global_weather.satellite_allsky_stage3_training \
    --dataset "${DATASET_DIR}" \
    --output outputs/s3_allsky_microwave_and_arctic_ice/training \
    --config "${CONFIG}" \
    --climatology "${CLIMATOLOGY}"
