#!/usr/bin/env bash
# Автономный опыт с допущенными ИК-кадрами и фиксированным станционным ядром.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

mode=launch
if [[ "${1:-}" == --worker ]]; then
    mode=worker
    shift
fi
[[ $# -eq 9 ]] || fail "Использование: $0 SHA PYTHON DATASET BASE_TRAINING RAW_MANIFEST MANIFEST_SHA DATASET_SHA BASE_SHA CLIMATOLOGY_SHA"
sha=$1
python=$2
dataset=$3
base_training=$4
raw_manifest=$5
manifest_sha=$6
dataset_sha=$7
base_sha=$8
climatology_sha=$9

[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail "Нужен полный SHA ревизии Git."
for digest in "$manifest_sha" "$dataset_sha" "$base_sha" "$climatology_sha"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "Нужны SHA256 паспорта, выборки, весов и климатологии."
done
for path in "$python" "$dataset" "$base_training" "$raw_manifest"; do
    [[ "$path" == /* ]] || fail "Укажите абсолютные пути входов и окружения."
done
[[ -x "$python" && -f "$python" ]] || fail "Не найден Python отдельного окружения."
[[ -f "$raw_manifest" && ! -L "$raw_manifest" ]] || fail "Отсутствует подготовленный спутниковый манифест."
[[ -d "$dataset" && -d "$base_training" ]] || fail "Отсутствует станционная выборка или базовое обучение."
run_root="$(cd "$(dirname "$raw_manifest")" && pwd -P)"
dataset="$(cd "$dataset" && pwd -P)"
base_training="$(cd "$base_training" && pwd -P)"
raw_manifest="$run_root/$(basename "$raw_manifest")"
[[ "$run_root" =~ ^/[a-zA-Z0-9._/-]+$ && -O "$run_root" ]] ||
    fail "Подготовьте паспорт в отдельном каталоге опыта, принадлежащем исполнителю."
for path in "$repo" "$dataset" "$base_training"; do
    [[ "$path" != "$run_root" && "$path" != "$run_root/"* && "$run_root" != "$path/"* ]] ||
        fail "Каталог опыта должен быть отдельным от исходников и станционных артефактов."
done
[[ "$(id -u)" != 0 ]] || fail "Запускайте от непривилегированного владельца данных."

config="$repo/configs/station_satellite_2023_2026_pilot.json"
frozen="$run_root/frozen.json"
training="$run_root/training"
test_report="$run_root/test.json"
log="$run_root/station-pilot.log"
reservation="$run_root/.station-pilot"
root_token="$(printf '%s' "$run_root" | sha256sum)"
unit="global-weather-station-ir-${root_token:0:16}"

assert_pinned() {
    local status
    [[ "$(git rev-parse HEAD)" == "$sha" ]] || fail "HEAD отличается от указанного SHA."
    status="$(GIT_OPTIONAL_LOCKS=0 git status --porcelain --untracked-files=normal)" ||
        fail "Не удалось проверить состояние Git."
    [[ -z "$status" ]] || fail "Для запуска нужна чистая рабочая копия."
}

check_hash() {
    local result
    result="$(sha256sum < "$1")" || fail "Не удалось прочитать $1."
    [[ "${result%% *}" == "$2" ]] || fail "Не совпадает SHA256 файла $1."
}

assert_pinned
check_hash "$raw_manifest" "$manifest_sha"
check_hash "$dataset/dataset.json" "$dataset_sha"
check_hash "$base_training/station_climatology.npz" "$climatology_sha"
for path in "$frozen" "$run_root/frozen.sensors.json" "$training" "$test_report" "$log" "$reservation"; do
    [[ ! -e "$path" && ! -L "$path" ]] || fail "Артефакт $path уже существует. Нужен новый каталог опыта."
done

if [[ "$mode" == launch ]]; then
    properties=(--property=Type=exec --property=UMask=0077 --property=PrivateNetwork=yes
        --property=NoNewPrivileges=yes --property=ProtectSystem=strict --property=ProtectHome=read-only
        "--property=ReadWritePaths=$run_root" --property=MemoryMax=24G
        --property=RuntimeMaxSec=2h --property=RemainAfterExit=yes)
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        properties+=("--setenv=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    fi
    sudo -n systemd-run --unit="$unit" --uid="$(id -u)" --gid="$(id -g)" \
        --working-directory="$repo" "${properties[@]}" \
        /bin/bash "$repo/scripts/run_station_satellite_detached.sh" --worker "$@"
    printf 'Служба: %s.service\nЖурнал: %s\n' "$unit" "$log"
    exit 0
fi

[[ "${INVOCATION_ID:-}" =~ ^[0-9a-f]{32}$ ]] || fail "Рабочий режим запускается через systemd."
umask 077
mkdir -- "$reservation" || fail "У этого каталога уже есть исполнитель."
set -o noclobber
exec >"$log" 2>&1
set +o noclobber
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$reservation" XDG_CACHE_HOME="$reservation"

stamp() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

run_stage() {
    stage=$1
    shift
    printf '%s %s' "$(stamp)" "$stage"
    printf ' %q' "$@"
    printf '\n'
    "$@"
    printf '%s stage=%s exit=0\n' "$(stamp)" "$stage"
}

stage=admission
trap 'status=$?; printf "%s stage=%s exit=%s\n" "$(stamp)" "$stage" "$status"' EXIT
printf '%s SHA=%s cwd=%s unit=%s input_sha256=%s dataset_sha256=%s base_sha256=%s climatology_sha256=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$(stamp)" "$sha" "$repo" "$unit" "$manifest_sha" "$dataset_sha" "$base_sha" "$climatology_sha" "${CUDA_VISIBLE_DEVICES-auto}"

run_stage admission "$python" -u - "$dataset" "$base_training" "$raw_manifest" "$base_sha" <<'PY'
from pathlib import Path
import sys

from global_weather.observation_training import checkpoint_path
from global_weather.pipeline.io import read_json
from global_weather.station_satellite_data import StationSatelliteData

if sys.prefix == sys.base_prefix:
    raise SystemExit("Для запуска требуется отдельное окружение Python.")
dataset, base, manifest = map(Path, sys.argv[1:4])
best = read_json(base / "best.json")
complete = read_json(base / "complete.json")
if (best["sha256"] != sys.argv[4] or best["epoch"] != complete["best_epoch"]
        or complete["status"] != "surface_climatology_anomaly_research_trained"):
    raise SystemExit("Базовый станционный прогон не совпадает с паспортом.")
checkpoint_path(base, best)
data = StationSatelliteData(manifest, dataset, _fit=True)
if set(sensor.source for sensor in data.sensors) != {"arktika_m", "electro_l"}:
    raise SystemExit("Выборка должна включать «Арктику-М» и «Электро-Л».")
for split in ("train", "validation", "test"):
    count = data.covered(split)
    print(f"{split}: {count} допущенных сроков по метаданным", flush=True)
    if not count:
        raise SystemExit(f"{split}: нет допустимых спутниковых сроков.")
PY

check_hash "$raw_manifest" "$manifest_sha"
run_stage fit "$python" -u -m global_weather.station_satellite_data fit \
    --dataset "$dataset" --manifest "$raw_manifest" --output "$frozen"
check_hash "$raw_manifest" "$manifest_sha"
assert_pinned
frozen_sha="$(sha256sum < "$frozen")"
frozen_sha="${frozen_sha%% *}"
run_stage train "$python" -u -m global_weather.station_anomaly_training \
    --dataset "$dataset" --output "$training" --config "$config" \
    --satellite-manifest "$frozen" --base-training "$base_training" \
    --expected-base-sha "$base_sha" --expected-climatology-sha "$climatology_sha" \
    --expected-dataset-sha "$dataset_sha"

assert_pinned
check_hash "$frozen" "$frozen_sha"
stage=checkpoint
best_sha="$("$python" - "$training/best.json" <<'PY'
import sys
from global_weather.pipeline.io import read_json

print(read_json(sys.argv[1])["sha256"])
PY
)"
[[ "$best_sha" =~ ^[0-9a-f]{64}$ ]] || fail "Не удалось выбрать лучшую контрольную точку."
run_stage test "$python" -u -m global_weather.station_anomaly_evaluation \
    --dataset "$dataset" --training "$training" --output "$test_report" \
    --satellite-manifest "$frozen" --expected-checkpoint-sha "$best_sha" \
    --expected-climatology-sha "$climatology_sha" --device auto
stage=evidence
check_hash "$frozen" "$frozen_sha"
sha256sum -- "$frozen" "$training/best.json" "$training/complete.json" "$test_report"
