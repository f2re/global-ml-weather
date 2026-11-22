#!/usr/bin/env bash
# Фоновый станционный опыт GHCNh 2023–2026 на вычислительном узле.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo"
unit=global-weather-station-2023-2026
data_root=data/extended_sample_2023_2026
run_root=outputs/extended_2023_2026/detached-station
training="$run_root/training"
config=configs/station_anomaly_extended_2023_2026.json
admission=assets/admission_851_stations.json
python="$repo/.venv/bin/python"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

assert_pinned() {
    local status
    [[ "$(git rev-parse HEAD)" == "$sha" ]] || fail "HEAD отличается от указанного SHA."
    status="$(GIT_OPTIONAL_LOCKS=0 git status --porcelain --untracked-files=normal)" ||
        fail "Не удалось проверить состояние Git."
    [[ -z "$status" ]] || fail "Для запуска нужна чистая рабочая копия."
}

mode=launch
if [[ "${1:-}" == --worker ]]; then
    mode=worker
    shift
fi
sha="${1:-}"
if [[ ! "$sha" =~ ^[0-9a-f]{40}$ || $# -gt 2 || ( $# -eq 2 && "$2" != --allow-network ) ]]; then
    fail "Использование: bash scripts/run_extended_station_campaign_detached.sh SHA [--allow-network]"
fi
assert_pinned
[[ -x "$python" ]] || fail "Не найдено отдельное окружение $python."

if [[ -e "$training" || -L "$training" ]]; then
    fail "Каталог $training уже существует. Сохраните его для проверки контрольных точек."
fi

if [[ "$mode" == launch ]]; then
    [[ "$(loginctl show-user "$(id -u)" --property=Linger --value)" == yes ]] ||
        fail "Менеджер systemd пользователя должен работать после выхода из SSH (Linger=yes)."
    systemd-run --user --collect --unit="$unit" --property=Type=exec --property=UMask=0077 \
        --working-directory="$repo" --setenv="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}" \
        /bin/bash "$repo/scripts/run_extended_station_campaign_detached.sh" --worker "$@"
    printf 'Журнал: journalctl --user -u %s\n' "$unit"
    exit 0
fi

umask 077
mkdir -p "$run_root"
exec 9>>"$run_root/campaign.lock"
flock --nonblock 9 || fail "У станционного опыта уже есть исполнитель."
[[ ! -e "$training" && ! -L "$training" ]] || fail "Каталог обучения уже занят."
printf '%s SHA=%s cwd=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$sha" "$repo" "${CUDA_VISIBLE_DEVICES:-auto}"

# Готовая выборка проходит повторную проверку. Новая загрузка использует
# отдельный кэш, чтобы старый исполнитель не менял файлы при подготовке.
dataset="$data_root/surface_dataset"
cache="$data_root/ghcnh_cache"
if [[ ! -e "$dataset" && ! -L "$dataset" ]]; then
    dataset="$data_root/detached-station/surface_dataset"
    cache="$data_root/detached-station/ghcnh_cache"
    if [[ ! -e "$dataset" && ! -L "$dataset" ]]; then
        [[ "${2:-}" == --allow-network ]] ||
            fail "Готовая выборка отсутствует. Для получения GHCNh передайте --allow-network."
        "$python" -u -m global_weather.observation_acquisition \
            --admission "$admission" --cache "$cache" --output "$run_root/ghcnh_acquisition.json" \
            --allow-network --years 2023 2024 2025 2026
    fi
fi

assert_pinned
"$python" -u -m global_weather.observation_data prepare \
    --source-cache "$cache" --admission "$admission" --output "$dataset" --campaign extended_2023_2026
"$python" -u - "$dataset" "$config" <<'PY'
import json
from pathlib import Path
import sys

from global_weather.observation_data import ObservationDataset

dataset = ObservationDataset(sys.argv[1])
if dataset.campaign != "extended_2023_2026":
    raise SystemExit("Период набора не совпадает с кампанией 2023–2026.")
config = json.loads(Path(sys.argv[2]).read_text())
for split, limit in (
    ("train", config["max_train_samples"]),
    ("validation", config["max_val_samples"]),
    ("test", None),
):
    usable = 0
    for index in dataset.subset(split)[:limit]:
        hour = dataset.samples[index]["hour"]
        input_hours = slice(hour - 11, hour + 1)
        causal_inputs = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
        measured_targets = dataset.mask[hour + 3:hour + 73:3]
        usable += bool(causal_inputs.any() and measured_targets.any())
    print(f"{split}: окон с причинным входом и измеренной целью {usable}", flush=True)
    if not usable:
        raise SystemExit(f"{split}: нет пригодных окон наблюдений.")
PY

assert_pinned
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$python" -u -m global_weather.station_anomaly_training \
    --dataset "$dataset" --output "$training" --config "$config"
