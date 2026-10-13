# R10 mesh8: финальный запуск Q4→Q5

Этот runbook относится только к `joint-mesh8-qualification-r10`. Он проверяет техническую работоспособность physical-v2 на 655362 ячейках и сроке +3 ч. Он не выполняет 72-часовую научную приёмку.

Каноническая identity mesh8 после повторной Q0-квалификации:

- fingerprint: `64c3e32f792a42ad7d363635e5217cd3c845743a9b92487fdc226da5f1937aa9`;
- schema: `mesh8-geometry-q1e-7-v1`;
- координаты для identity квантуются до `1e-7` на единичной сфере, примерно 0,64 м на Земле;
- прежний fingerprint `992e223c…` снят с эксплуатации: он зависел от суб-ULP различий NumPy/runtime, хотя топология и физическая геометрия сетки не менялись.

Артефакты mesh8, созданные с прежним fingerprint, для R10 нужно перестроить или мигрировать отдельной проверяемой процедурой. Не подменяйте fingerprint вручную в manifest.

## 1. Требования к узлу

Нужен Linux-узел с чистым checkout `main`, Python-окружением проекта, CUDA-сборкой PyTorch, одной доступной GPU с фактической памятью не меньше заявленного бюджета 22528 MiB и минимум 64 GiB свободного места в файловой системе каталога опыта.

В R10 не включается спутниковая ветвь. Источники обучения — подготовленный joint GHCNh+IGRA archive, его фиксированные GraphCast profile norms, train-only station climatology, внешний месячный NCEP1 context и независимый DEM/land artifact.

Не используйте симлинки для dataset, norms, climate, static или run root. Preflight завершится ошибкой.

## 2. Зафиксировать ревизию

```bash
git switch main
git pull --ff-only
git status --porcelain
git rev-parse HEAD
```

`git status --porcelain` перед запуском должен быть пустым.

```bash
export R10_SHA="$(git rev-parse HEAD)"
```

После этого не меняйте tracked-файлы до завершения опыта. Preflight и runner сравнивают фактический HEAD с `R10_SHA`.

## 3. Указать неизменяемые входы

Пример структуры:

```bash
export JOINT_DATA=/data/global-weather/joint-observations
export PROFILE_NORM=/data/global-weather/profile-normalization.json
export NCEP1_LTM=/data/global-weather/NCEP1-LTM-1991-2020
export CLIMATE_MESH4=/data/global-weather/seasonal-climatology-mesh4-r10
export DEM=/data/global-weather/earth_relief_30s_p.nc
export DEM_MANIFEST=assets/dem/earth_relief_30s_p.json
export LAND=/data/global-weather/natural-earth-land.geojson
export LAND_SHA256=<64-hex-sha256>
export STATIC_MESH8=/data/global-weather/joint-static-mesh8-r10
export RUN_ROOT=/data/global-weather/runs/joint-mesh8-qualification-r10-${R10_SHA}
```

Пути должны указывать на локальные immutable inputs. Не заменяйте их во время опыта. Для нового R10 предпочтительны новые каталоги climate/static, чтобы исключить случайное использование manifests с retired mesh identity.

## 4. Q2 — подготовить климатический context

NCEP1 не интерполируется заранее в многогигабайтный `12×655362×37×5` массив. Хранится context mesh4, который затем детерминированно отображается на mesh8.

```bash
python -m global_weather.seasonal_climatology \
  --source-dir "$NCEP1_LTM" \
  --output "$CLIMATE_MESH4" \
  --mesh-level 4 \
  --minimum-years 25
```

Если каталог уже существует, используйте только ранее проверенный immutable artifact. При наличии старого mesh8 mapping/fingerprint создайте новый каталог, а не редактируйте manifest на месте.

## 5. Q3 — подготовить static terrain на mesh8

```bash
python -m global_weather.joint_static \
  --dem "$DEM" \
  --dem-manifest "$DEM_MANIFEST" \
  --land-geojson "$LAND" \
  --land-sha256 "$LAND_SHA256" \
  --mesh-level 8 \
  --output "$STATIC_MESH8"
```

Static artifact обязан быть привязан к canonical grid fingerprint:

`64c3e32f792a42ad7d363635e5217cd3c845743a9b92487fdc226da5f1937aa9`.

Fingerprint schema должна быть `mesh8-geometry-q1e-7-v1`. Отрицательные высоты суши сохраняются; океан определяется независимым land polygon, а не знаком DEM.

## 6. Отдельно выполнить preflight

Preflight не обучает модель. Он должен завершиться со статусом `ready_for_q4_q5`.

```bash
python -m global_weather.mesh8_preflight \
  --dataset "$JOINT_DATA" \
  --norm "$PROFILE_NORM" \
  --climatology "$CLIMATE_MESH4" \
  --static "$STATIC_MESH8" \
  --config configs/joint_observation_training.json \
  --run-root "$RUN_ROOT" \
  --expected-sha "$R10_SHA"
```

Проверьте `$RUN_ROOT/preflight.json`. В нём должны быть:

- `status = ready_for_q4_q5`;
- `mesh.level = 8` и `mesh.cells = 655362`;
- fingerprint `64c3e32f…f1937aa9` и schema `mesh8-geometry-q1e-7-v1`;
- `joint_architecture = joint-measured-physical-v2`;
- `profile_architecture = hydrostatic-primitive-conditioned-v2`;
- hashes dataset/norm/climate/static;
- конкретная CUDA device identity и total VRAM;
- свободное место не ниже 64 GiB.

Если preflight упал, Q4 не запускайте. Исправляйте источник несоответствия, а не обходите gate.

## 7. Q4→Q5 одной фиксированной командой

Установите абсолютный deadline с часовым поясом. Он нужен для fail-closed остановки вычислительного процесса.

```bash
export R10_DEADLINE='2026-10-14T06:00:00+00:00'
```

Замените эту дату на реальный будущий срок запуска.

```bash
python scripts/mesh8-r10-runner.py \
  --dataset "$JOINT_DATA" \
  --norm "$PROFILE_NORM" \
  --climatology "$CLIMATE_MESH4" \
  --static "$STATIC_MESH8" \
  --config configs/joint_observation_training.json \
  --run-root "$RUN_ROOT" \
  --expected-sha "$R10_SHA" \
  --deadline "$R10_DEADLINE"
```

Runner выполняет только зафиксированные команды:

1. повторный preflight;
2. Q4: одну эпоху joint training с `hidden=4`, +3 ч и edge chunks 32768;
3. Q5: загрузку best checkpoint и два validation issues с frozen inference;
4. physical diagnostics и GRIB2;
5. final evidence verification.

Runner сохраняет отдельные stdout/stderr каждого шага и их SHA256. При deadline или падении свободного диска ниже 64 GiB дочерний process group завершается.

## 8. Что должно появиться

Минимальный набор:

```text
RUN_ROOT/
  preflight.json
  execution-evidence.json
  training/
    progress.json
    epoch-0001/
      state.pt
      metrics.json
    latest.json
    best.json
    complete.json
  forecast/
    batch.json
    <issue>/
      forecast.npz
      forecast.json
      forecast.grib2
      forecast.grib2.json
      completion.json
  final-verification.json
  logs/
    preflight.stdout.log
    preflight.stderr.log
    training.stdout.log
    training.stderr.log
    inference.stdout.log
    inference.stderr.log
    final-check.stdout.log
    final-check.stderr.log
```

## 9. Критерий успешного R10

`final-verification.json` должен содержать:

```text
status = r10_q4_q5_technically_qualified
scientific_acceptance = false
qualified_horizon_hours = 3
next_stage_authorized = false
```

Дополнительно verifier требует ненулевые finite gradients всех trainable branches, измеренные CUDA peak allocated/reserved, `q` без domain violations, `max |hydrostatic residual| <= 5 m2 s-2`, `max |continuity residual| <= 1e-6 s-1` и совпадение SHA256 NPZ/GRIB2/inventory.

Успех R10 означает только то, что physical-v2 технически проходит полный mesh8 тракт на +3 ч.

## 10. Что делать после R10

Не меняйте `rollout_hours` в этом опыте на 72. Следующий этап создаётся отдельным experiment ID. Перед ним нужны фактическая R10 CUDA telemetry, streaming/checkpointed rollout, streaming export и независимая метеорологическая верификация по переменным, уровням и срокам.
