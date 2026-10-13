# Решение 0018. Квалификационный контур mesh level 8

Дата: 13 октября 2026 года.
Идентификатор: `joint-mesh8-qualification-r10` / `GLOBAL-WEATHER-MESH8-R10`.
Статус: **Q0 geometry/graph пройден; Q1–Q5 runtime/CUDA ещё не закрыты; научная приёмка запрещена**.

## Цель

R10 должен доказать работоспособность полного технического тракта на `mesh_level=8`: сферическая сетка, реальные GHCNh/IGRA, климатический контекст, DEM, forward/backward, checkpoint, повторная загрузка, физическая постобработка и GRIB2.

R10 не является испытанием 72-часового skill. Горизонт квалификации намеренно ограничен +3 ч. Переход к 72 ч допускается только новым экспериментом после измерения реальных VRAM/RSS и реализации потокового rollout/export.

## Геометрия

Иерархия содержит `10*4^level+2` центров ячеек.

| Параметр | Значение |
|---|---:|
| mesh level | 8 |
| центры ячеек | 655362 |
| сферические треугольники | 1310720 |
| неориентированные рёбра | 1966080 |
| направленные рёбра | 3932160 |
| ячейки с 5 соседями | 12 |
| ячейки с 6 соседями | 655350 |
| `sqrt(mean cell area)` | 27,897963965 км |
| canonical grid fingerprint | `992e223cc99417d5f12f8ed6d314579d4006a788ca9d935af1f5abc22b2b76b9` |

27,898 км — геометрический масштаб средней площади, а не доказанное метеорологическое разрешение. Заявления «20 км» или «28 км skill» запрещены без отдельной спектральной и case-study верификации.

Сетки 0–7 сохраняют прежний порядок построения и `SphericalVoronoi`, чтобы не менять существующие fingerprints/checkpoints. Mesh8 строится только через явный `build_mesh8_grid()`: случайный `build_grid(8)` по-прежнему блокируется memory guard. Последний переход 7→8 векторизован. Полный Voronoi на 655362 центрах не материализуется; площадь вершиной-центрированной ячейки равна сумме одной трети площадей прилегающих сферических треугольников. Глобальная площадь обязана замыкаться к `4*pi*R^2`.

## Фактический Q0

Канонический workflow `mesh8-qualification`, run `38333408280`, ревизия `0c26d09ef97fe525411c5414ebc775ce4c229125`, выполнил реальный mesh8 и полный sparse neighbour exchange.

| Диагностика | Результат |
|---|---:|
| contract tests | 6 passed |
| grid fingerprint | `992e223c…2b76b9` |
| relative area closure error | `6,1267e-16` |
| geometry build | 5,988 с |
| full graph exchange | 0,385 с |
| peak RSS | 737333248 байт, около 703 MiB |
| artifact id | `11784180964` |

Q0 подтверждает топологию, площади, `locate()` и `GraphOps.neighbours()` на всех 655362 ячейках. Он не подтверждает CUDA training или качество прогноза.

## Замороженная конфигурация R10

Файл: `configs/joint_observation_training.json`.

| Параметр | Значение |
|---|---:|
| `mesh_level` | 8 |
| `hidden` | 4 |
| `rollout_hours` | 3 |
| `epochs` | 1 |
| train / validation / test issues | ≤4 / ≤2 / ≤2 |
| profile scalars/window | ≤2000 |
| `edge_chunk_size` | 32768 |
| surface/profile loss | 0,5 / 0,5 |
| declared activation budget | 22528 MiB |
| CUDA required | yes |
| `scientific_acceptance` | false |

`HydrostaticFlowProfileModel` выполняет wind-conditioned exchange по edge chunks; тест подтверждает эквивалентность chunked и исторического one-shot результата на малой сетке. Полный mesh8 exchange не материализует тензор всех 3932160 рёбер × 37 уровней одновременно.

## Q1. Реальные данные и нормировка

Joint dataset остаётся хэшированным GHCNh+IGRA архивом с 12-часовой историей и исходным 72-часовым guard. R10 читает только цели +3 ч, но guard не ослабляется.

GraphCast-нормы пяти профильных величин фиксированы. Станционная климатология рассчитывается только по train. Validation/test не участвуют в fit. Перед каждой эпохой dataset, нормы, station climatology, climate и static artifacts перепроверяются по identity/hash.

Статус: код и fail-closed проверки реализованы; фактический runtime report на вычислительном узле ещё не создан.

## Q2. Климатический контекст

NCEP1 1991–2020 существенно грубее mesh8. Хранить заранее интерполированный массив `12×655362×37×5` нерационально: это многогигабайтный артефакт без новой информации источника.

R10 требует подготовленный seasonal artifact на mesh level 4. `Mesh8SeasonalClimatology` детерминированно отображает его на mesh8 по ближайшему центру context-cell и включает source/target fingerprints в собственный fingerprint. Это внешний контекст, не наблюдение и не повышение разрешения NCEP1.

```bash
python -m global_weather.seasonal_climatology \
  --source-dir DATA/NCEP1-LTM-1991-2020 \
  --output DATA/seasonal-climatology-mesh4 \
  --mesh-level 4 \
  --minimum-years 25
```

Статус: adapter реализован; реальный artifact build на data node pending.

## Q3. Статические поля mesh8

`joint_static` явно использует `build_mesh8_grid()` при `--mesh-level 8`. DEM выбирается для всех 655362 центров, а суша определяется независимым GeoJSON, поэтому отрицательная высота суши не превращается в океан.

```bash
python -m global_weather.joint_static \
  --dem DATA/earth_relief_30s_p.nc \
  --dem-manifest assets/dem/earth_relief_30s_p.json \
  --land-geojson DATA/natural-earth-land.geojson \
  --land-sha256 LAND_SHA256 \
  --mesh-level 8 \
  --output DATA/joint-static-mesh8
```

Статус: код реализован; real static artifact/hash на data node pending.

## Q4. CUDA train/validation qualification

```bash
CUDA_VISIBLE_DEVICES=0 python -m global_weather.joint_observation_training \
  --dataset DATA/joint-observations \
  --norm DATA/profile-normalization.json \
  --climatology DATA/seasonal-climatology-mesh4 \
  --static DATA/joint-static-mesh8 \
  --config configs/joint_observation_training.json \
  --output outputs/joint-mesh8-qualification-r10
```

Условия успеха: фактический CUDA device; конечные loss/gradients; ненулевые gradients всех trainable tensors; реальные targets всех пяти профильных и шести native surface величин; peak VRAM ≤ заявленного бюджета; validation на +3 ч; атомарные checkpoint/complete; identity содержит commit, mesh fingerprint и все data/norm/static/climate hashes.

Статус: **pending executor run**. В GitHub-конфигурации репозитория не обнаружен готовый self-hosted/GPU runner, поэтому CUDA-проход не помечается как выполненный без реального артефакта.

## Q5. Frozen inference, физика и GRIB2

Frozen inference ограничен горизонтом checkpoint. Для R10 mesh8 разрешено только +3 ч; попытка запросить больший горизонт завершается ошибкой.

```bash
CUDA_VISIBLE_DEVICES=0 python -m global_weather.joint_observation_inference \
  --dataset DATA/joint-observations \
  --training outputs/joint-mesh8-qualification-r10 \
  --static DATA/joint-static-mesh8 \
  --output outputs/joint-mesh8-qualification-r10-inference \
  --split validation --limit 1 --horizon-hours 3 \
  --device cuda:0 --warmups 0 --repeats 1 \
  --spacing-degrees 0.5
```

Выход обязан пройти гидростатическую/геометрическую диагностику, continuity-диагностику omega, сохранение NPZ, GRIB2 inventory и SHA256. Регулярная GRIB-сетка — формат выдачи, не разрешение модели.

Статус: pending Q4.

## Критерии закрытия R10

R10 становится `qualified` только когда одновременно выполнены:

1. Q0 full mesh8 geometry/graph — **выполнено**.
2. Q1 data/norm runtime verification.
3. Q2 immutable climate artifact.
4. Q3 immutable mesh8 static artifact.
5. Q4 CUDA train + validation без OOM/nonfinite с записанной телеметрией.
6. Reload best checkpoint с совпадением fixed buffers и fingerprints.
7. Q5 frozen +3h inference, physical diagnostics, NPZ/GRIB2/inventory.
8. `scientific_acceptance=false` во всех manifests.

Машиночитаемое состояние gates: `experiments/joint-mesh8-qualification-r10.json`.
Распределение ролей: `agents/MESH8_EXPERIMENT_CONTRACT.md`.

## Переход к 72 часам

После успешного R10 создаётся новый experiment ID и новое решение. Перед 72-часовым mesh8-run обязательны streaming/checkpointed rollout и streaming export, чтобы не удерживать одновременно 25 полных глобальных профильных кадров. `hidden`, epochs и horizon не повышаются автоматически.
