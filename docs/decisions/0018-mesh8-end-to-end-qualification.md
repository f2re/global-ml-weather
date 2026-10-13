# Решение 0018. Квалификационный контур mesh level 8

Дата: 13 октября 2026 года.
Идентификатор: `joint-mesh8-qualification-r10` / `GLOBAL-WEATHER-MESH8-R10`.
Статус: подготовлен к вычислительному запуску; научная приёмка запрещена.
Базовая ревизия регистрации: `97cb0ff0bfe00edede37d4bc8a937801a63ecff5`. Точная ревизия запуска сохраняется самим training identity и имеет приоритет над этой ссылкой.

## Цель

Цель R10 — доказать, что глобальная joint-модель проходит полный технический тракт на `mesh_level=8`: построение сферической сетки, привязку реальных GHCNh/IGRA, климатический контекст, DEM, прямой и обратный проход, сохранение checkpoint, повторную загрузку, физическую постобработку и GRIB2.

R10 не является испытанием качества 72-часового прогноза. Горизонт намеренно ограничен 3 часами, чтобы сначала измерить реальную память и время на сетке из 655362 ячеек. После R10 переход к 72 часам оформляется отдельным опытом.

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
| остальные ячейки | 6 соседей |
| `sqrt(mean cell area)` | 27,898 км |

Величина 27,898 км является только геометрическим масштабом средней площади. Она не означает доказанное метеорологическое разрешение 28 или 20 км. Реальный разрешаемый масштаб определяется отдельными спектральными и case-study испытаниями.

Для малых сеток сохраняется прежний `SphericalVoronoi` и прежний порядок построения. Для больших сеток, включая level 8, полный Voronoi не материализуется. Площадь вершиной-центрированной ячейки считается как сумма одной трети площадей прилегающих сферических треугольников. Глобальная сумма обязана совпадать с `4*pi*R^2` в пределах `rtol=1e-9`.

## Почему климатология не хранится на mesh8

NCEP1 1991–2020 имеет существенно более грубое исходное разрешение, чем mesh8. Предварительное хранение массива `12 x 655362 x 37 x 5` создало бы многогигабайтный артефакт без новой информации источника.

В R10 климатический контекст готовится на `mesh_level=4`, после чего `Mesh8SeasonalClimatology` детерминированно сопоставляет каждой ячейке mesh8 ближайшую ячейку климатического контекста. Это относится только к внешнему месячному контексту. Состояние модели, динамика, рельеф, граф и выдаваемые поля остаются mesh8.

Контекст level 4 выбран как вычислительно компактный технический эталон R10. Он не повышает пространственную информативность NCEP1 и не является наблюдением или целью обучения.

## Замороженная конфигурация R10

Файл: `configs/joint_observation_training.json`.

| Параметр | Значение |
|---|---:|
| `mesh_level` | 8 |
| `hidden` | 4 |
| `rollout_hours` | 3 |
| `epochs` | 1 |
| train issues | не более 4 |
| validation issues | не более 2 |
| test issues | не более 2 |
| profile scalars/window | не более 2000 |
| `edge_chunk_size` | 32768 |
| surface/profile loss | 0,5 / 0,5 |
| declared activation budget | 22528 MiB |
| `scientific_acceptance` | false |

При float32 один скрытый state `[655362,37,4]` занимает около 370 MiB. Один полный профильный кадр `[655362,37,6]` занимает около 555 MiB. Два кадра для сроков 0 и +3 ч занимают около 1110 MiB без учёта масок, градиентов и служебных тензоров. Полный climate feature tensor `[655362,37,14]` занимает около 1295 MiB в float32. Эти оценки служат до запуска только верхней границей планирования; фактические `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes` и RSS являются обязательным результатом R10.

## Этапы

### Q0. Геометрия и граф

Автоматический workflow `mesh8-qualification` обязан:

- построить настоящую сетку level 8;
- проверить 655362 ячейки, 1310720 граней и 3932160 направленных рёбер;
- проверить 12 пятисоседних ячеек;
- проверить глобальный баланс площади;
- выполнить полный sparse neighbour exchange на 655362 ячейках;
- сохранить `outputs/mesh8/geometry.json`.

### Q1. Реальные данные и нормы

Joint dataset остаётся тем же хэшированным GHCNh+IGRA архивом с 12-часовой историей и 72-часовой защитой split. R10 использует только первые +3 ч целей, но не ослабляет исходный 72-часовой guard.

GraphCast-нормы пяти профильных величин не пересчитываются. Станционная климатология строится только по train-периоду. Никакие validation/test значения не участвуют в fit.

### Q2. Климатический контекст level 4

```bash
python -m global_weather.seasonal_climatology \
  --source-dir DATA/NCEP1-LTM-1991-2020 \
  --output DATA/seasonal-climatology-mesh4 \
  --mesh-level 4 \
  --minimum-years 25
```

На этапе загрузки mesh8 модель обязана проверить исходный manifest, SHA256, level 4 и детерминированное соответствие target grid fingerprint.

### Q3. Статические поля mesh8

```bash
python -m global_weather.joint_static \
  --dem DATA/earth_relief_30s_p.nc \
  --dem-manifest assets/dem/earth_relief_30s_p.json \
  --land-geojson DATA/natural-earth-land.geojson \
  --land-sha256 LAND_SHA256 \
  --mesh-level 8 \
  --output DATA/joint-static-mesh8
```

DEM выбирается для каждой из 655362 ячеек. Суша задаётся независимым полигональным источником, поэтому отрицательная высота суши не превращается в океан. Manifest обязан содержать mesh8 fingerprint.

### Q4. Один обучающий квалификационный проход

```bash
CUDA_VISIBLE_DEVICES=0 python -m global_weather.joint_observation_training \
  --dataset DATA/joint-observations \
  --norm DATA/profile-normalization.json \
  --climatology DATA/seasonal-climatology-mesh4 \
  --static DATA/joint-static-mesh8 \
  --config configs/joint_observation_training.json \
  --output outputs/joint-mesh8-qualification-r10
```

Обязательные условия успеха:

- CUDA используется фактически;
- все пять профильных и все шесть native surface величин имеют реальные цели;
- все обучаемые ветви получают конечные ненулевые градиенты;
- loss и gradients конечны;
- peak VRAM не превышает объявленный бюджет;
- validation выполняется на том же 3-часовом rollout;
- checkpoint и `complete.json` записываются атомарно;
- identity содержит commit, mesh8 fingerprint, grid area method, dataset/norm/static/climate hashes.

### Q5. Frozen checkpoint, физика и GRIB2

```bash
CUDA_VISIBLE_DEVICES=0 python -m global_weather.joint_observation_inference \
  --dataset DATA/joint-observations \
  --training outputs/joint-mesh8-qualification-r10 \
  --static DATA/joint-static-mesh8 \
  --output outputs/joint-mesh8-qualification-r10-inference \
  --split validation \
  --limit 1 \
  --horizon-hours 3 \
  --device cuda:0 \
  --warmups 0 \
  --repeats 1 \
  --spacing-degrees 0.5
```

Inference не имеет права запросить горизонт больше `rollout_hours` checkpoint. Для квалификационного checkpoint mesh8 разрешён только +3 ч.

Выход должен пройти гидростатическую проверку, геометрическую маску относительно проверенного DEM, диагностический расчёт omega из continuity, сохранение NPZ и GRIB2 с inventory и SHA256. Регулярная GRIB-сетка 0,5 градуса является только форматом квалификационной выдачи и не определяет разрешение модели.

## Критерии закрытия R10

R10 получает статус `qualified` только если одновременно выполнены:

1. CI geometry report успешен на полном level 8.
2. Static и climatology artifacts имеют неизменные SHA256 и ожидаемые grid identities.
3. Train и validation завершились без OOM/nonfinite.
4. Ненулевые gradients подтверждены для всех trainable parameter tensors.
5. Best checkpoint повторно загружается и совпадает по fixed buffers.
6. 3-часовой frozen inference завершён на mesh8.
7. Physical diagnostics и GRIB2 inventory созданы и проходят встроенные проверки.
8. Фактические VRAM, RSS и времена записаны.
9. `scientific_acceptance=false` сохранён во всех manifests.

Провал любого пункта сохраняется как отрицательный результат R10. Запрещено уменьшать mesh level и при этом сохранять идентификатор R10.

## Переход к 72 часам

После успешного R10 создаётся новый опыт. Минимально требуется отдельно проверить streaming/checkpointed rollout, чтобы не удерживать в памяти 25 полных глобальных кадров, а также streaming export. Увеличение `hidden`, количества эпох и горизонта не выполняется автоматически.

Распределение ответственности по ролям закреплено в `agents/MESH8_EXPERIMENT_CONTRACT.md`.
