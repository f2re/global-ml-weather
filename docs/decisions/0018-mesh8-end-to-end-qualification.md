# Решение 0018. Квалификационный контур mesh level 8

Дата: 13 октября 2026 года.
Идентификатор: `joint-mesh8-qualification-r10` / `GLOBAL-WEATHER-MESH8-R10`.
Статус: **Q0 geometry/graph повторно пройден под устойчивой identity; Q1–Q5 runtime/CUDA ещё не закрыты; научная приёмка запрещена**.

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
| canonical grid fingerprint | `64c3e32f792a42ad7d363635e5217cd3c845743a9b92487fdc226da5f1937aa9` |
| fingerprint schema | `mesh8-geometry-q1e-7-v1` |

27,898 км — геометрический масштаб средней площади, а не доказанное метеорологическое разрешение. Заявления «20 км» или «28 км skill» запрещены без отдельной спектральной и case-study верификации.

Сетки 0–7 сохраняют прежний порядок построения, `SphericalVoronoi` и исторический fingerprint contract. Mesh8 строится только через явный `build_mesh8_grid()`: случайный `build_grid(8)` по-прежнему блокируется memory guard. Последний переход 7→8 векторизован. Полный Voronoi на 655362 центрах не материализуется; площадь вершиной-центрированной ячейки равна сумме одной трети площадей прилегающих сферических треугольников.

### Стабилизация identity mesh8

Первоначальный Q0 использовал fingerprint `992e223c…2b76b9`, который хэшировал округлённые float64-координаты. При повторении на другой NumPy/runtime тот же код и та же топология дали иной hash из-за суб-ULP различий нормализации midpoint. Это делало identity среды частью identity сетки и мешало воспроизводимому preflight.

Для mesh8 введена schema `mesh8-geometry-q1e-7-v1`: координаты единичной сферы перед хэшированием квантуются в целые с шагом `1e-7`, то есть примерно 0,64 м на поверхности Земли, после чего хэшируются вместе с directed edges. Такой допуск на порядки меньше среднего масштаба ячейки и устраняет только численный шум runtime. Levels 0–7 не изменены.

Старый `992e223c…` fingerprint считается retired identity-format, а не другой физической сеткой. Любые mesh8 artifacts, где он записан в manifest, перед R10 нужно перестроить либо мигрировать отдельной проверяемой процедурой; ручная замена hash запрещена.

## Фактический Q0 после переквалификации

Канонический workflow `mesh8-qualification`, run `38389968579`, ревизия `145ba93e4aa3ff60246773b49ffb462cc9b58136`, выполнил полный mesh8, targeted physical/R10 tests и полный sparse neighbour exchange.

| Диагностика | Результат |
|---|---:|
| physical/R10 contract tests | 48 passed |
| grid fingerprint | `64c3e32f…f1937aa9` |
| relative area closure error | `1,2253e-16` |
| geometry build | 3,487 с |
| full graph exchange | 0,248 с |
| peak RSS | 675614720 байт, около 644 MiB |
| artifact id | `11808376808` |

Q0 подтверждает топологию, площади, `locate()`, `GraphOps.neighbours()`, physical-v2 targeted contracts и устойчивую identity на всех 655362 ячейках. Он не подтверждает CUDA training или качество прогноза.

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

Physical-v2 подробно зафиксирован в [решении 0019](0019-r10-physical-core-and-final-execution.md). Профильный processor выполняет гидростатическую проекцию, primitive-equation-conditioned directed exchange и stability-gated vertical coupling; joint surface head ограничивает термодинамические диапазоны. Эти механизмы не являются заявлением о полном решении примитивных уравнений или точном сохранении массы/энергии.

## Q1. Реальные данные и нормировка

Joint dataset остаётся хэшированным GHCNh+IGRA архивом с 12-часовой историей и исходным 72-часовым guard. R10 читает только цели +3 ч, но guard не ослабляется.

GraphCast-нормы пяти профильных величин фиксированы. Станционная климатология рассчитывается только по train. Validation/test не участвуют в fit. Preflight проверяет source identity, hashes и exact clean Git SHA до запуска Q4.

Статус: код и fail-closed проверки реализованы; фактический runtime report на вычислительном узле ещё не создан.

## Q2. Климатический контекст

NCEP1 1991–2020 существенно грубее mesh8. R10 требует подготовленный seasonal artifact на mesh level 4. `Mesh8SeasonalClimatology` детерминированно отображает его на mesh8 и включает source/target fingerprints в собственный fingerprint.

```bash
python -m global_weather.seasonal_climatology \
  --source-dir DATA/NCEP1-LTM-1991-2020 \
  --output DATA/seasonal-climatology-mesh4-r10 \
  --mesh-level 4 \
  --minimum-years 25
```

Статус: adapter реализован; real artifact build/preflight pending. Artifact, созданный под retired mesh8 identity, не допускается молчаливо.

## Q3. Статические поля mesh8

`joint_static` явно использует `build_mesh8_grid()` при `--mesh-level 8`. DEM выбирается для всех 655362 центров, а суша определяется независимым GeoJSON, поэтому отрицательная высота суши не превращается в океан.

```bash
python -m global_weather.joint_static \
  --dem DATA/earth_relief_30s_p.nc \
  --dem-manifest assets/dem/earth_relief_30s_p.json \
  --land-geojson DATA/natural-earth-land.geojson \
  --land-sha256 LAND_SHA256 \
  --mesh-level 8 \
  --output DATA/joint-static-mesh8-r10
```

Статус: код реализован; real static artifact/hash под canonical stable fingerprint pending.

## Q4 и Q5

Финальный порядок выполнения зафиксирован в [MESH8_R10_FINAL_RUN.md](../MESH8_R10_FINAL_RUN.md):

`preflight → Q4 CUDA training/validation → reload best checkpoint → Q5 frozen +3h inference → physical diagnostics/GRIB2 → final evidence gate`.

R10 считается технически закрытым только при `final-verification.json: status=r10_q4_q5_technically_qualified`. При этом `scientific_acceptance=false`, а переход к 72 часам остаётся запрещённым до нового experiment ID, фактической CUDA telemetry, streaming rollout/export и независимой skill verification.

Машиночитаемое состояние gates: `experiments/joint-mesh8-qualification-r10.json`.
Распределение ролей: `agents/MESH8_EXPERIMENT_CONTRACT.md`.
