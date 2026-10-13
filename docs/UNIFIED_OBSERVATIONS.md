# Unified observations: фактический статус

Этот документ сохраняет историю legacy-Unified экспериментов. После физического аудита 12 октября 2026 года данный контур **не считается совместным обучением поверхности, аэрологии и спутника**. Канонический путь на реальных GHCNh + IGRA реализован в `global_weather.joint_observation_training`; текущая квалификация mesh8 описана в [решении 0018](decisions/0018-mesh8-end-to-end-qualification.md).

Подробные исправления: [аудит физической постановки](audits/2026-10-12-physical-corrections.md), [уточнённое решение 0016](decisions/0016-high-res-terrain-satellite-10day.md) и [решение 0017](decisions/0017-joint-observation-pilot-and-diagnostic-products.md).

## Что фактически обучает legacy Unified

`global_weather.unified_observation_training` получает 12 часов реальных приземных наблюдений NOAA GHCNh и прогнозирует безразмерные аномалии шести родных станционных величин. Профильный и спутниковый интерфейсы есть в `UnifiedObservationModel`, но текущий train-loop не передаёт им реальные обучающие targets.

Поэтому:

- `supervised_branches = [surface]`;
- `unsupervised_branches = [profile, satellite]`;
- месячная станционная климатология рассчитывается только по train;
- NOAA/NCEP 1991–2020 используется как внешний 3-D сезонный контекст и не называется станционной нормой;
- научный допуск этого legacy-контура остаётся `false`.

Штатный `configs/unified_observation_training.json` после аудита ограничен 72 часами, `surface_weight=1`, `profile_weight=0`.

## Исторический завершённый 72-часовой surface-опыт

Эксперимент на RTX 4090 использовал 361 train, 177 validation и 180 test серий. Ранний останов сработал на 65-й эпохе; лучшая модель была на эпохе 15.

| Показатель | Значение |
|---|---:|
| Исходная ревизия | `ea6896b1a799ad7b8c5c2ac87da5f79f871dca90` |
| Время обучения | 53 мин 22 с |
| Лучшая эпоха | 15 |
| train loss | 0.059607 |
| validation loss | 0.075332 |
| test loss | 0.068648 |
| SHA256 весов лучшей эпохи | `94004fae4a9797f695588712d7ead0993f28cbe4f748930c920a569c4410b368` |

`test loss=0.068648` — Huber loss по безразмерным приземным аномалиям. Это не RMSE температуры, ветра или давления и не верификация 3-D атмосферы.

## Исторический 240-часовой инженерный опыт

12 октября был запущен 80-шаговый rollout до 240 часов с AMP bfloat16 и activation checkpointing. На момент фиксации было пройдено 98+ эпох; лучшая — 22 (`val_loss=0.077026`, `train_loss=0.060611`), VRAM около 4175 MiB.

Эти значения теперь трактуются **только как инженерная диагностика** по следующим причинам:

1. исходный `native-station-dataset-1` проектировался с 72-часовым временным guard;
2. train-loop фактически супервизировал только surface targets;
3. наличие профильного и спутникового encoder не доказывает их обучение;
4. mesh level 2 содержит 162 ячейки и имеет `sqrt(mean area) ≈ 1774 км`, поэтому не является сеткой 20 км.

`validate_forecast_windows()` запрещает использовать окно, которое пересекает замороженную границу train/validation/test. Для нового 240-часового опыта нужен новый dataset с 240-часовым guard и заново рассчитанными train-only статистиками.

## Канонический joint-контур GHCNh + IGRA

Физически предпочтительный путь использует `global_weather.joint_observation_training`. Историческая mesh2-конфигурация сохранена как `configs/joint_observation_training_mesh2_pilot.json`.

Контур:

- использует реальные GHCNh и IGRA;
- сохраняет фактические времена, координаты и давления аэрологии;
- использует фиксированную профильную нормировку;
- применяет гидростатически согласованный профильный decoder;
- использует независимую маску суши для статического рельефа;
- не содержит фиктивно обученной спутниковой ветви.

## Текущий эксперимент R10: mesh level 8

`configs/joint_observation_training.json` теперь является конфигурацией `joint-mesh8-qualification-r10`:

- `mesh_level=8` — 655362 глобальные ячейки;
- `sqrt(mean cell area)=27,898 км` — геометрический показатель, не доказанное разрешение;
- `hidden=4`;
- `rollout_hours=3`;
- 1 эпоха, максимум 4 train / 2 validation / 2 test issue;
- `edge_chunk_size=32768`;
- declared activation budget 22528 MiB;
- `scientific_acceptance=false`.

R10 проверяет полный путь geometry → real observations → climate/static context → backward pass → checkpoint → frozen inference → physical diagnostics → GRIB2. Он специально **не** проверяет 72-часовой skill.

Сезонная NCEP1 климатология не размножается до 655362 ячеек в долговременном артефакте. Для R10 она готовится на mesh level 4, после чего детерминированно отображается на mesh8 как внешний context. Рельеф и динамическое состояние модели остаются mesh8.

Полный level8 geometry/graph smoke вынесен в workflow `mesh8-qualification`. Вычислительный CUDA-запуск считается состоявшимся только после появления фактических `metrics.json`, `complete.json`, VRAM/RSS telemetry и frozen-inference manifests.

После успешного R10 отдельным решением должен быть создан mesh8 72-hour experiment. Для него потребуется streaming/checkpointed rollout и streaming export, чтобы не удерживать в памяти 25 полных глобальных профильных кадров.

Спутниковое обучение «Арктики-М» остаётся отдельным следующим опытом после подготовки синхронизированного набора реальных радиансов/яркостных температур с приборной калибровкой, масками, временем наблюдения и собственной train-only нормировкой канала.
