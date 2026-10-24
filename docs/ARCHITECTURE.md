# Архитектура глобальной модели

`global-ml-weather` содержит несколько исследовательских моделей, но текущую архитектуру следует читать как две параллельные линии: Adaptive/Multimodal и Joint/Hydrostatic. Они используют общую инфраструктуру данных, сферической геометрии, нормировки, provenance, checkpoints, verification и выдачи, но имеют разные внутренние состояния и несовместимые веса.

Научно подтверждённых погодных весов для общего эксплуатационного прогноза в репозитории пока нет.

[Актуальный визуальный путеводитель](ARCHITECTURE_VISUAL_GUIDE.md).

## 1. Ветка Adaptive / Multimodal

`GlobalWeatherModel` остаётся исходным сравнительным графовым вариантом.

`AdaptiveWeatherModel` добавляет:

- обучаемое сжатие полной вертикали в `K=4/8/16/38` latent slots;
- стандартную конфигурацию `K=8`;
- `AdaptiveProcessor` с направленными сообщениями;
- многоуровневую down/up-иерархию сферического графа;
- state-conditioned смешивание горизонтального, вертикального и локального механизмов;
- причинный `analysis_state` и собственный background cycling;
- декодирование обратно в 37 изобарических уровней и поверхность.

`MultimodalWeatherModel` расширяет эту ветвь:

- `SparseObservationEncoder` для разреженных наблюдений;
- `SatelliteBank`;
- `ResUNet` + `MaskedConvGRU` для imager-последовательностей;
- microwave MLP + GRU для СВЧ-потоков;
- `SourceFusion` с cross-attention.

Спутниковые каналы имеют собственные нормы и маски. Перенос спутниковых признаков на сферическую сетку не объявляется консервативной перепроекцией физических полей.

Упрощённый тракт:

```text
causal observations / satellite sequences / static fields
  → modality-specific encoders
  → SourceFusion
  → physical vertical representation
  → attention compression to K=8
  → AdaptiveProcessor
  → attention expansion
  → 37 pressure levels + surface heads
```

`K` не является физическим уровнем давления.

## 2. Ветка Joint / Hydrostatic

Для реальных совместных GHCNh + IGRA опытов используется `JointObservationModel`, наследующий `HydrostaticFlowProfileModel`, который в свою очередь наследует `PressureProfileModel`.

Ключевое отличие от Adaptive: скрытое состояние Joint существует непосредственно на 37 физических уровнях:

```text
state: [Ncell, 37, hidden]
```

Обязательного промежуточного `K=8` в этой ветви нет.

Текущий Joint-контур намеренно не использует спутниковые параметры.

### PressureProfileModel

`PressureProfileModel` сохраняет давление наблюдений и размещает профильные измерения по ближайшим стандартным изобарическим уровням. Входы используют фактические время, координаты, давление, переменную и маску наличия.

### HydrostaticFlowProfileModel

Текущий физически ориентированный processor остаётся learned surrogate, а не finite-volume решателем primitive equations.

Горизонтальный обмен обусловлен:

- направлением ветра;
- градиентом геопотенциала вдоль ребра;
- ускорением Кориолиса;
- длиной и направлением ребра.

На Mesh-8 обмен выполняется по edge chunks, чтобы не материализовать весь тензор `3 932 160 × 37` рёбер одновременно.

Вертикальный обмен использует сухую потенциальную температуру. Положительный устойчивый скачок `θ` уменьшает latent mixing через интерфейс слоя.

Декодер:

- параметризует `T > 0`;
- параметризует `q > 0`;
- строит геопотенциал через `hydrostatic_projection`;
- не выдаёт learned omega в основном profile head.

Важно: в коде уже существует `integrate_hydrostatic_geopotential` с привязкой к поверхности, но canonical `_decode()` `HydrostaticFlowProfileModel` на текущем `main` всё ещё вызывает `hydrostatic_projection` без surface anchor. Наличие helper-функции не следует описывать как уже включённую физику canonical forward.

### JointObservationModel

Surface-ветвь использует 12-часовую masked station history. `surface_encoder` и `GRUCell` вводят наземные наблюдения в профильное состояние с убывающим по log-pressure vertical weight.

Подготовленный static artifact добавляет:

- `terrain_elevation_m`;
- `terrain_land_fraction`;
- `terrain_slope`.

Surface head формирует station-native:

- `T`;
- `Td`;
- `u`, `v`;
- `Ps`;
- `MSLP`.

Hard-domain projection обеспечивает `T > 0`, `Td > 0`, `Td ≤ T`, `Ps > 0`, `MSLP > 0`.

## 3. Как связаны ветки

Ветки связаны общей инфраструктурой, а не последовательным вычислительным графом.

Сейчас нет автоматической цепочки `Adaptive → Joint` или `Joint → Adaptive`. Нет общего latent state и нет совместимых checkpoints.

Общие элементы:

- семейство сферических сеток и grid fingerprints;
- причинный отбор наблюдений;
- маски и provenance;
- атмосферные и source-specific нормы;
- подготовленный DEM и статические признаки;
- training/evaluation infrastructure;
- NPZ/GRIB2 issuance и telemetry.

Перспективное, но ещё не canonical направление — использовать Multimodal как learned analysis/assimilation encoder, декодировать его состояние в 37 физических уровней и передавать это состояние в HydrostaticFlow processor.

## 4. Сетка и физические признаки

Используется двойственная икосаэдрическая сетка, не H3 и не HEALPix.

Двенадцать ячеек имеют пять соседей, остальные обычно шесть. Иерархия содержит `10*4**level + 2` центров ячеек.

Для `mesh_level=8`:

- 655 362 центра ячеек;
- 1 310 720 сферических треугольников;
- 3 932 160 направленных рёбер;
- `sqrt(mean cell area) ≈ 27.898 км` как геометрический масштаб средней площади, а не доказанное метеорологическое разрешение.

HEALPix остаётся сравнительным направлением и не должен подписываться как текущая Mesh-8 геометрия.

## 5. Климатология и рельеф

NCEP1 1991–2020 существенно грубее Mesh-8. Контекст климатологии хранится на более грубой сетке и детерминированно отображается на Mesh-8. Это внешний условный признак, а не наблюдение и не способ повысить разрешение NCEP1.

Глобальный DEM подготавливается отдельно. На воде и в смешанных ячейках сохраняется независимая исходная орография, а не батиметрическая глубина.

## 6. Временная схема

Исходные атмосферные наблюдения используют 12-часовое окно. Будущие фактические наблюдения не подмешиваются в уже начатый выпуск.

Adaptive-контур имеет трёхчасовой шаг и архитектурный горизонт до 72 часов.

Для Mesh-8 Joint требуется различать:

- `joint-mesh8-qualification-r10`: ограниченная end-to-end квалификация `+3 ч`;
- `mesh8_research_training`: исследовательские конфигурации с rollout 3 ч, в том числе конфигурации до 4000 epochs;
- all-leads evaluation scripts: отдельный исследовательский путь, который не превращает frozen Mesh-8 checkpoint автоматически в квалифицированный 72-часовой operational inference.

Frozen Joint inference не имеет права превышать горизонт, записанный в checkpoint/config.

## 7. Postprocessing и новые physics helpers

Canonical issuance применяет `apply_forecast_physics` для профильных полей и формирует diagnostic omega и геометрическую below-ground mask.

Последние изменения добавили дополнительные механизмы:

- `apply_orographic_station_reduction`;
- `enforce_global_mass_and_tendency_bounds`;
- bias-aware masked loss helper;
- `integrate_hydrostatic_geopotential`;
- `ClimatologyAnomalyStepper`;
- `StationMicroclimateCorrector`.

Их статус различается. На текущем `main` глобальное mass/tendency bounding подключено в `scripts/evaluate_mesh8_all_leads.py` для surface rollout evaluation. `ClimatologyAnomalyStepper` и `StationMicroclimateCorrector` реализованы как классы, но не являются частью `JointObservationModel.forward()`. Terrain-anchored hydrostatic integration также не заменило canonical decoder path.

Документация должна различать «реализовано в репозитории» и «подключено к canonical train/inference graph».

## 8. Выходы и ограничения

На 37 изобарах выдаются `T`, `q`, `u`, `v`, геопотенциал и диагностический/маскированный `omega` согласно конкретному issuance path.

Joint surface outputs остаются station-native. Не следует автоматически называть их стандартными `2 m / 10 m` полями без соответствующего instrument-height контракта.

Регулярная GRIB-сетка является форматом выдачи и не определяет внутреннее разрешение модели.

Нормы GraphCast включены как реальные reference-файлы, но не покрывают все спутниковые каналы. Полный водно-энергетический бюджет, эксплуатационная калибровка всех приборов и подтверждённая многосезонная forecast skill остаются незакрытыми задачами.

[Обучение](TRAINING.md) · [Нормы](NORMALIZATION.md) · [Многомодальная ветвь](MULTIMODAL.md) · [Справочные данные](REFERENCE_DATA.md) · [План](ROADMAP.md) · [Mesh-8 R10](decisions/0018-mesh8-end-to-end-qualification.md)
