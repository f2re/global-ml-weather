# Получение и применение норм

Источник зафиксирован в `configs/normalization_sources.json`; смысл и ограничения —
в [решении 0002](decisions/0002-physically-adaptive.md).

## Получить базовые файлы отдельно от расчётного сервера

```bash
python -m pip install -e '.[data]'
python -m global_weather.import_climatology \
  --download data/normalization/upstream \
  --output data/normalization/graphcast_base.json
```

Это явная сетевая операция первоначальной подготовки. Модель не скачивает
ничего во время прогноза. Для сервера без сети скопируйте исходные NetCDF и
полученный JSON. Импорт локальных файлов:

```bash
python -m global_weather.import_climatology \
  --mean data/normalization/upstream/mean_by_level.nc \
  --std data/normalization/upstream/stddev_by_level.nc \
  --output data/normalization/graphcast_base.json
```

После проверки происхождения сохраните `provenance.artifact_sha256` в отдельный
JSON и передавайте его через `--expected-hashes`. Изменение любого исходного
файла приведёт к отказу импорта. Импорт файла с подходящей структурой сам по себе
не удостоверяет его происхождение: храните источники и журнал загрузки.

В этой ревизии импорт проверен на помеченных синтетических NetCDF. Реальные
облачные файлы не удалось получить в среде разработки; их численные значения
НЕ включены в репозиторий и не объявляются проверенными. Ревизия исходного кода,
пути и соответствие переменных проверены по GitHub. При недоступности источника
загрузка завершится ошибкой, а не создаст вымышленные коэффициенты.

## Подготовить полный набор

Базовый GraphCast: шесть высотных величин на 37 изобарах, T2, U10/V10, MSLP и
осадки за 6 часов. Для модели с шагом 3 часа требуется пересчитать именно
трёхчасовые суммы на обучающем периоде. Для Td2, ps и общей облачности также
нужны свои нормы. Все зарегистрированные спутниковые каналы имеют отдельные
статистики. Не заменяйте их атмосферными величинами с такими же единицами.

`weighted_statistics(values, mask, weights)` рассчитывает μ/σ по обучающим
значениям с площадными весами и масками. Ось 0 — выборка; остальные — сохраняемые
оси. Для [samples,levels] веса площади передаются как [samples,1]. Функция не
выбирает годы за пользователя: состав обучающего подмножества, единицы и сроки
должны быть проверены до вызова. Конвейер многолетней выгрузки пока не включён.

Полный `NormalizationBundle` имеет `schema_version=1`,
`kind=global_level_zscore`, словарь `variables` и `provenance`.
У переменной: `units`, `mean`, `std`, `pressure_pa`, `interval_hours`.
Поверхность — списки длины 1; профиль — списки на поддерживаемых давлениях.
У `precipitation_step` интервал обязан совпадать с шагом модели.
В `provenance` записываются репозиторий, ревизия, семейство данных, лицензия,
SHA256 источников и период расчёта (`null`, когда не установлен).

При объединении базовых и собственных норм храните происхождение каждого
компонента и все хэши. Нельзя наследовать известный период собственного
дополнения и объявлять им весь набор, если период базовой статистики неизвестен.
Не подставляйте инженерные коэффициенты из демонстрации вместо недостающих норм.

## Подключение в Python

```python
from global_weather.normalization import NormalizationBundle
from global_weather.observations import pack_observations
from global_weather.adaptive import AdaptiveWeatherModel

norm = NormalizationBundle.load('data/normalization/complete_era5_training.json')
# grids, records, registry, pressure_pa, issue_time подготавливаются адаптерами.
obs = pack_observations(records, grids[0], pressure_pa, issue_time,
                        registry, normalization=norm)
model = AdaptiveWeatherModel(grids, obs.vocabulary,
    observation_schema=obs.schema_fingerprint, hidden=128,
    latent_slots=8, step_hours=3, normalization=norm)
```

Неполный набор норм приводит к явному отказу. Входы, выходное преобразование и
масштабы обучающей ошибки используют один набор. Его отпечаток входит в схему
наблюдений, состояния и контрольной точки. Смена норм требует нового эксперимента;
она не совместима с молчаливой загрузкой старых весов.

`assert_independent_test(test_start)` не допускает пересечения периода расчёта
норм и теста. При неизвестном периоде проверка тоже завершается отказом. Это не
заменяет проверку лет предварительного обучения самих нейросетевых весов.
