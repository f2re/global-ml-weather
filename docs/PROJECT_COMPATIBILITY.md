# Совместимость пользовательских проектов — 0.3.1

Проверка исходников 3 октября 2026 года. Зафиксированы точные ревизии и Git blob
SHA1 в configs/upstream_contracts.json; репозитории-поставщики не изменяются.
Структурные тесты используют явно синтетические файлы, не реальные наблюдения.

| Поток | До изменения | Реализовано | Ограничение физического допуска |
|---|---|---|---|
| arktika-worker | Общий манифест, не native SQLite | Read-only assets/jobs + download.json + SHA256, каналы/время/маски | Калибровка, геометрия и время обработки проверяются отдельно |
| SatDump MSU-MR | Инвентаризация непрочитанных файлов | dataset.json + ограниченный CBOR + images/timestamps/projection/status | Нативный калибратор не исполняется; нужен числовой экспорт |
| SatDump MTVZA | Инвентаризация | hrpt30, 30 транспортных изображений, quality/no_data/partial | Физическая раскладка, калибровка и антенное пятно не готовы; обучение блокируется |
| Электро-Л через SatDump | Общий манифест | msu_gs LRIT-продукт и проверенный физический GeoTIFF-мост | Сырые MSU-GS-*.png без metadata не допускаются |
| Все физические растры | Не было перехода к контракту модели | Тайл GeoTIFF + per-pixel NPZ → причинный JSONL; связь с native metadata | Это не автоматическая калибровка/антенный оператор/научная валидация |

## Почему нельзя заявить безусловную совместимость

В просмотренном arktika/model.py PLATFORMS ограничен ARCM1/ARCM2. Файл
спектральной чувствительности Электро-L не является коннектором архива.
В processing.py разрешён исследовательский assumed DN=K; новый физический
мост его отвергает. download.json содержит время получения, а не время кадра.

В пользовательской ветке SatDump MTVZA/hrpt30 сохраняет исходные изображения
и геометрию без назначения калибратора в просмотренном модуле. Маркер CALIB_*
для MSU-MR не доказывает, что файл уже в физических единицах. LRIT использует
msu_gs/productizer, сырой MSU-GS модуль сохраняет изображения без полного
dataset-контракта. Эти случаи нельзя исправлять переименованием PNG в K.

## Настройка без изменения работающих проектов

Только в окружении модели:
```bash
python -m pip install -e '.[test,data,lab,compat]'
python -m global_weather.connectors.preflight --project-root . \
  --arktika-source /path/to/arktika-worker --satdump-source /path/to/SatDump \
  --report outputs/compat-001/preflight.json

python -m global_weather.connectors.project_bridge arktika \
  --database /path/to/arktika-state/catalog.sqlite \
  --data-root /path/to/arktika-downloads --output outputs/compat-001/arktika.json

python -m global_weather.connectors.project_bridge satdump \
  --dataset-dir /path/to/satdump-pass --output outputs/compat-001/satdump.json
```
Пути заменяются фактическими; нельзя использовать source checkout как архив
наблюдений. Рекомендуется монтирование архивов read-only. Отчёты пишутся только
в consumer outputs, исходники/БД/settings не меняются. Не следует импортировать
Store из поставщика: его конструктор изменяет состояния очереди.

## Контракт физического экспорта

`python -m global_weather.connectors.physical_bridge --review review.json
--data-root /path/to/physical-package --issue-time 2026-10-03T12:00:00Z
--output outputs/compat-001/channel.jsonl`

Рецензия JSON, schema `global-weather.physical-raster/1`, содержит:
- validation_status=reviewed; data_kind=real или synthetic; reviewer; license.
- producer.repository и фактическую 40-символьную revision; source, platform,
  instrument, channel_id, variable; channel_mapping_verified=true.
- calibration_reference, geometry_reference, time_reference, quality_reference;
  calibration_status metadata/declared/verified (не assumed); quantity, units,
  scale, offset и physical_valid_range. Scale/offset применяются к хранимому DN
  ровно один раз. Для уже физических значений явно указать 1 и 0.
- raster, geometry, native_metadata — локальные файлы внутри data-root; SHA256
  каждого в соответствующем поле *_sha256. Native metadata для Арктики —
  сохранённая нормализованная запись asset, для SatDump — product.cbor.
- available_at: готовность физической обработки; download_completed_at: получение.
- observation_id_prefix: устойчивый ID съёмки и исходной сетки;
  pixel_origin=[row,column]: положение тайла; revision: ревизия продукта.
  Это сохраняет идентичность пикселя при повторном экспорте/перекрытии тайлов.

Дополнительно для Арктики: asset_id, original_raster, download_receipt и
его SHA256. Журнал должен совпадать с реальным исходным растром по size/SHA/time.
Для SatDump: dataset_metadata, его SHA256 и platform_family=source; проверяются
платформа, membership продукта, прибор и наличие канала в native images.
Слово reviewed — заявление ответственного человека, не криптографическое
подтверждение истины. Реальные файлы/бинарник и сами reference проверяются отдельно.

Тайл — одноканальный GTiff с CRS, максимум 65536 пикселей по умолчанию. Явный
предел --max-records до 1000000. Геометрия NPZ без pickle: observed_at_unix,
view_zenith_deg, footprint_km, valid(bool) имеют ту же двумерную форму.
grid_transform — первые шесть affine коэффициентов; grid_crs — идентификатор
CRS. Для reflectance обязателен solar_zenith_deg. Маска каждого канала отдельна.
Координаты вычисляются по CRS растра. Переинтерполяции или автоматического
назначения единого времени/угла/размера пятна нет. МТВЗА здесь не превращается
в точечные наблюдения — до отдельного антенного оператора действует отказ.

Выход — физический JSONL плюс *.manifest.json с SHA256 и происхождением.
Он совместим с pack_observations и реестром Variable, но допуск к обучению
требует норм, оператора наблюдений и независимого QC. Поле quality=1 означает
прохождение переданной бинарной QC-маски, не уверенность в прогнозе. Нельзя
смешивать два экспорта одинаковых ID/ревизий с разными значениями.

## Проверки

52 новых локальных контрактных теста пройдены на синтетических примерах:
read-only SQLite, receipts, CBOR, matrix, partial/no_data, каналы, время,
плохие пути и хэши, физический экспорт Арктики и Электро-Л до pack_observations.
Набор не является проверкой захваченных спутниковых данных или сборки SatDump.
CI дополнительно сверяет реальные checkout точных upstream SHA и обязательные
инструкции агентов. Результат CI смотреть по опубликованному коммиту; наличие
workflow само по себе не означает его успех.

Аудит исходников:
- https://github.com/f2re/arktika-worker/tree/4d744570653ac60c8fd059d10e617c3806a6bafc
- https://github.com/f2re/SatDump/tree/394431e11d9fffe1a73d3e0670fb023ad7562241
Пути проверенных файлов и blob-хэши — в lock-файле. Правила остановки —
в протоколе 05 и agents/executor.md, verification.md, release-auditor.md.

GeoTIFF-мост ограничен драйвером GTiff, отключает PAM и сетевой PROJ. Внешние
маски/aux/overviews не принимаются молча: подготовьте автономный тайл с явной
QC-маской в geometry.valid. Для гарантии read-only на уровне файловой системы
рекомендуется read-only mount архивов; SQLite использует штатные WAL-блокировки.
