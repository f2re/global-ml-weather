# Коннекторы данных

Сетевая загрузка отделена от веб-стенда и требует явного --network.
Ни один файл не допускается в обучение только потому, что он скачан.

## Пользовательские спутниковые проекты: 0.3.1

Добавлены project_bridge (нативный каталог Арктики и CBOR SatDump),
physical_bridge (проверенный числовой GeoTIFF + геометрия → JSONL), preflight
(закреплённые ревизии и агентские правила). Полный контракт и команды —
[PROJECT_COMPATIBILITY.md](PROJECT_COMPATIBILITY.md).

Арктика читается без импорта её Store и без изменения jobs/settings. SatDump
проверяется именно по release/1.2.2. Обнаруженные raw/unknown/assumed не
подменяются кельвинами. MTVZA/hrpt30 читается как транспортный продукт и остаётся
заблокированным для точечного экспорта до калибровки и антенного оператора.
Электро-Л через msu_gs поддерживается на уровне native metadata и проверенного
физического экспорта; ARCM-only каталог Арктики не назван загрузчиком Электро-Л.
Эти адаптеры работают локальными командами, не кнопкой запуска произвольного URL.

## Существующие источники

JSONL: станции/аэрология и физические спутниковые записи с масками и временем.
Инспекция файла не заменяет независимый QC и проверку радиометрии.

GraphCast: ограниченная HTTPS-загрузка mean/std и SHA256, затем импорт норм.
NOAA ISD: годовой CSV, QC T/Td/ветра/MSLP и преобразование в JSONL.
ERA5 CDS: формирование запроса и отдельное выполнение через cdsapi.

```bash
python -m global_weather.connectors.acquire graphcast --network --output data/norms
python -m global_weather.import_climatology --mean data/norms/mean_by_level.nc \
  --std data/norms/stddev_by_level.nc --output data/graphcast_base.json
python -m global_weather.connectors.acquire noaa-isd --network \
  --station 26063099999 --year 2020 --output data/isd.csv
python -m global_weather.connectors.acquire era5-request \
  --date 2020-01-01 --output data/era5-request.json
python -m global_weather.connectors.acquire era5-download --network \
  --date 2020-01-01 --output data/era5-pressure.nc
```

CDS требует ключа вне репозитория и принятия условий; для поверхности добавить
--surface. Пример идентификатора NOAA не удостоверяет наличие архива.
Ветер переводится из направления «откуда»; без направления используется только
штиль. Сомнительный QC пропускается; осадки не делятся на произвольные часы.
Время получения архива не является историческим временем поступления сообщений.

Загрузка сохраняет URL, SHA256, время и content_verified=false. Публичный
загрузчик ограничивает хосты HTTPS, объём и перенаправления и не перезаписывает
файлы. Глобальный день 37 уровней ERA5 — не маленькая пробная загрузка.
Реальные сетевые загрузки и захваченные спутниковые продукты этой доработкой
не испытывались. Полноценная совместимость эксплуатации требует их проверки.

Первичная документация: https://cds.climate.copernicus.eu/how-to-api ;
https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database ;
https://www.sqlite.org/uri.html ; https://www.sqlite.org/wal.html ;
https://rasterio.readthedocs.io/en/stable/topics/masks.html .
