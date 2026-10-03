# Источники архитектурных решений

Проверены 3 октября 2026 года. Это ссылки на первичную документацию; они не
подтверждают качество собственной необученной модели.

- **S1. H3: устройство глобальной сетки и 12 пятиугольников.**
  https://h3geo.org/docs/core-library/overview/
- **S2. H3: размеры, неравные площади и количество ячеек.**
  https://h3geo.org/docs/core-library/restable/
  Использовано для сравнения вариантов. Реализованная сетка не является H3.
- **S3. ECMWF: ERA5 data documentation.**
  https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
  Различие 137 гибридных модельных уровней и 37 изобарических выходных уровней.
- **S4. SciPy: SphericalVoronoi.**
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html
  Построение сферических ячеек и вычисление их площадей.

Контракты времени и радиометрии перенесены из предоставленного в разговоре
`meteo_obs_foundation_v0.1.zip`. Региональный плоский кодировщик оттуда не
используется как сферический: в этом репозитории реализовано другое ядро.

## Дополнения 0.2

- Официальное семейство WeatherNext/GraphCast:
  https://github.com/google-deepmind/weathernext
  Ревизия источника: `f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3`.
- Источник ссылок на статистики:
  https://github.com/google-deepmind/weathernext/blob/f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3/docs/weathernext1_graph/graphcast_demo.ipynb
- Состав высотных/приземных величин и 37 уровней:
  https://github.com/google-deepmind/weathernext/blob/f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3/weathernext/weathernext1_graph/graphcast.py
- Проверенный альтернативный реестр (не основной источник 37 уровней):
  https://github.com/microsoft/aurora/blob/main/aurora/normalisation.py
  Blob SHA: `f10a1b56cab4ab0fce47392eef59608c67affad2`.

Нет копирования весов этих моделей. Код импорта в этом проекте написан отдельно.
README WeatherNext различает лицензию кода Apache-2.0 и остальных материалов
CC-BY-4.0, а также требует учитывать условия исходных данных ECMWF/Copernicus.
При получении и распространении статистик сохраняйте авторство и происхождение.
Численные файлы статистик в данный репозиторий пока не включены; период их
расчёта не установлен одним лишь просмотром ссылок и не выдумывается.
