# R8: поиск пространственной сезонной климатологии

Дата проверки: 9 октября 2026 года.
Основание: прикреплённый оператором перечень 20 NOAA NetCDF и двух коллекций NASA MERRA-2 V2.
Проверены официальные веб-каталоги и спецификации.
Бинарные данные локально не скачивались; расчёты и испытания модели не выполнялись.
Этот документ не меняет действующие контракты и не подтверждает выбор нового источника.
После исследования выбран ограниченный опыт NCEP-1 по [решению 0011](../decisions/0011-seasonal-climate-context.md).
Роль источника — климатические признаки; это решение не устанавливает его превосходство по точности.

## Рекомендация

Сохранить фиксированные погодные масштабы GraphCast на 37 уровнях.
Отдельно проверить пространственно-сезонное среднее как ограниченный климатический контекст нового опыта.
Климатическое среднее не является измерением, плотной целью или заменой маски пропуска.
Выбор источника и его новой роли требует отдельного решения до обучения.
Фактические входы и цели должны оставаться наблюдательными.

Для быстрого ограниченного опыта наиболее подтверждён доступ к готовым месячным средним NCEP-1 за 1991–2020.
Комплект пяти согласованных величин имеет 45962671 байт и не требует авторизации Earthdata.
Для более широкой вертикали перспективен 20CRv3, но проверенный каталог содержит большие временные ряды, а не готовую климатологию из 12 месяцев.
MERRA-2 V2 полезна как более современный пространственный климатический источник, однако её готовая профильная климатология имеет только 12 уровней.
Улучшение точности модели этими источниками пока не установлено.

## Что подтверждено

| Источник | Подтверждённое покрытие | Практический смысл |
|---|---|---|
| NCEP-1, `air.mon.ltm.1991-2020.nc` | 12 месяцев, 17 уровней 1000–10 гПа; 73×144 точек | Готовый небольшой сезонный контекст температуры |
| NCEP-2 | Официальная документация указывает 17 уровней | Верхние уровни модели до 1 гПа не покрыты |
| 20CRv2 | Официальные метаданные указывают 24 уровня 1000–10 гПа | Старый источник; конкретные файлы приложения требуют проверки переменных |
| 20CRv3 | Официальное описание указывает 28 уровней до 1 гПа | Предпочтительная вертикаль при подтверждении нужного файла и периода |
| MERRA-2 M2TCNPLTM V2 | 1991–2020; 12 уровней от 1000 до 10 гПа | Климатические средние T, U, V, RH и QV; не 42 уровня обычных продуктов MERRA-2 |

Источники: [NCEP-1 фактические атрибуты](https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/air.mon.ltm.1991-2020.nc.html), [NCEP-2 NOAA](https://upwell.pfeg.noaa.gov/erddap/griddap/noaa_psl_9c63_f301_a7da.html), [20CRv2 NOAA](https://data.noaa.gov/metaview/page?view=ISO19115Components-HTMLTable&xml=NOAA%2Foar%2Fesrl%2Fpsd%2Fiso%2Fxml%2FNOAA-CIRES_20th_Century_Reanalysis_V2.xml), [20CRv3 NOAA](https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.pressure.html), [NASA M2TCNPLTM V2](https://data.nasa.gov/dataset/merra-2-tavgc-3d-ltm-np-3d-long-term-mean-3-dimensional-meteorological-fields-based-on-199-8d6f1).

NASA определяет STD как межгодовой разброс месячных средних.
Это не σ отдельных погодных состояний и не разброс ансамбля 20CR.
Обычные MERRA-2 продукты имеют другую вертикаль; её нельзя приписывать готовой климатологии.
MERRA-2 ассимилирует наблюдения и не является независимой наблюдательной истиной. [Спецификация NASA](https://gmao.gsfc.nasa.gov/media/publications/zbly36ziNFDFbmYmvhQeVqPhUo/Collow1466.pdf).

## Конкретные файлы и доступ

Проверенная [карточка NCEP-1](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/air.mon.ltm.1991-2020.nc) указывает 9709277 байт и HTTPServer/OPeNDAP/NCSS.
Прямой HTTP-путь: `https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/Monthlies/pressure/air.mon.ltm.1991-2020.nc`.
Температура имеет единицу `degC`; при импорте в K прибавьте 273,15 ровно один раз.
Период берите из `climo_period`, а не условного года 0001 координаты времени.
Файл содержит `valid_yr_count`; полнота должна проверяться отдельно.

В [каталоге NCEP-1](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html) также подтверждены `hgt.mon.ltm.1991-2020.nc` — 8,874 МБ и `air.day.ltm.1991-2020.nc` — 272,9 МБ.
Значения каталога округлены; точные байты и единицы остальных переменных требуют удалённого чтения заголовков.

[Каталог 20CRv3 prsSI](https://psl.noaa.gov/thredds/catalog/Datasets/20thC_ReanV3/Monthlies/prsSI/catalog.html) показывает `air.mon.mean.nc` — 9,040 ГБ, `hgt.mon.mean.nc` — 8,855 ГБ, `uwnd.mon.mean.nc` — 12,94 ГБ, `vwnd.mon.mean.nc` — 14,18 ГБ и `shum.mon.mean.nc` — 8,448 ГБ.
Имена этих файлов не означают готовые 12 климатических месяцев.
Сначала проверьте NCSS-подмножество нужного базового периода; не начинайте полную загрузку десятков гигабайт.

NASA публикует 12 гранул M2TCNPLTM V2 через GES DISC.
Имена гранул и точные размеры в этой проверке не установлены; не угадывайте имена файлов.
CMR-запрос гранул оказался недоступен через веб-инструмент.
Для получения данных проверьте действующий маршрут Earthdata и доступ пользователя отдельно.
GES DISC требует Earthdata Login и разрешения приложения; старый адрес нельзя считать подтверждённым маршрутом загрузки.
Официальный сервер сообщает о переходе доступа в Earthdata Cloud. [Доступ NASA](https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/), [Earthdata Login](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget).

## Подтверждённый комплект NCEP-1 для ограниченного опыта

Все пять файлов относятся к одной базе 1991–2020 и содержат 12 климатических месяцев.
Горизонтальная сетка — 73×144, шаг 2,5°.
Важное уточнение: удельная влажность имеет только восемь уровней до 300 гПа.
Температура, ветер и высота имеют 17 уровней до 10 гПа.

| Файл | Байты по карточке | Единица источника | Допустимое преобразование |
|---|---:|---|---|
| `air.mon.ltm.1991-2020.nc` | 9709277 | degC | Прибавить 273,15 для K |
| `hgt.mon.ltm.1991-2020.nc` | 8874993 | m | Умножить на g0 один раз для м²/с² |
| `uwnd.mon.ltm.1991-2020.nc` | 10927383 | m/s | Сохранить м/с |
| `vwnd.mon.ltm.1991-2020.nc` | 11405646 | m/s | Сохранить м/с |
| `shum.mon.ltm.1991-2020.nc` | 5045372 | grams/kg | Умножить на 0,001 для кг/кг |

Сумма пяти размеров — 45962671 байт; она не заменяет проверку фактически полученных байтов и SHA256.
Карточки: [T](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/air.mon.ltm.1991-2020.nc), [H](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/hgt.mon.ltm.1991-2020.nc), [U](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/uwnd.mon.ltm.1991-2020.nc), [V](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/vwnd.mon.ltm.1991-2020.nc), [Q](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis/Monthlies/pressure/catalog.html?dataset=Datasets/ncep.reanalysis/Monthlies/pressure/shum.mon.ltm.1991-2020.nc).

Метаданные остальных величин: [H](https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/hgt.mon.ltm.1991-2020.nc.html), [U](https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/uwnd.mon.ltm.1991-2020.nc.html), [V](https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/vwnd.mon.ltm.1991-2020.nc.html), [Q](https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/pressure/shum.mon.ltm.1991-2020.nc.html).
HTTPServer использует указанный выше базовый путь и эти точные имена файлов.
Все пять переменных имеют `valid_yr_count` и fill value `-9.96921E36`.
Каталог допускает среднее при низкой полноте; перед использованием задайте собственный порог подтверждённых лет.
Например, порог 20 лет из 30 является предложением для нового опыта, а не свойством поставщика.
В принятом ограниченном R8 заранее выбран более строгий порог 25 лет из 30.
Отсутствующая поддержка q выше 300 гПа не заменяется нулём или климатическим продолжением.

NOAA просит указать PSL как источник в публикации.
Это требование атрибуции не является заявлением о единой лицензии всех внешних наборов. [Правило NOAA](https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.pressure.html).

## План допуска и сравнительного опыта

1. На вычислительном узле получите только разрешённые заголовки и ограниченные файлы выбранного источника.
2. Сохраните источник, период, байты, SHA256, единицы, fill values и давление каждой переменной.
3. Проверьте сезонный календарь, долготы, полюса и подземные уровни.
4. Разделите климатический контекст и наблюдательные маски; не экстраполируйте неподдержанную вертикаль.
5. Сохраните погодную σ GraphCast; межгодовую STD используйте только как отдельную диагностику.
6. Зафиксируйте новый опыт с одинаковыми наблюдениями и бюджетом контрольного сравнения.
7. Выбирайте эпоху по реальным измерениям validation; внешние реанализы оценивайте только после фиксации весов.

Ошибки доступа к карточкам NCEP-2 и 20CRv2 не удостоверяют отсутствие файлов.
Приложенные размеры и вертикальные границы каждого из 20 файлов пока не подтверждены побайтово.
Последующая удалённая проверка должна сохранять эти отрицательные результаты и ограничения.
