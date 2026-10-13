# Контракт ролей для эксперимента mesh8

Идентификатор: `GLOBAL-WEATHER-MESH8-R10`.
Эксперимент: `joint-mesh8-qualification-r10`.
Статус: инженерная квалификация, `scientific_acceptance=false`.

Контракт дополняет `GLOBAL-WEATHER-OPS-1` и `GLOBAL-WEATHER-REMOTE-1`. Запись роли в таблице означает назначение ответственности, а не утверждение о фактически выполненном запуске. Выполнение подтверждается только артефактом, журналом команды и хэшем ревизии.

| Роль | Задача | Обязательное доказательство |
|---|---|---|
| `coordinator` | Заморозить конфигурацию R10, зависимости стадий и критерии перехода к 72 ч | паспорт эксперимента и итоговая матрица gates |
| `data-steward` | Проверить GHCNh/IGRA split, 72-часовой guard, исходные NOAA NCEP1 и статические источники | хэши dataset/norm/climatology/static и отчёт `verify()` |
| `normalization` | Подтвердить неизменность GraphCast-норм и train-only станционной климатологии | hash норм, период train и отсутствие fit на validation/test |
| `physics` | Проверить гидростатику, влажность, геометрическую маску, continuity-диагностику omega | физические тесты и diagnostic physics в forecast manifest |
| `model-engineer` | Обеспечить 655362 ячейки, chunked edge exchange и ограниченный 3-часовой rollout | geometry report, nonzero gradients и отсутствие nonfinite |
| `radiometry` | Не допустить ложного включения спутниковой ветви в R10 | явная запись `satellite_supervision=false`; спутник относится к следующему опыту |
| `executor` | Выполнить R10 на CUDA-узле в замороженной ревизии | `progress.json`, `metrics.json`, `complete.json`, peak VRAM/RSS |
| `verification` | Перезагрузить best checkpoint и провести 3-часовой inference до GRIB2 | batch manifest, forecast NPZ, GRIB2 inventory, hashes |
| `release-auditor` | Проверить формулировки разрешения и научного статуса | отсутствие заявления «20 км skill», `scientific_acceptance=false` |

## Порядок зависимостей

1. `data-steward`, `normalization` и `physics` закрывают входные контракты.
2. `model-engineer` закрывает геометрию и вычислительный граф.
3. `executor` получает разрешение только после пунктов 1–2.
4. `verification` работает только с неизменяемым best checkpoint.
5. `release-auditor` закрывает эксперимент после проверки всех предыдущих артефактов.

Ни одна роль не имеет права автоматически повышать `rollout_hours`, `hidden`, число эпох или менять `mesh_level`. Переход от квалификации 3 ч к 72-часовому mesh8-опыту оформляется новым решением и новым идентификатором эксперимента.
