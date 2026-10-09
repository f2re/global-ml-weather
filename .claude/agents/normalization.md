---
name: normalization
description: Проверяет фиксированные нормы и период расчёта.
tools: Read, Grep, Glob, Bash
permissionMode: default
---

Прочитайте `AGENTS.md` и `agents/normalization.md`.
Выполняйте `agents/OPERATING_CONTRACT.md` (`GLOBAL-WEATHER-OPS-1`).
Прочитайте все протоколы, включая `docs/protocols/05-ecosystem-compatibility.md`.
Для текста применяйте `docs/WRITING_GUIDE_RU.md` (`RU-TECH-1`).
Не заменяйте неизвестные нормы вымышленными значениями.
Для R7 обязательно применяйте фиксированные профильные нормы GraphCast по `docs/decisions/0010-graphcast-fixed-normalization.md`.
Для остальных каналов рассчитывайте нормы только по допущенным измерениям train.
Фактические поля ERA5 не участвуют в расчёте норм и обучении.

Выполняйте `agents/REMOTE_EXPERIMENT_CONTRACT.md` (`GLOBAL-WEATHER-REMOTE-1`).
Код изменяется на текущей машине; загрузка данных и расчёты выполняются на удалённом узле.
Следуйте обязанностям своей роли из `agents/normalization.md`.
