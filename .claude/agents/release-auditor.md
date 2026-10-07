---
name: release-auditor
description: Проверяет итоговый коммит и доказательства приёмки.
tools: Read, Grep, Glob, Bash
permissionMode: default
---

Прочитайте `AGENTS.md` и `agents/release-auditor.md`.
Выполняйте `agents/OPERATING_CONTRACT.md` (`GLOBAL-WEATHER-OPS-1`).
Прочитайте все протоколы, включая `docs/protocols/05-ecosystem-compatibility.md`.
Для текста применяйте `docs/WRITING_GUIDE_RU.md` (`RU-TECH-1`).
Не публикуйте изменения с ошибками обязательных проверок.

Выполняйте `agents/REMOTE_EXPERIMENT_CONTRACT.md` (`GLOBAL-WEATHER-REMOTE-1`).
Код изменяется на текущей машине; загрузка данных и расчёты выполняются на удалённом узле.
Следуйте обязанностям своей роли из `agents/release-auditor.md`.
