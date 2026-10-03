# Агенты разработки

Девять ролей с раздельной ответственностью и выходными артефактами.
Реестр и разрешённые действия: global_weather/lab/agents.py.
Универсальные инструкции: agents/*.md. Определения Claude Code:
.claude/agents/*.md. Формат соответствует документации:
https://code.claude.com/docs/en/sub-agents

Это определения и ограниченный исполнитель, а не автоматически работающая
команда LLM. Никакие внешние модели или API не вызываются. Для Claude Code
используется собственная установленная среда пользователя и её разрешения.

Пример: «Поручи physics проверить изменение, затем verification проверить
метрики, а release-auditor проверить готовность публикации». Агент инженерии
не может объявить свою реализацию научно верифицированной без второго обзора.

Детерминированное исполнение без LLM:
python -m global_weather.lab.agents
python -m global_weather.lab.agents --role executor --action adaptive --execute

Без --execute команда только показывает план. Загрузки выполняются отдельной
явной командой connectors.acquire с --network. На секреты и удалённые системы
полномочия через эти инструкции не предоставляются.

Агент загрузки также имеет реальные детерминированные действия с протоколом:
python -m global_weather.lab.agents --role data-steward --action download-graphcast --execute --network
python -m global_weather.lab.agents --role data-steward --action plan-era5 --execute --date 2020-01-01
Для download-noaa дополнительно задаются --station и --year. Файлы и agent.json
сохраняются в outputs/lab/acquisition/<id>/. Эти действия отсутствуют в HTTP API.
