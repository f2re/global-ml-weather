# 0023. Multimodal analysis -> Hydrostatic forecast -> guarded feedback

Дата: 2026-10-24. Статус: принято как экспериментальная архитектура; skill не заявлен.

## Решение

`MultimodalWeatherModel` используется как причинный оператор усвоения разнородных наблюдений. Его latent analysis переводится обучаемым `LatentToPressureBridge` в 37-уровневое скрытое состояние `JointObservationModel`/`HydrostaticFlowProfileModel`. После T0 прогнозная ветвь развивается автономно.

Повторный ingest GHCNh/IGRA в Joint при coupled-запуске запрещён, потому что эти же факты уже могли войти в Multimodal analysis.

После поступления фактических наблюдений прогноз верифицируется. Свежий sample допускается в online optimizer только после двух независимых immutable receipts: target SHA должен совпасть с зарегистрированной target-версией (`fact_admitted_verified`), а исходный прогноз этого же sample должен получить error receipt (`forecast_verified_sample`). Для этого canonical feedback path использует `VerifiedReplayPlanner`.

Train-роли могут образовать candidate update через существующие `CampaignStore` и `StepTrainer`; validation/test/guard никогда не используются для optimizer step. Candidate публикуется только после guard-проверки. Отклонённый candidate не создаёт checkpoint generation или training_event.

## Темпы обучения

Fast loop обновляет bridge и прогнозную динамику/головы. Slow loop реже допускает обновление multimodal encoder/fusion при отдельной защите от деградации. Исторический replay обязателен для снижения catastrophic forgetting.

## Следствия

Coupled checkpoint несовместим с checkpoint каждой исходной ветви без bridge. До paired-обучения bridge новая архитектура не является заменой квалифицированного R10. 72-часовой coupled skill должен подтверждаться отдельным опытом.
