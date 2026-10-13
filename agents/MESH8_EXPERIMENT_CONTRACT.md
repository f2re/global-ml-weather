# Контракт ролей для эксперимента mesh8

Идентификатор: `GLOBAL-WEATHER-MESH8-R10`.
Эксперимент: `joint-mesh8-qualification-r10`.
Статус: physical-v2 реализован; Q0 переквалифицирован; Q1–Q3 runtime artifacts и Q4–Q5 CUDA execution pending; `scientific_acceptance=false`.

Контракт дополняет `GLOBAL-WEATHER-OPS-1` и `GLOBAL-WEATHER-REMOTE-1`. Запись роли означает назначение ответственности, а не утверждение о фактически выполненном запуске. Выполнение подтверждается только артефактом, журналом команды, SHA256 и хэшем ревизии.

| Роль | Задача | Обязательное доказательство |
|---|---|---|
| `coordinator` | Заморозить R10, canonical mesh identity, зависимости Q0–Q5 и запрет автоматического перехода к 72 ч | experiment passport, decisions 0018/0019, итоговая матрица gates |
| `data-steward` | Проверить GHCNh/IGRA split, 72-часовой guard, NCEP1 и static sources | successful preflight: dataset verify, source hashes, climate/static fingerprints |
| `normalization` | Подтвердить неизменность GraphCast-норм и train-only station climatology | norm source identity/hash, train-only period, отсутствие fit на validation/test |
| `physics` | Проверить hard hydrostatic/q/surface constraints, pressure-gradient/Coriolis conditioning, stability gate и continuity omega | targeted physical tests, diagnostic physics в Q5 manifest, decision 0019 |
| `model-engineer` | Обеспечить 655362 ячейки, stable identity, chunked graph/physics operators и +3 ч rollout | Q0 geometry artifact, canonical fingerprint, finite/nonzero gradients |
| `radiometry` | Не допустить ложного включения спутниковой supervision в R10 | `satellite_supervision=false`; спутниковый опыт остаётся отдельным |
| `executor` | Выполнить фиксированный `mesh8-r10-runner.py` на CUDA-узле без изменения frozen envelope | `preflight.json`, `execution-evidence.json`, Q4 metrics/complete, measured peak VRAM |
| `verification` | Перезагрузить best checkpoint и провести Q5 до физики/GRIB2 | `batch.json`, issue completion manifests, NPZ/GRIB2 hashes, `final-verification.json` |
| `release-auditor` | Закрыть формулировки и проверить отсутствие научных overclaims | `scientific_acceptance=false`, нет заявления «28 км skill», `next_stage_authorized=false` |

## Уже подтверждено автоматическими проверками

Q0 переквалифицирован на полной сетке с fingerprint `64c3e32f792a42ad7d363635e5217cd3c845743a9b92487fdc226da5f1937aa9`, schema `mesh8-geometry-q1e-7-v1`. Workflow `mesh8-qualification` run `38389968579` выполнил 48 targeted physical/R10 tests и full 655362-cell geometry/graph smoke.

Это закрывает программные обязанности `physics`/`model-engineer` только в части unit/contract/Q0 доказательств. Оно не является доказательством выполнения `data-steward`, `normalization`, `executor` или `verification` на реальных Q2–Q5 runtime artifacts.

## Порядок зависимостей последнего этапа

1. `data-steward` и `normalization` создают/проверяют Q2/Q3 artifacts под canonical stable mesh identity.
2. `physics` и `model-engineer` считаются допущенными по CI только если targeted workflow зелёный на той же source revision.
3. `executor` начинает Q4 только после `mesh8_preflight: ready_for_q4_q5`.
4. `verification` работает только с опубликованным best checkpoint и Q5 frozen inference +3 ч.
5. `release-auditor` принимает только `mesh8_final_check: r10_q4_q5_technically_qualified` и всё равно сохраняет `scientific_acceptance=false`.

Ни одна роль не имеет права автоматически повышать `rollout_hours`, `hidden`, число эпох или менять `mesh_level`. Переход от квалификации +3 ч к 72-часовому mesh8-опыту оформляется новым решением и новым идентификатором эксперимента после фактической CUDA telemetry, streaming rollout/export и независимой skill verification.
