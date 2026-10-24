# Индекс актуальных архитектурных схем

Канонический комплект — SVG, чтобы подписи оставались читаемыми на GitHub и при масштабировании.

| № | Файл | Назначение |
|---:|---|---|
| 1 | [01-global-architecture-overview.svg](images/architecture/01-global-architecture-overview.svg) | Две параллельные ветви и общая инфраструктура |
| 2 | [02-input-data-and-feature-preparation.svg](images/architecture/02-input-data-and-feature-preparation.svg) | Источники, причинность, нормализация и сферическая геометрия |
| 3 | [03-adaptive-multimodal-branch.svg](images/architecture/03-adaptive-multimodal-branch.svg) | Adaptive / Multimodal с `K=8` |
| 4 | [04-satellite-spatiotemporal-encoder.svg](images/architecture/04-satellite-spatiotemporal-encoder.svg) | ResU-Net, MaskedConvGRU, microwave MLP/GRU и проекция признаков |
| 5 | [05-feature-fusion-and-compressed-vertical.svg](images/architecture/05-feature-fusion-and-compressed-vertical.svg) | Source fusion и attention-сжатие вертикали |
| 6 | [06-global-adaptive-forecast-core.svg](images/architecture/06-global-adaptive-forecast-core.svg) | AdaptiveProcessor, иерархия графа и декодирование |
| 7 | [07-decoder-and-output-heads.svg](images/architecture/07-decoder-and-output-heads.svg) | Joint / Hydrostatic Mesh-8 на полной 37-уровневой вертикали |
| 8 | [08-full-training-analysis-and-forecast-flow.svg](images/architecture/08-full-training-analysis-and-forecast-flow.svg) | Активная физика, evaluation helpers, postprocessing и issuance |

`01-global-architecture-overview.webp` сохраняется как растровый preview для README. Каноническая редактируемая версия этой схемы — SVG.

Основной документ: [ARCHITECTURE_VISUAL_GUIDE.md](ARCHITECTURE_VISUAL_GUIDE.md).
