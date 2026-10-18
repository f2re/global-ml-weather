#!/usr/bin/env python3
"""Фоновый конвейер загрузки и формирования расширенной выборки 2023-2026.

Выполняет:
1. Загрузку и проверку каталога спутников «Арктика-М» и «Электро-Л» из arktika-worker;
2. Подготовку индексов синоптических сроков (2023-2024 train, 2025 val, 2026 test);
3. Ведение непрерывного журнала подготовки в outputs/dataset_pipeline_2023_2026.log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from global_weather.joint_observation_data import common_windows, BOUNDARIES_2023_2026
from scripts.fetch_arktika_worker_data import (
    list_available_arktika_scenes,
    get_scene_assets,
    compute_geometry_for_point,
    ARKTIKA_WORKER_ROOT,
)


def log(msg: str, log_file: Path | None = None):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_pipeline(
    output_dir: Path,
    log_file: Path,
    worker_root: Path = ARKTIKA_WORKER_ROOT,
    stride_hours: int = 6,
):
    log("=== Старт фонового конвейера подготовки выборки 2023–2026 ===", log_file)

    # 1. Формирование окон наблюдений по расширенному календарю
    log(f"Генерация синоптических окон с шагом {stride_hours} ч...", log_file)
    windows = common_windows(horizon_hours=72, issue_stride_hours=stride_hours, boundaries=BOUNDARIES_2023_2026)
    train_w = [w for w in windows if w["split"] == "train"]
    val_w = [w for w in windows if w["split"] == "validation"]
    test_w = [w for w in windows if w["split"] == "test"]

    log(f"Сформировано окон: Всего={len(windows)}, Train={len(train_w)}, Val={len(val_w)}, Test={len(test_w)}", log_file)

    # Сохранение расписания выборки
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "windows_2023_2026.json"
    with open(windows_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "extended-windows-2023-2026-v1",
            "stride_hours": stride_hours,
            "horizon_hours": 72,
            "total_windows": len(windows),
            "splits_summary": {
                "train_count": len(train_w),
                "val_count": len(val_w),
                "test_count": len(test_w),
            },
            "windows": windows,
        }, f, indent=2, ensure_ascii=False)
    log(f"Окна сохранены в {windows_path}", log_file)

    # 2. Сканирование спутниковых данных Арктика-М через arktika-worker
    log(f"Сканирование каталога arktika-worker: {worker_root}...", log_file)
    scenes = list_available_arktika_scenes(
        worker_root=worker_root,
        start_date="2024-01-01T00:00:00Z",
        end_date="2026-10-18T23:59:59Z",
    )
    log(f"Найдено спутниковых сцен: {len(scenes)}", log_file)

    # 3. Индексация сцен и проверка геометрии
    log("Проверка орбитальной геометрии SGP4 и доступности каналов...", log_file)
    indexed_scenes = []
    for idx, sc in enumerate(scenes):
        stamp_dt = datetime.fromisoformat(sc["stamp"].replace("Z", "+00:00"))
        assets = get_scene_assets(sc["item_id"], worker_root)
        try:
            geom = compute_geometry_for_point(sc["platform"], stamp_dt, 75.0, 40.0, worker_root)
            zenith = geom["zenith_deg"]
            sub_alt = geom["subpoint"]["alt_km"]
        except Exception:
            zenith, sub_alt = None, None

        indexed_scenes.append({
            "item_id": sc["item_id"],
            "platform": sc["platform"],
            "stamp": sc["stamp"],
            "channels_count": len(assets),
            "zenith_75N": zenith,
            "alt_km": sub_alt,
        })
        if (idx + 1) % 100 == 0 or idx == len(scenes) - 1:
            log(f"Проиндексировано спутниковых сцен: {idx + 1}/{len(scenes)}", log_file)

    scenes_manifest = output_dir / "arktika_scenes_indexed.json"
    with open(scenes_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "arktika-scenes-index-1",
            "indexed_count": len(indexed_scenes),
            "scenes": indexed_scenes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, ensure_ascii=False)
    log(f"Индекс спутниковых сцен сохранен в {scenes_manifest}", log_file)

    log("=== Формирование расширенной выборки успешно завершено ===", log_file)


def main():
    parser = argparse.ArgumentParser(description="Фоновый запуск подготовки выборки 2023-2026")
    parser.add_argument("--output-dir", type=Path, default=Path("data/extended_sample_2023_2026"))
    parser.add_argument("--log-file", type=Path, default=Path("outputs/dataset_pipeline_2023_2026.log"))
    parser.add_argument("--stride-hours", type=int, default=6)
    args = parser.parse_args()

    run_pipeline(args.output_dir, args.log_file, stride_hours=args.stride_hours)


if __name__ == "__main__":
    main()
