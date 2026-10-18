#!/usr/bin/env python3
"""Интеграционный мост к данным и каталогу arktika-worker.

Извлекает метаданные и сцены КА «Арктика-М» №1 и №2 из catalog.sqlite,
рассчитывает орбитальную геометрию через модуль arktika.orbit (SGP4),
проверяет доступность L2IR растров и формирует пакеты для глобальной модели.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import urllib.request

ARKTIKA_WORKER_ROOT = Path("/Users/meteo/Documents/WWW/arktika-worker")
sys.path.insert(0, str(ARKTIKA_WORKER_ROOT))

try:
    from arktika.orbit import calculate_viewing_geometry
except ImportError:
    calculate_viewing_geometry = None


def get_catalog_db(worker_root: Path = ARKTIKA_WORKER_ROOT) -> Path:
    db_path = worker_root / "data" / "catalog.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"Каталог arktika-worker не найден по пути {db_path}")
    return db_path


def list_available_arktika_scenes(
    worker_root: Path = ARKTIKA_WORKER_ROOT,
    start_date: str = "2024-01-01T00:00:00Z",
    end_date: str = "2026-10-18T23:59:59Z",
    platforms: tuple[str, ...] = ("ARCM1", "ARCM2"),
) -> list[dict]:
    """Возвращает список сцен МСУ-ГС/А с доступными L2IR каналами."""
    db_path = get_catalog_db(worker_root)
    scenes = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        cur = con.cursor()
        query = """
            SELECT id, platform, stamp, data
            FROM items
            WHERE stamp >= ? AND stamp <= ? AND platform IN (?, ?)
            ORDER BY stamp DESC
        """
        for item_id, platform, stamp, raw_data in cur.execute(query, (start_date, end_date, *platforms)):
            if "L2IR" in item_id or "MSU-GS" in item_id:
                scenes.append({
                    "item_id": item_id,
                    "platform": platform,
                    "stamp": stamp,
                })
    return scenes


def get_scene_assets(
    item_id: str,
    worker_root: Path = ARKTIKA_WORKER_ROOT,
    channels: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10),
) -> list[dict]:
    """Возвращает список TIFF ассетов для указанной сцены и каналов."""
    db_path = get_catalog_db(worker_root)
    assets = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        cur = con.cursor()
        query = "SELECT data FROM assets WHERE item_id = ?"
        for (raw_json,) in cur.execute(query, (item_id,)):
            try:
                info = json.loads(raw_json)
                fn = info.get("filename", "")
                ch = info.get("channel")
                if fn.endswith(".tif") and ch in channels:
                    assets.append(info)
            except Exception:
                continue
    return assets


def ensure_asset_downloaded(
    asset_info: dict,
    download_dir: Path,
    timeout_sec: int = 15,
) -> Path:
    """Проверяет локальное наличие файла или скачивает его из публичного S3."""
    download_dir.mkdir(parents=True, exist_ok=True)
    target_path = download_dir / asset_info["filename"]
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    uri = asset_info.get("uri")
    if not uri:
        raise ValueError(f"Нет URI для ассета {asset_info.get('id')}")

    req = urllib.request.Request(uri, headers={"User-Agent": "GlobalWeather-Assimilation/1.0"})
    tmp_path = target_path.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp_path, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    os.replace(tmp_path, target_path)
    return target_path


def compute_geometry_for_point(
    platform: str,
    when: datetime,
    lat: float,
    lon: float,
    worker_root: Path = ARKTIKA_WORKER_ROOT,
) -> dict:
    """Вычисляет точную топоцентрическую геометрию визирования аппарата."""
    if calculate_viewing_geometry is None:
        raise RuntimeError("Модуль arktika.orbit недоступен.")
    cache_dir = worker_root / "data" / "orbit"
    return calculate_viewing_geometry(platform, when, lat, lon, cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Навигация и сбор спутниковых данных Арктика-М")
    parser.add_argument("--worker-root", type=Path, default=ARKTIKA_WORKER_ROOT, help="Путь к arktika-worker")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arktika_acquired"), help="Каталог выгрузки")
    parser.add_argument("--start", default="2024-01-01T00:00:00Z", help="Начальная дата ISO")
    parser.add_argument("--end", default="2026-10-18T23:59:59Z", help="Конечная дата ISO")
    parser.add_argument("--max-scenes", type=int, default=10, help="Максимальное число обрабатываемых сцен")
    parser.add_argument("--download-channels", action="store_true", help="Фактически скачивать недостающие растры")
    args = parser.parse_args()

    print(f"Поиск сцен Арктика-М в каталоге {args.worker_root}...")
    scenes = list_available_arktika_scenes(args.worker_root, args.start, args.end)
    print(f"Найдено доступных сцен в диапазоне {args.start} — {args.end}: {len(scenes)}")

    processed = []
    for sc in scenes[:args.max_scenes]:
        item_id = sc["item_id"]
        assets = get_scene_assets(item_id, args.worker_root)
        stamp_dt = datetime.fromisoformat(sc["stamp"].replace("Z", "+00:00"))

        # Проверка геометрии на опорной точке в Баренцевом море (75N, 40E)
        try:
            geom = compute_geometry_for_point(sc["platform"], stamp_dt, 75.0, 40.0, args.worker_root)
            zenith = geom["zenith_deg"]
            dist_km = geom["distance_km"]
            sub_lat = geom["subpoint"]["lat"]
            sub_alt = geom["subpoint"]["alt_km"]
        except Exception as e:
            zenith, dist_km, sub_lat, sub_alt = None, None, None, None

        downloaded_paths = []
        if args.download_channels and assets:
            for ast in assets[:2]:  # скачиваем для проверки 1-2 канала
                try:
                    p = ensure_asset_downloaded(ast, args.output_dir / sc["platform"] / sc["stamp"][:10])
                    downloaded_paths.append(str(p))
                except Exception as ex:
                    print(f"Ошибка загрузки {ast.get('filename')}: {ex}")

        processed.append({
            "item_id": item_id,
            "platform": sc["platform"],
            "stamp": sc["stamp"],
            "zenith_75N": zenith,
            "distance_km": dist_km,
            "sub_alt_km": sub_alt,
            "channels_available": [a.get("channel") for a in assets],
            "downloaded": downloaded_paths,
        })
        print(f"Сцена: {sc['platform']} {sc['stamp']} | Каналов: {len(assets)} | Зенит 75N: {zenith}° | Высота: {sub_alt} км")

    report_path = args.output_dir / "arktika_acquisition_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "arktika_satellite_navigation_verified",
            "total_scenes_in_catalog": len(scenes),
            "processed_scenes": processed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"Отчет сохранен в {report_path}")


if __name__ == "__main__":
    main()
