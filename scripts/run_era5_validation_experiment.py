#!/usr/bin/env python3
"""Научный фоновый эксперимент по составлению полей прогноза и сравнению с реанализом ERA5.

Проводит независимое сопоставление прогнозов с эталонными полями реанализа ERA5:
  - 13 барических уровней: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 гПа;
  - 5 профильных величин: температура T (K), геопотенциал Z (м²/с²), зональный ветер U (м/с),
    меридиональный ветер V (м/с), удельная влажность Q (кг/кг);
  - 4 приземных поля: температура на 2 м T2M (K), ветер на 10 м U10/V10 (м/с),
    давление на уровне моря MSLP (Па);
  - Метрики: взвешенные по косинусу широты RMSE, MAE, Mean Bias, ACC и Skill Score.

Эксперимент полностью соблюдает контракты:
  - ERA5 используется исключительно как внешняя диагностическая оценка, а не для обучения;
  - Никаких будущих данных в начальное состояние не подмешивается;
  - scientific_acceptance: false (исследовательский эксперимент).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import xarray as xr


VARIABLES_PROFILE = [
    ("temperature", "T", "K"),
    ("geopotential", "Z", "m2 s-2"),
    ("u_component_of_wind", "U", "m s-1"),
    ("v_component_of_wind", "V", "m s-1"),
    ("specific_humidity", "Q", "kg kg-1"),
]

VARIABLES_SURFACE = [
    ("2m_temperature", "T2M", "K"),
    ("mean_sea_level_pressure", "MSLP", "Pa"),
    ("10m_u_component_of_wind", "U10", "m s-1"),
    ("10m_v_component_of_wind", "V10", "m s-1"),
]


def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {msg}", flush=True)


def compute_weighted_metrics(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray):
    """Вычисление взвешенных по площади ячеек (косинусу широты) метрик ошибки."""
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.any():
        return {"rmse": None, "mae": None, "bias": None, "count": 0}

    diff = (pred - truth)[valid]
    w = weights[valid]
    w_sum = np.sum(w)
    if w_sum <= 0:
        return {"rmse": None, "mae": None, "bias": None, "count": 0}

    w_norm = w / w_sum
    bias = float(np.sum(w_norm * diff))
    mae = float(np.sum(w_norm * np.abs(diff)))
    rmse = float(np.sqrt(np.sum(w_norm * (diff ** 2))))

    # Коэффициент корреляции аномалий (ACC) относительно пространственного среднего
    truth_anom = truth[valid] - np.sum(w_norm * truth[valid])
    pred_anom = pred[valid] - np.sum(w_norm * pred[valid])
    denom = np.sqrt(np.sum(w_norm * (truth_anom ** 2)) * np.sum(w_norm * (pred_anom ** 2)))
    acc = float(np.sum(w_norm * pred_anom * truth_anom) / denom) if denom > 1e-12 else None

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "acc": acc,
        "count": int(np.sum(valid)),
    }


def run_experiment(era5_path: str, output_dir: str, forecast_nc: str | None = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log(f"Запуск эксперимента валидации полей против реанализа ERA5: {era5_path}")
    ds_era5 = xr.open_dataset(era5_path, decode_times=False)

    lats = ds_era5.lat.values
    lons = ds_era5.lon.values
    levels = ds_era5.level.values
    n_lat, n_lon = len(lats), len(lons)
    log(f"Сетка реанализа: {n_lat} x {n_lon} ({len(levels)} барических уровней: {levels.tolist()})")

    # Матрица весов по косинусу широты
    lat_rad = np.deg2rad(lats)
    weights_1d = np.cos(lat_rad)
    weights_2d = np.repeat(weights_1d[:, None], n_lon, axis=1)

    time_steps = ds_era5.sizes["time"]
    log(f"Доступно временных шагов в ERA5: {time_steps}")

    # Загрузка или генерация прогноза
    ds_fc = None
    if forecast_nc and os.path.exists(forecast_nc):
        log(f"Загрузка ранее рассчитанного прогноза: {forecast_nc}")
        ds_fc = xr.open_dataset(forecast_nc, decode_times=False)

    results = []

    # Анализ шагов заблаговременности (T+6, T+12, T+18, T+24...)
    # T0 соответствует индексу 1 (после 6-часового окна наблюдения T-6, T0)
    base_idx = 1
    leads_available = range(base_idx + 1, time_steps)

    for step_idx in leads_available:
        lead_hours = (step_idx - base_idx) * 6
        log(f"--- Обработка заблаговременности T+{lead_hours} ч (индекс {step_idx}) ---")

        # 1. Приземные поля
        for var_key, var_abbr, units in VARIABLES_SURFACE:
            if var_key not in ds_era5:
                continue

            truth = ds_era5[var_key].isel(batch=0, time=step_idx).values
            persist = ds_era5[var_key].isel(batch=0, time=base_idx).values

            # Метрики сохранения начального состояния (Persistence)
            persist_metrics = compute_weighted_metrics(persist, truth, weights_2d)

            # Метрики модели (если доступен прогноз)
            fc_metrics = None
            skill_score = None
            if ds_fc is not None and var_key in ds_fc:
                fc_lead_idx = step_idx - base_idx - 1
                if fc_lead_idx < ds_fc.sizes["time"]:
                    fc_field = ds_fc[var_key].isel(batch=0, time=fc_lead_idx).values
                    fc_metrics = compute_weighted_metrics(fc_field, truth, weights_2d)
                    if fc_metrics["rmse"] and persist_metrics["rmse"] and persist_metrics["rmse"] > 1e-12:
                        skill_score = 1.0 - (fc_metrics["rmse"] / persist_metrics["rmse"])

            row = {
                "kind": "surface",
                "variable": var_abbr,
                "variable_full": var_key,
                "units": units,
                "level_hpa": "surface",
                "lead_hours": lead_hours,
                "persistence": persist_metrics,
                "model_forecast": fc_metrics,
                "skill_score_vs_persistence": skill_score,
            }
            results.append(row)
            log(f"  [{var_abbr}] Persist RMSE: {persist_metrics['rmse']:.3f} {units} | ACC: {persist_metrics['acc']:.3f}"
                + (f" | Model RMSE: {fc_metrics['rmse']:.3f} | Skill: {skill_score:+.2%}" if fc_metrics and fc_metrics['rmse'] else ""))

        # 2. Профильные поля на барических уровнях
        for var_key, var_abbr, units in VARIABLES_PROFILE:
            if var_key not in ds_era5:
                continue

            for lvl_idx, lvl in enumerate(levels):
                truth = ds_era5[var_key].isel(batch=0, time=step_idx, level=lvl_idx).values
                persist = ds_era5[var_key].isel(batch=0, time=base_idx, level=lvl_idx).values

                persist_metrics = compute_weighted_metrics(persist, truth, weights_2d)

                fc_metrics = None
                skill_score = None
                if ds_fc is not None and var_key in ds_fc:
                    fc_lead_idx = step_idx - base_idx - 1
                    if fc_lead_idx < ds_fc.sizes["time"]:
                        fc_field = ds_fc[var_key].isel(batch=0, time=fc_lead_idx, level=lvl_idx).values
                        fc_metrics = compute_weighted_metrics(fc_field, truth, weights_2d)
                        if fc_metrics["rmse"] and persist_metrics["rmse"] and persist_metrics["rmse"] > 1e-12:
                            skill_score = 1.0 - (fc_metrics["rmse"] / persist_metrics["rmse"])

                row = {
                    "kind": "profile",
                    "variable": var_abbr,
                    "variable_full": var_key,
                    "units": units,
                    "level_hpa": int(lvl),
                    "lead_hours": lead_hours,
                    "persistence": persist_metrics,
                    "model_forecast": fc_metrics,
                    "skill_score_vs_persistence": skill_score,
                }
                results.append(row)

                if lvl in (850, 500, 250):
                    log(f"  [{var_abbr}@{lvl}hPa] Persist RMSE: {persist_metrics['rmse']:.3f} {units} | ACC: {persist_metrics['acc']:.3f}"
                        + (f" | Model RMSE: {fc_metrics['rmse']:.3f} | Skill: {skill_score:+.2%}" if fc_metrics and fc_metrics['rmse'] else ""))

    ds_era5.close()
    if ds_fc:
        ds_fc.close()

    # Сохранение результатов
    json_path = output_path / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "era5-validation-experiment-v1",
                "experiment_time_utc": datetime.now(timezone.utc).isoformat(),
                "era5_source": str(era5_path),
                "forecast_source": str(forecast_nc) if forecast_nc else None,
                "scientific_acceptance": False,
                "metrics": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log(f"Полные метрики сохранены в: {json_path}")

    # Экспорт сводной таблицы CSV
    def fmt(val, prec=4):
        return f"{val:.{prec}f}" if val is not None else ""

    csv_path = output_path / "metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("kind,variable,level_hpa,lead_hours,persist_rmse,persist_bias,persist_acc,model_rmse,model_bias,model_acc,skill_score\n")
        for r in results:
            p = r["persistence"]
            m = r["model_forecast"] or {}
            ss = fmt(r['skill_score_vs_persistence'])
            p_rmse = fmt(p.get('rmse'))
            p_bias = fmt(p.get('bias'))
            p_acc = fmt(p.get('acc'))
            m_rmse = fmt(m.get('rmse'))
            m_bias = fmt(m.get('bias'))
            m_acc = fmt(m.get('acc'))
            f.write(f"{r['kind']},{r['variable']},{r['level_hpa']},{r['lead_hours']},"
                    f"{p_rmse},{p_bias},{p_acc},{m_rmse},{m_bias},{m_acc},{ss}\n")
    log(f"Сводная CSV-таблица сохранена в: {csv_path}")

    # Генерация Markdown отчета
    report_path = output_path / "VALIDATION_REPORT.md"
    generate_markdown_report(report_path, results, era5_path, forecast_nc)
    log(f"Итоговый научно-технический отчет сохранен в: {report_path}")

    log("=== ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН ===")


def generate_markdown_report(report_path: Path, results: list, era5_path: str, forecast_nc: str | None):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Научно-технический отчет: Валидация полей прогноза по реанализу ERA5\n\n")
        f.write(f"- **Дата формирования**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"- **Эталонный источник**: `{era5_path}` (ERA5 0.25°)\n")
        if forecast_nc:
            f.write(f"- **Источник прогноза**: `{forecast_nc}`\n")
        f.write("- **Статус**: Исследовательская валидация (`scientific_acceptance: false`)\n\n")

        f.write("## 1. Сводные метрики приземных полей\n\n")
        f.write("| Переменная | Горизонт | Инерция RMSE | Инерция ACC | Модель RMSE | Модель ACC | Прирост (Skill) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            if r["kind"] == "surface":
                p = r["persistence"]
                m = r["model_forecast"] or {}
                m_rmse = f"{m.get('rmse', 0):.2f}" if m.get('rmse') is not None else "—"
                m_acc = f"{m.get('acc', 0):.3f}" if m.get('acc') is not None else "—"
                ss = f"{r['skill_score_vs_persistence']:+.1%}" if r["skill_score_vs_persistence"] is not None else "—"
                f.write(f"| {r['variable']} ({r['units']}) | T+{r['lead_hours']}ч | {p['rmse']:.2f} | {p['acc']:.3f} | {m_rmse} | {m_acc} | {ss} |\n")

        f.write("\n## 2. Ключевые тропосферные уровни (Z500, T850, V250, Q700)\n\n")
        f.write("| Переменная / Уровень | Горизонт | Инерция RMSE | Инерция ACC | Модель RMSE | Модель ACC | Прирост (Skill) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        key_levels = {("Z", 500), ("T", 850), ("U", 250), ("V", 250), ("Q", 700), ("Z", 850), ("T", 500)}
        for r in results:
            if r["kind"] == "profile" and (r["variable"], r["level_hpa"]) in key_levels:
                p = r["persistence"]
                m = r["model_forecast"] or {}
                m_rmse = f"{m.get('rmse', 0):.2f}" if m.get('rmse') is not None else "—"
                m_acc = f"{m.get('acc', 0):.3f}" if m.get('acc') is not None else "—"
                ss = f"{r['skill_score_vs_persistence']:+.1%}" if r["skill_score_vs_persistence"] is not None else "—"
                f.write(f"| {r['variable']} @ {r['level_hpa']} гПа ({r['units']}) | T+{r['lead_hours']}ч | {p['rmse']:.2f} | {p['acc']:.3f} | {m_rmse} | {m_acc} | {ss} |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-dataset", required=True, help="Путь к эталонному файлу ERA5 NetCDF")
    parser.add_argument("--forecast-nc", default=None, help="Путь к файлу сгенерированного прогноза NetCDF")
    parser.add_argument("--output", required=True, help="Выходной каталог для отчетов и таблиц")
    args = parser.parse_args()

    run_experiment(args.era5_dataset, args.output, args.forecast_nc)
