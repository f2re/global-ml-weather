"""Evaluate unified observation model checkpoint in physical units against Climatology and Persistence.

Computes MAE and RMSE for:
  - Temperature (K / deg C)
  - Dew Point (K / deg C)
  - U Wind (m/s)
  - V Wind (m/s)
  - Surface Pressure (hPa)
  - Mean Sea Level Pressure (hPa)

Across forecast lead horizons from 3 hours to 240 hours (10 days).
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
import torch

from global_weather.grid import build_grid
from global_weather.observation_data import ObservationDataset, START
from global_weather.observations import utc
from global_weather.station_climatology import StationClimatology
from global_weather.terrain_features import extract_grid_terrain
from global_weather.unified_observation_model import UnifiedObservationModel

VAR_NAMES = [
    "Temperature (K)",
    "Dew Point (K)",
    "U Wind (m/s)",
    "V Wind (m/s)",
    "Surface Pressure (hPa)",
    "MSLP (hPa)",
]
VAR_KEYS = ["t2m", "td2m", "u10", "v10", "sp", "mslp"]
VAR_FACTORS = [1.0, 1.0, 1.0, 1.0, 0.01, 0.01]  # Pa -> hPa for pressure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to state.pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to observation dataset directory")
    parser.add_argument("--climatology", required=True, help="Path to climatology directory")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="JSON output file path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    cfg = state["config"]
    epoch = state.get("epoch")
    lead_hours = state.get("lead_hours", list(range(3, int(cfg.get("horizon_hours", 72)) + 1, int(cfg.get("lead_step_hours", 3)))))

    print(f"Loaded checkpoint epoch {epoch} with {len(lead_hours)} lead steps ({lead_hours[0]}h to {lead_hours[-1]}h)")

    dataset = ObservationDataset(args.dataset)
    grid = build_grid(cfg["mesh_level"])

    # Load 3-D climatology
    clim_path = Path(args.climatology) / "seasonal-climatology.npz"
    with np.load(clim_path, allow_pickle=False) as data:
        raw_clim = np.asarray(data["mean"], dtype=np.float32)
        support_clim = np.asarray(data["support"], dtype=bool)
    profile_scales = np.array([15.0, 0.005, 10.0, 10.0, 50000.0], dtype=np.float32)
    scaled_clim_mean = np.where(support_clim, raw_clim / profile_scales[None, None, None, :], 0.0).astype(np.float32)

    # Terrain
    dem_p = Path(cfg.get("dem_path", "data/dem/earth_relief_30s_p.nc"))
    dem_m = Path(cfg.get("dem_manifest", "assets/dem/earth_relief_30s_p.json"))
    terrain_elev, terrain_land, terrain_slope = extract_grid_terrain(dem_p, dem_m, grid)

    station_elev = np.array(
        [float(s["elevation_m"]) if s.get("elevation_m") is not None else 0.0 for s in dataset.manifest["stations"]],
        dtype=np.float32,
    )

    station_clim_path = checkpoint_path.parent.parent / "station_climatology.npz"
    if not station_clim_path.exists():
        station_clim_path = checkpoint_path.parent / "station_climatology.npz"
    station_climatology = StationClimatology.load(station_clim_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UnifiedObservationModel(
        grid=grid,
        station_latlon=dataset.latlon,
        surface_scales=dataset.std,
        profile_scales=profile_scales,
        hidden=cfg["hidden"],
        climatology_grid_mean=scaled_clim_mean,
        dropout=0.0,
        terrain_elevation_m=terrain_elev,
        terrain_land_fraction=terrain_land,
        terrain_slope=terrain_slope,
        station_elevation_m=station_elev,
        lead_hours=lead_hours,
    ).to(device)

    model.load_state_dict(state["model"], strict=True)
    model.eval()

    indices = dataset.subset(args.split)
    if args.max_samples is not None:
        indices = indices[:args.max_samples]

    n_stations = len(dataset.latlon)
    n_leads = len(lead_hours)

    print(f"Evaluating {len(indices)} samples from split '{args.split}' on {device}...")

    # Accumulators per lead and per variable:
    # sum_abs_err_model, sum_sq_err_model, count
    # sum_abs_err_clim, sum_sq_err_clim
    # sum_abs_err_pers, sum_sq_err_pers
    lead_model_abs = np.zeros((n_leads, 6), dtype=np.float64)
    lead_model_sq = np.zeros((n_leads, 6), dtype=np.float64)
    lead_clim_abs = np.zeros((n_leads, 6), dtype=np.float64)
    lead_clim_sq = np.zeros((n_leads, 6), dtype=np.float64)
    lead_pers_abs = np.zeros((n_leads, 6), dtype=np.float64)
    lead_pers_sq = np.zeros((n_leads, 6), dtype=np.float64)
    lead_counts = np.zeros((n_leads, 6), dtype=np.int64)

    batch_x_anom: list[np.ndarray] = []
    batch_x_mask: list[np.ndarray] = []
    batch_y_raw: list[np.ndarray] = []
    batch_y_mask: list[np.ndarray] = []
    batch_c_targets: list[np.ndarray] = []
    batch_pers_real: list[np.ndarray] = []
    batch_months: list[int] = []
    batch_hours: list[float] = []

    def flush_eval():
        if not batch_x_anom:
            return
        bx_anom = torch.as_tensor(np.stack(batch_x_anom), device=device)
        bx_mask = torch.as_tensor(np.stack(batch_x_mask), device=device)
        b_months = torch.as_tensor(batch_months, dtype=torch.long, device=device)
        b_hours = torch.as_tensor(batch_hours, dtype=torch.float32, device=device)

        with torch.no_grad():
            amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
                forecast = model(bx_anom, bx_mask, month_idx=b_months, issue_hour=b_hours)
                pred_anom = forecast.surface_anomalies.detach().float().cpu().numpy()

        for i in range(len(batch_x_anom)):
            y_r = batch_y_raw[i]            # [n_leads, n_st, 6]
            y_m = batch_y_mask[i]           # [n_leads, n_st, 6]
            c_tar = batch_c_targets[i]      # [n_leads, n_st, 6]
            p_r = batch_pers_real[i]        # [n_st, 6]
            # De-normalize model prediction to physical units
            pred_r = pred_anom[i] * station_climatology.std[None, None, :] + c_tar

            for l_idx in range(n_leads):
                for v_idx in range(6):
                    valid = y_m[l_idx, :, v_idx]
                    if not valid.any():
                        continue
                    n_v = valid.sum()
                    factor = VAR_FACTORS[v_idx]

                    # Model errors
                    err_m = (pred_r[l_idx, valid, v_idx] - y_r[l_idx, valid, v_idx]) * factor
                    lead_model_abs[l_idx, v_idx] += np.abs(err_m).sum()
                    lead_model_sq[l_idx, v_idx] += (err_m ** 2).sum()

                    # Climatology errors
                    err_c = (c_tar[l_idx, valid, v_idx] - y_r[l_idx, valid, v_idx]) * factor
                    lead_clim_abs[l_idx, v_idx] += np.abs(err_c).sum()
                    lead_clim_sq[l_idx, v_idx] += (err_c ** 2).sum()

                    # Persistence errors (last valid surface observation)
                    err_p = (p_r[valid, v_idx] - y_r[l_idx, valid, v_idx]) * factor
                    lead_pers_abs[l_idx, v_idx] += np.abs(err_p).sum()
                    lead_pers_sq[l_idx, v_idx] += (err_p ** 2).sum()

                    lead_counts[l_idx, v_idx] += n_v

        batch_x_anom.clear()
        batch_x_mask.clear()
        batch_y_raw.clear()
        batch_y_mask.clear()
        batch_c_targets.clear()
        batch_pers_real.clear()
        batch_months.clear()
        batch_hours.clear()

    for idx in indices:
        row = dataset.samples[idx]
        hour = row["hour"]
        input_hours = np.arange(hour - 11, hour + 1)
        target_hours = hour + np.array(lead_hours)

        x_raw = dataset.values[input_hours]
        x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
        if not x_mask.any():
            continue

        y_raw = np.zeros((n_leads, n_stations, 6), dtype=np.float32)
        y_mask = np.zeros((n_leads, n_stations, 6), dtype=bool)
        c_targets = np.zeros((n_leads, n_stations, 6), dtype=np.float32)

        in_bounds = target_hours < len(dataset.values)
        if in_bounds.any():
            actual_hours = target_hours[in_bounds]
            y_raw[in_bounds] = dataset.values[actual_hours]
            y_mask[in_bounds] = dataset.mask[actual_hours]
            t_targets = [START + timedelta(hours=int(h)) for h in actual_hours]
            c_targets[in_bounds] = station_climatology.for_window(t_targets)

        if not y_mask.any():
            continue

        t_inputs = [START + timedelta(hours=int(h)) for h in input_hours]
        c_inputs = station_climatology.for_window(t_inputs)
        x_anom = np.where(x_mask, (x_raw - c_inputs) / station_climatology.std, 0.0).astype(np.float32)

        # Persistence: latest observed surface value at hour
        pers = x_raw[-1].copy()

        issue_dt = utc(row["issue"])
        batch_x_anom.append(x_anom)
        batch_x_mask.append(x_mask)
        batch_y_raw.append(y_raw)
        batch_y_mask.append(y_mask)
        batch_c_targets.append(c_targets)
        batch_pers_real.append(pers)
        batch_months.append(issue_dt.month - 1)
        batch_hours.append(float(issue_dt.hour))

        if len(batch_x_anom) >= args.batch_size:
            flush_eval()

    flush_eval()

    # Results tabulation
    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "split": args.split,
        "lead_hours": lead_hours,
        "variables": VAR_KEYS,
        "by_lead": {},
        "overall": {},
    }

    print("\n" + "=" * 95)
    print(f"VERIFICATION METRICS ON SPLIT '{args.split.upper()}' (Checkpoint Epoch {epoch})")
    print("=" * 95)

    key_leads = [3, 6, 12, 24, 48, 72, 120, 168, 240]
    lead_indices = [lead_hours.index(h) for h in key_leads if h in lead_hours]

    for v_idx, v_name in enumerate(VAR_NAMES):
        tot_cnt = lead_counts[:, v_idx].sum()
        if tot_cnt == 0:
            continue
        tot_m_mae = lead_model_abs[:, v_idx].sum() / tot_cnt
        tot_m_rmse = np.sqrt(lead_model_sq[:, v_idx].sum() / tot_cnt)
        tot_c_mae = lead_clim_abs[:, v_idx].sum() / tot_cnt
        tot_c_rmse = np.sqrt(lead_clim_sq[:, v_idx].sum() / tot_cnt)
        tot_p_mae = lead_pers_abs[:, v_idx].sum() / tot_cnt
        tot_p_rmse = np.sqrt(lead_pers_sq[:, v_idx].sum() / tot_cnt)

        results["overall"][VAR_KEYS[v_idx]] = {
            "model_mae": float(tot_m_mae),
            "model_rmse": float(tot_m_rmse),
            "clim_mae": float(tot_c_mae),
            "clim_rmse": float(tot_c_rmse),
            "pers_mae": float(tot_p_mae),
            "pers_rmse": float(tot_p_rmse),
        }

        print(f"\n--- {v_name} ---")
        print(f"{'Lead':>6} | {'Model MAE':>10} {'RMSE':>8} | {'Clim MAE':>10} {'RMSE':>8} | {'Pers MAE':>10} {'RMSE':>8}")
        print("-" * 75)

        for l_idx in lead_indices:
            h = lead_hours[l_idx]
            cnt = lead_counts[l_idx, v_idx]
            if cnt == 0:
                continue
            m_mae = lead_model_abs[l_idx, v_idx] / cnt
            m_rmse = np.sqrt(lead_model_sq[l_idx, v_idx] / cnt)
            c_mae = lead_clim_abs[l_idx, v_idx] / cnt
            c_rmse = np.sqrt(lead_clim_sq[l_idx, v_idx] / cnt)
            p_mae = lead_pers_abs[l_idx, v_idx] / cnt
            p_rmse = np.sqrt(lead_pers_sq[l_idx, v_idx] / cnt)

            lead_str = f"+{h}h"
            print(f"{lead_str:>6} | {m_mae:10.2f} {m_rmse:8.2f} | {c_mae:10.2f} {c_rmse:8.2f} | {p_mae:10.2f} {p_rmse:8.2f}")

            if lead_str not in results["by_lead"]:
                results["by_lead"][lead_str] = {}
            results["by_lead"][lead_str][VAR_KEYS[v_idx]] = {
                "model_mae": float(m_mae),
                "model_rmse": float(m_rmse),
                "clim_mae": float(c_mae),
                "clim_rmse": float(c_rmse),
                "pers_mae": float(p_mae),
                "pers_rmse": float(p_rmse),
            }

    print("\n" + "=" * 95)
    print("OVERALL 10-DAY MEAN METRICS:")
    for v_idx, v_name in enumerate(VAR_NAMES):
        k = VAR_KEYS[v_idx]
        if k in results["overall"]:
            ov = results["overall"][k]
            print(f"  {v_name:<25}: Model MAE = {ov['model_mae']:6.2f} | Clim MAE = {ov['clim_mae']:6.2f} | Pers MAE = {ov['pers_mae']:6.2f}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"\nSaved detailed metrics to {out_p}")


if __name__ == "__main__":
    main()
