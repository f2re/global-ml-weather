"""Legacy surface-control training on real station observations.

This module intentionally supervises only the GHCNh surface branch. Upper-air
and satellite interfaces in UnifiedObservationModel are not trained here.
The scientifically preferred measured surface+IGRA path is
``global_weather.joint_observation_training``. Legacy Unified defaults to a
72-hour horizon because native-station-dataset-1 was prepared with a 72-hour
chronological guard. A 240-hour experiment requires a newly prepared dataset.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from .grid import build_grid
from .observation_data import ObservationDataset, START, TRAIN_END, VAL_END, END
from .observations import utc
from .observation_training import digest, save, sync_directory
from .station_climatology import StationClimatology
from .terrain_features import extract_grid_terrain
from .unified_observation_model import UnifiedObservationModel, unified_loss

DEFAULT_CONFIG = {
    'mesh_level': 1,
    'hidden': 64,
    'epochs': 300,
    'patience': 40,
    'threads': 8,
    'seed': 42,
    'learning_rate': 0.0005,
    'min_lr': 1e-6,
    'scheduler': 'cosine',
    'weight_decay': 0.01,
    'dropout': 0.0,
    'gradient_clip': 1.0,
    'surface_weight': 1.0,
    'profile_weight': 0.0,
    'batch_size': 4,
    'max_train_samples': None,
    'max_val_samples': None,
    'dem_path': 'data/dem/earth_relief_30s_p.nc',
    'dem_manifest': 'assets/dem/earth_relief_30s_p.json',
    'allow_missing_dem': False,
    'horizon_hours': 72,
    'lead_step_hours': 3,
    'prediction_target': (
        'surface-station dimensionless anomalies relative to train-only station '
        'climatology; NCEP1 monthly fields are external context only'
    ),
}


def load_climatology_array(climatology_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load fixed NOAA/NCEP monthly context and its explicit support mask."""
    npz_path = climatology_path / 'seasonal-climatology.npz' if climatology_path.is_dir() else climatology_path
    manifest_path = climatology_path / 'manifest.json' if climatology_path.is_dir() else climatology_path.with_name('manifest.json')
    with np.load(npz_path, allow_pickle=False) as data:
        raw = np.asarray(data['mean'], dtype=np.float32)
        support = np.asarray(data['support'], dtype=bool)
    if raw.shape != support.shape or raw.ndim != 4 or raw.shape[0] != 12 or raw.shape[-1] != 5:
        raise ValueError('Climatology mean/support must share shape [12, cells, levels, 5].')
    if not np.isfinite(raw[support]).all():
        raise ValueError('Supported climatology values must be finite.')
    mean_3d = np.where(support, raw, 0.0).astype(np.float32)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'period': '1991-2020'}
    return mean_3d, support, manifest


def validate_forecast_windows(samples: list[dict], lead_hours: list[int]) -> None:
    """Reject targets crossing frozen splits before fitting any statistics."""
    if not lead_hours or any(type(h) is not int or h <= 0 for h in lead_hours):
        raise ValueError('Forecast leads must be positive integer hours.')
    boundaries = {'train': (START, TRAIN_END), 'validation': (TRAIN_END, VAL_END),
                  'test': (VAL_END, END)}
    for row in samples:
        if row['split'] not in boundaries:
            raise ValueError('Unknown chronological split.')
        begin, end = boundaries[row['split']]
        issue = utc(row['issue'])
        if issue - timedelta(hours=12) < begin or issue + timedelta(hours=max(lead_hours)) >= end:
            raise ValueError(
                f"Forecast window crosses {row['split']} split at {row['issue']}; "
                'prepare a new horizon-compatible dataset, do not reuse contaminated statistics.'
            )


def evaluate(
    model: UnifiedObservationModel,
    dataset: ObservationDataset,
    station_climatology: StationClimatology,
    split: str,
    max_samples: int | None,
    device: torch.device,
    lead_hours: list[int],
    batch_size: int = 4,
) -> float:
    model.eval()
    losses = []
    indices = dataset.subset(split)
    if max_samples is not None:
        indices = indices[:max_samples]

    n_stations = len(dataset.latlon)
    n_leads = len(lead_hours)

    batch_x_anom: list[np.ndarray] = []
    batch_x_mask: list[np.ndarray] = []
    batch_y_anom: list[np.ndarray] = []
    batch_y_mask: list[np.ndarray] = []
    batch_months: list[int] = []
    batch_hours: list[float] = []

    def flush_eval_batch() -> None:
        if not batch_x_anom:
            return
        bx_anom = torch.as_tensor(np.stack(batch_x_anom), device=device)
        bx_mask = torch.as_tensor(np.stack(batch_x_mask), device=device)
        by_anom = torch.as_tensor(np.stack(batch_y_anom), device=device)
        by_mask = torch.as_tensor(np.stack(batch_y_mask), device=device)
        b_months = torch.as_tensor(batch_months, dtype=torch.long, device=device)
        b_hours = torch.as_tensor(batch_hours, dtype=torch.float32, device=device)

        batch_x_anom.clear()
        batch_x_mask.clear()
        batch_y_anom.clear()
        batch_y_mask.clear()
        batch_months.clear()
        batch_hours.clear()

        amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=amp_dtype):
            forecast = model(bx_anom, bx_mask, month_idx=b_months, issue_hour=b_hours)
            l = unified_loss(forecast, by_anom, by_mask, surface_weight=1.0, profile_weight=0.0)
        losses.append(float(l.detach().cpu()))

    with torch.no_grad():
        for idx in indices:
            row = dataset.samples[idx]
            hour = row['hour']
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
            y_anom = np.where(y_mask, (y_raw - c_targets) / station_climatology.std, 0.0).astype(np.float32)

            issue_dt = utc(row['issue'])
            batch_x_anom.append(x_anom)
            batch_x_mask.append(x_mask)
            batch_y_anom.append(y_anom)
            batch_y_mask.append(y_mask)
            batch_months.append(issue_dt.month - 1)
            batch_hours.append(float(issue_dt.hour))

            if len(batch_x_anom) >= batch_size:
                flush_eval_batch()

        flush_eval_batch()

    return float(np.mean(losses)) if losses else float('inf')


def train(
    dataset_path: str | Path,
    climatology_dir: str | Path,
    output: str | Path,
    config: dict | None = None,
) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    if float(cfg.get('profile_weight', 0.0)) != 0.0:
        raise ValueError(
            'Legacy Unified has no profile supervision. Use joint_observation_training '
            'for measured GHCNh+IGRA training.'
        )
    if float(cfg.get('surface_weight', 1.0)) <= 0.0:
        raise ValueError('Legacy Unified requires a positive surface loss weight.')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ObservationDataset(dataset_path)

    horizon_hours = int(cfg.get('horizon_hours', 72))
    lead_step = int(cfg.get('lead_step_hours', 3))
    if horizon_hours <= 0 or lead_step <= 0 or horizon_hours % lead_step:
        raise ValueError('Horizon must be positive and divisible by the lead step.')
    lead_hours = list(range(lead_step, horizon_hours + 1, lead_step))
    validate_forecast_windows(dataset.samples, lead_hours)

    # NOAA/NCEP monthly 3-D context; it is not the station target baseline.
    clim_mean, clim_support, clim_meta = load_climatology_array(Path(climatology_dir))
    grid = build_grid(cfg['mesh_level'])
    if clim_mean.shape[1] != grid.n_cells:
        raise ValueError('Climatology grid does not match model grid.')
    climatology_support_fraction = float(clim_support.mean())

    # Terrain is mandatory for a run that advertises DEM conditioning. Silent
    # all-zero substitution is allowed only in explicitly requested synthetic/debug runs.
    dem_p = Path(cfg.get('dem_path', 'data/dem/earth_relief_30s_p.nc'))
    dem_m = Path(cfg.get('dem_manifest', 'assets/dem/earth_relief_30s_p.json'))
    if dem_p.exists() and dem_m.exists():
        terrain_elev, terrain_land, terrain_slope = extract_grid_terrain(dem_p, dem_m, grid)
        terrain_source = 'GMT Earth Relief 30s NetCDF (896 MB)'
    elif cfg.get('allow_missing_dem'):
        terrain_elev = np.zeros(grid.n_cells, dtype=np.float32)
        terrain_land = np.zeros(grid.n_cells, dtype=np.float32)
        terrain_slope = np.zeros(grid.n_cells, dtype=np.float32)
        terrain_source = 'explicit-debug-zero-terrain'
    else:
        raise FileNotFoundError(
            'Verified DEM and manifest are required; set allow_missing_dem=true only for an explicit debug run.'
        )

    # Native station datasets do not guarantee instrument elevation. Missing
    # metadata remains an explicit limitation rather than being described as measured height.
    station_height_known = np.array([s.get('elevation_m') is not None for s in dataset.manifest['stations']], dtype=bool)
    station_elev = np.array(
        [float(s['elevation_m']) if s.get('elevation_m') is not None else 0.0 for s in dataset.manifest['stations']],
        dtype=np.float32,
    )

    surface_scales = dataset.std
    profile_scales = np.array([15.0, 0.005, 10.0, 10.0, 50000.0], dtype=np.float32)
    scaled_clim_mean = np.where(
        clim_support,
        clim_mean / profile_scales[None, None, None, :],
        0.0,
    ).astype(np.float32)

    n_stations = len(dataset.latlon)
    n_leads = len(lead_hours)
    station_clim_path = output / 'station_climatology.npz'
    if station_clim_path.exists():
        station_climatology = StationClimatology.load(station_clim_path)
    else:
        all_timestamps = [START + timedelta(hours=h) for h in range(len(dataset.values))]
        # The baseline may use the full train period, but never validation/test.
        train_hours_mask = np.array(
            [START <= timestamp < TRAIN_END for timestamp in all_timestamps],
            dtype=bool,
        )
        station_climatology = StationClimatology.fit(
            dataset.values, dataset.mask, all_timestamps, train_hours_mask
        )
        station_climatology.save(station_clim_path)

    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        model = UnifiedObservationModel(
            grid=grid,
            station_latlon=dataset.latlon,
            surface_scales=surface_scales,
            profile_scales=profile_scales,
            hidden=cfg['hidden'],
            climatology_grid_mean=scaled_clim_mean,
            dropout=cfg.get('dropout', 0.0),
            terrain_elevation_m=terrain_elev,
            terrain_land_fraction=terrain_land,
            terrain_slope=terrain_slope,
            station_elevation_m=station_elev,
            lead_hours=lead_hours,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
        )

        scheduler = None
        if cfg.get('scheduler') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg['epochs'],
                eta_min=cfg.get('min_lr', 1e-6),
            )

        best_loss = float('inf')
        best_epoch = None
        stale = 0
        train_indices = dataset.subset('train')
        if cfg.get('max_train_samples') is not None:
            train_indices = train_indices[:cfg['max_train_samples']]

        batch_size = max(1, int(cfg.get('batch_size', 4)))
        amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            epoch_losses = []
            gradients = {}
            random.shuffle(train_indices)

            batch_x_anom: list[np.ndarray] = []
            batch_x_mask: list[np.ndarray] = []
            batch_y_anom: list[np.ndarray] = []
            batch_y_mask: list[np.ndarray] = []
            batch_months: list[int] = []
            batch_hours: list[float] = []

            def flush_train_batch() -> None:
                if not batch_x_anom:
                    return
                bx_anom = torch.as_tensor(np.stack(batch_x_anom), device=device)
                bx_mask = torch.as_tensor(np.stack(batch_x_mask), device=device)
                by_anom = torch.as_tensor(np.stack(batch_y_anom), device=device)
                by_mask = torch.as_tensor(np.stack(batch_y_mask), device=device)
                b_months = torch.as_tensor(batch_months, dtype=torch.long, device=device)
                b_hours = torch.as_tensor(batch_hours, dtype=torch.float32, device=device)

                batch_x_anom.clear()
                batch_x_mask.clear()
                batch_y_anom.clear()
                batch_y_mask.clear()
                batch_months.clear()
                batch_hours.clear()

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=amp_dtype):
                    forecast = model(bx_anom, bx_mask, month_idx=b_months, issue_hour=b_hours)
                    loss = unified_loss(
                        forecast,
                        by_anom,
                        by_mask,
                        surface_weight=1.0,
                        profile_weight=0.0,
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite unified surface anomaly loss.')
                loss.backward()

                for name, param in model.named_parameters():
                    gradients[name] = bool(
                        param.grad is not None
                        and torch.isfinite(param.grad).all().item()
                        and (param.grad.abs().sum() > 0).item()
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            for idx in train_indices:
                row = dataset.samples[idx]
                hour = row['hour']
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
                y_anom = np.where(y_mask, (y_raw - c_targets) / station_climatology.std, 0.0).astype(np.float32)

                issue_dt = utc(row['issue'])
                batch_x_anom.append(x_anom)
                batch_x_mask.append(x_mask)
                batch_y_anom.append(y_anom)
                batch_y_mask.append(y_mask)
                batch_months.append(issue_dt.month - 1)
                batch_hours.append(float(issue_dt.hour))

                if len(batch_x_anom) >= batch_size:
                    flush_train_batch()

            flush_train_batch()

            if scheduler is not None:
                scheduler.step()
                current_lr = float(optimizer.param_groups[0]['lr'])
            else:
                current_lr = float(cfg['learning_rate'])

            val_loss = evaluate(
                model, dataset, station_climatology, 'validation',
                cfg.get('max_val_samples'), device, lead_hours=lead_hours, batch_size=batch_size
            )
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1

            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'current_lr': current_lr,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
                'climatology_period': clim_meta.get('period'),
                'climatology_support_fraction': climatology_support_fraction,
                'lead_hours': lead_hours,
            }
            torch.save(state, temp_dir / 'state.pt')
            metrics = {
                'epoch': epoch,
                'train_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_loss': val_loss,
                'improved': improved,
                'learning_rate': current_lr,
                'nonzero_gradients': gradients,
                'prediction_target': cfg['prediction_target'],
                'climatology_source': 'NCEP1-1991-2020-external-context; train-only-station-target-baseline',
                'climatology_support_fraction': climatology_support_fraction,
                'supervised_branches': ['surface'],
                'unsupervised_branches': ['profile', 'satellite'],
                'terrain_source': terrain_source,
                'station_height_known_fraction': float(station_height_known.mean()),
                'horizon_hours': horizon_hours,
                'scientific_acceptance': False,
            }
            save(temp_dir / 'metrics.json', metrics)
            sync_directory(temp_dir)
            final_dir = output / f'epoch-{epoch:04d}'
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)
            sync_directory(output)

            ref = {'directory': final_dir.name, 'epoch': epoch, 'sha256': digest(final_dir / 'state.pt')}
            save(output / 'latest.json', ref)
            if improved:
                save(output / 'best.json', ref)

            print(json.dumps(metrics), flush=True)
            if stale >= cfg['patience']:
                print(f"Early stopping triggered at epoch {epoch} (patience {cfg['patience']})", flush=True)
                break

        completion = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'status': 'legacy_unified_surface_control_completed',
            'climatology_source': 'NCEP1-1991-2020-external-context; train-only-station-target-baseline',
            'climatology_support_fraction': climatology_support_fraction,
            'supervised_branches': ['surface'],
            'unsupervised_branches': ['profile', 'satellite'],
            'terrain_source': terrain_source,
            'station_height_known_fraction': float(station_height_known.mean()),
            'horizon_hours': horizon_hours,
            'prediction_target': cfg['prediction_target'],
            'scientific_acceptance': False,
        }
        save(output / 'complete.json', completion)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--climatology', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    train(args.dataset, args.climatology, args.output, config)


if __name__ == '__main__':
    main()
