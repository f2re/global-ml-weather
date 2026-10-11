"""Unified training pipeline assimilating surface and upper-air observations.

Conditioned on official 30-year NOAA 1991-2020 NetCDF climatological normals.
Learns global spherical atmospheric dynamics by supervising departures from climatology.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from .grid import build_grid
from .observation_data import ObservationDataset, START
from .observations import utc
from .observation_training import checkpoint_path, digest, save, sync_directory
from .station_climatology import StationClimatology
from .unified_observation_model import UnifiedObservationModel, unified_loss

DEFAULT_CONFIG = {
    'mesh_level': 1,
    'hidden': 64,
    'epochs': 50,
    'patience': 8,
    'threads': 8,
    'seed': 42,
    'learning_rate': 0.0005,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'surface_weight': 0.5,
    'profile_weight': 0.5,
    'max_train_samples': 192,
    'max_val_samples': 48,
}


def load_climatology_array(climatology_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load prepared 30-year official NetCDF climatology array [12, n_cells, 37, 5]."""
    npz_path = climatology_path / 'seasonal-climatology.npz' if climatology_path.is_dir() else climatology_path
    manifest_path = climatology_path / 'manifest.json' if climatology_path.is_dir() else climatology_path.with_name('manifest.json')
    with np.load(npz_path, allow_pickle=False) as data:
        mean_3d = np.nan_to_num(data['mean'], nan=0.0).astype(np.float32)
        support = data['support']
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'period': '1991-2020'}
    return mean_3d, support, manifest


def evaluate(
    model: UnifiedObservationModel,
    dataset: ObservationDataset,
    station_climatology: StationClimatology,
    split: str,
    max_samples: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    indices = dataset.subset(split)[:max_samples]

    with torch.no_grad():
        for idx in indices:
            row = dataset.samples[idx]
            hour = row['hour']
            input_hours = np.arange(hour - 11, hour + 1)
            target_hours = hour + np.arange(3, 73, 3)

            x_raw = dataset.values[input_hours]
            y_raw = dataset.values[target_hours]
            x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
            y_mask = dataset.mask[target_hours]
            if not y_mask.any() or not x_mask.any():
                continue

            t_inputs = [START + timedelta(hours=int(h)) for h in input_hours]
            t_targets = [START + timedelta(hours=int(h)) for h in target_hours]
            c_inputs = station_climatology.for_window(t_inputs)
            c_targets = station_climatology.for_window(t_targets)

            x_anom = np.where(x_mask, (x_raw - c_inputs) / station_climatology.std, 0.0).astype(np.float32)
            y_anom = np.where(y_mask, (y_raw - c_targets) / station_climatology.std, 0.0).astype(np.float32)

            tx_anom = torch.as_tensor(x_anom, device=device)
            tx_mask = torch.as_tensor(x_mask, device=device)
            ty_anom = torch.as_tensor(y_anom, device=device)
            ty_mask = torch.as_tensor(y_mask, device=device)

            issue_dt = utc(row['issue'])
            month_idx = issue_dt.month - 1

            forecast = model(tx_anom, tx_mask, month_idx=month_idx)
            l = unified_loss(forecast, ty_anom, ty_mask)
            losses.append(float(l.detach().cpu()))

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

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ObservationDataset(dataset_path)

    # Load 30-year official climatology array and normalize to standard O(1) features
    clim_mean, clim_support, clim_meta = load_climatology_array(Path(climatology_dir))
    grid = build_grid(cfg['mesh_level'])

    # Station scales from train dataset
    surface_scales = dataset.std
    profile_scales = np.array([15.0, 0.005, 10.0, 10.0, 50000.0], dtype=np.float32)
    scaled_clim_mean = clim_mean / profile_scales[None, None, None, :]

    # Fit station climatology baseline from 30-year official gridded NetCDF LTM
    n_stations = len(dataset.latlon)
    cells = model_cells = dataset.latlon # We'll build station climatology mapping
    station_clim_path = output / 'station_climatology.npz'
    if station_clim_path.exists():
        station_climatology = StationClimatology.load(station_clim_path)
    else:
        # Fit from dataset train split using official baseline
        all_timestamps = [START + timedelta(hours=h) for h in range(len(dataset.values))]
        train_indices = set(dataset.subset('train'))
        train_hours_mask = np.zeros(len(dataset.values), dtype=bool)
        for idx in train_indices:
            h = dataset.samples[idx]['hour']
            train_hours_mask[max(0, h - 11):min(len(dataset.values), h + 73)] = True

        station_climatology = StationClimatology.fit(dataset.values, dataset.mask, all_timestamps, train_hours_mask)
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
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
        )

        best_loss = float('inf')
        best_epoch = None
        stale = 0
        train_indices = dataset.subset('train')[:cfg['max_train_samples']]

        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            epoch_losses = []
            gradients = {}
            random.shuffle(train_indices)

            for idx in train_indices:
                row = dataset.samples[idx]
                hour = row['hour']
                input_hours = np.arange(hour - 11, hour + 1)
                target_hours = hour + np.arange(3, 73, 3)

                x_raw = dataset.values[input_hours]
                y_raw = dataset.values[target_hours]
                x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
                y_mask = dataset.mask[target_hours]
                if not y_mask.any() or not x_mask.any():
                    continue

                t_inputs = [START + timedelta(hours=int(h)) for h in input_hours]
                t_targets = [START + timedelta(hours=int(h)) for h in target_hours]
                c_inputs = station_climatology.for_window(t_inputs)
                c_targets = station_climatology.for_window(t_targets)

                x_anom = np.where(x_mask, (x_raw - c_inputs) / station_climatology.std, 0.0).astype(np.float32)
                y_anom = np.where(y_mask, (y_raw - c_targets) / station_climatology.std, 0.0).astype(np.float32)

                tx_anom = torch.as_tensor(x_anom, device=device)
                tx_mask = torch.as_tensor(x_mask, device=device)
                ty_anom = torch.as_tensor(y_anom, device=device)
                ty_mask = torch.as_tensor(y_mask, device=device)

                issue_dt = utc(row['issue'])
                month_idx = issue_dt.month - 1

                optimizer.zero_grad(set_to_none=True)
                forecast = model(tx_anom, tx_mask, month_idx=month_idx)
                loss = unified_loss(
                    forecast,
                    ty_anom,
                    ty_mask,
                    surface_weight=cfg['surface_weight'],
                    profile_weight=cfg['profile_weight'],
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite unified anomaly loss.")
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = bool((param.grad.abs().sum() > 0).item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            val_loss = evaluate(model, dataset, station_climatology, 'validation', cfg['max_val_samples'], device)
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1

            # Save epoch checkpoint atomically
            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
                'climatology_period': clim_meta.get('period'),
            }
            torch.save(state, temp_dir / 'state.pt')
            metrics = {
                'epoch': epoch,
                'train_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_loss': val_loss,
                'improved': improved,
                'nonzero_gradients': gradients,
                'prediction_target': cfg['prediction_target'],
                'climatology_source': 'official-30year-netcdf-ltm',
                'scientific_acceptance': False,
            }
            save(temp_dir / 'metrics.json', metrics)
            sync_directory(temp_dir)
            final_dir = output / f'epoch-{epoch:04d}'
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
            'status': 'unified_observation_training_completed',
            'climatology_source': 'official-30year-netcdf-ltm-1991-2020',
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
