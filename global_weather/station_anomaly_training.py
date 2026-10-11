"""Training pipeline for station anomaly models.

Learns surface atmospheric dynamics by supervising departures from climatology.
All operations are fully deterministic and track active gradient propagation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from .grid import build_pyramid
from .observation_data import ObservationDataset, START
from .observations import utc
from .observation_training import (
    checkpoint_path,
    digest,
    save,
    sync_directory,
)
from .station_anomaly_model import (
    StationAnomalyModel,
    anomaly_loss,
)
from .station_climatology import StationClimatology

DEFAULT_CONFIG = {
    'mesh_level': 0,
    'hidden': 16,
    'epochs': 10,
    'patience': 3,
    'threads': 4,
    'seed': 42,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'max_train_samples': 48,
    'max_val_samples': 16,
}


def create_model(dataset: ObservationDataset, climatology: StationClimatology, config: dict, device: torch.device):
    grids = build_pyramid(config['mesh_level'])
    return StationAnomalyModel(
        grids,
        dataset.latlon,
        climatology.std,
        hidden=config['hidden'],
    ).to(device)


def evaluate(model: StationAnomalyModel, dataset: ObservationDataset, climatology: StationClimatology,
             split: str, max_samples: int, device: torch.device) -> float:
    model.eval()
    losses = []
    indices = dataset.subset(split)[:max_samples]
    with torch.no_grad():
        for idx in indices:
            row = dataset.samples[idx]
            hour = row['hour']
            input_hours = np.arange(hour - 11, hour + 1)
            target_hours = hour + np.arange(3, 73, 3)

            # Raw values and masks
            x_raw = dataset.values[input_hours]
            y_raw = dataset.values[target_hours]
            x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
            y_mask = dataset.mask[target_hours]
            if not y_mask.any():
                continue

            from datetime import timedelta
            t_inputs = [START + timedelta(hours=int(h)) for h in input_hours]
            t_targets = [START + timedelta(hours=int(h)) for h in target_hours]

            c_inputs = climatology.for_window(t_inputs)
            c_targets = climatology.for_window(t_targets)

            # Normalized anomalies
            x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)
            y_anom = np.where(y_mask, (y_raw - c_targets) / climatology.std, 0.0).astype(np.float32)

            tx_anom = torch.as_tensor(x_anom, device=device)
            tx_mask = torch.as_tensor(x_mask, device=device)
            ty_anom = torch.as_tensor(y_anom, device=device)
            ty_mask = torch.as_tensor(y_mask, device=device)

            forecast = model(tx_anom, tx_mask)
            l = anomaly_loss(forecast.predicted_anomalies, ty_anom, ty_mask)
            losses.append(float(l.detach().cpu()))

    return float(np.mean(losses)) if losses else float('inf')


def train(dataset_path: str | Path, output: str | Path, config: dict | None = None,
          climatology_path: str | Path | None = None) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ObservationDataset(dataset_path)

    # Climatology setup
    if climatology_path and Path(climatology_path).exists():
        climatology = StationClimatology.load(climatology_path)
    else:
        # Fit from dataset's train split
        from datetime import timedelta
        all_timestamps = [START + timedelta(hours=h) for h in range(len(dataset.values))]
        train_indices = set(dataset.subset('train'))
        train_hours_mask = np.zeros(len(dataset.values), dtype=bool)
        for idx in train_indices:
            h = dataset.samples[idx]['hour']
            # Window covers h-11 to h+72
            train_hours_mask[max(0, h - 11):min(len(dataset.values), h + 73)] = True

        climatology = StationClimatology.fit(dataset.values, dataset.mask, all_timestamps, train_hours_mask)
        climatology.save(output / 'station_climatology.npz')

    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        model = create_model(dataset, climatology, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        best_loss = float('inf')
        best_epoch = None
        stale = 0
        from datetime import timedelta

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
                c_inputs = climatology.for_window(t_inputs)
                c_targets = climatology.for_window(t_targets)

                x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)
                y_anom = np.where(y_mask, (y_raw - c_targets) / climatology.std, 0.0).astype(np.float32)

                tx_anom = torch.as_tensor(x_anom, device=device)
                tx_mask = torch.as_tensor(x_mask, device=device)
                ty_anom = torch.as_tensor(y_anom, device=device)
                ty_mask = torch.as_tensor(y_mask, device=device)

                optimizer.zero_grad(set_to_none=True)
                forecast = model(tx_anom, tx_mask)
                loss = anomaly_loss(forecast.predicted_anomalies, ty_anom, ty_mask)

                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite anomaly loss.")
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = bool(param.grad.abs().sum() > 0)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            val_loss = evaluate(model, dataset, climatology, 'validation', cfg['max_val_samples'], device)
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1

            # Save epoch checkpoint
            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
            }
            torch.save(state, temp_dir / 'state.pt')
            metrics = {
                'epoch': epoch,
                'train_anomaly_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_anomaly_loss': val_loss,
                'improved': improved,
                'nonzero_gradients': gradients,
                'prediction_target': 'dimensionless anomaly departure from climatology',
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
                break

        completion = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'status': 'surface_climatology_anomaly_research_trained',
            'prediction_target': 'dimensionless anomaly departure from climatology',
            'climatology_source': 'station-monthly-climatology-1',
            'scientific_acceptance': False,
        }
        save(output / 'complete.json', completion)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--climatology')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    train(args.dataset, args.output, config, args.climatology)


if __name__ == '__main__':
    main()
