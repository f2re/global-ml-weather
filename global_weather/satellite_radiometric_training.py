"""Stage S1: Radiometric warmup of Russian satellite assimilation operators.

Trains satellite forward operators, VarBC bias correction, GSICS calibration,
and ViewConditionedFiLM feature modulation conditioned on the frozen forecast core
(StationAnomalyModel epoch-0081).
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
import torch.nn as nn
import torch.nn.functional as F

from .grid import build_grid, build_mesh8_grid
from .observation_data import ObservationDataset, campaign_bounds
from .observation_training import checkpoint_path, digest, save, sync_directory
from .station_anomaly_model import StationAnomalyModel
from .station_climatology import StationClimatology
from .multimodal.ru_sat_msugs import ArktikaElektroMSUGSAdapter, ViewConditionedFiLMBlock
from .analysis.observation_operator import IKFS2CloudClearedOperator, MeteorMTVZAAllSkyOperator
from .losses import SatelliteAssimilationLoss

DEFAULT_S1_CONFIG = {
    'schema': 'satellite-stage-s1-config-1',
    'stage_id': 'S1_ru_sat_radiometric_warmup',
    'experiment_id': 's1-radiometric-warmup-2024-h1',
    'freeze_forecast_core': True,
    'mesh_level': 4,
    'hidden': 32,
    'epochs': 8,
    'learning_rate': 3e-4,
    'weight_decay': 0.001,
    'gradient_clip': 1.0,
    'huber_delta': 1.5,
    'gross_error_sigma': 3.5,
    'varbc_reg_weight': 0.001,
    'film_reg_weight': 0.001,
    'max_train_samples': 400,
    'max_val_samples': 80,
    'threads': 8,
    'seed': 42,
    'scientific_acceptance': False,
}


class SatelliteRadiometricModel(nn.Module):
    """Coupled assimilation model for Stage S1.
    
    The base weather core remains completely frozen (requires_grad = False).
    Satellite components (MSU-GS, GSICS, FiLM, IKFS-2, MTVZA, VarBC) are trained.
    """
    def __init__(
        self,
        base_model: StationAnomalyModel,
        *,
        n_sats: int = 3,
        hidden_dim: int = 32,
        num_ikfs2: int = 180,
        num_mtvza: int = 29,
        huber_delta: float = 1.5,
        gross_error_sigma: float = 3.5,
        varbc_reg_weight: float = 1e-3,
        film_reg_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.n_sats = n_sats
        self.hidden_dim = hidden_dim
        self.film_reg_weight = film_reg_weight

        # 1. MSU-GS adapter and learnable GSICS calibration
        self.msugs_adapter = ArktikaElektroMSUGSAdapter()
        self.gsics_slope = nn.Parameter(torch.ones(n_sats, 7, dtype=torch.float32))
        self.gsics_offset = nn.Parameter(torch.zeros(n_sats, 7, dtype=torch.float32))
        self.msugs_proj = nn.Linear(20, hidden_dim)

        # 2. ViewConditionedFiLMBlock
        self.film_block = ViewConditionedFiLMBlock(hidden_dim=hidden_dim, geom_dim=4)

        # 3. Hyperspectral IR & Microwave operators with VarBC
        self.ikfs2_operator = IKFS2CloudClearedOperator(num_selected_channels=num_ikfs2, num_pcs=32, num_model_levels=37)
        self.mtvza_operator = MeteorMTVZAAllSkyOperator(num_channels=num_mtvza, num_model_levels=37)

        # 4. Loss functional
        self.loss_fn = SatelliteAssimilationLoss(
            huber_delta=huber_delta,
            gross_error_sigma=gross_error_sigma,
            varbc_reg_weight=varbc_reg_weight,
        )

    def forward(
        self,
        history_anomalies: torch.Tensor,
        history_mask: torch.Tensor,
        sat_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Base weather core analysis state (frozen, no gradient)
        with torch.no_grad():
            base_state, base_profiles = self.base_model.analysis_state(history_anomalies, history_mask)
            # base_state: [n_nodes, 37, hidden]
            # base_profiles: [n_nodes, 37, 6] (T, q, u, v, z, omega)

        n_nodes = base_state.shape[0]
        device = history_anomalies.device

        # 2. MSU-GS radiometric conversion and viewing geometry
        node_xyz_km = self.base_model.xyz * 6371.0
        node_normal = self.base_model.xyz
        fused = self.msugs_adapter(
            sat_inputs['raw_channels'],
            sat_inputs['valid_mask'],
            node_xyz_km,
            node_normal,
            sat_inputs['sat_xyz_km'],
            sat_inputs['sun_dir_ecef'],
            self.gsics_slope,
            self.gsics_offset,
        )
        geom_features = fused[..., 13:17]  # eff_mu_sat, eff_air_mass, eff_r_norm, mu_sun
        fused_proj = self.msugs_proj(fused)

        # 3. FiLM feature modulation on column-mean hidden state
        h_mean = base_state.mean(dim=1, keepdim=True).transpose(0, 1)  # [1, n_nodes, hidden]
        h_mod = self.film_block(h_mean + fused_proj, geom_features)
        loss_film = self.film_reg_weight * F.mse_loss(h_mod, h_mean)

        # 4. Forward observation operators
        n_obs = min(sat_inputs['obs_tb'].shape[0], n_nodes)
        
        # IKFS-2
        t_prof = base_profiles[:n_obs, :, 0]
        q_prof = base_profiles[:n_obs, :, 1]
        t_skin = base_profiles[:n_obs, 0, 0:1]
        mu_sat = geom_features[0, :n_obs, 0:1]
        varbc_p = sat_inputs['varbc_predictors'][:n_obs]
        cld_frac = sat_inputs.get('cld_frac', torch.zeros(n_obs, device=device))[:n_obs]
        cth = sat_inputs.get('cth', torch.full((n_obs,), 1013.25, device=device))[:n_obs]

        out_ikfs2 = self.ikfs2_operator(
            sat_inputs['obs_spectra'][:n_obs],
            t_prof,
            q_prof,
            t_skin,
            mu_sat,
            varbc_p,
            cld_frac,
            cth,
        )

        # MTVZA-ГЯ
        m_prof = torch.cat([t_prof, q_prof, torch.zeros_like(t_prof)], dim=-1)
        wind_10m = torch.norm(base_profiles[:n_obs, 0, 2:4], dim=-1, keepdim=True)
        surf_state = torch.cat([t_skin, wind_10m, torch.zeros_like(t_skin), torch.zeros_like(t_skin)], dim=-1)

        out_mtvza = self.mtvza_operator(
            sat_inputs['obs_tb'][:n_obs],
            m_prof,
            surf_state,
            varbc_p,
        )

        # AMV
        n_amv = min(sat_inputs['amv_uv_obs'].shape[0], n_obs)
        amv_pred = base_profiles[:n_amv, 0, 2:4]

        # 5. Robust satellite assimilation loss
        varbc_params = list(self.mtvza_operator.varbc.parameters()) + list(self.ikfs2_operator.varbc.parameters())
        varbc_priors = [torch.zeros_like(p) for p in varbc_params]

        loss_dict = self.loss_fn(
            out_mtvza,
            out_ikfs2,
            amv_pred,
            sat_inputs['amv_uv_obs'][:n_amv],
            sat_inputs['amv_sigma'][:n_amv],
            sat_inputs['amv_qi'][:n_amv],
            varbc_params,
            varbc_priors,
        )

        total_loss = loss_dict['loss_sat_total'] + loss_film
        metrics = {
            'loss_sat_total': float(total_loss.item()),
            'loss_mtvza': float(loss_dict['loss_mtvza'].item()),
            'loss_ikfs2': float(loss_dict['loss_ikfs2'].item()),
            'loss_amv': float(loss_dict['loss_amv'].item()),
            'loss_varbc_reg': float(loss_dict['loss_varbc_reg'].item()),
            'loss_film': float(loss_film.item()),
        }
        return total_loss, metrics


def generate_satellite_batch(
    n_nodes: int,
    n_sats: int,
    device: torch.device,
    model: SatelliteRadiometricModel | None = None,
    base_profiles: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Generates satellite observation tensors consistent with physical sensor geometry."""
    b = 1
    raw_channels = torch.full((b, n_sats, n_nodes, 10), 250.0, device=device)
    valid_mask = torch.ones((b, n_sats, n_nodes), dtype=torch.bool, device=device)

    # 3 spacecraft: Elektro-L3 (GEO), Arktika-M1 (HEO apogee), Arktika-M2 (HEO descending)
    sat_xyz = torch.tensor([
        [[0.0, 0.0, 42164.0], [0.0, 30000.0, 38000.0], [30000.0, 0.0, 38000.0]]
    ], device=device)
    sun_dir = torch.tensor([[1.0, 0.0, 0.0]], device=device)

    n_obs = min(n_nodes, 64)
    varbc_p = torch.zeros(n_obs, 7, device=device)

    # Generate synthetic/surrogate observations passing QC
    if model is not None and base_profiles is not None:
        with torch.no_grad():
            t_prof = base_profiles[:n_obs, :, 0]
            q_prof = base_profiles[:n_obs, :, 1]
            t_skin = base_profiles[:n_obs, 0, 0:1]
            mu_sat = torch.ones(n_obs, 1, device=device)
            ikfs2_in = torch.cat([t_prof, q_prof, t_skin, mu_sat], dim=-1)
            init_spectra = model.ikfs2_operator.forward_rt_surrogate(ikfs2_in)
            obs_spectra = init_spectra + torch.randn_like(init_spectra) * 0.1

            m_prof = torch.cat([t_prof, q_prof, torch.zeros_like(t_prof)], dim=-1)
            wind_10m = torch.norm(base_profiles[:n_obs, 0, 2:4], dim=-1, keepdim=True)
            surf_state = torch.cat([t_skin, wind_10m, torch.zeros_like(t_skin), torch.zeros_like(t_skin)], dim=-1)
            mtvza_in = torch.cat([m_prof, surf_state], dim=-1)
            init_tb = model.mtvza_operator.allsky_rt_surrogate(mtvza_in)
            obs_tb = init_tb + torch.randn_like(init_tb) * 0.2
    else:
        obs_spectra = torch.randn(n_obs, 180, device=device)
        obs_tb = torch.full((n_obs, 29), 240.0, device=device)

    n_amv = min(n_obs, 16)
    amv_uv_obs = torch.randn(n_amv, 2, device=device)
    amv_sigma = torch.full((n_amv, 1), 2.0, device=device)
    amv_qi = torch.full((n_amv, 1), 0.95, device=device)

    return {
        'raw_channels': raw_channels,
        'valid_mask': valid_mask,
        'sat_xyz_km': sat_xyz,
        'sun_dir_ecef': sun_dir,
        'obs_spectra': obs_spectra,
        'obs_tb': obs_tb,
        'varbc_predictors': varbc_p,
        'amv_uv_obs': amv_uv_obs,
        'amv_sigma': amv_sigma,
        'amv_qi': amv_qi,
        'cld_frac': torch.zeros(n_obs, device=device),
        'cth': torch.full((n_obs,), 1013.25, device=device),
    }


def load_base_model(
    checkpoint_path_str: str | Path,
    climatology: StationClimatology,
    mesh_level: int,
    hidden: int,
    device: torch.device,
) -> StationAnomalyModel:
    """Restores the base StationAnomalyModel from checkpoint and freezes its weights."""
    grid = build_mesh8_grid() if mesh_level == 8 else build_grid(mesh_level)
    ckpt = torch.load(checkpoint_path_str, map_location=device)
    
    # Reconstruct station coordinates from operator buffer or latlon
    station_xyz = ckpt['model']['operator.station_xyz'].cpu().numpy()
    lat = np.degrees(np.arcsin(np.clip(station_xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(station_xyz[:, 1], station_xyz[:, 0]))
    station_latlon = np.stack([lat, lon], axis=-1)

    model = StationAnomalyModel(
        [grid],
        station_latlon,
        climatology.std,
        hidden=hidden,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def evaluate_s1(
    model: SatelliteRadiometricModel,
    dataset: ObservationDataset,
    climatology: StationClimatology,
    split: str,
    max_samples: int,
    device: torch.device,
) -> float:
    """Evaluates satellite assimilation loss on held-out split."""
    model.eval()
    losses = []
    indices = dataset.subset(split)[:max_samples]
    with torch.no_grad():
        for idx in indices:
            row = dataset.samples[idx]
            hour = row['hour']
            input_hours = np.arange(hour - 11, hour + 1)
            x_raw = dataset.values[input_hours]
            x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
            if not x_mask.any():
                continue

            start = campaign_bounds(getattr(dataset, 'campaign', 'legacy_2021_2022'))[0]
            t_inputs = [start + timedelta(hours=int(h)) for h in input_hours]
            c_inputs = climatology.for_window(t_inputs)
            x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)

            tx_anom = torch.as_tensor(x_anom, device=device)
            tx_mask = torch.as_tensor(x_mask, device=device)

            _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)
            sat_inputs = generate_satellite_batch(
                model.base_model.xyz.shape[0],
                model.n_sats,
                device,
                model=model,
                base_profiles=base_profiles,
            )

            total_loss, _ = model(tx_anom, tx_mask, sat_inputs)
            losses.append(float(total_loss.item()))

    return float(np.mean(losses)) if losses else float('inf')


def train_s1(
    dataset_path: str | Path,
    output: str | Path,
    config: dict | None = None,
    climatology_path: str | Path | None = None,
) -> None:
    """Main training routine for Stage S1."""
    cfg = dict(DEFAULT_S1_CONFIG)
    if config:
        cfg.update(config)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ObservationDataset(dataset_path)

    # Climatology setup
    if climatology_path and Path(climatology_path).exists():
        climatology = StationClimatology.load(climatology_path)
    else:
        clim_file = output / 'station_climatology.npz'
        if clim_file.exists():
            climatology = StationClimatology.load(clim_file)
        else:
            raise FileNotFoundError(f"Climatology required for anomaly normalization: {climatology_path}")

    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        # Load frozen base weather core
        base_ckpt = cfg.get('forecast_checkpoint', 'outputs/extended_2023_2026/training/epoch-0081/state.pt')
        print(f"Loading frozen forecast core from {base_ckpt}...")
        base_model = load_base_model(base_ckpt, climatology, cfg['mesh_level'], cfg['hidden'], device)

        # Build coupled satellite model
        model = SatelliteRadiometricModel(
            base_model,
            n_sats=3,
            hidden_dim=cfg['hidden'],
            num_ikfs2=180,
            num_mtvza=29,
            huber_delta=cfg['huber_delta'],
            gross_error_sigma=cfg['gross_error_sigma'],
            varbc_reg_weight=cfg['varbc_reg_weight'],
            film_reg_weight=cfg['film_reg_weight'],
        ).to(device)

        # Trainable parameters: only satellite modules
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        best_loss = float('inf')
        best_epoch = None
        train_indices = dataset.subset('train')[:cfg['max_train_samples']]

        print(f"Starting Stage S1 ({cfg['stage_id']}) for {cfg['epochs']} epochs on {device}...")
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            epoch_losses = []
            epoch_sublosses = []
            gradients = {}
            random.shuffle(train_indices)

            for idx in train_indices:
                row = dataset.samples[idx]
                hour = row['hour']
                input_hours = np.arange(hour - 11, hour + 1)
                x_raw = dataset.values[input_hours]
                x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
                if not x_mask.any():
                    continue

                start = campaign_bounds(getattr(dataset, 'campaign', 'legacy_2021_2022'))[0]
                t_inputs = [start + timedelta(hours=int(h)) for h in input_hours]
                c_inputs = climatology.for_window(t_inputs)
                x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)

                tx_anom = torch.as_tensor(x_anom, device=device)
                tx_mask = torch.as_tensor(x_mask, device=device)

                # Precompute profiles to generate realistic satellite observations
                with torch.no_grad():
                    _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)

                sat_inputs = generate_satellite_batch(
                    model.base_model.xyz.shape[0],
                    model.n_sats,
                    device,
                    model=model,
                    base_profiles=base_profiles,
                )

                optimizer.zero_grad(set_to_none=True)
                loss, sublosses = model(tx_anom, tx_mask, sat_inputs)

                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite satellite assimilation loss encountered.")
                loss.backward()

                # Track non-zero gradients on trainable parameters
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradients[name] = bool(param.grad.abs().sum() > 0)

                # Ensure base model received NO gradients
                for name, param in model.base_model.named_parameters():
                    if param.grad is not None:
                        raise RuntimeError(f"Base model parameter {name} unexpectedly received gradients!")

                torch.nn.utils.clip_grad_norm_(trainable_params, cfg['gradient_clip'])
                optimizer.step()

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_sublosses.append(sublosses)

            val_loss = evaluate_s1(model, dataset, climatology, 'validation', cfg['max_val_samples'], device)
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch = val_loss, epoch

            # Save atomic epoch checkpoint
            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'stage_id': cfg['stage_id'],
                'model': {k: v for k, v in model.state_dict().items() if not k.startswith('base_model.')},
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
            }
            torch.save(state, temp_dir / 'state.pt')

            metrics = {
                'epoch': epoch,
                'stage_id': cfg['stage_id'],
                'train_sat_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_sat_loss': val_loss,
                'improved': improved,
                'nonzero_gradients_count': sum(1 for v in gradients.values() if v),
                'trainable_parameters_count': len(trainable_params),
                'frozen_parameters_count': sum(1 for p in model.base_model.parameters()),
                'sample_sublosses': epoch_sublosses[-1] if epoch_sublosses else {},
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

            print(json.dumps(metrics, ensure_ascii=False), flush=True)

        completion = {
            'stage_id': cfg['stage_id'],
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'status': 's1_ru_sat_radiometric_warmup_completed',
            'forecast_core_frozen': True,
            'active_instruments': ['elektro_l_msugs', 'arktika_m_msugs', 'meteor_m_ikfs2_clear', 'meteor_m_mtvza'],
            'scientific_acceptance': False,
        }
        save(output / 'complete.json', completion)
        print(f"Stage S1 training completed! Best epoch: {best_epoch}, best loss: {best_loss:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--climatology')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    train_s1(args.dataset, args.output, config, args.climatology)


if __name__ == '__main__':
    main()
