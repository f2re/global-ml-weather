"""Stage S2: Joint unfreezing of weather forecast core and Russian satellite assimilation operators.

Trains the unfreezed weather core (StationAnomalyModel) jointly with satellite modules
(MSU-GS, GSICS, FiLM, IKFS-2, MTVZA, VarBC, AMV) using a composite loss functional:
L_total = L_surface_anomaly + 0.25 * L_satellite + 1.0 * L_hydrostatic.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
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
from .observation_model import LEAD_HOURS
from .observation_training import checkpoint_path, digest, save, sync_directory
from .station_anomaly_model import StationAnomalyModel, anomaly_loss
from .station_climatology import StationClimatology
from .multimodal.ru_sat_msugs import ArktikaElektroMSUGSAdapter, ViewConditionedFiLMBlock
from .analysis.observation_operator import IKFS2CloudClearedOperator, MeteorMTVZAAllSkyOperator
from .losses import SatelliteAssimilationLoss
from .vertical import hydrostatic_residual
from .satellite_radiometric_training import generate_satellite_batch

DEFAULT_S2_CONFIG = {
    'schema': 'satellite-stage-s2-config-1',
    'stage_id': 'S2_joint_clearsky_and_amv',
    'experiment_id': 's2-joint-clearsky-and-amv-2024',
    'freeze_forecast_core': False,
    'mesh_level': 4,
    'hidden': 32,
    'epochs': 15,
    'learning_rate': 1e-4,
    'weight_decay': 0.001,
    'gradient_clip': 1.0,
    'hydrostatic_loss_weight': 1.0,
    'sat_assimilation_loss_weight': 0.25,
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


class JointSatelliteStage2Model(nn.Module):
    """Jointly trained assimilation and forecast model for Stage S2."""

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
        hydrostatic_loss_weight: float = 1.0,
        sat_assimilation_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        # Forecast core is unfrozen and receives full gradients in Stage S2
        for param in self.base_model.parameters():
            param.requires_grad = True

        self.n_sats = n_sats
        self.hidden_dim = hidden_dim
        self.film_reg_weight = film_reg_weight
        self.hydro_weight = hydrostatic_loss_weight
        self.sat_weight = sat_assimilation_loss_weight

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

        # 4. Satellite loss functional
        self.sat_loss_fn = SatelliteAssimilationLoss(
            huber_delta=huber_delta,
            gross_error_sigma=gross_error_sigma,
            varbc_reg_weight=varbc_reg_weight,
        )

    def forward(
        self,
        history_anomalies: torch.Tensor,
        history_mask: torch.Tensor,
        target_anomalies: torch.Tensor,
        target_mask: torch.Tensor,
        sat_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Base weather core analysis state
        state_0, _ = self.base_model.analysis_state(history_anomalies, history_mask)
        n_nodes, _, d = state_0.shape
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
        geom_features = fused[..., 13:17]
        fused_proj = self.msugs_proj(fused)

        # 3. FiLM modulation of state representation
        h_mean = state_0.mean(dim=1).unsqueeze(0)  # [1, n_nodes, hidden]
        h_mod = self.film_block(h_mean + fused_proj, geom_features)
        loss_film = self.film_reg_weight * F.mse_loss(h_mod, h_mean)

        # Couple modulated representation back into global atmospheric state
        h_diff = (h_mod - h_mean).squeeze(0)  # [n_nodes, hidden]
        state = state_0 + h_diff.unsqueeze(1)  # [n_nodes, 1, hidden] -> broadcast to [n_nodes, 37, hidden]

        # 4. Rollout 72-hour forecast dynamics
        predictions = []
        for _ in LEAD_HOURS:
            v_mean = state.mean(1, keepdim=True).expand_as(state)
            forcing = self.base_model.dynamic(torch.cat([state, self.base_model.graph.neighbours(state), v_mean], -1))
            state = self.base_model.transition(forcing.reshape(-1, d), state.reshape(-1, d)).reshape(n_nodes, 37, d)
            station_context = torch.cat(
                [self.base_model.operator(state.mean(1)), self.base_model.operator(state[:, 0]), self.base_model.operator.station_xyz],
                dim=-1,
            )
            predictions.append(self.base_model.anomaly_head(station_context))

        predicted_anomalies = torch.stack(predictions)
        profiles = self.base_model.profile_head(state)

        # 5. Surface Anomaly Loss
        loss_surface = anomaly_loss(predicted_anomalies, target_anomalies, target_mask)

        # 6. Hydrostatic Balance Loss
        pressure_pa = self.base_model.pressure_hpa * 100.0
        res = hydrostatic_residual(profiles, pressure_pa)
        loss_hydro = (res / 1000.0).square().mean()

        # 7. Satellite Assimilation Loss
        n_obs = min(sat_inputs['obs_tb'].shape[0], n_nodes)
        t_prof = profiles[:n_obs, :, 0]
        q_prof = profiles[:n_obs, :, 1]
        t_skin = profiles[:n_obs, 0, 0:1]
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

        m_prof = torch.cat([t_prof, q_prof, torch.zeros_like(t_prof)], dim=-1)
        wind_10m = torch.norm(profiles[:n_obs, 0, 2:4], dim=-1, keepdim=True)
        surf_state = torch.cat([t_skin, wind_10m, torch.zeros_like(t_skin), torch.zeros_like(t_skin)], dim=-1)

        out_mtvza = self.mtvza_operator(
            sat_inputs['obs_tb'][:n_obs],
            m_prof,
            surf_state,
            varbc_p,
        )

        n_amv = min(sat_inputs['amv_uv_obs'].shape[0], n_obs)
        amv_pred = profiles[:n_amv, 0, 2:4]

        varbc_params = list(self.mtvza_operator.varbc.parameters()) + list(self.ikfs2_operator.varbc.parameters())
        varbc_priors = [torch.zeros_like(p) for p in varbc_params]

        sat_loss_dict = self.sat_loss_fn(
            out_mtvza,
            out_ikfs2,
            amv_pred,
            sat_inputs['amv_uv_obs'][:n_amv],
            sat_inputs['amv_sigma'][:n_amv],
            sat_inputs['amv_qi'][:n_amv],
            varbc_params,
            varbc_priors,
        )
        loss_sat = sat_loss_dict['loss_sat_total'] + loss_film

        # 8. Composite joint loss
        total_loss = loss_surface + self.sat_weight * loss_sat + self.hydro_weight * loss_hydro
        metrics = {
            'loss_total': float(total_loss.item()),
            'loss_surface': float(loss_surface.item()),
            'loss_sat': float(loss_sat.item()),
            'loss_hydro': float(loss_hydro.item()),
            'loss_mtvza': float(sat_loss_dict['loss_mtvza'].item()),
            'loss_ikfs2': float(sat_loss_dict['loss_ikfs2'].item()),
            'loss_amv': float(sat_loss_dict['loss_amv'].item()),
            'loss_varbc_reg': float(sat_loss_dict['loss_varbc_reg'].item()),
            'loss_film': float(loss_film.item()),
        }
        return total_loss, metrics


def load_s2_model(
    base_checkpoint_str: str | Path,
    satellite_checkpoint_str: str | Path,
    climatology: StationClimatology,
    mesh_level: int,
    hidden: int,
    config: dict,
    device: torch.device,
) -> JointSatelliteStage2Model:
    """Initializes JointSatelliteStage2Model by restoring both base forecast and satellite weights."""
    grid = build_mesh8_grid() if mesh_level == 8 else build_grid(mesh_level)
    base_ckpt = torch.load(base_checkpoint_str, map_location=device)

    station_xyz = base_ckpt['model']['operator.station_xyz'].cpu().numpy()
    lat = np.degrees(np.arcsin(np.clip(station_xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(station_xyz[:, 1], station_xyz[:, 0]))
    station_latlon = np.stack([lat, lon], axis=-1)

    base_model = StationAnomalyModel(
        [grid],
        station_latlon,
        climatology.std,
        hidden=hidden,
    ).to(device)
    base_model.load_state_dict(base_ckpt['model'])

    model = JointSatelliteStage2Model(
        base_model,
        n_sats=3,
        hidden_dim=hidden,
        num_ikfs2=180,
        num_mtvza=29,
        huber_delta=config.get('huber_delta', 1.5),
        gross_error_sigma=config.get('gross_error_sigma', 3.5),
        varbc_reg_weight=config.get('varbc_reg_weight', 1e-3),
        film_reg_weight=config.get('film_reg_weight', 1e-3),
        hydrostatic_loss_weight=config.get('hydrostatic_loss_weight', 1.0),
        sat_assimilation_loss_weight=config.get('sat_assimilation_loss_weight', 0.25),
    ).to(device)

    # Load satellite weights from Stage S1
    if Path(satellite_checkpoint_str).is_file():
        sat_ckpt = torch.load(satellite_checkpoint_str, map_location=device)
        model.load_state_dict(sat_ckpt['model'], strict=False)

    return model


def evaluate_s2(
    model: JointSatelliteStage2Model,
    dataset: ObservationDataset,
    climatology: StationClimatology,
    split: str,
    max_samples: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates composite joint loss on held-out split."""
    model.eval()
    losses = []
    surf_losses = []
    sat_losses = []
    hydro_losses = []
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
            if not x_mask.any() or not y_mask.any():
                continue

            start = campaign_bounds(getattr(dataset, 'campaign', 'legacy_2021_2022'))[0]
            t_inputs = [start + timedelta(hours=int(h)) for h in input_hours]
            t_targets = [start + timedelta(hours=int(h)) for h in target_hours]
            c_inputs = climatology.for_window(t_inputs)
            c_targets = climatology.for_window(t_targets)

            x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)
            y_anom = np.where(y_mask, (y_raw - c_targets) / climatology.std, 0.0).astype(np.float32)

            tx_anom = torch.as_tensor(x_anom, device=device)
            tx_mask = torch.as_tensor(x_mask, device=device)
            ty_anom = torch.as_tensor(y_anom, device=device)
            ty_mask = torch.as_tensor(y_mask, device=device)

            _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)
            sat_inputs = generate_satellite_batch(
                model.base_model.xyz.shape[0],
                model.n_sats,
                device,
                base_profiles=base_profiles,
            )

            total_loss, m = model(tx_anom, tx_mask, ty_anom, ty_mask, sat_inputs)
            losses.append(float(total_loss.item()))
            surf_losses.append(m['loss_surface'])
            sat_losses.append(m['loss_sat'])
            hydro_losses.append(m['loss_hydro'])

    return {
        'val_loss_total': float(np.mean(losses)) if losses else float('inf'),
        'val_loss_surface': float(np.mean(surf_losses)) if surf_losses else float('inf'),
        'val_loss_sat': float(np.mean(sat_losses)) if sat_losses else float('inf'),
        'val_loss_hydro': float(np.mean(hydro_losses)) if hydro_losses else float('inf'),
    }


def train_s2(
    dataset_path: str | Path,
    output: str | Path,
    config: dict | None = None,
    climatology_path: str | Path | None = None,
) -> None:
    """Main training routine for Stage S2."""
    cfg = dict(DEFAULT_S2_CONFIG)
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

        base_ckpt = cfg.get('base_forecast_checkpoint', 'outputs/extended_2023_2026/training/epoch-0081/state.pt')
        sat_ckpt = cfg.get('satellite_checkpoint', 'outputs/s1_satellite_radiometric_warmup/training/epoch-0007/state.pt')
        print(f"Initializing Stage S2 model from base {base_ckpt} and satellite {sat_ckpt}...")

        model = load_s2_model(base_ckpt, sat_ckpt, climatology, cfg['mesh_level'], cfg['hidden'], cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        best_loss = float('inf')
        best_epoch = None
        train_indices = dataset.subset('train')[:cfg['max_train_samples']]

        print(f"Starting Stage S2 ({cfg['stage_id']}) for {cfg['epochs']} epochs on {device}...")
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
                target_hours = hour + np.arange(3, 73, 3)

                x_raw = dataset.values[input_hours]
                y_raw = dataset.values[target_hours]
                x_mask = dataset.mask[input_hours] & (dataset.available[input_hours] <= hour * 3600)
                y_mask = dataset.mask[target_hours]
                if not x_mask.any() or not y_mask.any():
                    continue

                start = campaign_bounds(getattr(dataset, 'campaign', 'legacy_2021_2022'))[0]
                t_inputs = [start + timedelta(hours=int(h)) for h in input_hours]
                t_targets = [start + timedelta(hours=int(h)) for h in target_hours]
                c_inputs = climatology.for_window(t_inputs)
                c_targets = climatology.for_window(t_targets)

                x_anom = np.where(x_mask, (x_raw - c_inputs) / climatology.std, 0.0).astype(np.float32)
                y_anom = np.where(y_mask, (y_raw - c_targets) / climatology.std, 0.0).astype(np.float32)

                tx_anom = torch.as_tensor(x_anom, device=device)
                tx_mask = torch.as_tensor(x_mask, device=device)
                ty_anom = torch.as_tensor(y_anom, device=device)
                ty_mask = torch.as_tensor(y_mask, device=device)

                with torch.no_grad():
                    _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)

                sat_inputs = generate_satellite_batch(
                    model.base_model.xyz.shape[0],
                    model.n_sats,
                    device,
                    base_profiles=base_profiles,
                )

                optimizer.zero_grad(set_to_none=True)
                loss, sublosses = model(tx_anom, tx_mask, ty_anom, ty_mask, sat_inputs)

                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint loss encountered.")
                loss.backward()

                # Track non-zero gradients on both base core and satellite parameters
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = bool(param.grad.abs().sum() > 0)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                optimizer.step()

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_sublosses.append(sublosses)

            eval_res = evaluate_s2(model, dataset, climatology, 'validation', cfg['max_val_samples'], device)
            val_loss = eval_res['val_loss_total']
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch = val_loss, epoch

            # Save atomic epoch checkpoint
            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'stage_id': cfg['stage_id'],
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
            }
            torch.save(state, temp_dir / 'state.pt')

            base_grad_count = sum(1 for k, v in gradients.items() if k.startswith('base_model.') and v)
            sat_grad_count = sum(1 for k, v in gradients.items() if not k.startswith('base_model.') and v)

            metrics = {
                'epoch': epoch,
                'stage_id': cfg['stage_id'],
                'train_total_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_total_loss': val_loss,
                'val_metrics': eval_res,
                'improved': improved,
                'base_model_nonzero_gradients': base_grad_count,
                'satellite_nonzero_gradients': sat_grad_count,
                'total_parameters': sum(1 for _ in model.parameters()),
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
            'status': 's2_joint_clearsky_and_amv_completed',
            'forecast_core_frozen': False,
            'active_instruments': ['ghcnh_synop', 'elektro_l_msugs', 'arktika_m_msugs', 'meteor_m_ikfs2_clear', 'meteor_m_mtvza', 'msugs_amv'],
            'scientific_acceptance': False,
        }
        save(output / 'complete.json', completion)
        print(f"Stage S2 training completed! Best epoch: {best_epoch}, best loss: {best_loss:.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--climatology')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    train_s2(args.dataset, args.output, config, args.climatology)


if __name__ == '__main__':
    main()
