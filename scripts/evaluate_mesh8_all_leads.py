"""Scientific physical evaluation of Mesh-8 across all lead horizons (3h to 72h).

Evaluates every 3-hour lead step:
  - Surface variables: t2m, td2m, u10, v10, surface_pressure, mslp
  - Profile variables: temperature, specific_humidity, u, v, geopotential
Across validation and test splits with exact RMSE, MAE, Bias and observation counts.
Computes persistence and climatology baselines and skill scores for each lead horizon.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from global_weather.grid import unit_xyz
from global_weather.joint_observation_data import JointObservationDataset
from global_weather.joint_observation_training import (
    _build,
    _device,
    _state,
    normalized_history,
    supported_issues,
)
from global_weather.observation_data import NATIVE_VARIABLES, UNITS
from global_weather.observation_training import checkpoint_path
from global_weather.observations import utc
from global_weather.profile_normalization import load_normalization
from global_weather.profile_physics import screen_humidity
from global_weather.profile_training import VARIABLES
from global_weather.profile_training_v2 import _prediction
from global_weather.station_climatology import StationClimatology
from global_weather.vertical import PRESSURE_HPA, PROFILE_UNITS


LEAD_HOURS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72]
MIN_PROFILE_COUNT = 25


def compute_skill_score(model_metric: dict, baseline_metric: dict, min_count: int = 1) -> float | None:
    if (
        model_metric.get("count", 0) < min_count
        or baseline_metric.get("count", 0) < min_count
        or model_metric.get("rmse") is None
        or baseline_metric.get("rmse") is None
        or baseline_metric["rmse"] <= 1e-12
    ):
        return None
    return float(1.0 - (model_metric["rmse"] / baseline_metric["rmse"]))



def load_model_checkpoint(dataset_path, training, device='auto'):
    training = Path(training)
    completion = json.loads((training / 'complete.json').read_text())
    ref = json.loads((training / 'best.json').read_text())
    state = torch.load(checkpoint_path(training, ref), map_location='cpu', weights_only=True)
    expected = completion['identity']

    dataset = JointObservationDataset(dataset_path)
    norms = load_normalization(json.loads(Path(expected['norm_path']).read_text()))
    station = StationClimatology.load(expected['station_climatology_path'])
    dev = _device(device)
    model = _build(dataset, norms, expected['config'], station, dev, expected['climatology_path'], expected.get('static_path'))
    _state(model, state['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.trained_variable_mask = np.asarray(state['trained_variable_mask'], dtype=bool)
    model.frozen_metadata = {
        'config': expected['config'],
        'epoch': state['epoch'],
        'identity': expected,
        'trained_variable_mask': state['trained_variable_mask'],
    }
    model.frozen_checkpoint_sha256 = ref['sha256']
    return model, dataset


def frame_to_cpu(frame):
    for attr in dir(frame):
        if not attr.startswith('__'):
            val = getattr(frame, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(frame, attr, val.cpu())
    return frame


def streaming_forward(model, inputs, issue, surface_inputs, surface_mask, horizon_hours=72):
    """Memory-efficient streaming forward pass moving large profile grids to host RAM."""
    model._surface_history = (surface_inputs, surface_mask)
    try:
        issue = utc(issue)
        n, d = model.grid.n_cells, model.hidden
        pressure = model.pressure_encoder(model.log_pressure) + model.levels
        state = model._initial_state(issue, pressure)

        # 12-hour input assimilation
        for hour in range(12):
            rows = [
                r for r in inputs
                if issue - timedelta(hours=12 - hour) < utc(r['observed_at']) <= issue - timedelta(hours=11 - hour)
                and utc(r['available_at']) <= issue
            ]
            rows = [r for r in rows if model.normalization.at(r['variable'], r['pressure_pa'])[2]]
            if not rows:
                continue
            variables = torch.tensor([VARIABLES.index(r['variable']) for r in rows], device=state.device)
            pressures = np.array([r['pressure_pa'] for r in rows])
            levels = np.abs(np.log(pressures[:, None]) - np.log(np.array(PRESSURE_HPA)[None] * 100)).argmin(1)
            cells = model.grid.locate([r['latitude'] for r in rows], [r['longitude'] for r in rows])
            indices = torch.tensor(cells * 37 + levels, device=state.device)
            normalized = [model.normalization.normalize(r['variable'], r['value'], r['pressure_pa']) for r in rows]
            xyz = unit_xyz([r['latitude'] for r in rows], [r['longitude'] for r in rows])
            extra = np.column_stack((
                normalized,
                np.log(pressures / 100000),
                [(issue - utc(r['observed_at'])).total_seconds() / 43200 for r in rows],
                xyz,
            ))
            features = torch.tensor(extra, dtype=state.dtype, device=state.device)
            tokens = model.token(torch.cat((features, model.variable(variables)), -1))
            sums = state.new_zeros(n * 37, d).index_add(0, indices, tokens)
            counts = state.new_zeros(n * 37).index_add(0, indices, torch.ones(len(rows), device=state.device))
            has_obs = (counts > 0)
            if has_obs.any():
                obs_idx = has_obs.nonzero(as_tuple=True)[0]
                sub_forcing = sums[obs_idx] / counts[obs_idx, None]
                flat_state = state.reshape(-1, d)
                sub_state = flat_state[obs_idx]
                sub_updated = model.ingest(sub_forcing, sub_state)
                flat_state = flat_state.clone()
                flat_state[obs_idx] = sub_updated
                state = flat_state.reshape(n, 37, d)

        # Initial frame (lead 0) offloaded to CPU
        frames = [frame_to_cpu(model._decode(state, issue, 0))]

        # Autoregressive rollout: advance on GPU, move decoded fields to CPU
        for lead in range(3, horizon_hours + 1, 3):
            state = model._advance_state(state, pressure, issue, lead)
            frames.append(frame_to_cpu(model._decode(state, issue, lead)))

        return frames
    finally:
        model._surface_history = None


def get_persistence_surface(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Last available native observation across past 12-hour history."""
    result = np.full(raw.shape[1:], np.nan, dtype=np.float32)
    for hour in range(len(raw)):
        result = np.where(mask[hour], raw[hour], result)
    return result


def get_persistence_profiles(inputs: list[dict]) -> dict:
    """Map of (variable, profile_id, pressure_pa) to latest observed physical value."""
    latest = {}
    for row in inputs:
        key = (row['variable'], row.get('profile_id'), float(row['pressure_pa']))
        if key not in latest or utc(row['observed_at']) > utc(latest[key]['observed_at']):
            latest[key] = row
    return {key: float(row['value']) for key, row in latest.items()}


def get_climate_profile_frame(model, when):
    """3D NOAA seasonal climatology profile frame for observation interpolation."""
    if not hasattr(model, 'climatology'):
        return None
    mean, mask = model.climatology.sample(when)
    values = np.concatenate((mean, np.full(mean.shape[:-1] + (1,), np.nan)), -1)
    support = np.concatenate((mask, np.zeros(mask.shape[:-1] + (1,), bool)), -1)
    from types import SimpleNamespace
    return SimpleNamespace(
        valid_time=when,
        profiles=torch.as_tensor(values, dtype=torch.float32),
        profile_variable_mask=torch.as_tensor(support),
        profile_mask=torch.as_tensor(mask.any(-1)),
        wind_basis='local_enu_vector',
    )


def compute_metrics(diffs: list[float]) -> dict:
    if not diffs:
        return {'rmse': None, 'mae': None, 'bias': None, 'count': 0}
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        'rmse': float(np.sqrt(np.mean(arr ** 2))),
        'mae': float(np.mean(np.abs(arr))),
        'bias': float(np.mean(arr)),
        'count': int(len(arr)),
    }


def evaluate_split(model, dataset, config, split: str, device: torch.device, horizon_hours: int = 72):
    policy = model.physical_policy
    limit_issues = config['max_' + split + '_issues']
    issues = supported_issues(dataset, split, limit_issues, horizon_hours)

    leads = [lead for lead in LEAD_HOURS if lead <= horizon_hours]
    num_steps = len(leads)

    lead_surf_diffs = {
        lead: {
            'model': [[] for _ in range(6)],
            'persistence': [[] for _ in range(6)],
            'climatology': [[] for _ in range(6)],
        }
        for lead in leads
    }

    lead_prof_diffs = {
        lead: {
            'model': [[] for _ in range(5)],
            'persistence': [[] for _ in range(5)],
            'climatology': [[] for _ in range(5)],
        }
        for lead in leads
    }

    print(f"[{split}] Evaluating {len(issues)} issues across {num_steps} lead horizons (3h..{horizon_hours}h)...")
    for issue_idx, issue in enumerate(issues, 1):
        print(f"  Issue {issue_idx}/{len(issues)}: {issue.isoformat()}")
        raw, mask, target, target_mask, inputs, targets = dataset.sample(
            issue, config['max_records_per_window'], target_horizon_hours=horizon_hours
        )

        x, m = normalized_history(model, raw, mask, issue)
        inputs_screened, _ = screen_humidity(inputs, policy)
        cutoff = utc(issue) + timedelta(hours=horizon_hours)
        targets = [row for row in targets if utc(row['observed_at']) <= cutoff]
        targets_screened, _ = screen_humidity(targets, policy)

        target = target[:num_steps]
        target_mask = target_mask[:num_steps]

        with torch.inference_mode():
            frames = streaming_forward(model, inputs_screened, issue, surface_inputs=x, surface_mask=m, horizon_hours=horizon_hours)

        # 1. Surface evaluation across all leads
        pers_surf = get_persistence_surface(raw, mask)
        for step_idx, frame in enumerate(frames[1:num_steps + 1]):
            lead_h = leads[step_idx]
            month = frame.valid_time.month - 1
            if hasattr(frame, 'native_physical') and frame.native_physical is not None:
                mod_surf = frame.native_physical.numpy()
            else:
                mod_surf = (frame.native_normalized * model.surface_scales + model.surface_monthly[month]).cpu().numpy()

            clim_surf = model.surface_monthly[month].cpu().numpy()

            for var_idx in range(6):
                valid = target_mask[step_idx, :, var_idx]
                if not np.any(valid):
                    continue
                obs = target[step_idx, valid, var_idx]
                mod = mod_surf[valid, var_idx]
                lead_surf_diffs[lead_h]['model'][var_idx].extend((mod - obs).tolist())

                # Persistence
                pers = pers_surf[valid, var_idx]
                pers_valid = np.isfinite(pers)
                if np.any(pers_valid):
                    lead_surf_diffs[lead_h]['persistence'][var_idx].extend((pers[pers_valid] - obs[pers_valid]).tolist())

                # Climatology
                clim = clim_surf[valid, var_idx]
                clim_valid = np.isfinite(clim)
                if np.any(clim_valid):
                    lead_surf_diffs[lead_h]['climatology'][var_idx].extend((clim[clim_valid] - obs[clim_valid]).tolist())

        # 2. Profile evaluation across all leads
        pers_prof_map = get_persistence_profiles(inputs)
        clim_prof_frames = [
            get_climate_profile_frame(model, issue + timedelta(hours=h))
            for h in [0] + leads
        ]
        clim_prof_frames = [f for f in clim_prof_frames if f is not None]

        # Keep model.pressure_pa on CPU for CPU observation operator
        pressure_pa_cpu = model.pressure_pa.cpu()
        for record in targets_screened:
            var_idx = VARIABLES.index(record['variable'])
            if var_idx == 1 and record['pressure_pa'] < policy.humidity_min_pressure_pa:
                continue
            _, std, supported = model.normalization.at(var_idx, record['pressure_pa'])
            if not supported:
                continue

            dt_hours = (utc(record['observed_at']) - issue).total_seconds() / 3600.0
            step_bin = int(np.clip(round(dt_hours / 3.0), 1, num_steps))
            lead_h = leads[step_bin - 1]

            result = _prediction(frames, model.grid, record, pressure_pa_cpu, 3)
            if result is None:
                continue
            prediction_val = float(result[0])
            obs_val = float(record['value'])
            lead_prof_diffs[lead_h]['model'][var_idx].append(prediction_val - obs_val)

            # Persistence
            pers_key = (record['variable'], record.get('profile_id'), float(record['pressure_pa']))
            if pers_key in pers_prof_map:
                lead_prof_diffs[lead_h]['persistence'][var_idx].append(pers_prof_map[pers_key] - obs_val)

            # Climatology
            if clim_prof_frames:
                res_clim = _prediction(clim_prof_frames, model.grid, record, pressure_pa_cpu, 3)
                if res_clim is not None:
                    clim_val = float(res_clim[0])
                    lead_prof_diffs[lead_h]['climatology'][var_idx].append(clim_val - obs_val)

        # Garbage collect frame memory before next issue
        del frames
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Compile results per lead
    leads_output = {}
    for lead_h in leads:
        surf_model = {}
        surf_pers = {}
        surf_clim = {}
        surf_skill = {}
        for var_idx, name in enumerate(NATIVE_VARIABLES):
            m_res = compute_metrics(lead_surf_diffs[lead_h]['model'][var_idx])
            m_res['unit'] = UNITS[var_idx]
            surf_model[name] = m_res

            p_res = compute_metrics(lead_surf_diffs[lead_h]['persistence'][var_idx])
            p_res['unit'] = UNITS[var_idx]
            surf_pers[name] = p_res

            c_res = compute_metrics(lead_surf_diffs[lead_h]['climatology'][var_idx])
            c_res['unit'] = UNITS[var_idx]
            surf_clim[name] = c_res

            surf_skill[name] = {
                'vs_persistence': compute_skill_score(m_res, p_res, min_count=10),
                'vs_climatology': compute_skill_score(m_res, c_res, min_count=10),
            }

        prof_model = {}
        prof_pers = {}
        prof_clim = {}
        prof_skill = {}
        for var_idx, name in enumerate(VARIABLES):
            m_res = compute_metrics(lead_prof_diffs[lead_h]['model'][var_idx])
            m_res['unit'] = PROFILE_UNITS[var_idx]
            m_res['sufficient_sample'] = bool(m_res['count'] >= MIN_PROFILE_COUNT)
            prof_model[name] = m_res

            p_res = compute_metrics(lead_prof_diffs[lead_h]['persistence'][var_idx])
            p_res['unit'] = PROFILE_UNITS[var_idx]
            p_res['sufficient_sample'] = bool(p_res['count'] >= MIN_PROFILE_COUNT)
            prof_pers[name] = p_res

            c_res = compute_metrics(lead_prof_diffs[lead_h]['climatology'][var_idx])
            c_res['unit'] = PROFILE_UNITS[var_idx]
            c_res['sufficient_sample'] = bool(c_res['count'] >= MIN_PROFILE_COUNT)
            prof_clim[name] = c_res

            prof_skill[name] = {
                'vs_persistence': compute_skill_score(m_res, p_res, min_count=MIN_PROFILE_COUNT),
                'vs_climatology': compute_skill_score(m_res, c_res, min_count=MIN_PROFILE_COUNT),
            }

        leads_output[str(lead_h)] = {
            'lead_hours': lead_h,
            'model': {'surface': surf_model, 'profile': prof_model},
            'baselines': {
                'persistence': {'surface': surf_pers, 'profile': prof_pers},
                'climatology': {'surface': surf_clim, 'profile': prof_clim},
            },
            'skill_scores': {'surface': surf_skill, 'profile': prof_skill},
        }

    return {
        'leads': leads_output,
        'issues_evaluated': [issue.isoformat() for issue in issues],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help="Path to observation dataset directory")
    parser.add_argument('--training', default="outputs/mesh8-500-run/training", help="Training root with checkpoints")
    parser.add_argument('--output', default="outputs/mesh8-500-run/evaluation_metrics_all_leads_epoch2000.json")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--horizon-hours', type=int, default=72)
    args = parser.parse_args(argv)

    print(f"Loading checkpoint from {args.training} on device {args.device}...")
    model, dataset = load_model_checkpoint(args.dataset, args.training, device=args.device)
    config = model.frozen_metadata['config']
    epoch = model.frozen_metadata['epoch']
    sha256_hash = model.frozen_checkpoint_sha256
    print(f"Checkpoint loaded: epoch {epoch}, SHA256: {sha256_hash}")

    eval_out = {
        'schema': 'mesh8-all-leads-physical-evaluation-1',
        'epoch': epoch,
        'checkpoint_sha256': sha256_hash,
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mesh_level': model.grid.level,
        'grid_cells': model.grid.n_cells,
        'nominal_area_scale_km': model.grid.nominal_area_scale_km,
        'horizon_hours': args.horizon_hours,
        'device': str(model.xyz.device),
    }

    # Evaluate validation and test across all leads
    val_metrics = evaluate_split(model, dataset, config, 'validation', model.xyz.device, args.horizon_hours)
    test_metrics = evaluate_split(model, dataset, config, 'test', model.xyz.device, args.horizon_hours)

    eval_out['validation'] = val_metrics
    eval_out['test'] = test_metrics

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_out, indent=2, allow_nan=False) + '\n')
    print(f"Full horizon evaluation metrics written to {output_path}")


if __name__ == '__main__':
    main()
