"""Scientific physical evaluation of Mesh-8 2000-epoch checkpoint against baselines.

Evaluates:
  - Surface variables: t2m, td2m, u10, v10, surface_pressure, mslp
  - Profile variables: temperature, specific_humidity, u, v, geopotential
Across validation and test splits with exact RMSE, MAE, Bias and observation counts.
Computes persistence and climatology baselines, skill scores, and compares with epoch 500.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from global_weather.joint_observation_data import JointObservationDataset
from global_weather.joint_observation_training import load_frozen, normalized_history, supported_issues
from global_weather.observation_data import NATIVE_VARIABLES, UNITS
from global_weather.observations import utc
from global_weather.profile_physics import screen_humidity
from global_weather.profile_training import VARIABLES
from global_weather.profile_training_v2 import _prediction
from global_weather.vertical import PRESSURE_HPA, PROFILE_UNITS


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


def evaluate_split(model, dataset, config, split: str, device: torch.device):
    policy = model.physical_policy
    rollout_hours = config['rollout_hours']
    limit_issues = config['max_' + split + '_issues']
    issues = supported_issues(dataset, split, limit_issues, rollout_hours)

    # Accumulators for errors (prediction - observation)
    mod_surf_diffs = [[] for _ in range(6)]
    pers_surf_diffs = [[] for _ in range(6)]
    clim_surf_diffs = [[] for _ in range(6)]

    mod_prof_diffs = [[] for _ in range(5)]
    pers_prof_diffs = [[] for _ in range(5)]
    clim_prof_diffs = [[] for _ in range(5)]

    print(f"[{split}] Evaluating {len(issues)} issues with rollout {rollout_hours}h...")
    for issue_idx, issue in enumerate(issues, 1):
        print(f"  Issue {issue_idx}/{len(issues)}: {issue.isoformat()}")
        try:
            raw, mask, target, target_mask, inputs, targets = dataset.sample(
                issue, config['max_records_per_window'], target_horizon_hours=rollout_hours
            )
        except TypeError:
            raw, mask, target, target_mask, inputs, targets = dataset.sample(
                issue, config['max_records_per_window']
            )

        x, m = normalized_history(model, raw, mask, issue)
        inputs_screened, _ = screen_humidity(inputs, policy)
        cutoff = utc(issue) + timedelta(hours=rollout_hours)
        targets = [row for row in targets if utc(row['observed_at']) <= cutoff]
        targets_screened, _ = screen_humidity(targets, policy)

        steps = rollout_hours // 3
        target = target[:steps]
        target_mask = target_mask[:steps]

        with torch.inference_mode():
            frames = model(inputs_screened, issue, surface_inputs=x, surface_mask=m, horizon_hours=rollout_hours)

        # 1. Surface evaluation
        pers_surf = get_persistence_surface(raw, mask)
        for step_idx, frame in enumerate(frames[1:]):
            month = frame.valid_time.month - 1
            if hasattr(frame, 'native_physical') and frame.native_physical is not None:
                mod_surf = frame.native_physical.cpu().numpy()
            else:
                mod_surf = (frame.native_normalized * model.surface_scales + model.surface_monthly[month]).cpu().numpy()

            clim_surf = model.surface_monthly[month].cpu().numpy()

            for var_idx in range(6):
                valid = target_mask[step_idx, :, var_idx]
                if not np.any(valid):
                    continue
                obs = target[step_idx, valid, var_idx]
                mod = mod_surf[valid, var_idx]
                mod_diff = mod - obs
                mod_surf_diffs[var_idx].extend(mod_diff.tolist())

                # Persistence
                pers = pers_surf[valid, var_idx]
                pers_valid = np.isfinite(pers)
                if np.any(pers_valid):
                    pers_diff = pers[pers_valid] - obs[pers_valid]
                    pers_surf_diffs[var_idx].extend(pers_diff.tolist())

                # Climatology
                clim = clim_surf[valid, var_idx]
                clim_valid = np.isfinite(clim)
                if np.any(clim_valid):
                    clim_diff = clim[clim_valid] - obs[clim_valid]
                    clim_surf_diffs[var_idx].extend(clim_diff.tolist())

        # 2. Profile evaluation
        pers_prof_map = get_persistence_profiles(inputs)
        clim_prof_frame = get_climate_profile_frame(model, issue + timedelta(hours=rollout_hours))

        for record in targets_screened:
            var_idx = VARIABLES.index(record['variable'])
            if var_idx == 1 and record['pressure_pa'] < policy.humidity_min_pressure_pa:
                continue
            _, std, supported = model.normalization.at(var_idx, record['pressure_pa'])
            if not supported:
                continue

            result = _prediction(frames, model.grid, record, model.pressure_pa, 3)
            if result is None:
                continue
            prediction_val = float(result[0].cpu().numpy())
            obs_val = float(record['value'])
            mod_prof_diffs[var_idx].append(prediction_val - obs_val)

            # Persistence
            pers_key = (record['variable'], record.get('profile_id'), float(record['pressure_pa']))
            if pers_key in pers_prof_map:
                pers_prof_diffs[var_idx].append(pers_prof_map[pers_key] - obs_val)

            # Climatology
            if clim_prof_frame is not None:
                res_clim = _prediction([clim_prof_frame], model.grid, record, model.pressure_pa, 3)
                if res_clim is not None:
                    clim_val = float(res_clim[0].cpu().numpy())
                    clim_prof_diffs[var_idx].append(clim_val - obs_val)

    # Compile results
    surface_results = {}
    surface_pers = {}
    surface_clim = {}
    for var_idx, name in enumerate(NATIVE_VARIABLES):
        res = compute_metrics(mod_surf_diffs[var_idx])
        res['unit'] = UNITS[var_idx]
        surface_results[name] = res

        pers_res = compute_metrics(pers_surf_diffs[var_idx])
        pers_res['unit'] = UNITS[var_idx]
        surface_pers[name] = pers_res

        clim_res = compute_metrics(clim_surf_diffs[var_idx])
        clim_res['unit'] = UNITS[var_idx]
        surface_clim[name] = clim_res

    profile_results = {}
    profile_pers = {}
    profile_clim = {}
    for var_idx, name in enumerate(VARIABLES):
        res = compute_metrics(mod_prof_diffs[var_idx])
        res['unit'] = PROFILE_UNITS[var_idx]
        profile_results[name] = res

        pers_res = compute_metrics(pers_prof_diffs[var_idx])
        pers_res['unit'] = PROFILE_UNITS[var_idx]
        profile_pers[name] = pers_res

        clim_res = compute_metrics(clim_prof_diffs[var_idx])
        clim_res['unit'] = PROFILE_UNITS[var_idx]
        profile_clim[name] = clim_res

    # Skill scores
    surface_skill = {}
    for name in NATIVE_VARIABLES:
        m_rmse = surface_results[name]['rmse']
        p_rmse = surface_pers[name]['rmse']
        c_rmse = surface_clim[name]['rmse']
        surface_skill[name] = {
            'vs_persistence': float(1.0 - m_rmse / p_rmse) if m_rmse and p_rmse and p_rmse > 0 else None,
            'vs_climatology': float(1.0 - m_rmse / c_rmse) if m_rmse and c_rmse and c_rmse > 0 else None,
        }

    profile_skill = {}
    for name in VARIABLES:
        m_rmse = profile_results[name]['rmse']
        p_rmse = profile_pers[name]['rmse']
        c_rmse = profile_clim[name]['rmse']
        profile_skill[name] = {
            'vs_persistence': float(1.0 - m_rmse / p_rmse) if m_rmse and p_rmse and p_rmse > 0 else None,
            'vs_climatology': float(1.0 - m_rmse / c_rmse) if m_rmse and c_rmse and c_rmse > 0 else None,
        }

    return {
        'model': {'surface': surface_results, 'profile': profile_results},
        'baselines': {
            'persistence': {'surface': surface_pers, 'profile': profile_pers},
            'climatology': {'surface': surface_clim, 'profile': profile_clim},
        },
        'skill_scores': {'surface': surface_skill, 'profile': profile_skill},
        'issues_evaluated': [issue.isoformat() for issue in issues],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help="Path to observation dataset directory")
    parser.add_argument('--training', default="outputs/mesh8-500-run/training", help="Training root with checkpoints")
    parser.add_argument('--output', default="outputs/mesh8-500-run/evaluation_metrics_epoch2000.json")
    parser.add_argument('--baseline-epoch500', default="outputs/mesh8-500-run/evaluation_metrics.json")
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args(argv)

    print(f"Loading frozen model from {args.training} on device {args.device}...")
    model, dataset = load_frozen(args.dataset, args.training, device=args.device)
    config = model.frozen_metadata['config']
    epoch = model.frozen_metadata['epoch']
    sha256_hash = model.frozen_checkpoint_sha256
    print(f"Frozen checkpoint loaded: epoch {epoch}, SHA256: {sha256_hash}")

    eval_out = {
        'schema': 'mesh8-physical-evaluation-2',
        'epoch': epoch,
        'checkpoint_sha256': sha256_hash,
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mesh_level': model.grid.level,
        'grid_cells': model.grid.n_cells,
        'nominal_area_scale_km': model.grid.nominal_area_scale_km,
        'rollout_hours': config.get('rollout_hours', 3),
        'device': str(model.xyz.device),
    }

    # Evaluate validation and test
    val_metrics = evaluate_split(model, dataset, config, 'validation', model.xyz.device)
    test_metrics = evaluate_split(model, dataset, config, 'test', model.xyz.device)

    eval_out['validation'] = val_metrics
    eval_out['test'] = test_metrics

    # Compare with Epoch 500 if available
    epoch500_path = Path(args.baseline_epoch500)
    if epoch500_path.is_file():
        print(f"Comparing with Epoch 500 baseline from {epoch500_path}...")
        epoch500 = json.loads(epoch500_path.read_text())
        comparison = {}
        for split in ('validation', 'test'):
            comparison[split] = {'surface': {}, 'profile': {}}
            if split in epoch500:
                # Surface comparison
                e500_surf = epoch500[split].get('surface', {})
                e2000_surf = eval_out[split]['model']['surface']
                for var, vdata in e2000_surf.items():
                    if var in e500_surf and e500_surf[var].get('rmse') is not None:
                        rmse500 = e500_surf[var]['rmse']
                        rmse2000 = vdata['rmse']
                        impr = (rmse500 - rmse2000) / rmse500 * 100.0
                        comparison[split]['surface'][var] = {
                            'epoch_500_rmse': rmse500,
                            'epoch_2000_rmse': rmse2000,
                            'improvement_pct': float(impr),
                            'unit': vdata['unit'],
                        }
                # Profile comparison
                e500_prof = epoch500[split].get('profile', {})
                e2000_prof = eval_out[split]['model']['profile']
                for var, vdata in e2000_prof.items():
                    if var in e500_prof and e500_prof[var].get('rmse') is not None:
                        rmse500 = e500_prof[var]['rmse']
                        rmse2000 = vdata['rmse']
                        impr = (rmse500 - rmse2000) / rmse500 * 100.0
                        comparison[split]['profile'][var] = {
                            'epoch_500_rmse': rmse500,
                            'epoch_2000_rmse': rmse2000,
                            'improvement_pct': float(impr),
                            'unit': vdata['unit'],
                        }
        eval_out['epoch_500_comparison'] = comparison

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_out, indent=2, allow_nan=False) + '\n')
    print(f"Evaluation metrics written to {output_path}")


if __name__ == '__main__':
    main()
