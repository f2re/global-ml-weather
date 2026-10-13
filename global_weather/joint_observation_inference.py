"""Immutable batches of causal, physical joint-observation research forecasts.

Run on the compute node. Forecast windows read past inputs only; GRIB2 encodes
all 37 standard pressure levels with independent masks and a field inventory.
Mesh-8 checkpoints may not be rolled beyond the horizon that was explicitly
qualified during training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import torch

from .grib_export import MISSING_CENTRE, export_forecast, sha256
from .observation_data import NATIVE_VARIABLES, UNITS
from .observations import utc
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS


def selected_issues(dataset, *, issues=None, split='validation', limit=3):
    """Select dates deterministically without opening any target window."""
    if issues:
        dates = [utc(value) for value in issues]
    else:
        dates = list(dataset.issues(split, limit))
    if not dates or len(set(dates)) != len(dates):
        raise ValueError('A nonempty list of distinct forecast issues is required.')
    if any(value.minute or value.second or value.microsecond for value in dates):
        raise ValueError('Forecast issues require whole UTC hours.')
    return sorted(dates)


def validate_history(values, mask, profiles, issue):
    """Reject malformed, late or future inputs before their assimilation."""
    if (values.ndim != 3 or values.shape[0] != 12 or values.shape[-1] != 6
            or values.shape != mask.shape or mask.dtype != np.bool_):
        raise ValueError('Need 12-hour native surface values and boolean variable masks.')
    if not np.isfinite(values[mask]).all():
        raise ValueError('Valid surface input is nonfinite.')
    for row in profiles:
        if not (issue - timedelta(hours=12) < utc(row['observed_at']) <= issue
                and utc(row['available_at']) <= issue):
            raise ValueError('Future, late or outside-history profile input.')
    if not mask.any() or not profiles:
        raise ValueError('Joint forecast needs both admitted surface and profile inputs.')


def timed_forecast(model, values, mask, profiles, issue, *, horizon_hours, device,
                   warmups=1, repeats=3):
    from .joint_observation_training import normalized_history
    if warmups < 0 or repeats < 1:
        raise ValueError('Warmups >= 0 and repeats >= 1 are required.')
    inputs, masks = normalized_history(model, values, mask, issue)
    inputs, masks = inputs.to(device), masks.to(device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def forward():
        return model(profiles, issue, surface_inputs=inputs, surface_mask=masks,
                     horizon_hours=horizon_hours)

    with torch.inference_mode():
        for _ in range(warmups):
            forward()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        seconds = []
        frames = None
        for _ in range(repeats):
            frames = None
            sync()
            started = time.perf_counter()
            frames = forward()
            sync()
            seconds.append(time.perf_counter() - started)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return frames, {
        'timing_scope': 'batch_one_model_only_all_requested_leads_both_heads_no_io',
        'warmups': warmups, 'repeats': repeats, 'seconds': seconds,
        'median_seconds': float(np.median(seconds)),
        'p95_seconds': float(np.percentile(seconds, 95)),
        'parameters_total': sum(p.numel() for p in model.parameters()),
        'mesh_level': model.grid.level, 'grid_cells': model.grid.n_cells,
        'nominal_area_scale_km': model.grid.nominal_area_scale_km,
        'grid_area_method': model.grid.area_method,
        'device': str(device), 'torch_version': str(torch.__version__),
        'python_version': platform.python_version(), 'precision': 'float32',
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        'process_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024),
        'rss_scope': 'whole_process_including_loading_not_training',
    }


def save_issue(dataset, model, state, checkpoint_hash, issue, destination, *,
               horizon_hours=72, device=torch.device('cpu'), warmups=1, repeats=3,
               spacing=5., centre=MISSING_CENTRE, subcentre=0, source_revision=None,
               static_fields=None):
    """Stage one complete bundle, then atomically publish without overwriting."""
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Forecast issue output already exists.')
    values, input_mask, profiles = dataset.forecast_history(
        issue, limit=state['config'].get('max_records_per_window', 2000))
    validate_history(values, input_mask, profiles, issue)
    from .profile_physics import screen_humidity
    profiles, input_qc = screen_humidity(profiles, model.physical_policy)
    frames, telemetry = timed_forecast(model, values, input_mask, profiles, issue,
                                       horizon_hours=horizon_hours, device=device,
                                       warmups=warmups, repeats=repeats)
    leads = np.asarray([frame.lead_hours for frame in frames], dtype=np.int32)
    expected = np.arange(0, horizon_hours + 1, 3)
    if not np.array_equal(leads, expected):
        raise ValueError('Model rollout does not contain the complete 3-hour horizon.')
    prediction = np.stack([frame.profiles.cpu().numpy() for frame in frames])
    mask = np.stack([frame.profile_variable_mask.cpu().numpy() for frame in frames])
    if prediction.shape != (len(leads), model.grid.n_cells, 37, 6) or mask.shape != prediction.shape:
        raise ValueError('Model must output every standard level and profile variable.')
    if mask.dtype != np.bool_ or not np.isfinite(prediction[mask]).all() or mask[..., 5].any():
        raise ValueError('Nonfinite issued profile or unsupported omega issuance.')
    if not np.isfinite(prediction[..., :5]).all():
        raise FloatingPointError('Nonfinite joint physical prediction.')
    if static_fields is None:
        raise ValueError('Verified static elevation is required for geometric issuance masking.')
    from .joint_forecast_physics import apply_forecast_physics
    prediction, mask, diagnostic_physics = apply_forecast_physics(
        prediction, mask, model.grid, static_fields['elevation_m'])
    surface = torch.stack([frame.native_physical for frame in frames]).detach().cpu().numpy()
    if surface.shape != (len(leads), len(dataset.surface.latlon), 6):
        raise ValueError('Surface prediction geometry or variable count differs.')
    if not np.isfinite(surface).all():
        raise FloatingPointError('Nonfinite station-native physical prediction.')
    if (np.any(surface[..., 0] <= 0) or np.any(surface[..., 1] <= 0)
            or np.any(surface[..., 1] > surface[..., 0])
            or np.any(surface[..., 4] <= 0) or np.any(surface[..., 5] <= 0)):
        raise FloatingPointError('Station-native prediction left the reviewed thermodynamic domain.')
    # These are learned station-native products. An impossible value is an
    # inference failure, not something the issuance mask is allowed to hide.
    surface_mask = np.ones(surface.shape, dtype=bool)
    surface_physics = {
        'temperature_min_k': float(surface[..., 0].min()),
        'dewpoint_min_k': float(surface[..., 1].min()),
        'dewpoint_minus_temperature_max_k': float((surface[..., 1]-surface[..., 0]).max()),
        'surface_pressure_min_pa': float(surface[..., 4].min()),
        'mean_sea_level_pressure_min_pa': float(surface[..., 5].min()),
        'domain_violations': 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.joint-issue-', dir=destination.parent) as staging:
        stage = Path(staging) / 'product'
        stage.mkdir()
        np.savez_compressed(stage / 'inputs.npz', surface=values, surface_mask=input_mask,
                            station_latlon=dataset.surface.latlon)
        (stage / 'profile_inputs.json').write_text(json.dumps(profiles, allow_nan=False) + '\n')
        np.savez_compressed(stage / 'forecast.npz', profiles=prediction,
                            profile_variable_mask=mask, pressure_pa=np.asarray(PRESSURE_HPA) * 100,
                            xyz=model.grid.xyz, lead_hours=leads, surface=surface,
                            surface_issue_mask=surface_mask, station_latlon=dataset.surface.latlon,
                            terrain_elevation_m=static_fields['elevation_m'],
                            profile_training_support=np.asarray(state['trained_variable_mask'], dtype=bool))
        manifest = {
            'schema': 'joint-observation-research-forecast-1', 'issue_time': issue.isoformat(),
            'lead_hours': leads.tolist(), 'valid_times': [frame.valid_time.isoformat() for frame in frames],
            'profile_variables': list(PROFILE_VARIABLES), 'profile_units': list(PROFILE_UNITS),
            'surface_variables': list(NATIVE_VARIABLES), 'surface_units': list(UNITS),
            'profile_dimensions': ['lead', 'cell', 'pressure', 'variable'],
            'surface_dimensions': ['lead', 'station', 'variable'],
            'pressure_pa': (np.asarray(PRESSURE_HPA) * 100).tolist(),
            'wind_basis': 'local_enu_vector', 'grid_fingerprint': model.grid.fingerprint,
            'mesh_level': model.grid.level, 'grid_cells': model.grid.n_cells,
            'nominal_area_scale_km': model.grid.nominal_area_scale_km,
            'grid_area_method': model.grid.area_method,
            'checkpoint_sha256': checkpoint_hash, 'checkpoint_epoch': state['epoch'],
            'checkpoint_config': state['config'], 'identity': state['identity'],
            'normalization_sha256': state['identity'].get('norm_sha256'),
            'profile_training_support': state.get('trained_variable_mask'),
            'diagnostic_physics': diagnostic_physics,
            'surface_physics': surface_physics,
            'static_provenance': {key: value for key, value in static_fields.items()
                                  if key not in ('elevation_m', 'land_mask', 'slope', 'xyz')},
            'field_origins': {**{variable: 'learned_measured_joint_profile' for variable in PROFILE_VARIABLES[:5]},
                              'geopotential': 'learned_anchor_with_hydrostatic_projection',
                              'omega': 'diagnostic_pressure_continuity'},
            'source_revision': source_revision, 'inference_source_sha256': sha256(Path(__file__)),
            'forecast_npz_sha256': sha256(stage / 'forecast.npz'),
            'input_provenance': {'dataset_fingerprint': dataset.fingerprint,
                                 'surface_input_sha256': sha256(stage / 'inputs.npz'),
                                 'profile_input_sha256': sha256(stage / 'profile_inputs.json'),
                                 'profile_scalar_count': len(profiles),
                                 'profile_input_qc': input_qc,
                                 'surface_valid_scalar_count': int(input_mask.sum())},
            'input_history_hours': 12, 'future_targets_read': False,
            'scientific_acceptance': False, 'data_kind': 'real_observations_research_forecast',
            'forecast_status': 'research_dense_neural_fields_unaccepted',
            'below_ground_mask_status': 'geometric_forecast_geopotential_above_verified_terrain; no_global_surface_pressure',
            'limitations': [
                'Dense neural profile predictions do not imply dense observation support or verified accuracy at every level.',
                'Nominal area scale is a geometry diagnostic, not a demonstrated meteorological resolution.',
                'All 37 levels are encoded; omega is a continuity diagnostic with an imposed 1 hPa boundary, not a learned target.',
                'Geometric issuance masking uses forecast geopotential and verified elevation; it does not replace target-pressure training masks.',
                'Surface outputs remain station-native; sensor heights are unknown. GRIB2 exports supported MSLP only.',
                'Nearest-cell product sampling does not increase model resolution or conserve area integrals.',
                'Historical observation availability follows the admitted archive assumption, not verified operational delays.',
            ],
            'telemetry': telemetry, 'started_utc': started_utc,
            'execution_argv': list(sys.argv), 'execution_working_directory': str(Path.cwd()),
        }
        (stage / 'forecast.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        grib = export_forecast(stage, stage / 'forecast.grib2', spacing=spacing,
                               centre=centre, subcentre=subcentre)
        manifest['grib_sha256'] = grib['grib_sha256']
        manifest['grib_inventory_sha256'] = sha256(stage / 'forecast.grib2.json')
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['pipeline_seconds'] = time.perf_counter() - started
        (stage / 'completion.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Forecast issue output appeared during execution.')
        os.rename(stage, destination)
    return {'issue_time': issue.isoformat(), 'directory': destination.name,
            'forecast_npz_sha256': manifest['forecast_npz_sha256'],
            'grib_sha256': grib['grib_sha256'], 'messages': len(grib['messages']),
            'median_model_seconds': telemetry['median_seconds'],
            'pipeline_seconds': manifest['pipeline_seconds']}


def forecast_batch(dataset, training, output, *, issues=None, split='validation', limit=3,
                   horizon_hours=None, device='auto', warmups=1, repeats=3, threads=8,
                   spacing=5., centre=MISSING_CENTRE, subcentre=0, source_revision=None,
                   static=None):
    from .joint_observation_training import load_frozen
    if threads < 1:
        raise ValueError('threads must be positive.')
    output = Path(output)
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise FileExistsError('Choose a new batch output without symlink parents.')
    torch.set_num_threads(threads)
    selected = torch.device('cuda:0' if device == 'auto' and torch.cuda.is_available()
                            else 'cpu' if device == 'auto' else device)
    model, joint = load_frozen(dataset, training, device=selected)
    state, checkpoint_hash = model.frozen_metadata, model.frozen_checkpoint_sha256
    qualified_horizon = int(state['config'].get('rollout_hours', 72))
    if horizon_hours is None:
        horizon_hours = qualified_horizon
    if type(horizon_hours) is not int or horizon_hours < 3 or horizon_hours > 72 or horizon_hours % 3:
        raise ValueError('Need a 3-hour horizon within 3..72.')
    if horizon_hours > qualified_horizon:
        raise ValueError('Requested inference horizon exceeds the checkpoint-qualified rollout.')
    if model.grid.level == 8:
        if horizon_hours != 3:
            raise ValueError('Mesh-8 qualification checkpoint is limited to the reviewed 3-hour inference path.')
        if selected.type != 'cuda':
            raise ValueError('Mesh-8 end-to-end inference qualification requires CUDA.')
        if limit > 2:
            raise ValueError('Mesh-8 qualification limits one batch to at most two issues.')
    from .joint_static import load_static
    if static is None:
        raise ValueError('A verified prepared static artifact is required.')
    static_fields = load_static(static, model.grid)
    if static_fields['static_sha256'] != state['identity'].get('static_fingerprint'):
        raise ValueError('Forecast terrain differs from the trained static identity.')
    dates = selected_issues(joint, issues=issues, split=split, limit=limit)
    output.mkdir(parents=True)
    report = {'schema': 'joint-research-forecast-batch-1', 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_sha256': checkpoint_hash, 'checkpoint_epoch': state['epoch'],
              'source_revision': source_revision, 'horizon_hours': horizon_hours,
              'qualified_horizon_hours': qualified_horizon,
              'mesh_level': model.grid.level, 'grid_cells': model.grid.n_cells,
              'grid_fingerprint': model.grid.fingerprint,
              'nominal_area_scale_km': model.grid.nominal_area_scale_km,
              'scientific_acceptance': False, 'issues_requested': [x.isoformat() for x in dates],
              'products': []}

    def save():
        temporary = output / '.batch.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, output / 'batch.json')

    save()
    try:
        for issue in dates:
            row = save_issue(joint, model, state, checkpoint_hash, issue,
                             output / issue.strftime('%Y%m%dT%H%M%SZ'),
                             horizon_hours=horizon_hours, device=selected,
                             warmups=warmups, repeats=repeats, spacing=spacing,
                             centre=centre, subcentre=subcentre, source_revision=source_revision,
                             static_fields=static_fields)
            report['products'].append(row)
            save()
        joint.verify()
        rechecked_static = load_static(static, model.grid)
        if (rechecked_static['static_sha256'] != static_fields['static_sha256']
                or rechecked_static['manifest_sha256'] != static_fields['manifest_sha256']):
            raise ValueError('Static artifact changed during forecast batch.')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--static', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--issue', dest='issues', action='append')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--horizon-hours', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--spacing-degrees', dest='spacing', type=float, default=5.)
    parser.add_argument('--centre', type=int, default=MISSING_CENTRE)
    parser.add_argument('--subcentre', type=int, default=0)
    parser.add_argument('--source-revision')
    print(json.dumps(forecast_batch(**vars(parser.parse_args(argv))), allow_nan=False))


if __name__ == '__main__':
    main()
