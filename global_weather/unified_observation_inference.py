"""Causal research forecast and synchronized benchmark for legacy unified checkpoints.

Run on the compute node. No future observations enter the model; unsupported,
untrained vertical fields are retained only as explicitly masked diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

from .grid import build_grid, latlon
from .observation_data import NATIVE_VARIABLES, UNITS, ObservationDataset, START
from .observations import utc
from .station_climatology import StationClimatology
from .unified_observation_model import UnifiedObservationModel
from .vertical import PRESSURE_HPA

PROFILE_NAMES = ('temperature', 'specific_humidity', 'u', 'v', 'geopotential')
PROFILE_UNITS = ('K', 'kg kg-1', 'm s-1', 'm s-1', 'm2 s-2')


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_hash(path: str | Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Artifact hash mismatch: {Path(path).name}')
    return actual


def causal_history(dataset, climatology: StationClimatology, issue: str):
    """Select only (issue-12h, issue] and enforce recorded availability."""
    when = utc(issue)
    hour_float = (when - START).total_seconds() / 3600
    hour = int(hour_float)
    if hour != hour_float or hour < 11 or hour >= len(dataset.values):
        raise ValueError('Issue must be an archived whole UTC hour with 12 hours of history.')
    if climatology.n_stations != len(dataset.latlon):
        raise ValueError('Station climatology geometry size differs.')
    window = slice(hour - 11, hour + 1)
    values = dataset.values[window]
    mask = dataset.mask[window] & (dataset.available[window] <= hour * 3600)
    if not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError('No usable finite observations at issue time.')
    timestamps = [when - timedelta(hours=11 - i) for i in range(12)]
    anomalies = np.where(mask, (values - climatology.for_window(timestamps)) / climatology.std, 0.)
    return anomalies.astype(np.float32), mask, when


def load_model(checkpoint: Path, expected_sha256: str, station_latlon: np.ndarray,
               dataset_std: np.ndarray, horizon_hours: int, device: torch.device):
    checked_hash(checkpoint, expected_sha256)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']
    leads = list(state['lead_hours'])
    if (not leads or any(type(h) is not int or h <= 0 for h in leads)
            or leads != list(range(leads[0], leads[-1] + leads[0], leads[0]))):
        raise ValueError('Checkpoint lead times must have a uniform positive cadence.')
    if not 1 <= horizon_hours <= 240 or horizon_hours not in leads:
        raise ValueError('Requested horizon must be a checkpoint lead within 1..240 hours.')
    weights = state['model']
    grid = build_grid(int(cfg['mesh_level']))
    model = UnifiedObservationModel(
        grid, station_latlon, weights['surface_scales'], weights['profile_scales'],
        hidden=int(cfg['hidden']), dropout=float(cfg.get('dropout', 0.)),
        climatology_grid_mean=weights['climatology_mean'].numpy(),
        lead_hours=[h for h in leads if h <= horizon_hours],
    )
    # Geometry/scales are matched before loading; load_state_dict would overwrite them.
    for key in ('xyz', 'pressure_hpa', 'station_operator.cells', 'station_operator.station_xyz'):
        if not torch.equal(model.state_dict()[key], weights[key]):
            raise ValueError(f'Checkpoint geometry differs: {key}')
    if not np.array_equal(np.asarray(dataset_std), weights['surface_scales'].numpy()):
        raise ValueError('Checkpoint dataset normalization differs.')
    for key in ('surface_scales', 'profile_scales'):
        if not torch.isfinite(weights[key]).all() or bool((weights[key] <= 0).any()):
            raise ValueError('Invalid checkpoint scales.')
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Nonfinite checkpoint weights or buffers.')
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), state


def benchmark(model, anomalies, mask, when, device, *, warmups=2, repeats=5):
    """Model-only, batch-one timings include rollout and both heads; exclude I/O."""
    if warmups < 0 or repeats < 1:
        raise ValueError('Warmups >= 0 and repeats >= 1 are required.')
    x = torch.as_tensor(anomalies, device=device)
    m = torch.as_tensor(mask, device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def forward():
        return model(x, m, month_idx=when.month - 1, issue_hour=when.hour)

    with torch.inference_mode():
        for _ in range(warmups):
            forward()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        forecast = None
        for _ in range(repeats):
            # Free the preceding output, avoiding an artificial double-output peak.
            forecast = None
            sync()
            start = time.perf_counter()
            forecast = forward()
            sync()
            timings.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    telemetry = {
        'timing_scope': 'batch_one_model_only_all_requested_leads_both_heads_no_io',
        'precision': 'float32', 'warmups': warmups, 'repeats': repeats,
        'seconds': timings, 'median_seconds': float(np.median(timings)),
        'p95_seconds': float(np.percentile(timings, 95)),
        'parameters_total': sum(p.numel() for p in model.parameters()),
        'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'process_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024),
        'rss_scope': 'whole_process_high_water_mark_including_archive_loading_not_training',
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'torch_version': str(torch.__version__), 'python_version': platform.python_version(),
        'threads': torch.get_num_threads(),
    }
    return forecast, telemetry


def forecast(dataset, checkpoint, checkpoint_sha256, station_climatology,
             station_climatology_sha256, issue, output, *, horizon_hours=72,
             device='auto', warmups=2, repeats=5, threads=8, source_revision=None):
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    if threads < 1:
        raise ValueError('threads must be positive.')
    torch.set_num_threads(threads)
    selected_device = torch.device('cuda:0' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    output = Path(output)
    # A forecast is immutable: do not silently replace earlier issue products.
    if output.exists():
        raise FileExistsError(f'Output directory exists: {output}')
    dataset = ObservationDataset(dataset)
    clim_hash = checked_hash(station_climatology, station_climatology_sha256)
    climatology = StationClimatology.load(station_climatology)
    anomalies, input_mask, when = causal_history(dataset, climatology, issue)
    model, state = load_model(Path(checkpoint), checkpoint_sha256, dataset.latlon, dataset.std, horizon_hours, selected_device)
    prediction, telemetry = benchmark(model, anomalies, input_mask, when, selected_device, warmups=warmups, repeats=repeats)
    leads = np.asarray(prediction.lead_hours, dtype=np.int32)
    valid_times = [when + timedelta(hours=int(lead)) for lead in leads]
    surface = prediction.surface_anomalies.float().cpu().numpy() * climatology.std + climatology.for_window(valid_times)
    profile_anomaly = prediction.profile_anomalies.float().cpu().numpy()
    # Buffer contains scaled issue-month background; its missing support was discarded
    # by legacy training. These values therefore remain strictly diagnostic.
    scales = model.profile_scales.cpu().numpy()
    profile = (profile_anomaly + model.climatology_mean[when.month - 1].cpu().numpy()) * scales
    if not np.isfinite(surface).all() or not np.isfinite(profile).all():
        raise FloatingPointError('Nonfinite forecast output.')
    issue_mask = np.isfinite(surface)
    issue_mask[..., [0, 1, 4, 5]] &= surface[..., [0, 1, 4, 5]] > 0
    output.mkdir(parents=True)
    arrays_path = output / 'forecast.npz'
    np.savez_compressed(arrays_path, surface=surface, profile=profile,
                        surface_anomalies=prediction.surface_anomalies.float().cpu().numpy(),
                        profile_anomalies=profile_anomaly, surface_issue_mask=issue_mask,
                        profile_issue_mask=np.zeros(profile.shape, dtype=bool),
                        input_mask=input_mask, station_latlon=dataset.latlon,
                        grid_latlon=latlon(model.grid.xyz), pressure_hpa=np.asarray(PRESSURE_HPA),
                        lead_hours=leads, station_elevation_m=model.station_elevation.cpu().numpy())
    manifest = {
        'schema': 'unified-research-forecast-1', 'issue_time': when.isoformat(),
        'valid_times': [t.isoformat() for t in valid_times],
        'surface_variables': list(NATIVE_VARIABLES), 'surface_units': list(UNITS),
        'profile_variables': list(PROFILE_NAMES), 'profile_units': list(PROFILE_UNITS),
        'surface_dimensions': ['lead', 'station', 'variable'],
        'profile_dimensions': ['lead', 'cell', 'pressure', 'variable'],
        'lead_hours': leads.tolist(), 'pressure_hpa': list(PRESSURE_HPA),
        'station_ids': [s['id'] for s in dataset.manifest['stations']],
        'grid_fingerprint': model.grid.fingerprint, 'grid_type': 'icosahedral_voronoi_native',
        'checkpoint_sha256': checkpoint_sha256, 'checkpoint_epoch': state['epoch'],
        'checkpoint_config': state['config'],
        'inference_source_sha256': sha256(Path(__file__)),
        'model_source_sha256': sha256(Path(__file__).with_name('unified_observation_model.py')),
        'execution_working_directory': str(Path.cwd()),
        'execution_argv': list(sys.argv),
        'station_climatology_sha256': clim_hash, 'dataset_manifest_sha256': sha256(dataset.root / 'dataset.json'),
        'dataset_hourly_sha256': dataset.manifest['hourly_sha256'],
        'dataset_norm_sha256': dataset.manifest['norm_sha256'],
        'forecast_npz_sha256': sha256(arrays_path), 'source_revision': source_revision,
        'input_history_hours': 12, 'future_observations_used': False, 'wind_basis': 'east_north_local',
        'scientific_acceptance': False, 'data_kind': 'real_observations_research_forecast',
        'forecast_status': 'diagnostic_only_unaccepted_checkpoint',
        'limitations': [
            'Legacy training used 240-hour targets with dataset split guards designed for 72 hours; validation contamination invalidates independent skill claims.',
            'All outputs are diagnostic products; serialization compatibility is not scientific or operational acceptance.',
            'Legacy checkpoint does not bind training source revision or station climatology hash; supplied hashes identify current artifacts only.',
            'Profile head lacks supervised profile training; profile_issue_mask is false everywhere.',
            'Profile diagnostic background uses issue month and lacks original climatology support masks.',
            'Surface outputs are station-native; station elevation is not a 2 m or 10 m measurement-height declaration.',
            'No globally forecast surface pressure is available to establish below-ground profile issuance masks.',
        ],
        'telemetry': telemetry, 'started_utc': start_utc,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'forecast_pipeline_seconds_before_manifest_write': time.perf_counter() - start,
    }
    (output / 'forecast.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--station-climatology', required=True)
    parser.add_argument('--station-climatology-sha256', required=True)
    parser.add_argument('--issue', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon-hours', type=int, default=72)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--source-revision')
    args = vars(parser.parse_args(argv))
    print(json.dumps(forecast(**args), allow_nan=False))


if __name__ == '__main__':
    main()
