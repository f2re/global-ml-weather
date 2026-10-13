"""Verify preserved evidence from the final mesh8 R10 Q4/Q5 execution.

Passing this gate means the frozen mesh8 path trained once, reloaded its best
checkpoint and produced physically diagnosed +3 h products on CUDA. It does
NOT establish meteorological skill or authorise the later 72 h experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .joint_observation_model import ARCHITECTURE as JOINT_ARCHITECTURE
from .joint_observation_training import STATUS as TRAINING_STATUS
from .mesh8_preflight import CANONICAL_MESH8_FINGERPRINT, _path
from .observation_training import digest, save
from .profile_hydrostatic_model import ARCHITECTURE as PROFILE_ARCHITECTURE

HYDROSTATIC_MAX_ABS_M2_S2 = 5.0
CONTINUITY_MAX_ABS_S1 = 1e-6


def _json(path):
    path = _path(path, directory=False)
    return json.loads(path.read_text())


def _same_sha(path, expected, label):
    path = _path(path, directory=False)
    actual = digest(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(label+' SHA256 differs from preserved manifest.')
    return actual


def _surface_evidence(path, manifest):
    """Independently verify physical station-native values in the hashed NPZ."""
    path = _path(path, directory=False)
    with np.load(path, allow_pickle=False) as archive:
        required = {'surface', 'surface_issue_mask', 'lead_hours'}
        if not required.issubset(archive.files):
            raise ValueError('Forecast NPZ lacks station-native physical evidence.')
        surface = archive['surface']
        issue_mask = archive['surface_issue_mask']
        leads = archive['lead_hours']
    if (surface.ndim != 3 or surface.shape[0] != 2 or surface.shape[-1] != 6
            or issue_mask.shape != surface.shape or issue_mask.dtype != np.bool_
            or not issue_mask.all() or not np.array_equal(leads, np.array([0, 3], dtype=leads.dtype))
            or not np.isfinite(surface).all()):
        raise ValueError('Q5 station-native output geometry/mask/leads are invalid.')
    violations = ((surface[..., 0] <= 0) | (surface[..., 1] <= 0)
                  | (surface[..., 1] > surface[..., 0])
                  | (surface[..., 4] <= 0) | (surface[..., 5] <= 0))
    if np.any(violations):
        raise ValueError('Q5 saved station-native output left the reviewed thermodynamic domain.')
    declared = manifest.get('surface_physics', {})
    if declared.get('domain_violations') != 0:
        raise ValueError('Q5 surface-physics manifest does not confirm zero domain violations.')
    return {
        'temperature_min_k': float(surface[..., 0].min()),
        'dewpoint_min_k': float(surface[..., 1].min()),
        'dewpoint_minus_temperature_max_k': float((surface[..., 1]-surface[..., 0]).max()),
        'surface_pressure_min_pa': float(surface[..., 4].min()),
        'mean_sea_level_pressure_min_pa': float(surface[..., 5].min()),
        'domain_violations': 0,
    }


def verify(run_root, expected_sha, output=None):
    root = _path(run_root, directory=True)
    preflight = _json(root/'preflight.json')
    if (preflight.get('status') != 'ready_for_q4_q5'
            or preflight.get('source_commit') != expected_sha
            or preflight.get('mesh', {}).get('fingerprint') != CANONICAL_MESH8_FINGERPRINT
            or preflight.get('joint_architecture') != JOINT_ARCHITECTURE
            or preflight.get('profile_architecture') != PROFILE_ARCHITECTURE):
        raise ValueError('Preflight evidence does not authorise this source/mesh/physical architecture.')

    training = _path(root/'training', directory=True)
    complete = _json(training/'complete.json')
    if (complete.get('status') != TRAINING_STATUS
            or complete.get('mesh_level') != 8 or complete.get('grid_cells') != 655362
            or complete.get('rollout_hours') != 3 or complete.get('scientific_acceptance') is not False):
        raise ValueError('Q4 completion does not match the frozen R10 training contract.')
    identity = complete.get('identity', {})
    config = identity.get('config', {})
    if (identity.get('grid_fingerprint') != CANONICAL_MESH8_FINGERPRINT
            or identity.get('experiment_id') != 'joint-mesh8-qualification-r10'
            or identity.get('experiment_role') != 'mesh8_end_to_end_qualification'
            or identity.get('architecture') != JOINT_ARCHITECTURE
            or identity.get('rollout_hours') != 3
            or config.get('mesh_level') != 8
            or config.get('physical_policy', {}).get('architecture') != 'hydrostatic_flow'
            or not isinstance(identity.get('joint_source_sha256'), dict)
            or 'profile_hydrostatic_model.py' not in identity['joint_source_sha256']
            or 'joint_observation_model.py' not in identity['joint_source_sha256']):
        raise ValueError('Q4 frozen physical-v2 identity differs from R10.')
    best = _json(training/'best.json')
    if best.get('epoch') != complete.get('best_epoch'):
        raise ValueError('Published best checkpoint does not match complete.json.')
    state = training/best['directory']/'state.pt'
    _same_sha(state, best.get('sha256'), 'Best checkpoint')
    metrics = _json(training/best['directory']/'metrics.json')
    if (metrics.get('cuda_peak_allocated_bytes') is None
            or metrics.get('cuda_peak_reserved_bytes') is None
            or metrics.get('mesh_level') != 8 or metrics.get('rollout_hours') != 3
            or metrics.get('scientific_acceptance') is not False):
        raise ValueError('Q4 lacks measured CUDA telemetry for the mesh8 run.')
    if not metrics.get('nonzero_gradients') or not all(metrics['nonzero_gradients'].values()):
        raise ValueError('Q4 does not prove nonzero gradients through every trainable branch.')

    forecast = _path(root/'forecast', directory=True)
    batch = _json(forecast/'batch.json')
    if (batch.get('status') != 'complete' or batch.get('source_revision') != expected_sha
            or batch.get('mesh_level') != 8 or batch.get('grid_fingerprint') != CANONICAL_MESH8_FINGERPRINT
            or batch.get('horizon_hours') != 3 or batch.get('scientific_acceptance') is not False
            or not batch.get('products')):
        raise ValueError('Q5 forecast batch differs from the frozen R10 identity.')

    products = []
    for product in batch['products']:
        directory = product.get('directory')
        if not isinstance(directory, str) or not directory or '/' in directory or '\\' in directory or directory in ('.', '..'):
            raise ValueError('Q5 product directory is not a single immutable child name.')
        issue = _path(forecast/directory, directory=True)
        if issue.parent != forecast:
            raise ValueError('Q5 product escaped the forecast root.')
        manifest = _json(issue/'completion.json')
        if (manifest.get('source_revision') != expected_sha
                or manifest.get('grid_fingerprint') != CANONICAL_MESH8_FINGERPRINT
                or manifest.get('lead_hours') != [0, 3]
                or manifest.get('scientific_acceptance') is not False):
            raise ValueError('Q5 issue manifest differs from R10.')
        physics = manifest.get('diagnostic_physics', {})
        humidity = physics.get('specific_humidity', {})
        hydro = physics.get('hydrostatic_residual', {})
        continuity = physics.get('continuity_residual', {})
        if humidity.get('physical_domain_violations') != 0:
            raise ValueError('Q5 contains specific-humidity physical-domain violations.')
        if (hydro.get('count', 0) <= 0 or hydro.get('max_abs_m2_s2') is None
                or hydro['max_abs_m2_s2'] > HYDROSTATIC_MAX_ABS_M2_S2):
            raise ValueError('Q5 hydrostatic residual exceeds the technical R10 gate.')
        if (continuity.get('count', 0) <= 0 or continuity.get('max_abs_s-1') is None
                or continuity['max_abs_s-1'] > CONTINUITY_MAX_ABS_S1):
            raise ValueError('Q5 pressure-continuity residual exceeds the technical R10 gate.')
        npz_path = issue/'forecast.npz'
        npz_hash = _same_sha(npz_path, manifest.get('forecast_npz_sha256'), 'Forecast NPZ')
        surface = _surface_evidence(npz_path, manifest)
        grib_hash = _same_sha(issue/'forecast.grib2', manifest.get('grib_sha256'), 'Forecast GRIB2')
        inventory_hash = _same_sha(issue/'forecast.grib2.json', manifest.get('grib_inventory_sha256'), 'GRIB inventory')
        products.append({'issue_time': manifest['issue_time'], 'directory': directory,
                         'forecast_npz_sha256': npz_hash, 'grib_sha256': grib_hash,
                         'grib_inventory_sha256': inventory_hash,
                         'hydrostatic_max_abs_m2_s2': hydro['max_abs_m2_s2'],
                         'continuity_max_abs_s-1': continuity['max_abs_s-1'],
                         'humidity_domain_violations': 0,
                         'surface_physics': surface,
                         'telemetry': manifest.get('telemetry')})

    report = {
        'schema': 'mesh8-r10-final-verification-1',
        'status': 'r10_q4_q5_technically_qualified',
        'scientific_acceptance': False,
        'source_commit': expected_sha,
        'joint_architecture': JOINT_ARCHITECTURE,
        'profile_architecture': PROFILE_ARCHITECTURE,
        'mesh_level': 8, 'grid_cells': 655362,
        'grid_fingerprint': CANONICAL_MESH8_FINGERPRINT,
        'qualified_horizon_hours': 3,
        'best_epoch': best['epoch'], 'best_checkpoint_sha256': best['sha256'],
        'training_cuda_peak_allocated_bytes': metrics['cuda_peak_allocated_bytes'],
        'training_cuda_peak_reserved_bytes': metrics['cuda_peak_reserved_bytes'],
        'products': products,
        'next_stage_authorized': False,
        'next_stage_requirement': 'new experiment identity for 72 h plus measured memory/streaming design and independent skill verification',
    }
    destination = _path(output) if output else root/'final-verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    save(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.run_root, args.expected_sha, args.output), allow_nan=False))


if __name__ == '__main__':
    main()
