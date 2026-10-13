"""Verify preserved evidence from the final mesh8 R10 Q4/Q5 execution.

Passing this gate means the frozen mesh8 path trained once, reloaded its best
checkpoint and produced physically diagnosed +3 h products on CUDA. It does
NOT establish meteorological skill or authorise the later 72 h experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .joint_observation_training import STATUS as TRAINING_STATUS
from .mesh8_preflight import CANONICAL_MESH8_FINGERPRINT
from .observation_training import digest, save

HYDROSTATIC_MAX_ABS_M2_S2 = 5.0
CONTINUITY_MAX_ABS_S1 = 1e-6


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Required immutable R10 JSON is missing or a symlink: '+str(path))
    return json.loads(path.read_text())


def _same_sha(path, expected, label):
    actual = digest(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(label+' SHA256 differs from preserved manifest.')
    return actual


def verify(run_root, expected_sha, output=None):
    root = Path(run_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('R10 run root is missing or a symlink.')
    preflight = _json(root/'preflight.json')
    if (preflight.get('status') != 'ready_for_q4_q5'
            or preflight.get('source_commit') != expected_sha
            or preflight.get('mesh', {}).get('fingerprint') != CANONICAL_MESH8_FINGERPRINT):
        raise ValueError('Preflight evidence does not authorise this source/mesh.')

    training = root/'training'
    complete = _json(training/'complete.json')
    if (complete.get('status') != TRAINING_STATUS
            or complete.get('mesh_level') != 8 or complete.get('grid_cells') != 655362
            or complete.get('rollout_hours') != 3 or complete.get('scientific_acceptance') is not False):
        raise ValueError('Q4 completion does not match the frozen R10 training contract.')
    identity = complete.get('identity', {})
    if (identity.get('grid_fingerprint') != CANONICAL_MESH8_FINGERPRINT
            or identity.get('experiment_id') != 'joint-mesh8-qualification-r10'):
        raise ValueError('Q4 frozen identity differs from R10.')
    best = _json(training/'best.json')
    if best.get('epoch') != complete.get('best_epoch'):
        raise ValueError('Published best checkpoint does not match complete.json.')
    state = training/best['directory']/'state.pt'
    _same_sha(state, best.get('sha256'), 'Best checkpoint')
    metrics = _json(training/best['directory']/'metrics.json')
    if (metrics.get('cuda_peak_allocated_bytes') is None
            or metrics.get('cuda_peak_reserved_bytes') is None
            or metrics.get('mesh_level') != 8 or metrics.get('rollout_hours') != 3):
        raise ValueError('Q4 lacks measured CUDA telemetry for the mesh8 run.')
    if not metrics.get('nonzero_gradients') or not all(metrics['nonzero_gradients'].values()):
        raise ValueError('Q4 does not prove nonzero gradients through every trainable branch.')

    forecast = root/'forecast'
    batch = _json(forecast/'batch.json')
    if (batch.get('status') != 'complete' or batch.get('source_revision') != expected_sha
            or batch.get('mesh_level') != 8 or batch.get('grid_fingerprint') != CANONICAL_MESH8_FINGERPRINT
            or batch.get('horizon_hours') != 3 or batch.get('scientific_acceptance') is not False
            or not batch.get('products')):
        raise ValueError('Q5 forecast batch differs from the frozen R10 identity.')

    products = []
    for product in batch['products']:
        issue = forecast/product['directory']
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
        npz_hash = _same_sha(issue/'forecast.npz', manifest.get('forecast_npz_sha256'), 'Forecast NPZ')
        grib_hash = _same_sha(issue/'forecast.grib2', manifest.get('grib_sha256'), 'Forecast GRIB2')
        inventory_hash = _same_sha(issue/'forecast.grib2.json', manifest.get('grib_inventory_sha256'), 'GRIB inventory')
        products.append({'issue_time': manifest['issue_time'], 'directory': product['directory'],
                         'forecast_npz_sha256': npz_hash, 'grib_sha256': grib_hash,
                         'grib_inventory_sha256': inventory_hash,
                         'hydrostatic_max_abs_m2_s2': hydro['max_abs_m2_s2'],
                         'continuity_max_abs_s-1': continuity['max_abs_s-1'],
                         'humidity_domain_violations': 0,
                         'telemetry': manifest.get('telemetry')})

    report = {
        'schema': 'mesh8-r10-final-verification-1',
        'status': 'r10_q4_q5_technically_qualified',
        'scientific_acceptance': False,
        'source_commit': expected_sha,
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
    destination = Path(output).resolve() if output else root/'final-verification.json'
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
