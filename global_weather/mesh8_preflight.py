"""Fail-closed preflight for the final mesh8 R10 CUDA qualification.

This command performs no training and no network access. It verifies the exact
source revision, frozen experiment envelope, full mesh8 fingerprint, admitted
GHCNh+IGRA bundle, profile normalization provenance, seasonal context, static
terrain, CUDA memory and disk reserve. Only a successful JSON report with
``status=ready_for_q4_q5`` authorises the fixed R10 runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from .grid import build_mesh8_grid
from .joint_observation_data import JointObservationDataset
from .joint_observation_model import ARCHITECTURE as JOINT_ARCHITECTURE
from .joint_observation_training import MESH8_ROLE, _config
from .joint_static import load_static
from .mesh8_climatology import Mesh8SeasonalClimatology
from .observation_training import digest, save
from .profile_hydrostatic_model import ARCHITECTURE as PROFILE_ARCHITECTURE
from .profile_normalization import load_normalization

# Qualified after replacing the environment-sensitive 1e-12 float-byte hash
# with mesh8-geometry-q1e-7-v1. Levels 0..7 keep their historical identities.
CANONICAL_MESH8_FINGERPRINT = '64c3e32f792a42ad7d363635e5217cd3c845743a9b92487fdc226da5f1937aa9'
MINIMUM_FREE_DISK_GIB = 64


def _path(path, *, directory=None):
    """Return an absolute path only after rejecting every symlink component."""
    candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError('R10 paths and parents must not be symlinks: '+str(component))
    if directory is True and not candidate.is_dir():
        raise ValueError('Required R10 directory is missing: '+str(candidate))
    if directory is False and not candidate.is_file():
        raise ValueError('Required R10 file is missing: '+str(candidate))
    return candidate


def _git_identity(root, expected_sha):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if not isinstance(expected_sha, str) or len(expected_sha) != 40 or actual != expected_sha:
        raise ValueError('Checked-out source revision differs from --expected-sha.')
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True)
    if dirty.strip():
        raise ValueError('Tracked source tree is dirty; R10 requires an immutable revision.')
    return actual


def _cuda_identity(config, require_cuda=True):
    budget = int(config['physical_policy']['memory_budget_mib']) * 1024**2
    if not require_cuda:
        return {'required': False, 'available': bool(torch.cuda.is_available()),
                'declared_memory_budget_bytes': budget}
    if not torch.cuda.is_available():
        raise ValueError('R10 Q4/Q5 requires CUDA; no CUDA device is available.')
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    total = int(properties.total_memory)
    if total < budget:
        raise ValueError(f'CUDA device memory {total} bytes is below declared R10 budget {budget}.')
    uuid = None
    try:
        line = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader', '-i', '0'], text=True).strip()
        uuid = line.splitlines()[0] if line else None
    except (OSError, subprocess.CalledProcessError):
        pass
    return {'required': True, 'available': True, 'index': 0,
            'name': properties.name, 'uuid': uuid,
            'compute_capability': [int(properties.major), int(properties.minor)],
            'total_memory_bytes': total, 'declared_memory_budget_bytes': budget,
            'torch': torch.__version__}


def _normalization(dataset, norm_path):
    norms = load_normalization(json.loads(norm_path.read_text()))
    expected = {
        'dataset_manifest_sha256': dataset.profile.manifest_sha256,
        **{key: dataset.profile.manifest[key]
           for key in ('database_sha256', 'source_sha256', 'admission_sha256')},
    }
    if norms.payload.get('source_identity') != expected:
        raise ValueError('Profile normalization belongs to another admitted IGRA dataset.')
    norms.verify_sources()
    return norms, expected


def preflight(dataset_path, norm_path, climatology_path, static_path, config_path,
              run_root, expected_sha, *, report_path=None, require_cuda=True,
              minimum_free_disk_gib=MINIMUM_FREE_DISK_GIB):
    root = Path(__file__).resolve().parents[1]
    revision = _git_identity(root, expected_sha)
    config_path = _path(config_path, directory=False)
    config = _config(json.loads(config_path.read_text()))
    if (config['experiment_id'] != 'joint-mesh8-qualification-r10'
            or config['experiment_role'] != MESH8_ROLE or config['mesh_level'] != 8
            or config['rollout_hours'] != 3 or config['scientific_acceptance'] is not False):
        raise ValueError('Configuration is not the frozen R10 mesh8 qualification envelope.')

    # Build the real 655362-cell geometry. Formula-only checks are insufficient
    # for authorising a final run because fingerprints include vertex order.
    grid = build_mesh8_grid()
    if grid.fingerprint != CANONICAL_MESH8_FINGERPRINT:
        raise ValueError('Mesh8 grid fingerprint differs from the qualified Q0 geometry.')

    dataset_path = _path(dataset_path, directory=True)
    norm_path = _path(norm_path, directory=False)
    climatology_path = _path(climatology_path, directory=True)
    static_path = _path(static_path, directory=True)
    joint = JointObservationDataset(dataset_path)
    joint.verify()
    if joint.horizon_hours != 72:
        raise ValueError('R10 requires the reviewed 72-hour guarded joint archive even though rollout is +3 h.')
    _, norm_identity = _normalization(joint, norm_path)
    climate = Mesh8SeasonalClimatology(climatology_path, grid)
    climate.verify_sources()
    static = load_static(static_path, grid)

    run_root = _path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    # Re-check the now-existing output after mkdir to close the common symlink
    # substitution case between argument parsing and artifact creation.
    run_root = _path(run_root, directory=True)
    free = int(shutil.disk_usage(run_root).free)
    minimum_free = int(minimum_free_disk_gib) * 1024**3
    if free < minimum_free:
        raise ValueError(f'Free disk {free} bytes is below R10 reserve {minimum_free}.')
    cuda = _cuda_identity(config, require_cuda=require_cuda)

    report = {
        'schema': 'mesh8-r10-preflight-1', 'status': 'ready_for_q4_q5',
        'scientific_acceptance': False, 'source_commit': revision,
        'experiment_id': config['experiment_id'], 'experiment_role': config['experiment_role'],
        'joint_architecture': JOINT_ARCHITECTURE, 'profile_architecture': PROFILE_ARCHITECTURE,
        'config': {'path': str(config_path), 'sha256': digest(config_path),
                   'mesh_level': config['mesh_level'], 'hidden': config['hidden'],
                   'rollout_hours': config['rollout_hours'], 'epochs': config['epochs'],
                   'edge_chunk_size': config['edge_chunk_size'],
                   'physical_policy': config['physical_policy']},
        'mesh': {'level': grid.level, 'cells': grid.n_cells,
                 'fingerprint': grid.fingerprint, 'area_method': grid.area_method,
                 'nominal_area_scale_km': grid.nominal_area_scale_km,
                 'fingerprint_schema': 'mesh8-geometry-q1e-7-v1'},
        'dataset': {'path': str(dataset_path), 'fingerprint': joint.fingerprint,
                    'bundle_sha256': getattr(joint, 'bundle_hash', None),
                    'horizon_hours': joint.horizon_hours, 'input_hours': 12},
        'normalization': {'path': str(norm_path), 'sha256': digest(norm_path),
                          'source_identity': norm_identity},
        'climatology': {'path': str(climatology_path), 'fingerprint': climate.fingerprint,
                        'source_fingerprint': climate.source.fingerprint,
                        'target_grid_fingerprint': climate.target_grid_fingerprint},
        'static': {'path': str(static_path), 'artifact_sha256': static['static_sha256'],
                   'manifest_sha256': static['manifest_sha256'],
                   'grid_fingerprint': static['manifest']['grid_fingerprint']},
        'runtime': {'cuda': cuda, 'run_root': str(run_root),
                    'free_disk_bytes': free, 'minimum_free_disk_bytes': minimum_free},
        'authorization': 'Q4 training then Q5 frozen inference only; no 72 h promotion and no scientific acceptance',
    }
    report_path = _path(report_path) if report_path else run_root/'preflight.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and report_path.is_symlink():
        raise ValueError('Preflight report destination may not be a symlink.')
    save(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--norm', required=True)
    parser.add_argument('--climatology', required=True)
    parser.add_argument('--static', required=True)
    parser.add_argument('--config', default='configs/joint_observation_training.json')
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--report')
    args = parser.parse_args(argv)
    report = preflight(args.dataset, args.norm, args.climatology, args.static,
                       args.config, args.run_root, args.expected_sha, report_path=args.report)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
