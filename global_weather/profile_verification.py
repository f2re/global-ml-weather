"""Separate ERA5 diagnostics for an already frozen observation-trained model.

No statistics, epoch selection, gradient or weight updates are performed here.
The diagnostic source can assimilate observations and is not independent truth.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid
import numpy as np
import torch
from .observation_training import checkpoint_path, digest, save
from .profile_training import load_frozen as _legacy_load_frozen, bounded_records
from .observations import utc
from .pipeline.era5 import prepare_targets
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS


def load_frozen(dataset_path,training):
    completion=json.loads((Path(training)/'complete.json').read_text())
    if completion.get('status') in ('measured_pressure_profile_research_trained','measured_graphcast_profile_research_trained'):
        from .profile_training_v2 import load_frozen as load_pressure_model
        return load_pressure_model(dataset_path,training)
    return _legacy_load_frozen(dataset_path,training)


def _exact_tensor_equal(left, right):
    """Compare frozen tensors without densifying sparse CUDA buffers."""
    if (left.shape != right.shape or left.dtype != right.dtype or
            left.device != right.device or left.layout != right.layout):
        return False
    if left.layout == torch.sparse_coo:
        left, right = left.coalesce(), right.coalesce()
        return (left.sparse_dim() == right.sparse_dim() and left.dense_dim() == right.dense_dim()
                and torch.equal(left.indices(), right.indices())
                and _exact_tensor_equal(left.values(), right.values()))
    if left.layout != torch.strided:
        raise ValueError('Unsupported frozen tensor layout; comparison cannot be skipped.')
    if left.is_floating_point() or left.is_complex():
        # Unchanged masked NaNs are allowed; finite entries remain exact.
        if left.is_complex():
            return _exact_tensor_equal(torch.view_as_real(left), torch.view_as_real(right))
        return bool(torch.all((left == right) | (torch.isnan(left) & torch.isnan(right))).item())
    return torch.equal(left, right)


def _input_hash(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _identity(dataset_path, training, pressure, surface, issue):
    ref = json.loads((training/'best.json').read_text())
    completion=json.loads((training/'complete.json').read_text())
    extra=[('pressure_norms',Path(completion['identity']['norm_path']))] if completion.get('status') in ('measured_pressure_profile_research_trained','measured_graphcast_profile_research_trained') else []
    if completion.get('status') == 'measured_graphcast_profile_research_trained':
        from .import_climatology import bundled_directory, PINNED_HASHES
        directory = bundled_directory()
        extra += [('graphcast_' + name, directory / name) for name in sorted(PINNED_HASHES)]
    def files(value):
        paths=list(value) if isinstance(value,(list,tuple)) else [value]
        if not paths: raise ValueError('External source file list is empty.')
        return [{'path':str(Path(path).resolve()),'sha256':digest(path)} for path in paths]
    return {'schema': 'frozen-profile-verification-2', 'issue': utc(issue).isoformat(),
            'paths': {'dataset':str(Path(dataset_path).resolve()),'training':str(training.resolve())},
            'external_files':{'pressure':files(pressure),'surface':files(surface)},
            'hashes': {name: digest(path) for name, path in
                       (('dataset', Path(dataset_path)/'dataset.json'),
                        ('completion', training/'complete.json'), ('best', training/'best.json'),
                        ('checkpoint', checkpoint_path(training, ref)), ('verification_code', __file__),*extra)}}


@contextmanager
def _destination_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Verification destination cannot contain symbolic links.')
    fd = os.open(output.parent/('.'+output.name+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+b') as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Unsafe verification lock.')
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def verify(dataset_path: str | Path, training: str | Path,
           pressure_netcdf: str | Path | list, surface_netcdf: str | Path | list,
           issue: str, output: str | Path) -> list:
    """Publish a verified diagnostic atomically, preserving failed attempts."""
    training, output = Path(training), Path(output).absolute()
    if not (training/'complete.json').is_file():
        raise ValueError('Complete training and freeze weights before ERA5 verification.')
    with _destination_lock(output):
        identity = _identity(dataset_path, training, pressure_netcdf, surface_netcdf, issue)
        if output.exists() and (output/'complete.json').is_file():
            receipt = json.loads((output/'complete.json').read_text())
            if receipt.get('identity') != identity:
                raise ValueError('Completed verification identity differs; use a new destination.')
            for name in ('verification.json', 'era5-diagnostic.npz'):
                if (output/name).is_symlink() or digest(output/name) != receipt['files'][name]:
                    raise ValueError('Completed verification artifact hash differs.')
            model, dataset = load_frozen(dataset_path, training)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('External verification requires a frozen evaluation model.')
            when = utc(issue)
            limit = json.loads((training/'complete.json').read_text())['identity']['config']['max_records_per_window']
            inputs = bounded_records(dataset.records(when-timedelta(hours=12), when, issue=when, split='test'), limit)
            dataset.verify()
            report = json.loads((output/'verification.json').read_text())
            if report['input_sha256'] != _input_hash(inputs) or identity != _identity(dataset_path, training, pressure_netcdf, surface_netcdf, issue):
                raise ValueError('Completed verification inputs or sources changed.')
            return report['metrics']
        previous = None
        if output.exists():
            if not output.is_dir():
                raise ValueError('Existing verification destination is not a directory.')
            previous = output.parent/('.'+output.name+'.failed-'+uuid.uuid4().hex)
            output.rename(previous)
            save(previous/'restart.json', {'reason': 'incomplete_verification_preserved', 'next_identity': identity})
        stage = output.parent/('.'+output.name+'.attempt-'+uuid.uuid4().hex)
        try:
            rows = _verify(dataset_path, training, pressure_netcdf, surface_netcdf, issue, stage)
            if identity != _identity(dataset_path, training, pressure_netcdf, surface_netcdf, issue):
                raise ValueError('Verification identity changed during execution.')
            save(stage/'complete.json', {'identity': identity,
                 'files': {name: digest(stage/name) for name in ('verification.json', 'era5-diagnostic.npz')},
                 'preserved_previous': str(previous) if previous else None})
            stage.rename(output)
            directory = os.open(output.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
            return rows
        except Exception as exc:
            if stage.exists():
                save(stage/'failure.json', {'identity': identity, 'exception': type(exc).__name__, 'detail': str(exc)})
            raise


def _verify(dataset_path, training, pressure_netcdf, surface_netcdf, issue, output):
    training, output = Path(training), Path(output)
    if not (training/'complete.json').is_file():
        raise ValueError('Complete training and freeze weights before ERA5 verification.')
    completion = json.loads((training/'complete.json').read_text())
    if completion.get('status') not in ('measured_upper_air_research_trained','measured_pressure_profile_research_trained','measured_graphcast_profile_research_trained'):
        raise ValueError('Only observation-trained profile checkpoints are admitted.')
    ref = json.loads((training/'best.json').read_text())
    checkpoint = checkpoint_path(training, ref)
    before = digest(checkpoint)
    if before != ref['sha256']: raise ValueError('Frozen checkpoint hash differs.')
    if completion.get('best_epoch') != ref['epoch']:
        raise ValueError('Completion and best checkpoint differ.')
    completion_hash = digest(training/'complete.json')
    issue = utc(issue)
    if not utc('2022-07-01T00:00:00Z') + timedelta(hours=54) <= issue < utc('2023-01-01T00:00:00Z') - timedelta(hours=114):
        raise ValueError('Use the held-out test period with chronological guards.')
    output.mkdir(parents=True, exist_ok=False)
    model, dataset = load_frozen(dataset_path, training)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('External verification requires a frozen evaluation model.')
    manifest_hash = digest(dataset.root/'dataset.json')
    frozen_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    limit = completion['identity']['config']['max_records_per_window']
    inputs = bounded_records(dataset.records(issue-timedelta(hours=12), issue, issue=issue, split='test'), limit)
    if not inputs: raise ValueError('No causal measured inputs for external verification.')
    # Model prediction is constructed before opening external target fields.
    with torch.no_grad(): frames = model(inputs, issue)
    target_path = output/'era5-diagnostic.npz'
    provenance = prepare_targets(pressure_netcdf, surface_netcdf, target_path,
             issue_time=issue.isoformat(), mesh_level=model.grid.level, horizon_hours=72,
             step_hours=3, confirm_utc=True)
    with np.load(target_path, allow_pickle=False) as arrays:
        target = arrays['profiles']; target_mask = arrays['profile_mask']
        pressure = arrays['pressure_hpa']; leads = arrays['lead_hours']
    shape = (25, model.grid.n_cells, 37, 6)
    if (target.shape != shape or target_mask.shape != shape or target_mask.dtype != bool
            or not np.array_equal(pressure, np.asarray(PRESSURE_HPA))
            or not np.array_equal(leads, np.arange(0, 73, 3))
            or len(frames) != 25 or [frame.lead_hours for frame in frames] != leads.tolist()):
        raise ValueError('External target grid, variable mask, pressure or forecast leads differ.')
    rows = []
    areas = model.grid.areas_m2
    for index, frame in enumerate(frames):
        prediction = frame.profiles.detach().cpu().numpy()
        prediction_mask = frame.profile_variable_mask.detach().cpu().numpy()
        for level in range(37):
            for variable in range(5):
                valid = (target_mask[index,:,level,variable] & prediction_mask[:,level,variable]
                         & frame.profile_mask[:,level].detach().cpu().numpy()
                         & np.isfinite(target[index,:,level,variable]) & np.isfinite(prediction[:,level,variable]))
                if not valid.any(): continue
                error = prediction[valid,level,variable] - target[index,valid,level,variable]
                weight = areas[valid]; weight = weight/weight.sum()
                rows.append({'lead_hours':frame.lead_hours,'pressure_pa':float(model.pressure_pa[level]),
                     'variable':PROFILE_VARIABLES[variable],'units':PROFILE_UNITS[variable],
                     'mae':float(np.sum(weight*np.abs(error))), 'rmse':float(np.sqrt(np.sum(weight*error**2))),
                     'bias':float(np.sum(weight*error)), 'cells':int(valid.sum())})
    if not rows: raise ValueError('No jointly supported external diagnostic fields.')
    dataset.verify()
    current_state = model.state_dict()
    if (set(current_state) != set(frozen_state) or
            any(not _exact_tensor_equal(current_state[name], tensor) for name, tensor in frozen_state.items())):
        raise ValueError('Frozen model weights or normalization buffers changed during verification.')
    if (digest(checkpoint) != before or json.loads((training/'best.json').read_text()) != ref
            or digest(training/'complete.json') != completion_hash
            or digest(dataset.root/'dataset.json') != manifest_hash):
        raise ValueError('Frozen checkpoint changed during verification.')
    save(output/'verification.json', {'source_role':'separate_frozen_model_verification_only',
         'checkpoint':ref,'issue':issue.isoformat(),'external_source':'ERA5',
         'target_provenance':provenance,'metrics':rows,'weight_updates':0,
         'input_sha256': _input_hash(inputs),
         'norm_updates':0,'epoch_selection':False,'scientific_acceptance':False,
         'independent_truth':False,'omega_verification':False,'future_targets_used_for_inference':False})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','training','pressure-netcdf','surface-netcdf','issue','output'):
        parser.add_argument('--'+name, required=True)
    args=parser.parse_args(argv)
    verify(args.dataset,args.training,args.pressure_netcdf,args.surface_netcdf,args.issue,args.output)


if __name__=='__main__': main()
