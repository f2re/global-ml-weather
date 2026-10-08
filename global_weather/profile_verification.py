"""Separate ERA5 diagnostics for an already frozen observation-trained model.

No statistics, epoch selection, gradient or weight updates are performed here.
The diagnostic source can assimilate observations and is not independent truth.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import torch
from .observation_training import checkpoint_path, digest, save
from .profile_training import load_frozen, bounded_records
from .observations import utc
from .pipeline.era5 import prepare_targets
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS


def verify(dataset_path, training, pressure_netcdf, surface_netcdf, issue, output):
    training, output = Path(training), Path(output)
    if not (training/'complete.json').is_file():
        raise ValueError('Complete training and freeze weights before ERA5 verification.')
    completion = json.loads((training/'complete.json').read_text())
    if completion.get('status') != 'measured_upper_air_research_trained':
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
            any(not torch.equal(current_state[name], tensor) for name, tensor in frozen_state.items())):
        raise ValueError('Frozen model weights or normalization buffers changed during verification.')
    if (digest(checkpoint) != before or json.loads((training/'best.json').read_text()) != ref
            or digest(training/'complete.json') != completion_hash
            or digest(dataset.root/'dataset.json') != manifest_hash):
        raise ValueError('Frozen checkpoint changed during verification.')
    save(output/'verification.json', {'source_role':'separate_frozen_model_verification_only',
         'checkpoint':ref,'issue':issue.isoformat(),'external_source':'ERA5',
         'target_provenance':provenance,'metrics':rows,'weight_updates':0,
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
