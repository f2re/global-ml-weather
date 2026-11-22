"""Score a frozen station anomaly checkpoint on the chronological GHCNh test split."""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from .observation_data import NATIVE_VARIABLES, UNITS, ObservationDataset
from .observation_model import LEAD_HOURS
from .observation_training import checkpoint_path
from .observations import utc
from .pipeline.io import atomic_json, read_json, sha256
from .station_anomaly_training import create_model
from .station_climatology import StationClimatology


def score(model, dataset, climatology, device):
    baselines = ('model', 'climatology', 'persistence')
    shape = (len(LEAD_HOURS), len(NATIVE_VARIABLES))
    count = np.zeros(shape, dtype=np.int64)
    target_count = np.zeros_like(count)
    totals = np.zeros((len(baselines), 3, *shape), dtype=np.float64)
    issues = dataset.subset('test')
    scored_issues = 0
    model.eval()
    with torch.inference_mode():
        for index in issues:
            sample = dataset.sample(index)
            observed = sample['input_mask']
            target_count += sample['target_mask'].sum(axis=1)
            common = sample['target_mask'] & observed.any(axis=0)
            if not common.any():
                continue
            issue = utc(sample['issue'])
            history_climate = climatology.for_window([issue + timedelta(hours=h) for h in range(-11, 1)])
            target_climate = climatology.for_window([issue + timedelta(hours=h) for h in LEAD_HOURS])
            history = np.where(observed, (sample['input'] - history_climate) / climatology.std, 0)
            forecast = model(torch.as_tensor(history, device=device), torch.as_tensor(observed, device=device))
            prediction = model.reconstruct_physical(
                forecast.predicted_anomalies, torch.as_tensor(target_climate, device=device)).cpu().numpy()
            last_index = len(observed) - 1 - observed[::-1].argmax(axis=0)
            persistence = np.take_along_axis(sample['input'], last_index[None], axis=0)
            for i, estimate in enumerate((prediction, target_climate, persistence)):
                error = np.where(common, estimate - sample['target'], 0).astype(np.float64)
                if not np.isfinite(error).all():
                    raise FloatingPointError('Nonfinite forecast error on the common test mask.')
                totals[i, 0] += np.abs(error).sum(axis=1)
                totals[i, 1] += np.square(error).sum(axis=1)
                totals[i, 2] += error.sum(axis=1)
            count += common.sum(axis=1)
            scored_issues += 1
            if scored_issues % 25 == 0:
                print(f'Scored {scored_issues}/{len(issues)} test issues', flush=True)
    if not scored_issues:
        raise ValueError('No causal test observations with a common target and persistence baseline.')

    rows = []
    for lead_index, lead in enumerate(LEAD_HOURS):
        for variable_index, (variable, units) in enumerate(zip(NATIVE_VARIABLES, UNITS)):
            cell = (lead_index, variable_index)
            n = int(count[cell])
            row = {'lead_hours': lead, 'variable': variable, 'units': units,
                   'common_targets': n, 'observed_targets': int(target_count[cell])}
            for i, name in enumerate(baselines):
                absolute, squared, signed = totals[i, :, lead_index, variable_index]
                row[name] = {'mae': float(absolute / n), 'rmse': float(np.sqrt(squared / n)),
                             'bias': float(signed / n)} if n else None
            rows.append(row)
    return {'test_issues': len(issues), 'scored_issues': scored_issues, 'metrics': rows}


def evaluate(dataset_path, training, checkpoint_sha, climatology_sha, device_name):
    best = read_json(training / 'best.json')
    completion = read_json(training / 'complete.json')
    if (completion['status'] != 'surface_climatology_anomaly_research_trained'
            or best['epoch'] != completion['best_epoch'] or best['sha256'] != checkpoint_sha):
        raise ValueError('Completion, validation selection and expected checkpoint disagree.')
    checkpoint = checkpoint_path(training, best)
    climate_path = training / 'station_climatology.npz'
    if sha256(climate_path) != climatology_sha:
        raise ValueError('Unexpected station climatology hash.')
    climatology = StationClimatology.load(climate_path)
    manifest_path = dataset_path / 'dataset.json'
    dataset_sha = sha256(manifest_path)
    print('Verifying the prepared observation dataset', flush=True)
    dataset = ObservationDataset(dataset_path)
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(8)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['epoch'] != best['epoch'] or state['best_loss'] != completion['best_loss']:
        raise ValueError('Checkpoint does not match the completion record.')
    model = create_model(dataset, climatology, state['config'], device)
    current = model.state_dict()
    for key in ('scales', 'xyz', 'pressure_hpa', 'operator.cells', 'operator.station_xyz'):
        if not torch.equal(current[key].cpu(), state['model'][key]):
            raise ValueError(f'Checkpoint and evaluation setup differ on {key}.')
    model.load_state_dict(state['model'], strict=True)
    results = score(model, dataset, climatology, device)
    if sha256(manifest_path) != dataset_sha or sha256(climate_path) != climatology_sha:
        raise ValueError('Evaluation input changed during scoring.')
    return {
        'schema': 'station-anomaly-test-1', 'status': 'research_test_completed',
        'scientific_acceptance': False, 'checkpoint_epoch': best['epoch'],
        'checkpoint_sha256': checkpoint_sha, 'climatology_sha256': climatology_sha,
        'dataset_sha256': dataset_sha, 'campaign': dataset.campaign,
        'availability': dataset.manifest['availability'], 'device': str(device),
        'mask': 'measured target and at least one causal input for the same station and variable',
        'weighting': 'equal weight per station-target pair',
        'persistence': 'most recent available measurement in the 12-hour history',
        **results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha', required=True)
    parser.add_argument('--expected-climatology-sha', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu'), default='auto')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output already exists. Choose a new evidence file.')
    result = evaluate(args.dataset, args.training, args.expected_checkpoint_sha,
                      args.expected_climatology_sha, args.device)
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
