"""Read-only diagnosis of a completed joint run; losses do not certify weather skill."""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime
import json
import math
from pathlib import Path
import re

from .observation_training import checkpoint_path, digest

SCHEMA = 'joint-training-audit-1'
SUPPORTED_CONFIG_KEYS = frozenset(('mesh_level', 'hidden', 'epochs', 'patience', 'threads', 'seed',
    'learning_rate', 'weight_decay', 'gradient_clip', 'max_train_issues', 'max_validation_issues',
    'max_test_issues', 'max_records_per_window', 'surface_weight', 'physical_policy'))


def _read(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Audit evidence symlinks are forbidden: '+str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Audit evidence must be a JSON object.')
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError('Invalid positive integer: '+name)
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid finite scalar: '+name)
    return float(value)


def _bound_file(root, name, expected_hash):
    if not isinstance(name, str) or not isinstance(expected_hash, str):
        raise ValueError('Missing bound monthly artifact descriptor.')
    root = Path(root).resolve()
    artifact = root/name
    if artifact.is_symlink() or root not in artifact.resolve().parents or digest(artifact) != expected_hash:
        raise ValueError('Monthly artifact path or hash differs: '+name)


def _monthly(path, checkpoint_sha256, identity):
    if path is None:
        return {'status': 'missing', 'observation_comparison_completed': False,
                'weather_skill_accepted': False}
    path = Path(path)
    value = _read(path)
    known = value.get('schema') == 'joint-monthly-verification-1'
    complete = known and value.get('stage') == 'verify' and value.get('status') == 'complete'
    if complete:
        if value.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('Monthly verification checkpoint differs from audited best weights.')
        if value.get('identity') != identity:
            raise ValueError('Monthly verification training identity differs.')
        month = value.get('month')
        if not isinstance(month, str) or not re.fullmatch(r'\d{4}-\d{2}', month):
            raise ValueError('Monthly verification has no valid month.')
        year, month_number = map(int, month.split('-'))
        if not 1 <= month_number <= 12:
            raise ValueError('Monthly verification month is invalid.')
        expected_days = calendar.monthrange(year, month_number)[1]
        days = value.get('days')
        if not isinstance(days, list) or len(days) != expected_days:
            raise ValueError('Monthly verification is not a complete month.')
        times = [datetime.fromisoformat(day['issue_time'].replace('Z', '+00:00')) for day in days]
        if ({time.day for time in times} != set(range(1, expected_days+1))
                or any(time.strftime('%Y-%m') != month or time.utcoffset() is None
                       or time.utcoffset().total_seconds() != 0 for time in times)
                or len({(time.hour, time.minute, time.second) for time in times}) != 1):
            raise ValueError('Monthly verification issue dates are incomplete or inconsistent.')
        for name, key in (('aggregate.jsonl', 'aggregate_sha256'),
                          ('accumulators.npz', 'accumulators_sha256')):
            artifact = path.parent/name
            if artifact.is_symlink() or digest(artifact) != value.get(key):
                raise ValueError('Monthly verification artifact hash differs: '+name)
        for day in days:
            _bound_file(path.parent, day.get('inventory'), day.get('inventory_sha256'))
            for filename, hashkey in (('pairs', 'pairs_sha256'), ('rejections', 'rejections_sha256')):
                _bound_file(path.parent, day.get(filename), day.get(hashkey))
        forecast_path = Path(value.get('forecast_manifest_path', path.parent.parent/'forecasts'/'month.json'))
        forecast = _read(forecast_path)
        if (forecast.get('schema') != 'joint-monthly-verification-1'
                or forecast.get('stage') != 'forecast' or forecast.get('status') != 'complete'
                or forecast.get('future_targets_read') is not False
                or forecast.get('checkpoint_sha256') != checkpoint_sha256):
            raise ValueError('Monthly forecast completion does not bind audited weights.')
        products = forecast.get('products')
        if (not isinstance(products, list) or [row.get('issue_time') for row in products]
                != [row['issue_time'] for row in days]):
            raise ValueError('Monthly frozen products do not cover verified issue dates.')
        for product in products:
            folder = product.get('directory')
            if not isinstance(folder, str) or Path(folder).name != folder:
                raise ValueError('Invalid monthly product directory.')
            base = forecast_path.parent/folder
            for name, key in (('verification-state.npz', 'verification_state_sha256'),
                              ('inputs.npz', 'inputs_sha256'), ('profile_inputs.json', 'profile_inputs_sha256'),
                              ('product/forecast.grib2', 'grib_sha256')):
                _bound_file(base, name, product.get(key))
        if value.get('forecast_manifest_sha256') != digest(forecast_path):
            raise ValueError('Monthly verification lacks valid forecast-completion hash binding.')
    return {'status': 'complete' if complete else 'unverified', 'schema': value.get('schema'),
            'sha256': digest(path), 'observation_comparison_completed': complete,
            'weather_skill_accepted': False}


def audit(training, *, reference_config=None, verification=None):
    root = Path(training)
    if root.is_symlink():
        raise ValueError('Training evidence symlinks are forbidden.')
    complete = _read(root/'complete.json')
    if complete.get('status') != 'joint_measured_observation_research_trained':
        raise ValueError('Joint completed training evidence is missing.')
    identity = complete.get('identity')
    if not isinstance(identity, dict) or not isinstance(identity.get('config'), dict):
        raise ValueError('Completed training has no immutable configuration.')
    config = identity['config']
    budget = _positive(config.get('epochs'), 'epochs')
    patience = _positive(config.get('patience'), 'patience')
    latest = _read(root/'latest.json'); best = _read(root/'best.json')
    for ref in (latest, best):
        _positive(ref.get('epoch'), 'checkpoint epoch')
        checkpoint_path(root, ref)
    actual = latest['epoch']
    if actual > budget or best['epoch'] != complete.get('best_epoch') or best['epoch'] > actual:
        raise ValueError('Completed epoch references contradict configuration or completion.')
    directories = {}
    partial = []
    for path in root.iterdir():
        match = re.fullmatch(r'epoch-(\d{4})', path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise ValueError('Completed epoch directory is invalid.')
            directories[int(match[1])] = path
        elif path.name.startswith('.epoch-'):
            partial.append(path.name)
    if sorted(directories) != list(range(1, actual+1)):
        raise ValueError('Completed epochs are missing, noncontiguous or ahead of latest reference.')
    rows = []; evidence = {}; gradient_reports = []; profile_counts = []; native_counts = []
    for epoch, folder in sorted(directories.items()):
        row = _read(folder/'metrics.json')
        if row.get('epoch') != epoch or not (folder/'state.pt').is_file() or (folder/'state.pt').is_symlink():
            raise ValueError('Completed epoch metrics or state are missing or inconsistent.')
        validation = row.get('validation')
        if not isinstance(validation, dict):
            raise ValueError('Completed epoch has no validation evidence.')
        loss = _number(validation.get('joint'), 'validation.joint')
        rows.append({'epoch': epoch, 'joint': loss,
                     'profile': _number(validation.get('profile'), 'validation.profile'),
                     'surface': _number(validation.get('surface'), 'validation.surface')})
        gradient_reports.append(row.get('nonzero_gradients'))
        profile_counts.append(row.get('profile_target_counts'))
        native_counts.append(row.get('native_target_counts'))
        evidence[str(epoch)] = {'metrics_sha256': digest(folder/'metrics.json'),
                                'state_sha256': digest(folder/'state.pt')}
    losses = [row['joint'] for row in rows]
    computed_best = min(range(actual), key=lambda i: losses[i])+1
    if computed_best != best['epoch']:
        raise ValueError('Best checkpoint contradicts completed validation history.')
    stale = actual-computed_best
    stop = 'early_stopping' if stale >= patience else ('epoch_budget' if actual == budget else 'unknown')
    decreasing = all(b < a for a, b in zip(losses, losses[1:])) if actual > 1 else None
    last_improved = actual > 1 and losses[-1] < min(losses[:-1])
    gradients_ready = all(isinstance(row, dict) and bool(row) and all(v is True for v in row.values())
                          for row in gradient_reports)
    def supported(counts, size):
        return all(isinstance(row, list) and len(row) == size
                   and all(type(v) is int and v > 0 for v in row) for row in counts)
    monthly = _monthly(verification, best['sha256'], identity)
    blockers = ['independent_scientific_acceptance_not_established',
                'normalized_training_losses_are_not_physical_weather_metrics']
    if not monthly['observation_comparison_completed']:
        blockers.append('monthly_observation_verification_missing_or_unverified')
    if not gradients_ready:
        blockers.append('gradient_evidence_missing_or_incomplete')
    report = {'schema': SCHEMA, 'status': 'complete', 'source_identity': identity,
              'training_artifact_sha256': {name: digest(root/name) for name in
                                           ('complete.json', 'latest.json', 'best.json')},
              'epoch_artifact_sha256': evidence, 'configured_epochs': budget,
              'completed_epochs': actual, 'best_epoch': best['epoch'], 'best_is_last': best['epoch'] == actual,
              'stop_reason': stop, 'stale_epochs': stale, 'strictly_decreasing_validation': decreasing,
              'last_epoch_improved': last_improved, 'convergence_not_established': True,
              'convergence_evidence': 'best_is_last_and_still_improving' if last_improved else
                                      'loss_history_alone_does_not_establish_convergence',
              'validation_history': rows, 'partial_epochs_ignored': sorted(partial),
              'normalized_loss_is_weather_accuracy': False, 'scientific_acceptance': False,
              'readiness': {'all_reported_parameter_gradients_nonzero': gradients_ready,
                           'all_profile_variables_have_targets_each_epoch': supported(profile_counts, 5),
                           'all_native_variables_have_targets_each_epoch': supported(native_counts, 6),
                           'all_levels_trained': False, 'weather_acceptance': 'blocked', 'blockers': blockers},
              'stage_inventory': {'training': {'status': 'complete', 'epochs': actual},
                                  'forecast_format': {'status': 'not_provided_in_this_audit'},
                                  'monthly_observation_verification': monthly}}
    if reference_config is not None:
        reference = _read(reference_config)
        common = sorted(set(reference) & SUPPORTED_CONFIG_KEYS)
        report['reference_configuration'] = {'sha256': digest(reference_config),
            'comparison': {key: {'reference': reference[key], 'actual': config.get(key),
                                'equal': reference[key] == config.get(key)} for key in common},
            'unsupported_reference_keys': sorted(set(reference)-SUPPORTED_CONFIG_KEYS),
            'reference_tuning_inherited': False,
            'interpretation': 'different architecture; equality of individual settings does not establish inherited tuning'}
    return report


def write_artifact(path, report):
    """Create a new artifact exclusively; preserve existing or interrupted evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True); parser.add_argument('--reference-config')
    parser.add_argument('--verification'); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = audit(args.training, reference_config=args.reference_config, verification=args.verification)
    write_artifact(args.output, report)
    print(json.dumps({'schema': SCHEMA, 'stop_reason': report['stop_reason'],
                      'weather_acceptance': 'blocked', 'output': args.output}))


if __name__ == '__main__':
    main()
