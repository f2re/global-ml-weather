"""Read-only audit of budget termination, incomplete evidence and acceptance claims."""
import hashlib
import json

import pytest

from global_weather.joint_training_audit import audit, write_artifact


def put(path, value):
    path.write_text(json.dumps(value))


def run(tmp_path, losses, *, budget=3, patience=2):
    root = tmp_path/'training'; root.mkdir()
    refs = []
    for epoch, loss in enumerate(losses, 1):
        folder = root/f'epoch-{epoch:04d}'; folder.mkdir()
        state = folder/'state.pt'; state.write_bytes(f'synthetic state {epoch}'.encode())
        refs.append({'epoch': epoch, 'directory': folder.name,
                     'sha256': hashlib.sha256(state.read_bytes()).hexdigest()})
        put(folder/'metrics.json', {'epoch': epoch, 'validation': {'joint': loss, 'profile': loss, 'surface': loss},
            'nonzero_gradients': {'synthetic_branch': True}, 'profile_target_counts': [1]*5,
            'native_target_counts': [1]*6})
    best = min(range(len(losses)), key=lambda i: losses[i])
    put(root/'latest.json', refs[-1]); put(root/'best.json', refs[best])
    put(root/'complete.json', {'status': 'joint_measured_observation_research_trained',
        'identity': {'config': {'epochs': budget, 'patience': patience, 'hidden': 32}},
        'best_epoch': best+1, 'scientific_acceptance': False})
    return root


def test_epoch_cap_is_not_convergence_or_weather_accuracy(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    (root/'.epoch-0004').mkdir()
    report = audit(root)
    assert report['stop_reason'] == 'epoch_budget'
    assert report['best_is_last'] and report['last_epoch_improved']
    assert report['strictly_decreasing_validation'] is True
    assert report['convergence_not_established'] is True
    assert report['normalized_loss_is_weather_accuracy'] is False
    assert report['readiness']['weather_acceptance'] == 'blocked'
    assert report['partial_epochs_ignored'] == ['.epoch-0004']
    assert report['stage_inventory']['monthly_observation_verification']['status'] == 'missing'


def test_patience_reports_early_stop_without_asserting_convergence(tmp_path):
    report = audit(run(tmp_path, [.2, .3, .4], budget=20, patience=2))
    assert report['stop_reason'] == 'early_stopping'
    assert report['stale_epochs'] == 2
    assert report['convergence_not_established'] is True


def test_missing_completed_metrics_blocks_diagnosis(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    (root/'epoch-0002'/'metrics.json').unlink()
    with pytest.raises(FileNotFoundError):
        audit(root)


def test_reference_hash_tamper_is_rejected(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    (root/'epoch-0003'/'state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit(root)


def test_noncontiguous_completed_epochs_rejected(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    (root/'epoch-0002'/'metrics.json').unlink()
    (root/'epoch-0002'/'state.pt').unlink()
    (root/'epoch-0002').rmdir()
    with pytest.raises(ValueError, match='noncontiguous'):
        audit(root)


def test_verification_execution_and_unsupported_tuning_never_accept_weather(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    verification_dir = tmp_path/'verification'; verification_dir.mkdir()
    verification = verification_dir/'verification.json'
    aggregate = verification_dir/'aggregate.jsonl'; aggregate.write_bytes(b'{}\n')
    accumulators = verification_dir/'accumulators.npz'; accumulators.write_bytes(b'synthetic accumulator')
    forecast_dir = tmp_path/'forecasts'; forecast_dir.mkdir()
    forecast = forecast_dir/'month.json'
    best = json.loads((root/'best.json').read_text())
    days = []; products = []
    for day in range(1, 31):
        issue = f'2022-09-{day:02d}T06:00:00+00:00'
        inventory = verification_dir/f'day-{day}.jsonl'; inventory.write_bytes(b'{}\n')
        days.append({'issue_time': issue, 'inventory': inventory.name,
                     'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest()})
        for name, key in (('pairs', 'pairs_sha256'), ('rejections', 'rejections_sha256')):
            artifact = verification_dir/f'day-{day}-{name}.jsonl.gz'
            artifact.write_bytes(b'synthetic paired or rejected target stream')
            days[-1][name] = artifact.name
            days[-1][key] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        folder = forecast_dir/f'day-{day}'; folder.mkdir(); (folder/'product').mkdir()
        product = {'issue_time': issue, 'directory': folder.name}
        for name, key in (('verification-state.npz', 'verification_state_sha256'),
                          ('inputs.npz', 'inputs_sha256'), ('profile_inputs.json', 'profile_inputs_sha256'),
                          ('product/forecast.grib2', 'grib_sha256')):
            artifact = folder/name; artifact.write_bytes(b'synthetic artifact')
            product[key] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        products.append(product)
    put(forecast, {'schema': 'joint-monthly-verification-1', 'stage': 'forecast', 'status': 'complete',
                   'checkpoint_sha256': best['sha256'], 'future_targets_read': False, 'products': products})
    put(verification, {'schema': 'joint-monthly-verification-1', 'stage': 'verify', 'status': 'complete',
                       'month': '2022-09', 'checkpoint_sha256': best['sha256'],
                       'identity': json.loads((root/'complete.json').read_text())['identity'],
                       'days': days,
                       'aggregate_sha256': hashlib.sha256(aggregate.read_bytes()).hexdigest(),
                       'accumulators_sha256': hashlib.sha256(accumulators.read_bytes()).hexdigest(),
                       'forecast_manifest_sha256': hashlib.sha256(forecast.read_bytes()).hexdigest(),
                       'scientific_acceptance': True})
    reference = tmp_path/'unified.json'
    put(reference, {'epochs': 500, 'hidden': 128, 'scheduler': 'cosine', 'batch_size': 4})
    report = audit(root, reference_config=reference, verification=verification)
    assert report['stage_inventory']['monthly_observation_verification']['observation_comparison_completed'] is True
    assert report['scientific_acceptance'] is False
    assert report['readiness']['weather_acceptance'] == 'blocked'
    assert report['reference_configuration']['unsupported_reference_keys'] == ['batch_size', 'scheduler']
    assert report['reference_configuration']['reference_tuning_inherited'] is False
    output = tmp_path/'audit.json'; write_artifact(output, report)
    with pytest.raises(FileExistsError):
        write_artifact(output, report)
    (verification_dir/days[0]['inventory']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact path or hash differs'):
        audit(root, verification=verification)


def test_bare_monthly_complete_claim_rejected(tmp_path):
    root = run(tmp_path, [.4, .3, .2])
    verification = tmp_path/'monthly.json'
    put(verification, {'schema': 'joint-monthly-verification-1', 'stage': 'verify', 'status': 'complete'})
    with pytest.raises(ValueError, match='checkpoint differs'):
        audit(root, verification=verification)
