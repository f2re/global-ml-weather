"""Physical report gates and additive pooling on unequal common supports."""
import csv
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pytest

from global_weather import joint_monthly_report as module


def fixture(tmp_path):
    root = tmp_path / 'verification'
    root.mkdir()
    profile = np.zeros((25, 37, 6, 5, 4))
    native = np.zeros((25, 6, 5, 4))
    # Two bins: one error 1, three errors 3. Pooled RMSE is sqrt(7), not 2.
    profile[1, 0, 0, 0] = [1, 1, 1, 1]
    profile[2, 1, 0, 0] = [9, 27, 9, 3]
    profile[1, 0, 0, 1] = [1, 1, 1, 1]
    profile[1, 0, 0, 2] = [2, 4, 2, 1]
    np.savez_compressed(root / 'accumulators.npz', profile=profile, native=native,
                        channels=np.asarray(module.CHANNELS), sums=np.asarray(module.SUMS))
    (root / 'aggregate.jsonl').write_text('{}\n')
    report = {'schema': 'joint-monthly-verification-1', 'stage': 'verify', 'status': 'complete',
              'month': '2022-09', 'days': [{'issue_time': (datetime(2022, 9, 1, 6, tzinfo=timezone.utc)
                                                         + timedelta(days=day)).isoformat()} for day in range(30)],
              'accumulators_sha256': module.file_hash(root / 'accumulators.npz'),
              'aggregate_sha256': module.file_hash(root / 'aggregate.jsonl')}
    (root / 'verification.json').write_text(json.dumps(report))
    recount = tmp_path / 'recount.json'
    checked = {'schema': 'joint-verification-recount-1', 'status': 'passed',
               'verification_manifest_sha256': module.file_hash(root / 'verification.json'),
               'verified_issues': 30, 'recounted_scalar_pairs': 4}
    recount.write_text(json.dumps(checked))
    return root, recount, report, checked


def test_report_pools_physical_error_sums_and_retains_all_empty_rows(tmp_path):
    root, recount, *_ = fixture(tmp_path)
    output = tmp_path / 'report'
    summary = module.build_report(root, recount, output)
    assert summary['inventory_rows'] == 5700
    temperature = summary['variables'][0]['all_leads']
    assert temperature['model_rmse'] == pytest.approx(math.sqrt(7))
    assert temperature['model_count'] == 4
    assert temperature['persistence_model_paired_count'] == temperature['persistence_count'] == 1
    assert temperature['persistence_skill_rmse'] == .5
    assert temperature['climatology_skill_rmse'] is None
    with (output / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5700
    assert all(row['model_rmse'] == '' and row['status'] == 'no_observed_target'
               for row in rows if row['variable'] == 'omega')
    assert summary['plots'] == []


@pytest.mark.parametrize('change', ['incomplete', 'recount_hash', 'failed_recount', 'wrong_count', 'artifact'])
def test_invalid_completion_recount_or_hash_blocks_output(tmp_path, change):
    root, recount, report, checked = fixture(tmp_path)
    if change == 'incomplete':
        report['status'] = 'running'
    elif change == 'recount_hash':
        checked['verification_manifest_sha256'] = '0' * 64
    elif change == 'failed_recount':
        checked['status'] = 'failed'
    elif change == 'wrong_count':
        checked['recounted_scalar_pairs'] = 3
    else:
        (root / 'accumulators.npz').write_bytes(b'changed')
    (root / 'verification.json').write_text(json.dumps(report))
    recount.write_text(json.dumps(checked))
    output = tmp_path / 'report'
    with pytest.raises(ValueError):
        module.build_report(root, recount, output)
    assert not output.exists()


def test_mismatched_control_pair_counts_are_rejected():
    native = np.zeros((25, 6, 5, 4))
    native[1, 0, 0] = [2, 4, 2, 1]
    native[1, 0, 2] = [1, 1, 1, 1]
    with pytest.raises(ValueError, match='paired counts'):
        module._valid_sums(native)


def test_manifest_mutation_after_gate_cannot_rebind_summary(tmp_path, monkeypatch):
    root, recount, report, _ = fixture(tmp_path)
    original = module.load_verified

    def changed_after_validation(*args):
        result = original(*args)
        report['month'] = '2022-10'
        (root / 'verification.json').write_text(json.dumps(report))
        return result

    monkeypatch.setattr(module, 'load_verified', changed_after_validation)
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='Verification changed'):
        module.build_report(root, recount, output)
    assert not output.exists()
