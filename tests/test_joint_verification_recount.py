"""Persisted pairs must independently reproduce metric values and counts."""
import gzip
import json

import numpy as np
import pytest

from global_weather.grib_export import sha256
from global_weather.joint_monthly_verification import accumulate, empty_accumulator, inventory_rows
import global_weather.joint_verification_recount as module


def fixture(tmp_path, monkeypatch):
    issue = module.monthly_issues()[0]
    monkeypatch.setattr(module, 'monthly_issues', lambda *args: [issue])
    row = {'kind': 'station_native', 'issue_time': issue.isoformat(),
           'observed_at': issue.replace(hour=7).isoformat(),
           'verification_variable': 'station_temperature', 'predicted': 282.,
           'observed': 280., 'persistence': 281., 'climatology': None}
    pairs = tmp_path/'pairs.jsonl.gz'
    with gzip.open(pairs, 'wt') as stream:
        stream.write(json.dumps(row)+'\n')
    profile, native = empty_accumulator(), empty_accumulator(False)
    accumulate(native[1, 0], 282., 280., persistence=281.)
    inventory = tmp_path/'daily.jsonl'
    rows = list(inventory_rows(profile)) + list(inventory_rows(native, profile=False))
    inventory.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    np.savez_compressed(tmp_path/'accumulators.npz', profile=profile, native=native)
    report = {'schema': module.SCHEMA, 'stage': 'verify', 'status': 'complete', 'month': '2022-09',
              'accumulators_sha256': sha256(tmp_path/'accumulators.npz'),
              'days': [{'issue_time': issue.isoformat(), 'pairs': pairs.name,
                        'pairs_sha256': sha256(pairs), 'inventory': inventory.name,
                        'inventory_sha256': sha256(inventory)}]}
    (tmp_path/'verification.json').write_text(json.dumps(report))
    return inventory, rows, report


def test_recount_paired_counts_and_empty_levels(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    assert module.recount(tmp_path)['recounted_scalar_pairs'] == 1


def test_changed_metrics_fail_even_with_recomputed_file_hash(tmp_path, monkeypatch):
    inventory, rows, report = fixture(tmp_path, monkeypatch)
    target = next(row for row in rows if row['kind'] == 'station_native'
                  and row['variable'] == 'station_temperature' and row['lead_bin_upper_hours'] == 3)
    target['metrics']['model']['rmse'] = 1.
    inventory.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    report['days'][0]['inventory_sha256'] = sha256(inventory)
    (tmp_path/'verification.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='reproduced'):
        module.recount(tmp_path)
