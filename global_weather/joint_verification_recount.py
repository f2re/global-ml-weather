"""Recompute monthly scalar scores from the persisted observation pairs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .grib_export import sha256
from .joint_monthly_verification import (
    SCHEMA, accumulate, empty_accumulator, inventory_rows, monthly_issues, reporting_bins,
)
from .observation_data import NATIVE_VARIABLES
from .observations import utc
from .vertical import PROFILE_VARIABLES


def _compare_inventory(path, expected_sha256, profiles, native):
    if path.is_symlink() or sha256(path) != expected_sha256:
        raise ValueError('Metric inventory hash differs.')
    expected = list(inventory_rows(profiles)) + list(inventory_rows(native, profile=False))
    actual = [json.loads(line) for line in path.read_text().splitlines()]
    if len(actual) != len(expected):
        raise ValueError('Metric inventory is incomplete.')
    for observed, calculated in zip(actual, expected):
        for key in ('kind', 'variable', 'units', 'lead_bin_upper_hours',
                    'pressure_reporting_bin_hpa', 'metrics', 'skill_rmse', 'status'):
            if observed.get(key) != calculated.get(key):
                raise ValueError('Metrics cannot be reproduced from persisted pairs: '+key)


def recount(verification):
    root = Path(verification)
    manifest = root / 'verification.json'
    report = json.loads(manifest.read_text())
    if (report.get('schema') != SCHEMA or report.get('stage') != 'verify'
            or report.get('status') != 'complete'):
        raise ValueError('Only a completed observation comparison can be recounted.')
    days = report['days']
    if [utc(day['issue_time']) for day in days] != monthly_issues(
            report['month'], utc(days[0]['issue_time']).hour):
        raise ValueError('Monthly issue inventory is incomplete.')
    profiles, native = empty_accumulator(), empty_accumulator(False)
    total = 0
    for day in days:
        issue = utc(day['issue_time'])
        pairs = root / day['pairs']
        if pairs.is_symlink() or sha256(pairs) != day['pairs_sha256']:
            raise ValueError('Persisted pair hash differs.')
        pday, nday = empty_accumulator(), empty_accumulator(False)
        with gzip.open(pairs, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if utc(row['issue_time']) != issue:
                    raise ValueError('Pair belongs to another issue.')
                if row['kind'] == 'profile':
                    lead, level = reporting_bins(issue, row)
                    cell = pday[lead, level, PROFILE_VARIABLES.index(row['verification_variable'])]
                elif row['kind'] == 'station_native':
                    hours = (utc(row['observed_at']) - issue).total_seconds() / 3600
                    if not 0 < hours <= 72:
                        raise ValueError('Native pair is outside forecast horizon.')
                    cell = nday[int(np.ceil(hours / 3)), NATIVE_VARIABLES.index(row['verification_variable'])]
                else:
                    raise ValueError('Unknown pair kind.')
                accumulate(cell, row['predicted'], row['observed'],
                           persistence=row['persistence'], climatology=row['climatology'])
                total += 1
        _compare_inventory(root / day['inventory'], day['inventory_sha256'], pday, nday)
        profiles += pday
        native += nday
    artifact = root / 'accumulators.npz'
    if artifact.is_symlink() or sha256(artifact) != report['accumulators_sha256']:
        raise ValueError('Monthly accumulator hash differs.')
    with np.load(artifact, allow_pickle=False) as arrays:
        for name, calculated in (('profile', profiles), ('native', native)):
            if not np.array_equal(arrays[name], calculated):
                raise ValueError('Monthly accumulators cannot be reproduced from persisted pairs.')
    _compare_inventory(root / 'aggregate.jsonl', report['aggregate_sha256'], profiles, native)
    return {'schema': 'joint-verification-recount-1', 'status': 'passed',
            'verification_manifest_sha256': sha256(manifest), 'verified_issues': len(days),
            'recounted_scalar_pairs': total, 'scientific_acceptance': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = recount(args.verification)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
