"""Frozen common windows for native GHCNh and actual-pressure IGRA scalars.

No measurements are interpolated onto standard levels, and inference reads only
past observations. Preparation references immutable admitted datasets; it neither
copies nor changes their observations or normalization statistics.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import numpy as np

from .observation_data import (
    ObservationDataset,
    START,
    TRAIN_END,
    VAL_END,
    END,
    EXTENDED_START,
    EXTENDED_TRAIN_END,
    EXTENDED_VAL_END,
    EXTENDED_END,
)
from .observation_training import digest, save
from .observations import utc
from .profile_training import ProfileDataset, bounded_records

BOUNDARIES = {'train': (START, TRAIN_END), 'validation': (TRAIN_END, VAL_END),
              'test': (VAL_END, END)}
BOUNDARIES_2023_2026 = {
    'train': (EXTENDED_START, EXTENDED_TRAIN_END),
    'validation': (EXTENDED_TRAIN_END, EXTENDED_VAL_END),
    'test': (EXTENDED_VAL_END, EXTENDED_END),
}
SCHEMA = 'joint-measured-observation-windows-1'


def common_windows(horizon_hours=72, issue_stride_hours=24, boundaries=None):
    """Keep entire histories/targets inside frozen parts and IGRA group guards."""
    if (type(horizon_hours) is not int or horizon_hours < 3 or horizon_hours > 240
            or horizon_hours % 3 or type(issue_stride_hours) is not int
            or issue_stride_hours < 1):
        raise ValueError('Need a 3-hour horizon within 3..240 and positive issue stride.')
    if boundaries is None:
        boundaries = BOUNDARIES
    start_time = boundaries['train'][0]
    rows = []
    for split, (begin, end) in boundaries.items():
        # Existing IGRA admission excludes 42 hours beside every split boundary.
        issue = begin + timedelta(hours=54)
        while issue + timedelta(hours=horizon_hours + 42) < end:
            rows.append({'issue': issue.isoformat(), 'split': split,
                         'hour': int((issue - start_time).total_seconds() / 3600)})
            issue += timedelta(hours=issue_stride_hours)
    return rows


def _validate_issue(issue, horizon_hours, *, guarded=True, boundaries=None):
    if boundaries is None:
        boundaries = BOUNDARIES
    issue = utc(issue)
    start_time = boundaries['train'][0]
    seconds = (issue - start_time).total_seconds()
    if seconds % 3600:
        raise ValueError('Native hourly dataset requires an integer-hour issue.')
    for split, (begin, end) in boundaries.items():
        if begin <= issue < end:
            guard = 42 if guarded else 0
            if (issue - timedelta(hours=12 + guard) < begin
                    or issue + timedelta(hours=horizon_hours + guard) >= end):
                raise ValueError('Joint forecast window crosses split/group guard.')
            return issue, split, int(seconds / 3600)
    raise ValueError('Issue lies outside frozen observation archive.')


class JointObservationDataset:
    """Paired sources with a new horizon-specific, chronologically guarded index."""

    def __init__(self, path, profile_root=None, *, horizon_hours=72, issue_stride_hours=24, boundaries=None):
        self.bundle_path = None
        if profile_root is None:
            self.bundle_path = Path(path) / 'dataset.json'
            if self.bundle_path.is_symlink():
                raise ValueError('Joint manifest symlink is forbidden.')
            self.bundle_hash = digest(self.bundle_path)
            self.manifest = json.loads(self.bundle_path.read_text())
            if self.manifest.get('schema') != SCHEMA or self.manifest.get('data_kind') != 'real':
                raise ValueError('Wrong joint dataset schema or source kind.')
            horizon_hours = self.manifest['horizon_hours']
            issue_stride_hours = self.manifest['issue_stride_hours']
            path = self.manifest['surface_root']
            profile_root = self.manifest['profile_root']
            if 'boundaries' in self.manifest:
                boundaries = {k: (utc(v[0]), utc(v[1])) for k, v in self.manifest['boundaries'].items()}
        else:
            self.manifest = None
        if boundaries is None:
            boundaries = BOUNDARIES
        self.boundaries = boundaries
        self.samples = common_windows(horizon_hours, issue_stride_hours, boundaries=boundaries)
        self.horizon_hours = horizon_hours
        self.lead_hours = list(range(3, horizon_hours + 1, 3))
        self.issue_stride_hours = issue_stride_hours
        self.surface = ObservationDataset(path)
        self.profile = ProfileDataset(profile_root)
        self.fingerprint = {
            'surface_dataset_sha256': digest(self.surface.root / 'dataset.json'),
            'profile_dataset_sha256': self.profile.manifest_sha256,
            'surface_hourly_sha256': self.surface.manifest['hourly_sha256'],
            'surface_norm_sha256': self.surface.manifest['norm_sha256'],
            'profile_database_sha256': self.profile.manifest['database_sha256'],
            'horizon_hours': horizon_hours, 'issue_stride_hours': issue_stride_hours,
        }
        if self.manifest is not None:
            if (self.manifest.get('fingerprint') != self.fingerprint
                    or self.manifest.get('samples') != self.samples):
                raise ValueError('Joint frozen identity or windows changed.')
        self.verify()

    def verify(self):
        self.profile.verify()
        if self.surface.root.is_symlink():
            raise ValueError('Surface dataset symlink is forbidden.')
        if digest(self.surface.root / 'dataset.json') != self.fingerprint['surface_dataset_sha256']:
            raise ValueError('Surface dataset manifest changed.')
        for filename, key in (('hourly.npz', 'hourly_sha256'), ('norm.json', 'norm_sha256'),
                              ('observations.sqlite', 'records_sha256')):
            if ((self.surface.root / filename).is_symlink()
                    or digest(self.surface.root / filename) != self.surface.manifest[key]):
                raise ValueError('Surface dataset artifact changed.')
        for source in self.surface.manifest['sources']:
            if Path(source['path']).is_symlink() or digest(source['path']) != source['sha256']:
                raise ValueError('Surface source changed.')
        if self.bundle_path is not None and digest(self.bundle_path) != self.bundle_hash:
            raise ValueError('Joint manifest changed.')

    def issues(self, split, limit=None):
        if split not in BOUNDARIES:
            raise ValueError('Unknown joint dataset split.')
        rows = [utc(row['issue']) for row in self.samples if row['split'] == split]
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError('Issue limit must be a positive integer.')
            if len(rows) > limit:
                rows = [rows[i] for i in np.linspace(0, len(rows) - 1, limit, dtype=int)]
        return rows

    def forecast_history(self, issue, limit=2000):
        """Return raw surface history/masks and scalars; never query future targets."""
        if type(limit) is not int or limit < 1:
            raise ValueError('Profile scalar limit must be a positive integer.')
        issue = utc(issue)
        hour_float = (issue - START).total_seconds() / 3600
        if hour_float % 1 or not 11 <= hour_float < len(self.surface.values):
            raise ValueError('Forecast history requires a supported integer-hour issue.')
        hour = int(hour_float)
        hours = np.arange(hour - 11, hour + 1)
        mask = self.surface.mask[hours] & (self.surface.available[hours] <= hour * 3600)
        values = np.where(mask, self.surface.values[hours], 0).astype(np.float32)
        profiles = bounded_records(self.profile.records(issue - timedelta(hours=12), issue,
                                                         issue=issue), limit)
        return values, mask.copy(), profiles

    def sample(self, issue, limit=2000, *, target_horizon_hours=None):
        """Raw surface inputs/targets plus unmodified actual-pressure profile scalars."""
        if type(limit) is not int or limit < 1:
            raise ValueError('Profile scalar limit must be a positive integer.')
        issue, split, hour = _validate_issue(issue, self.horizon_hours)
        hours = np.arange(hour - 11, hour + 1)
        mask = self.surface.mask[hours] & (self.surface.available[hours] <= hour * 3600)
        inputs = np.where(mask, self.surface.values[hours], 0).astype(np.float32)
        target_hours = hour + np.asarray(self.lead_hours)
        targets = self.surface.values[target_hours].copy()
        target_mask = self.surface.mask[target_hours].copy()
        target_horizon = self.horizon_hours if target_horizon_hours is None else target_horizon_hours
        profile_inputs = bounded_records(self.profile.records(issue - timedelta(hours=12), issue,
                                                              issue=issue, split=split), limit)
        profile_targets = bounded_records(self.profile.records(issue, issue + timedelta(hours=target_horizon),
                                                               split=split), limit)
        return inputs, mask.copy(), targets, target_mask, profile_inputs, profile_targets


def prepare_joint(surface_root, profile_root, output, *, horizon_hours=72, issue_stride_hours=24):
    """Atomically publish a reference bundle; reuse only exactly matching inputs."""
    output = Path(output)
    dataset = JointObservationDataset(surface_root, profile_root, horizon_hours=horizon_hours,
                                      issue_stride_hours=issue_stride_hours)
    manifest = {'schema': SCHEMA, 'data_kind': 'real',
                'surface_root': str(dataset.surface.root.resolve()),
                'profile_root': str(dataset.profile.root.resolve()),
                'horizon_hours': horizon_hours, 'issue_stride_hours': issue_stride_hours,
                'fingerprint': dataset.fingerprint, 'samples': dataset.samples,
                'input_hours': 12, 'lead_hours': dataset.lead_hours,
                'profile_operator': 'actual scalar pressure/location/time; no fabricated standard-level targets',
                'surface_operator': dataset.surface.manifest['operator'],
                'availability': dataset.surface.manifest['availability'],
                'scientific_acceptance': False, 'omega_supervision': False}
    if output.exists():
        existing = JointObservationDataset(output)
        if existing.manifest != manifest:
            raise ValueError('Joint preparation differs; use a new dataset ID.')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.joint-prepare-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        stage.mkdir()
        save(stage / 'dataset.json', manifest)
        dataset.verify()
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon-hours', type=int, default=72)
    parser.add_argument('--issue-stride-hours', type=int, default=24)
    args = parser.parse_args(argv)
    prepare_joint(args.surface, args.profile, args.output, horizon_hours=args.horizon_hours,
                  issue_stride_hours=args.issue_stride_hours)


if __name__ == '__main__':
    main()
