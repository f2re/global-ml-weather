"""Report physical monthly scores only after an independently saved pair recount.

This module deliberately avoids importing training code or torch. Plotting is
optional and imports matplotlib only after the completed-artifact gate passes.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

PRESSURES = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700,
             650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175,
             150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1)
PROFILE = ('temperature', 'specific_humidity', 'u', 'v', 'geopotential', 'omega')
NATIVE = ('station_temperature', 'station_dew_point', 'station_eastward_wind',
          'station_northward_wind', 'station_pressure', 'station_mean_sea_level_pressure')
PROFILE_UNITS = ('K', 'kg kg-1', 'm s-1', 'm s-1', 'm2 s-2', 'Pa s-1')
NATIVE_UNITS = ('K', 'K', 'm s-1', 'm s-1', 'Pa', 'Pa')
CHANNELS = ('model', 'persistence_model_paired', 'persistence',
            'climatology_model_paired', 'climatology')
SUMS = ('abs', 'square', 'signed', 'count')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _checked_file(root, name, expected):
    path = root / name
    if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
        raise ValueError('Monthly report artifact hash differs: ' + name)
    return path


def _valid_sums(array):
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite monthly error accumulator.')
    absolute, square, signed, count = (array[..., i] for i in range(4))
    if (np.any(absolute < 0) or np.any(square < 0) or np.any(count < 0)
            or np.any(count != np.floor(count))
            or np.any(np.abs(signed) > absolute + 1e-7 * np.maximum(1., absolute))
            or np.any(array[count == 0] != 0)):
        raise ValueError('Invalid monthly error sums or counts.')
    for paired, control in ((1, 2), (3, 4)):
        if (not np.array_equal(count[..., paired], count[..., control])
                or np.any(count[..., paired] > count[..., 0])):
            raise ValueError('Monthly controls do not have identical paired counts.')


def load_verified(verification, recount):
    """Bind recount to this immutable completed manifest and checked sums."""
    root, recount = Path(verification), Path(recount)
    manifest = root / 'verification.json'
    if root.is_symlink() or manifest.is_symlink() or recount.is_symlink():
        raise ValueError('Monthly report inputs cannot be symlinks.')
    manifest_bytes = manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    report = json.loads(manifest_bytes)
    checked = json.loads(recount.read_text())
    if (report.get('schema') != 'joint-monthly-verification-1'
            or report.get('stage') != 'verify' or report.get('status') != 'complete'):
        raise ValueError('Monthly report requires completed verification.')
    if (checked.get('schema') != 'joint-verification-recount-1'
            or checked.get('status') != 'passed'
            or checked.get('verification_manifest_sha256') != manifest_hash
            or checked.get('verified_issues') != len(report['days'])):
        raise ValueError('Passed recount is not bound to this verification manifest.')
    try:
        year, month = map(int, report['month'].split('-'))
        dates = [datetime.fromisoformat(day['issue_time'].replace('Z', '+00:00'))
                 for day in report['days']]
        if not dates or any(date.tzinfo is None for date in dates):
            raise ValueError('Monthly issue dates need UTC offsets.')
        dates = [date.astimezone(timezone.utc) for date in dates]
        first = datetime(year, month, 1, dates[0].hour, tzinfo=timezone.utc)
        expected_dates = [first + timedelta(days=day) for day in range(calendar.monthrange(year, month)[1])]
        if dates != expected_dates:
            raise ValueError('Monthly report issue inventory is incomplete.')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Malformed monthly report issue inventory.') from error
    artifact = _checked_file(root, 'accumulators.npz', report['accumulators_sha256'])
    _checked_file(root, 'aggregate.jsonl', report['aggregate_sha256'])
    with np.load(artifact, allow_pickle=False) as archive:
        profile, native = archive['profile'].copy(), archive['native'].copy()
        if ('channels' in archive and tuple(archive['channels'].tolist()) != CHANNELS
                or 'sums' in archive and tuple(archive['sums'].tolist()) != SUMS):
            raise ValueError('Monthly accumulator channel interpretation changed.')
    if profile.shape != (25, 37, 6, 5, 4) or native.shape != (25, 6, 5, 4):
        raise ValueError('Monthly report needs all 37 pressure bins and 25 leads.')
    _valid_sums(profile)
    _valid_sums(native)
    total = int(profile[..., 0, 3].sum() + native[..., 0, 3].sum())
    if checked.get('recounted_scalar_pairs') != total:
        raise ValueError('Recount scalar total differs from monthly accumulators.')
    if np.any(profile[:, :, 5] != 0):
        raise ValueError('This observation archive contains no admitted omega targets.')
    if file_hash(manifest) != manifest_hash:
        raise ValueError('Verification changed during report input validation.')
    return report, checked, profile, native


def metrics(sums):
    absolute, square, signed, count = map(float, sums)
    return {'count': int(count), 'mae': absolute / count if count else None,
            'rmse': math.sqrt(square / count) if count else None,
            'bias': signed / count if count else None}


def _scores(sums):
    result = {}
    for index, channel in enumerate(CHANNELS):
        for name, value in metrics(sums[index]).items():
            result[channel + '_' + name] = value
    for paired, control, label in ((1, 2, 'persistence'), (3, 4, 'climatology')):
        p, c = metrics(sums[paired]), metrics(sums[control])
        result[label + '_skill_rmse'] = (1 - p['rmse'] / c['rmse']
                                        if c['count'] and c['rmse'] > 0 else None)
    return result


def flat_rows(profile, native):
    """Return 5700 inventory rows; pressure values are reporting labels only."""
    for kind, array, variables, units in (
            ('profile', profile, PROFILE, PROFILE_UNITS),
            ('station_native', native, NATIVE, NATIVE_UNITS)):
        for lead in range(25):
            for level in range(37) if kind == 'profile' else (None,):
                for variable, name in enumerate(variables):
                    cell = array[lead, level, variable] if level is not None else array[lead, variable]
                    row = {'kind': kind, 'variable': name, 'units': units[variable],
                           'pressure_bin_hpa': PRESSURES[level] if level is not None else None,
                           'lead_bin_upper_hours': 3 * lead,
                           'bin_center_below_q_training_pressure': bool(kind == 'profile' and variable == 1
                                                                       and PRESSURES[level] < 300),
                           'humidity_pressure_policy_bin': (
                               'mixed_boundary_bin' if kind == 'profile' and variable == 1 and PRESSURES[level] == 300
                               else 'outside_direct_training' if kind == 'profile' and variable == 1 and PRESSURES[level] < 300
                               else 'within_pressure_policy' if kind == 'profile' and variable == 1 else 'not_applicable'),
                           'status': ('no_observed_target' if name == 'omega'
                                      else 'evaluated' if cell[0, 3] else 'no_supported_targets')}
                    yield {**row, **_scores(cell)}


def variable_summary(profile, native):
    """Pool additive physical error sums, never averages of bin RMSE values."""
    rows = []
    for kind, array, variables, units in (
            ('profile', profile, PROFILE, PROFILE_UNITS),
            ('station_native', native, NATIVE, NATIVE_UNITS)):
        pooled = array.sum(axis=(0, 1)) if kind == 'profile' else array.sum(axis=0)
        forecast = array[1:].sum(axis=(0, 1)) if kind == 'profile' else array[1:].sum(axis=0)
        for index, name in enumerate(variables):
            rows.append({'kind': kind, 'variable': name, 'units': units[index],
                         'all_leads': _scores(pooled[index]),
                         'forecast_leads_3_to_72': _scores(forecast[index]),
                         'status': 'no_observed_target' if name == 'omega' else
                                   'evaluated' if pooled[index, 0, 3] else 'no_supported_targets'})
    return rows


def _csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(root, profile, native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outputs = []
    leads = np.arange(0, 73, 3)
    pressures = np.asarray(PRESSURES)
    for variable, name in enumerate(PROFILE[:5]):
        sums = profile[:, :, variable, 0]
        count = sums[..., 3]
        values = np.sqrt(np.divide(sums[..., 1], count,
                                  out=np.zeros_like(count), where=count > 0))
        fig, ax = plt.subplots(figsize=(9, 6))
        colour = ax.pcolormesh(leads, pressures, np.ma.masked_where(count.T == 0, values.T),
                               shading='nearest')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlabel('Actual-time error grouped into upper lead bins (hours)')
        ax.set_ylabel('Nearest log-pressure reporting bin (hPa)')
        ax.set_title(name + ': physical RMSE; empty bins masked')
        if variable == 1:
            ax.axhline(300, color='white', linestyle='--')
            ax.text(.02, .96, 'p < 300 hPa: outside direct humidity training',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox={'facecolor': 'white', 'alpha': .85})
        fig.colorbar(colour, ax=ax, label=PROFILE_UNITS[variable])
        fig.tight_layout()
        path = root / (name + '-rmse.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path.name)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for variable, ax in enumerate(axes.flat):
        def curve(channel):
            sums = native[:, variable, channel]
            count = sums[:, 3]
            return np.sqrt(np.divide(sums[:, 1], count, out=np.full_like(count, np.nan), where=count > 0))
        ax.plot(leads, curve(0), color='black', label='Model, own support')
        for paired, control, label, colour in ((1, 2, 'Persistence', 'tab:blue'),
                                              (3, 4, 'Climate', 'tab:orange')):
            ax.plot(leads, curve(control), color=colour, label=label + ', common pairs')
            ax.plot(leads, curve(paired), color=colour, linestyle='--', label='Model on ' + label.lower() + ' pairs')
        ax.set_title(NATIVE[variable])
        ax.set_xlabel('Upper lead reporting bin (hours)')
        ax.set_ylabel('RMSE (' + NATIVE_UNITS[variable] + ')')
        ax.legend(fontsize=7)
    fig.suptitle('Native measured targets; heights unverified; supports differ between controls')
    fig.tight_layout(rect=(0, 0, 1, .96))
    path = root / 'native-rmse.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path.name)
    return outputs


def build_report(verification, recount, output, *, plots=False):
    input_recount_hash = file_hash(recount)
    report, checked, profile, native = load_verified(verification, recount)
    output = Path(output)
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise FileExistsError('Choose a new monthly report output without symlink parents.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.monthly-report-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'report'
        stage.mkdir()
        rows = list(flat_rows(profile, native))
        _csv(stage / 'metrics.csv', rows)
        count_keys = ['kind', 'variable', 'units', 'pressure_bin_hpa', 'lead_bin_upper_hours', 'status']
        _csv(stage / 'counts.csv', [{key: row[key] for key in count_keys +
                                    [channel + '_count' for channel in CHANNELS]} for row in rows])
        images = _plots(stage, profile, native) if plots else []
        summary = {'schema': 'joint-monthly-physical-report-1', 'status': 'complete',
                   'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'report_source_sha256': file_hash(Path(__file__)),
                   'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                   'platform': platform.platform(),
                                   'matplotlib': sys.modules['matplotlib'].__version__ if plots else None},
                   'month': report['month'], 'verified_issues': checked['verified_issues'],
                   'inventory_rows': len(rows), 'verification_manifest_sha256': checked['verification_manifest_sha256'],
                   'recount_sha256': input_recount_hash, 'accumulators_sha256': report['accumulators_sha256'],
                   'diagnostic_only': True, 'scientific_acceptance': False,
                   'count_scope': 'forecast-observation scalar comparisons; reused targets and neighbouring launches/levels are dependent',
                   'pressure_scope': '37 reporting bins; scored at actual pressure, not manufactured standard-level truth',
                   'humidity_scope': 'Flags describe bin centres; the 300 hPa boundary bin mixes actual target pressure regimes. Exact regimes require persisted actual-pressure pairs.',
                   'metrics_pooling': 'sum absolute/squared/signed errors and counts before division; no mean of RMSE',
                   'climate_scope': {'profile': report.get('profile_climatology'), 'native': report.get('surface_climatology')},
                   'uncertainty': 'not calculated; no significance claim',
                   'variables': variable_summary(profile, native), 'plots': images,
                   'artifacts': {path.name: file_hash(path) for path in stage.iterdir()}}
        (stage / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        if file_hash(Path(verification) / 'verification.json') != summary['verification_manifest_sha256']:
            raise ValueError('Verification changed during report generation.')
        if file_hash(recount) != input_recount_hash:
            raise ValueError('Recount changed during report generation.')
        _checked_file(Path(verification), 'accumulators.npz', report['accumulators_sha256'])
        _checked_file(Path(verification), 'aggregate.jsonl', report['aggregate_sha256'])
        stage.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--recount', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(argv)
    report = build_report(args.verification, args.recount, args.output, plots=args.plots)
    print(json.dumps({'status': report['status'], 'inventory_rows': report['inventory_rows'],
                      'plots': report['plots']}, allow_nan=False))


if __name__ == '__main__':
    main()
