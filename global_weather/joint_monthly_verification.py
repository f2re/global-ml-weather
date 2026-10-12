"""Two-stage monthly diagnosis: freeze every forecast before opening any target.

Actual-pressure observations are evaluated with the existing space/pressure/time
operator. Pressure bins describe reporting only, not interpolated observations.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import gzip
import itertools
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from .analysis.observation_operator import _prediction
from .grib_export import sha256
from .joint_observation_inference import timed_forecast, validate_history, save_issue
from .observation_data import NATIVE_VARIABLES, UNITS, VARIABLES, START, TRAIN_END, _checked
from .observation_training import save, sync_directory
from .observations import utc
from .profile_physics import parse_policy, screen_humidity
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS

SCHEMA = 'joint-monthly-verification-1'
LEADS = np.arange(0, 73, 3)
CHANNELS = ('model', 'persistence_model_paired', 'persistence', 'climatology_model_paired', 'climatology')


class Budget:
    def __init__(self, seconds=3600, maximum_bytes=3000000000):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0 or type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError('Positive finite resource budgets are required.')
        self.deadline = time.monotonic()+seconds
        self.maximum_bytes = maximum_bytes
        self.root = None
        self.extra_roots = []

    def check(self, root, *, size=False):
        if time.monotonic() >= self.deadline: raise TimeoutError('Monthly elapsed-time budget exhausted.')
        if self.root is None: self.root = Path(root)
        roots = [self.root]+[path for path in self.extra_roots if not path.resolve().is_relative_to(self.root.resolve())]
        if size and sum(path.stat().st_size for root in roots for path in root.rglob('*') if path.is_file()) > self.maximum_bytes:
            raise RuntimeError('Monthly new-artifact disk budget exhausted.')


def validate_config(config):
    required = {'schema': SCHEMA, 'issue_stride_hours': 24, 'horizon_hours': 72,
                'lead_step_hours': 3, 'profile_target_limit': None,
                'humidity_min_pressure_pa': 0., 'humidity_qc': True,
                'diagnostic_only': True, 'allow_training': False, 'allow_downloads': False,
                'profile_variables': list(PROFILE_VARIABLES), 'native_variables': list(NATIVE_VARIABLES),
                'profile_pressure_bin_count': 37}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Monthly config changes a fixed scientific evaluation policy.')
    if config.get('expected_issue_count') != len(monthly_issues(config['month'], config['issue_hour'])):
        raise ValueError('Configured month issue inventory differs.')
    if type(config.get('checkpoint_epoch')) is not int or config['checkpoint_epoch'] < 1 or len(config.get('checkpoint_sha256', '')) != 64:
        raise ValueError('Explicit frozen epoch and checkpoint SHA256 are required.')
    Budget(config['maximum_elapsed_seconds'], config['maximum_new_artifact_bytes'])
    return config


def _bind_config(state, checkpoint_hash, config):
    if config is not None and (state['epoch'] != config['checkpoint_epoch']
                              or checkpoint_hash != config['checkpoint_sha256']
                              or state['config']['max_records_per_window'] != config['input_profile_limit']):
        raise ValueError('Configured checkpoint or causal input budget differs from frozen training.')


def monthly_issues(month='2022-09', issue_hour=6):
    if type(issue_hour) is not int or not 0 <= issue_hour <= 23:
        raise ValueError('Issue hour must be an integer in 0..23 UTC.')
    year, number = map(int, month.split('-'))
    days = calendar.monthrange(year, number)[1]
    return [datetime(year, number, day, issue_hour, tzinfo=timezone.utc) for day in range(1, days+1)]


def empty_accumulator(profile=True):
    return np.zeros((25, 37, 6, 5, 4) if profile else (25, 6, 5, 4), dtype=np.float64)


def metrics(sums):
    absolute, squared, signed, count = map(float, sums)
    if count == 0:
        return {'mae': None, 'rmse': None, 'bias': None, 'count': 0}
    return {'mae': absolute/count, 'rmse': math.sqrt(squared/count),
            'bias': signed/count, 'count': int(count)}


def accumulate(cell, prediction, observation, *, persistence=None, climatology=None):
    """Each control and its paired model use precisely the same accepted target."""
    if not math.isfinite(prediction) or not math.isfinite(observation):
        raise ValueError('Nonfinite accepted prediction or observation.')
    def add(channel, value):
        error = value - observation
        cell[channel] += [abs(error), error*error, error, 1]
    add(0, prediction)
    for model_channel, control_channel, value in ((1, 2, persistence), (3, 4, climatology)):
        if value is not None:
            if not math.isfinite(value):
                raise ValueError('Nonfinite supported control.')
            add(model_channel, prediction)
            add(control_channel, value)


def reporting_bins(issue, record):
    hours = (utc(record['observed_at']) - issue).total_seconds()/3600
    if not 0 <= hours <= 72:
        raise ValueError('Target is outside the complete forecast horizon.')
    lead = int(math.ceil(hours/3))
    p = float(record['pressure_pa'])
    if not math.isfinite(p) or p <= 0:
        raise ValueError('Target pressure must be finite and positive.')
    level = int(np.abs(np.log(np.asarray(PRESSURE_HPA)*100)-math.log(p)).argmin())
    return lead, level


def _new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise FileExistsError('Choose a new immutable monthly output without symlink parents.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def forecast_month(dataset, training, static, output, *, month='2022-09', issue_hour=6,
                   device='auto', spacing_degrees=5., source_revision=None, budget=None, config=None):
    """Freeze raw forecasts and all products for the whole month, without targets."""
    from .joint_observation_training import load_frozen
    from .joint_static import load_static
    output = _new_output(output)
    budget = budget or Budget(); budget.check(output)
    model, joint = load_frozen(dataset, training, device=device)
    selected_device = model.xyz.device
    state, checkpoint_hash = model.frozen_metadata, model.frozen_checkpoint_sha256
    _bind_config(state, checkpoint_hash, config)
    static_fields = load_static(static, model.grid)
    if static_fields['static_sha256'] != state['identity']['static_fingerprint']:
        raise ValueError('Forecast static identity differs from frozen training.')
    dates = monthly_issues(month, issue_hour)
    output.mkdir()
    report = {'schema': SCHEMA, 'stage': 'forecast', 'status': 'running',
              'month': month, 'issue_hour': issue_hour, 'horizon_hours': 72,
              'lead_hours': LEADS.tolist(), 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_epoch': state['epoch'], 'identity': state['identity'],
              'dataset_fingerprint': joint.fingerprint,
              'static_sha256': static_fields['static_sha256'],
              'source_revision': source_revision, 'monthly_source_sha256': sha256(Path(__file__)),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'products': [],
              'future_targets_read': False, 'scientific_acceptance': False}
    report['config'] = config
    report['config_sha256'] = config.get('configuration_sha256') if config else None
    save(output/'month.json', report)
    try:
        for issue in dates:
            budget.check(output, size=True)
            raw, mask, inputs = joint.forecast_history(issue, state['config']['max_records_per_window'])
            validate_history(raw, mask, inputs, issue)
            inputs, input_qc = screen_humidity(inputs, model.physical_policy)
            frames, telemetry = timed_forecast(model, raw, mask, inputs, issue, horizon_hours=72,
                                               device=selected_device, warmups=0, repeats=1)
            profiles = np.stack([frame.profiles.cpu().numpy() for frame in frames])
            norm_mask = np.stack([frame.profile_variable_mask.cpu().numpy() for frame in frames])
            native = np.stack([frame.native_physical.cpu().numpy() for frame in frames])
            if (profiles.shape != (25, model.grid.n_cells, 37, 6)
                    or not np.isfinite(profiles[..., :5]).all() or not np.isfinite(native).all()
                    or not np.array_equal([frame.lead_hours for frame in frames], LEADS)):
                raise ValueError('Incomplete or nonfinite raw forecast before issuance masking.')
            # Raw verification state intentionally precedes predicted-terrain
            # issuance masking. An incorrect prediction cannot hide its error.
            folder = output/issue.strftime('%Y%m%dT%H%M%SZ')
            with tempfile.TemporaryDirectory(prefix='.monthly-issue-', dir=output) as temporary:
                stage = Path(temporary)/'issue'; stage.mkdir()
                np.savez_compressed(stage/'verification-state.npz', profiles=profiles,
                                    profile_norm_mask=norm_mask, native=native,
                                    lead_hours=LEADS, xyz=model.grid.xyz,
                                    station_latlon=joint.surface.latlon,
                                    pressure_pa=np.asarray(PRESSURE_HPA)*100)
                np.savez_compressed(stage/'inputs.npz', raw_surface=raw, surface_mask=mask)
                (stage/'profile_inputs.json').write_text(json.dumps(inputs, allow_nan=False)+'\n')
                product = save_issue(joint, model, state, checkpoint_hash, issue, stage/'product',
                                     horizon_hours=72, device=selected_device, warmups=0, repeats=1,
                                     spacing=spacing_degrees, source_revision=source_revision,
                                     static_fields=static_fields)
                descriptor = {'issue_time': issue.isoformat(), 'directory': folder.name,
                              'verification_state_sha256': sha256(stage/'verification-state.npz'),
                              'inputs_sha256': sha256(stage/'inputs.npz'),
                              'profile_inputs_sha256': sha256(stage/'profile_inputs.json'),
                              'checkpoint_sha256': checkpoint_hash, 'input_qc': input_qc,
                              'raw_mask_policy': 'fixed normalization support only; no predicted pressure/geopotential filter',
                              'telemetry': telemetry, 'grib_sha256': product['grib_sha256']}
                save(stage/'issue.json', descriptor)
                sync_directory(stage); stage.rename(folder)
            report['products'].append(descriptor)
            save(output/'month.json', report)
            budget.check(output, size=True)
        joint.verify()
        if load_static(static, model.grid)['static_sha256'] != static_fields['static_sha256']:
            raise ValueError('Static artifact changed during monthly forecasting.')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'; report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output/'month.json', report)
    return report


def iter_target_groups(root, issue, *, profile=True, chunk_size=2048):
    """Stream all immutable SQLite targets; keep paired T/q within a profile group."""
    database = Path(root)/'observations.sqlite' if Path(root).is_dir() else Path(root)
    if database.is_symlink(): raise ValueError('Target database symlink is forbidden.')
    start, end = issue.isoformat(), (issue+timedelta(hours=72)).isoformat()
    if profile:
        query = ("SELECT record FROM records WHERE observed>? AND observed<=? "
                 "ORDER BY observed,json_extract(record,'$.profile_id'),id")
    else:
        query = 'SELECT record FROM observations WHERE observed>? AND observed<=? ORDER BY observed,id'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        cursor = connection.execute(query, (start, end))
        def records():
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows: return
                for (text,) in rows: yield json.loads(text)
        key = (lambda row: (row['observed_at'], row.get('profile_id'))) if profile else (lambda row: row['observed_at'])
        for _, rows in itertools.groupby(records(), key):
            yield list(rows)


def iter_native_targets(source, issue, stations, *, chunk_size=2048):
    """Read exact station/time primary-key ranges without copying a raw archive."""
    start, end = issue, issue+timedelta(hours=72)
    database = Path(source)/'observations.sqlite'
    if database.is_symlink(): raise ValueError('Native source symlink is forbidden.')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as origin:
        for station in stations:
            prefix = 'GHCNh/'+station+'/'
            cursor = origin.execute('SELECT id,record FROM observations WHERE id>? AND id<=? ORDER BY id',
                                    (prefix+start.isoformat()+'/~', prefix+end.isoformat()+'/~'))
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows: break
                records = []
                for key, text in rows:
                    row = json.loads(text); when = utc(row['observed_at'])
                    if not start < when <= end or row['observation_id'] != key:
                        raise ValueError('Native indexed identity/time disagrees with actual observation.')
                    records.append(row)
                yield records


def _frames(archive, issue):
    values, masks = archive['profiles'], archive['profile_norm_mask']
    if masks.dtype != np.bool_ or values.shape != masks.shape or values.shape[0] != 25:
        raise ValueError('Invalid frozen raw-profile masks or horizon.')
    return [SimpleNamespace(valid_time=issue+timedelta(hours=int(lead)), profiles=torch.as_tensor(values[i]),
                            profile_variable_mask=torch.as_tensor(masks[i]),
                            profile_mask=torch.as_tensor(masks[i, ..., :5].any(-1)),
                            wind_basis='local_enu_vector') for i, lead in enumerate(LEADS)]


def persistence_profiles(inputs):
    latest = {}
    for row in inputs:
        key = (row['variable'], float(row['latitude']), float(row['longitude']), float(row['pressure_pa']))
        if key not in latest or utc(row['observed_at']) > utc(latest[key]['observed_at']):
            latest[key] = row
    return {key: float(row['value']) for key, row in latest.items()}


def persistence_surface(raw, mask):
    result = np.full(raw.shape[1:], np.nan)
    for hour in range(len(raw)):
        result = np.where(mask[hour], raw[hour], result)
    return result


def _native_prediction(values, lead, station, variable):
    if not 0 <= lead <= 72: return None
    left = int(math.floor(lead/3)); right = int(math.ceil(lead/3))
    alpha = lead/3-left
    result = (1-alpha)*float(values[left, station, variable])+alpha*float(values[right, station, variable])
    if not math.isfinite(result): raise ValueError('Nonfinite raw native forecast; cannot mask its error.')
    return result


def _climate_frame(model, when):
    if not hasattr(model, 'climatology'): return None
    mean, mask = model.climatology.sample(when)
    values = np.concatenate((mean, np.full(mean.shape[:-1]+(1,), np.nan)), -1)
    support = np.concatenate((mask, np.zeros(mask.shape[:-1]+(1,), bool)), -1)
    return SimpleNamespace(valid_time=when, profiles=torch.as_tensor(values, dtype=torch.float32),
                           profile_variable_mask=torch.as_tensor(support),
                           profile_mask=torch.as_tensor(mask.any(-1)), wind_basis='local_enu_vector')


def inventory_rows(accumulator, *, profile=True, training_support=None):
    for lead in range(25):
        for level in range(37) if profile else (None,):
            for variable in range(6):
                sums = accumulator[lead, level, variable] if profile else accumulator[lead, variable]
                row = {'kind': 'profile' if profile else 'station_native', 'lead_bin_upper_hours': int(LEADS[lead]),
                       'variable': (PROFILE_VARIABLES if profile else NATIVE_VARIABLES)[variable],
                       'units': (PROFILE_UNITS if profile else UNITS)[variable],
                       'metrics': {name: metrics(sums[index]) for index, name in enumerate(CHANNELS)}}
                row['skill_rmse'] = {}
                for model_index, control_index, name in ((1, 2, 'persistence'), (3, 4, 'climatology')):
                    model_metric, control_metric = metrics(sums[model_index]), metrics(sums[control_index])
                    row['skill_rmse'][name] = (1-model_metric['rmse']/control_metric['rmse']
                                              if control_metric['count'] and control_metric['rmse'] > 0 else None)
                if profile:
                    row['pressure_reporting_bin_hpa'] = PRESSURE_HPA[level]
                    row['pressure_truth'] = 'actual_measurements_grouped_by_reporting_bin; no generated standard-level targets'
                    row['training_target_support'] = (bool(training_support[level, variable])
                                                      if variable < 5 and training_support is not None else False)
                    row['humidity_training_sensitivity_region'] = bool(variable == 1 and PRESSURE_HPA[level] < 300)
                row['status'] = ('no_admitted_omega_targets' if profile and variable == 5
                                 else 'evaluated' if sums[0, 3] else 'no_supported_admitted_targets')
                yield row


def _write_inventory(path, profiles, native, training_support):
    with path.open('w') as stream:
        for row in itertools.chain(inventory_rows(profiles, training_support=training_support),
                                   inventory_rows(native, profile=False)):
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def verify_month(dataset, training, static, forecasts, output, *, month='2022-09', issue_hour=6, neighbours=3,
                 budget=None, config=None):
    """Only a fully published monthly forecast permits subsequent target reads."""
    from .joint_observation_training import load_frozen
    from .joint_static import load_static
    forecasts, output = Path(forecasts), _new_output(output)
    budget = budget or Budget(); budget.check(output)
    budget.extra_roots.append(forecasts)
    monthly_hash = sha256(forecasts/'month.json')
    frozen = json.loads((forecasts/'month.json').read_text())
    dates = monthly_issues(month, issue_hour)
    if (frozen.get('schema') != SCHEMA or frozen.get('stage') != 'forecast'
            or frozen.get('status') != 'complete' or frozen.get('future_targets_read') is not False
            or [row['issue_time'] for row in frozen['products']] != [x.isoformat() for x in dates]
            or frozen.get('lead_hours') != LEADS.tolist()):
        raise ValueError('All daily complete raw forecasts must be frozen before verification.')
    # Bind every issue before querying even the first target database.
    for row in frozen['products']:
        folder = forecasts/utc(row['issue_time']).strftime('%Y%m%dT%H%M%SZ')
        if folder.name != row['directory']: raise ValueError('Invalid frozen issue directory.')
        for name, key in (('verification-state.npz', 'verification_state_sha256'),
                          ('inputs.npz', 'inputs_sha256'), ('profile_inputs.json', 'profile_inputs_sha256')):
            if (folder/name).is_symlink() or sha256(folder/name) != row[key]:
                raise ValueError('Monthly frozen raw forecast or inputs changed.')
    model, joint = load_frozen(dataset, training, device='cpu')
    state = model.frozen_metadata
    _bind_config(state, model.frozen_checkpoint_sha256, config)
    if (model.frozen_checkpoint_sha256 != frozen['checkpoint_sha256'] or state['identity'] != frozen['identity']
            or joint.fingerprint != frozen['dataset_fingerprint']
            or load_static(static, model.grid)['static_sha256'] != frozen['static_sha256']):
        raise ValueError('Monthly forecast/checkpoint/dataset/static identity differs.')
    eval_payload = model.physical_policy.payload(); eval_payload['humidity_min_pressure_pa'] = 0.
    evaluation_policy = parse_policy(eval_payload)
    station_indices = {row['id']: index for index, row in enumerate(joint.surface.manifest['stations'])}
    train_hours = int((TRAIN_END-START).total_seconds()/3600)
    train_month = np.asarray([(START+timedelta(hours=hour)).month-1 for hour in range(train_hours)])
    native_climate_support = np.stack([joint.surface.mask[:train_hours][train_month == month].any(0) for month in range(12)])
    training_support = np.asarray(state['trained_variable_mask'], dtype=bool)
    profile_total, native_total = empty_accumulator(), empty_accumulator(False)
    output.mkdir()
    report = {'schema': SCHEMA, 'stage': 'verify', 'status': 'running', 'month': month,
              'forecast_month_sha256': monthly_hash, 'checkpoint_sha256': frozen['checkpoint_sha256'],
              'forecast_manifest_sha256': monthly_hash, 'forecast_manifest_path': str((forecasts/'month.json').resolve()),
              'identity': state['identity'], 'dataset_fingerprint': joint.fingerprint,
              'static_sha256': frozen['static_sha256'], 'evaluation_qc_policy': eval_payload,
              'neighbours': neighbours, 'verification_source_sha256': sha256(Path(__file__)),
              'execution_argv': list(sys.argv), 'execution_working_directory': str(Path.cwd()),
              'target_record_limit': None, 'started_utc': datetime.now(timezone.utc).isoformat(), 'days': [],
              'pressure_bin_definition': 'nearest log-pressure standard bin; prediction evaluated at actual measured pressure',
              'lead_bin_definition': '(previous 3-hour lead, reported upper lead]; +0 has no future targets',
              'target_mask': 'admitted measurement support and model fixed norm support; no predicted terrain/pressure screen',
              'metric_weighting': 'each admitted scalar equally weighted; reused targets across issues are dependent',
              'profile_climatology': 'official NOAA monthly physical masked context via observation operator; not GraphCast means',
              'surface_climatology': 'fixed train station monthly means; controls exclude station-month-variable fallback entries without measured train support',
              'persistence': 'last available measurement in identical frozen model input; exact profile location/pressure/variable',
              'diagnostic_only': True, 'scientific_acceptance': False}
    report['config'] = config
    report['config_sha256'] = config.get('configuration_sha256') if config else None
    report['metric_definitions'] = {'error': 'prediction-observation', 'mae': 'sum(abs(error))/count',
                                    'rmse': 'sqrt(sum(error^2)/count)', 'bias': 'sum(error)/count',
                                    'skill_rmse': '1-RMSE_paired_model/RMSE_control; null with empty pairs or zero control RMSE'}
    save(output/'verification.json', report)
    writers = ExitStack()
    try:
        report['native_target_scope'] = 'all admitted raw scalars at frozen station inventory; indexed GHCNh station/time IDs'
        with torch.inference_mode():
            for issue, descriptor in zip(dates, frozen['products']):
                budget.check(output, size=True)
                pairpath = output/(issue.strftime('%Y%m%dT%H%M%SZ')+'.pairs.jsonl.gz')
                rejectpath = output/(issue.strftime('%Y%m%dT%H%M%SZ')+'.rejections.jsonl.gz')
                pairtemp, rejecttemp = pairpath.with_suffix('.partial'), rejectpath.with_suffix('.partial')
                pairs = writers.enter_context(gzip.open(pairtemp, 'wt'))
                rejections = writers.enter_context(gzip.open(rejecttemp, 'wt'))
                def record_pair(row, predicted, observed, persistence, climatology, kind):
                    data = {key: row.get(key) for key in ('observation_id', 'profile_id', 'provider', 'observed_at',
                            'latitude', 'longitude', 'pressure_pa', 'variable', 'units', 'quality', 'provider_qc')}
                    data.update(kind=kind, issue_time=issue.isoformat(), predicted=predicted, observed=observed,
                                persistence=persistence, climatology=climatology)
                    data['verification_variable'] = (row['variable'] if kind == 'profile'
                                                     else NATIVE_VARIABLES[VARIABLES.index(row['variable'])])
                    threshold = state['config']['physical_policy']['humidity_min_pressure_pa']
                    data['training_regime_label'] = ('unsupported_training' if kind == 'profile'
                         and row['variable'] == 'specific_humidity' and float(row['pressure_pa']) < threshold else 'measured_training_regime')
                    pairs.write(json.dumps(data, allow_nan=False)+'\n')
                def record_rejection(row, reason, kind):
                    rejections.write(json.dumps({'observation_id': row.get('observation_id'), 'observed_at': row.get('observed_at'),
                                                 'kind': kind, 'reason': reason}, allow_nan=False)+'\n')
                folder = forecasts/descriptor['directory']
                with np.load(folder/'verification-state.npz', allow_pickle=False) as archive:
                    if (not np.array_equal(archive['lead_hours'], LEADS)
                            or not np.array_equal(archive['xyz'], model.grid.xyz)
                            or not np.array_equal(archive['station_latlon'], joint.surface.latlon)
                            or not np.array_equal(archive['pressure_pa'], np.asarray(PRESSURE_HPA)*100)):
                        raise ValueError('Frozen forecast geometry, levels or horizon differs.')
                    frames = _frames(archive, issue); native_forecast = archive['native'].copy()
                with np.load(folder/'inputs.npz', allow_pickle=False) as archive:
                    raw, mask = archive['raw_surface'], archive['surface_mask']
                    surface_control = persistence_surface(raw, mask)
                inputs = json.loads((folder/'profile_inputs.json').read_text())
                validate_history(raw, mask, inputs, issue)
                profile_control = persistence_profiles(inputs)
                profile_sums, native_sums = empty_accumulator(), empty_accumulator(False)
                counts, rejected, qc_reasons = Counter(), Counter(), Counter()
                cached_when, climate = None, None
                for group in iter_target_groups(joint.profile.root, issue):
                    budget.check(output)
                    counts['profile_before_qc'] += len(group)
                    admitted, qc = screen_humidity(group, evaluation_policy)
                    qc_reasons.update(qc['reasons']); counts['profile_qc_excluded'] += qc['excluded_records']
                    kept = {row['observation_id'] for row in admitted}
                    for row in group:
                        if row['observation_id'] not in kept:
                            record_rejection(row, 'humidity_physical_gross_error_screen', 'profile')
                    for row in admitted:
                        variable = PROFILE_VARIABLES.index(row['variable'])
                        counts['profile_attempted'] += 1
                        result = _prediction(frames, model.grid, row, model.pressure_pa.cpu(), neighbours)
                        if result is None:
                            rejected['profile_outside_norm_or_forecast_support'] += 1
                            record_rejection(row, 'profile_outside_norm_or_forecast_support', 'profile'); continue
                        predicted, observed = float(result[0]), float(row['value'])
                        lead, level = reporting_bins(issue, row)
                        key = (row['variable'], float(row['latitude']), float(row['longitude']), float(row['pressure_pa']))
                        when = utc(row['observed_at'])
                        if when != cached_when: climate = _climate_frame(model, when); cached_when = when
                        baseline = None if climate is None else _prediction([climate], model.grid, row, model.pressure_pa.cpu(), neighbours)
                        persistence = profile_control.get(key)
                        climatology = None if baseline is None else float(baseline[0])
                        accumulate(profile_sums[lead, level, variable], predicted, observed,
                                   persistence=persistence, climatology=climatology)
                        record_pair(row, predicted, observed, persistence, climatology, 'profile')
                        counts['profile_accepted'] += 1
                for group in iter_native_targets(joint.surface.root, issue, station_indices):
                    budget.check(output)
                    for row in group:
                        counts['native_attempted'] += 1
                        checked = _checked(row)
                        if checked is None:
                            rejected['native_provider_qc'] += 1
                            record_rejection(row, 'native_provider_qc', 'station_native'); continue
                        station, _, variable, _, when = checked
                        if station not in station_indices:
                            rejected['native_not_in_frozen_station_inventory'] += 1
                            record_rejection(row, 'native_not_in_frozen_station_inventory', 'station_native'); continue
                        station_index = station_indices[station]
                        if not np.allclose([row['latitude'], row['longitude']], joint.surface.latlon[station_index], rtol=0, atol=1e-5):
                            raise ValueError('Native target station geometry changed.')
                        hours = (when-issue).total_seconds()/3600
                        lead = int(math.ceil(hours/3))
                        predicted = _native_prediction(native_forecast, hours, station_index, variable)
                        if predicted is None:
                            rejected['native_outside_forecast'] += 1
                            record_rejection(row, 'native_outside_forecast', 'station_native'); continue
                        previous = surface_control[station_index, variable]
                        climatology = (float(model.surface_monthly[when.month-1, station_index, variable].cpu())
                                       if native_climate_support[when.month-1, station_index, variable] else None)
                        persistence = float(previous) if math.isfinite(previous) else None
                        accumulate(native_sums[lead, variable], predicted, float(row['value']),
                                   persistence=persistence,
                                   climatology=climatology)
                        record_pair(row, predicted, float(row['value']), persistence, climatology, 'station_native')
                        counts['native_accepted'] += 1
                profile_total += profile_sums; native_total += native_sums
                pairs.close(); rejections.close()
                pairtemp.rename(pairpath); rejecttemp.rename(rejectpath)
                daypath = output/(issue.strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
                with tempfile.TemporaryDirectory(prefix='.monthly-scores-', dir=output) as temporary:
                    staged = Path(temporary)/'scores.jsonl'
                    _write_inventory(staged, profile_sums, native_sums, training_support)
                    staged.rename(daypath)
                summary = {'issue_time': issue.isoformat(), 'inventory': daypath.name,
                           'inventory_sha256': sha256(daypath), 'counts': dict(counts),
                           'pairs': pairpath.name, 'pairs_sha256': sha256(pairpath),
                           'rejections': rejectpath.name, 'rejections_sha256': sha256(rejectpath),
                           'rejected': dict(rejected), 'humidity_qc_reasons': dict(qc_reasons)}
                report['days'].append(summary); save(output/'verification.json', report)
                budget.check(output, size=True)
        _write_inventory(output/'aggregate.jsonl', profile_total, native_total, training_support)
        np.savez_compressed(output/'accumulators.npz', profile=profile_total, native=native_total,
                            channels=np.asarray(CHANNELS), sums=np.asarray(['abs', 'square', 'signed', 'count']))
        report['aggregate_sha256'] = sha256(output/'aggregate.jsonl')
        report['accumulators_sha256'] = sha256(output/'accumulators.npz')
        joint.verify()
        if sha256(forecasts/'month.json') != monthly_hash: raise ValueError('Monthly forecast changed during verification.')
        for row in frozen['products']:
            folder = forecasts/row['directory']
            for name, key in (('verification-state.npz', 'verification_state_sha256'), ('inputs.npz', 'inputs_sha256'), ('profile_inputs.json', 'profile_inputs_sha256')):
                if sha256(folder/name) != row[key]: raise ValueError('Frozen issue archive changed during verification.')
        budget.check(output, size=True)
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'; report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        writers.close()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat(); save(output/'verification.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['forecast', 'verify', 'all'], default='all')
    parser.add_argument('--dataset', required=True); parser.add_argument('--training', required=True)
    parser.add_argument('--static', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--forecasts'); parser.add_argument('--month')
    parser.add_argument('--issue-hour', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'])
    parser.add_argument('--spacing-degrees', type=float)
    parser.add_argument('--source-revision'); parser.add_argument('--neighbours', type=int, default=3)
    args = parser.parse_args(argv)
    config = None
    if args.config:
        config = validate_config(json.loads(args.config.read_text()))
        config = {**config, 'configuration_sha256': sha256(args.config)}
    for key, default in (('month', '2022-09'), ('issue_hour', 6), ('device', 'auto'), ('spacing_degrees', 5.)):
        provided = getattr(args, key)
        configured = config.get(key, default) if config else default
        if config and provided is not None and provided != configured:
            parser.error('CLI override conflicts with pinned config: '+key)
        setattr(args, key, provided if provided is not None else configured)
    if args.neighbours < 1: parser.error('--neighbours must be positive')
    budget = Budget(config['maximum_elapsed_seconds'], config['maximum_new_artifact_bytes']) if config else Budget()
    if args.stage == 'all':
        root = _new_output(args.output); root.mkdir()
        budget.check(root)
        forecasts = root/'forecasts'; verification = root/'verification'
    else:
        forecasts = Path(args.output) if args.stage == 'forecast' else Path(args.forecasts or '')
        verification = Path(args.output)
    if args.stage in ('forecast', 'all'):
        forecast_report = forecast_month(args.dataset, args.training, args.static, forecasts, month=args.month,
                       issue_hour=args.issue_hour, device=args.device,
                       spacing_degrees=args.spacing_degrees, source_revision=args.source_revision,
                       budget=budget, config=config)
        if args.stage == 'forecast':
            print(json.dumps({'schema': forecast_report['schema'], 'status': forecast_report['status'],
                              'month': forecast_report['month'], 'forecast_issues': len(forecast_report['products'])}))
    if args.stage in ('verify', 'all'):
        if args.stage == 'verify' and not args.forecasts: parser.error('--forecasts is required for verification')
        report = verify_month(args.dataset, args.training, args.static, forecasts, verification,
                              month=args.month, issue_hour=args.issue_hour, neighbours=args.neighbours,
                              budget=budget, config=config)
        print(json.dumps({'schema': report['schema'], 'status': report['status'],
                          'month': report['month'], 'verified_issues': len(report['days']),
                          'checkpoint_sha256': report['checkpoint_sha256']}, allow_nan=False))


if __name__ == '__main__':
    main()
