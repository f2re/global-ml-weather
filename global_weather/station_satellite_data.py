"""Bind prepared infrared sequences to frozen GHCNh forecast issues."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import timedelta
import json
import math
from os import PathLike
from pathlib import Path

import numpy as np

from .multimodal.contracts import fingerprint, hash_value, is_station_ir, utc
from .multimodal.io import load_sensors, load_sequence, read_json, reference, regular, resolve, sha256, write_json
from .multimodal.prepare import _admitted_scan, from_capsules, sequence_from_capsules
from .observation_data import campaign_bounds


def _frame_index(sequence):
    if sequence.scan_start_unix is None:
        raise ValueError('Нет начала съёмки в спутниковой последовательности.')
    return [dict(id=ident, scan_start=start, scan_end=end, available_at=ready)
            for ident, start, end, ready in zip(sequence.frame_ids, sequence.scan_start_unix.tolist(),
                                               sequence.observed_unix.tolist(), sequence.available_unix.tolist())]


def _require_supported_sensor(sensor):
    if sensor.data_kind != 'real' or not is_station_ir(sensor):
        raise ValueError('Станционный опыт принимает реальные ИК-каналы «Арктики-М» и «Электро-Л» в K.')


def prepare_station_sequence(plan_path, output, report_path):
    """Run the existing capsule admission and attach its immutable issue-independent frame index."""
    plan_path = regular(plan_path)
    output, report_path = Path(output).absolute(), Path(report_path).absolute()
    if (report_path.parent != plan_path.parent or report_path == output
            or report_path.exists() or report_path.is_symlink()):
        raise ValueError('Нужен новый паспорт рядом с исходным планом капсул.')
    plan_ref = reference(report_path.parent, plan_path)
    plan = read_json(plan_path)
    sensors = load_sensors(resolve(plan_path.parent, plan['sensor_registry']))
    sensor = next(s for s in sensors if s.id == plan['sensor'])
    _require_supported_sensor(sensor)
    result = from_capsules(plan_path, output)
    sequence = load_sequence(output, sensor)
    if sequence.source_sha256 != result['sha256'] or sha256(plan_path) != plan_ref['sha256']:
        raise ValueError('Результат подготовки или план изменился.')
    passport = dict(result, schema='station-satellite-sequence-1', sensor=sensor.id,
                    plan=plan_ref, frame_index=_frame_index(sequence))
    write_json(report_path, passport)
    return passport


def _station_schedule(root):
    path = Path(root) / 'dataset.json'
    if Path(root).is_file():
        path = Path(root)
        if path.name != 'dataset.json':
            raise ValueError('Нужен станционный dataset.json.')
    station_sha = sha256(regular(path))
    manifest = read_json(path)
    if manifest.get('schema') != 'native-station-dataset-1' or manifest.get('data_kind') != 'real':
        raise ValueError('Нужна фиксированная выборка реальных станционных наблюдений.')
    start, train_end, val_end, end, _ = campaign_bounds(manifest.get('campaign', 'legacy_2021_2022'))
    boundaries = {'train': (start, train_end), 'validation': (train_end, val_end), 'test': (val_end, end)}
    schedule = {}
    for index, row in enumerate(manifest['samples']):
        issue, split = utc(row['issue']), row['split']
        if split not in boundaries or issue in schedule:
            raise ValueError('Неверный или повторный срок станционной выборки.')
        lo, hi = boundaries[split]
        if not lo <= issue - timedelta(hours=12) or not issue + timedelta(hours=72) < hi:
            raise ValueError('Станционный срок пересекает границы частей выборки.')
        schedule[issue] = (index, split)
    if sha256(path) != station_sha:
        raise ValueError('Станционная выборка изменилась при чтении.')
    return station_sha, schedule, manifest['samples']


def _check_replayed_sequence(sequence, replayed):
    for field in fields(sequence):
        if field.name == 'source_sha256':
            continue
        actual, expected = getattr(sequence, field.name), getattr(replayed, field.name)
        if hasattr(expected, 'numpy'):
            same = (hasattr(actual, 'numpy') and actual.dtype == expected.dtype
                    and np.array_equal(actual.numpy(), expected.numpy(), equal_nan=True))
        else:
            same = not hasattr(actual, 'numpy') and actual == expected
        if not same:
            raise ValueError(f'Поле спутниковой последовательности {field.name} не совпадает с исходными капсулами.')


def _capsule_provenance(report_path, report, sensor, *, verify_raw, sequence=None):
    plan_path = resolve(report_path.parent, report['plan'])
    root = plan_path.parent
    plan = read_json(plan_path)
    declarations = plan.get('declarations', {})
    if (plan.get('schema') != 'multimodal-capsule-plan-1' or plan.get('sensor') != sensor.id
            or plan.get('data_kind') != 'real' or 'real_admission' not in declarations):
        raise ValueError('Нужен план физического допуска исходных капсул.')
    admission_path = resolve(root, declarations['real_admission'])
    admission = read_json(admission_path)
    if (admission.get('schema') != 'multimodal-ir-admission-1' or not admission.get('capsules')
            or not isinstance(plan.get('frames'), list) or len(plan['frames']) != report.get('frames')):
        raise ValueError('План капсул с подтверждённым допуском не совпадает с паспортом подготовки.')
    registry = resolve(root, plan['sensor_registry'])
    if sensor.measurement_signature not in {s.measurement_signature for s in load_sensors(registry)}:
        raise ValueError('Реестр подготовки и калибровка станционного входа различаются.')
    used, evidence, recovered = set(), {}, []
    for frame in plan['frames']:
        channels = frame.get('channels', {})
        if set(channels) != {ch.id for ch in sensor.channels}:
            raise ValueError('Каналы паспорта капсул и реестра различаются.')
        scans, identities = [], set()
        for ch in sensor.channels:
            entry = channels[ch.id]
            if entry is None:
                continue
            if not isinstance(entry, dict) or set(entry) != {'manifest', 'pixels'}:
                raise ValueError('Нужны проверенные файлы капсулы.')
            manifest_path = resolve(root, entry['manifest'])
            digest = entry['manifest']['sha256']
            metadata = read_json(manifest_path)
            if (digest in used or metadata.get('source') != sensor.source or metadata.get('platform') != sensor.platform
                    or str(metadata.get('channel_id')) != ch.id or metadata.get('calibration_reference') != ch.calibration_id
                    or metadata.get('arrays_sha256') != entry['pixels']['sha256']
                    or not metadata.get('geometry_reference')):
                raise ValueError('Повторная капсула или несовместимый физический канал.')
            scans.append(_admitted_scan(manifest_path, digest, admission['capsules'].get(digest), metadata, root,
                                        evidence))
            identities.add(metadata['native_scene_id'])
            evidence[manifest_path] = digest
            used.add(digest)
        if len(identities) != 1:
            raise ValueError('Пустой спутниковый кадр или разные нативные сцены каналов.')
        starts, ends, readiness = zip(*scans)
        recovered.append(dict(id=fingerprint({'source': sensor.source, 'platform': sensor.platform,
                                               'native_scene_id': identities.pop()}),
                              scan_start=min(starts), scan_end=max(ends), available_at=max(readiness)))
    if set(admission['capsules']) != used or recovered != report.get('frame_index'):
        raise ValueError('Индекс кадра не соответствует подтверждённой нативной сцене.')
    if any(sha256(path) != digest for path, digest in evidence.items()):
        raise ValueError('Подтверждение происхождения изменилось.')
    if sha256(plan_path) != report['plan']['sha256'] or sha256(admission_path) != declarations['real_admission']['sha256']:
        raise ValueError('План подготовки изменился.')
    if verify_raw:
        replayed = sequence_from_capsules(plan_path)
        if replayed.source_sha256 != report['plan']['sha256']:
            raise ValueError('План подготовки изменился при повторной проверке.')
        if sequence is not None:
            _check_replayed_sequence(sequence, replayed)


def _preparation(root, row, issue, sensor, *, verify_raw=False, sequence=None):
    report_path = resolve(root, row['preparation'])
    report = read_json(report_path)
    if (report.get('schema') != 'station-satellite-sequence-1' or report.get('status') != 'sequence_prepared'
            or report.get('sensor') != sensor.id or report.get('data_kind') != 'real'
            or report.get('meteorologically_validated') is not False
            or report.get('sha256') != row['file']['sha256']):
        raise ValueError('Паспорт подготовки и спутниковый файл не совпадают.')
    _capsule_provenance(report_path, report, sensor, verify_raw=verify_raw, sequence=sequence)
    index = report.get('frame_index')
    shape = report.get('shape')
    if (not isinstance(index, list) or not 1 <= len(index) <= 48 or report.get('frames') != len(index)
            or not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [len(index), len(sensor.channels)]
            or any(type(n) is not int or n <= 0 for n in shape) or min(shape[2:]) < 8 or shape[2] * shape[3] > 262_144):
        raise ValueError('Неверные размеры спутникового паспорта.')
    issued = issue.timestamp()
    previous_end = -math.inf
    for frame in index:
        if not isinstance(frame, dict) or set(frame) != {'id', 'scan_start', 'scan_end', 'available_at'}:
            raise ValueError('Неверный индекс спутникового кадра.')
        hash_value(frame['id'])
        start, end, ready = (frame[key] for key in ('scan_start', 'scan_end', 'available_at'))
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end, ready))
                or not issued - 43_200 < start <= end <= ready <= issued or end <= previous_end):
            raise ValueError('Спутниковый кадр нарушает причинность 12-часового окна.')
        previous_end = end
    if sha256(report_path) != row['preparation']['sha256']:
        raise ValueError('Паспорт подготовки изменился.')
    return tuple(shape), index


class StationSatelliteData:
    """Validate hashes, the issue index and frozen norms before lazy access to pixel arrays.

    A verified preparation passport binds the capsule route to the sequence. Its
    integrity cannot establish the truth of the upstream physical review.
    """

    def __init__(self, path, dataset, *, _fit=False):
        self.path = regular(path)
        self.root = self.path.parent
        self.signature = sha256(self.path)
        spec = read_json(self.path)
        if (not isinstance(spec, dict) or set(spec) != {'schema', 'station_dataset_sha256', 'sensors', 'scenes'}
                or spec['schema'] != 'station-satellite-input-1'):
            raise ValueError('Неизвестный манифест спутниковых входов станции.')
        dataset_root = dataset if isinstance(dataset, (str, PathLike)) else dataset.root
        station_sha, schedule, samples = _station_schedule(dataset_root)
        if station_sha != spec['station_dataset_sha256'] or (hasattr(dataset, 'samples') and samples != dataset.samples):
            raise ValueError('Спутниковый манифест связан с другой станционной выборкой.')
        self.spec = spec
        self.station_sha = station_sha
        self.registry = resolve(self.root, spec['sensors'])
        self.sensors = load_sensors(self.registry)
        self._sensors = {s.id: s for s in self.sensors}
        for sensor in self.sensors:
            _require_supported_sensor(sensor)
            if _fit and sensor.fit_end is not None:
                raise ValueError('Для расчёта норм нужен новый физический реестр без статистики.')
            if not _fit:
                sensor.require_normalization()
        self._issues = {}
        self._indices = {split: [] for split in ('train', 'validation', 'test')}
        scenes = spec['scenes']
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError('Спутниковый манифест пуст.')
        seen = set()
        for label, rows in scenes.items():
            issue = utc(label)
            if issue not in schedule:
                raise ValueError('Спутниковый срок отсутствует в станционной выборке.')
            sample_id, split = schedule[issue]
            if sample_id in self._issues or not isinstance(rows, list) or not 1 <= len(rows) <= 32:
                raise ValueError('Дублированный срок или неверное число последовательностей.')
            verified = []
            pixels = 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {'sensor', 'file', 'preparation'} or row['sensor'] not in self._sensors:
                    raise ValueError('Неизвестная ссылка на спутниковую последовательность.')
                file = row['file']
                if (not isinstance(file, dict) or set(file) != {'path', 'sha256'}
                        or Path(file['path']).is_absolute() or '..' in Path(file['path']).parts):
                    raise ValueError('Спутниковый файл должен находиться под каталогом манифеста.')
                hash_value(file['sha256'])
                regular(self.root / file['path'])
                shape, index = _preparation(self.root, row, issue, self._sensors[row['sensor']])
                ids = {f['id'] for f in index}
                if len(ids) != len(index) or seen & ids:
                    raise ValueError('Повторное усвоение спутникового кадра.')
                seen.update(ids)
                pixels += math.prod(shape)
                if pixels > 8_000_000:
                    raise ValueError('Превышен предел декодированных спутниковых значений на срок.')
                verified.append((row, shape, index))
            self._issues[sample_id] = (issue, tuple(verified))
            self._indices[split].append(sample_id)
        for indices in self._indices.values():
            indices.sort()
        self.norm_sources = self._normalization_sources()
        if not _fit:
            self._check_normalization()
        if sha256(self.path) != self.signature or sha256(self.registry) != spec['sensors']['sha256']:
            raise ValueError('Спутниковый манифест изменился при проверке.')

    def _normalization_sources(self):
        sources = {sensor.id: {} for sensor in self.sensors}
        for sample_id in self._indices['train']:
            issue, rows = self._issues[sample_id]
            for row, _, _ in rows:
                files = sources[row['sensor']].setdefault(issue, {})
                files[row['file']['path']] = row['file']['sha256']
        if any(not issues for issues in sources.values()):
            raise ValueError('Для каждого спутникового прибора нужны кадры train.')
        return sources

    def _check_normalization(self):
        refs = read_json(self.registry)['statistics']
        for sensor in self.sensors:
            path = resolve(self.registry.parent, refs[sensor.id])
            norms = read_json(path)
            if any(norms.get(key) != value for key, value in self._norm_provenance(sensor).items()):
                raise ValueError('Нормы не подтверждены точным списком обучающих спутниковых кадров.')
            holdout = [self._issues[n][0] for split in ('validation', 'test') for n in self._indices[split]]
            if holdout and utc(sensor.fit_end) >= min(holdout) - timedelta(hours=12):
                raise ValueError('Спутниковые нормы пересекают проверочное окно.')
            if sha256(path) != sensor.normalization_sha256:
                raise ValueError('Файл спутниковых норм изменился.')

    def _norm_provenance(self, sensor):
        issues = self.norm_sources[sensor.id]
        return {'schema': 'multimodal-statistics-1', 'sensor': sensor.id,
                'fit_end': max(issues).isoformat(), 'sample_ids': [ts.isoformat() for ts in sorted(issues)],
                'source_sha256': {p: digest for files in issues.values() for p, digest in files.items()},
                'data_kind': sensor.data_kind, 'weighting': 'pixel_area',
                'measurement_signature': sensor.measurement_signature, 'source_dataset_sha256': self.station_sha}

    def indices(self, split):
        return list(self._indices[split])

    def covered(self, split, indices=None):
        available = self._indices[split]
        return len(available) if indices is None else len(set(available).intersection(indices))

    def sequences(self, index):
        if index not in self._issues:
            return ()
        issue, rows = self._issues[index]
        sequences = []
        for row, shape, frame_index in rows:
            sensor = self._sensors[row['sensor']]
            sequence = load_sequence(resolve(self.root, row['file']), sensor)
            _preparation(self.root, row, issue, sensor, verify_raw=True, sequence=sequence)
            if (tuple(sequence.values.shape) != shape or _frame_index(sequence) != frame_index
                    or sequence.source_sha256 != row['file']['sha256']):
                raise ValueError('Спутниковые массивы не совпадают с паспортом подготовки.')
            admitted = sequence.causal(issue, sensor)
            if (admitted is None or admitted.frame_ids != sequence.frame_ids
                    or not sequence.valid.flatten(1).any(dim=1).all()):
                raise ValueError('Недоступный или пустой спутниковый кадр.')
            sequences.append(sequence)
        return tuple(sequences)


def fit_station_sensors(dataset_path, path, output):
    """Write existing sensor-normalization format from unique train pixels only."""
    data = StationSatelliteData(path, dataset_path, _fit=True)
    output = Path(output).absolute()
    if output.parent != data.root:
        raise ValueError('Новый манифест должен находиться рядом с исходным.')
    registry = output.with_name(output.stem + '.sensors.json')
    norm_paths = {sensor.id: output.with_name(f'{output.stem}.{sensor.id}.norms.json') for sensor in data.sensors}
    if any(p.exists() or p.is_symlink() for p in (output, registry, *norm_paths.values())):
        raise FileExistsError('Нельзя перезаписать спутниковую нормировку.')

    states = {s.id: np.zeros((3, len(s.channels)), dtype=np.float64) for s in data.sensors}
    for sample_id in data.indices('train'):
        for sequence in data.sequences(sample_id):
            count, mean, m2 = states[sequence.sensor_id]
            x = sequence.values.numpy().astype(np.float64)
            mask = sequence.valid.numpy()
            weights = np.where(mask, sequence.area_m2.numpy(), 0.0).astype(np.float64)
            values = np.where(mask, x, 0.0)
            axes = (0, 2, 3)
            amount = weights.sum(axes)
            batch_mean = (weights * values).sum(axes) / np.maximum(amount, 1e-300)
            batch_m2 = (weights * (values - batch_mean[None, :, None, None]) ** 2).sum(axes)
            total = count + amount
            delta = batch_mean - mean
            denominator = np.maximum(total, 1e-300)
            mean += delta * amount / denominator
            m2 += batch_m2 + delta ** 2 * count * amount / denominator
            count[:] = total

    payloads = {}
    for sensor in data.sensors:
        count, mean, m2 = states[sensor.id]
        std = np.sqrt(m2 / np.maximum(count, 1e-300))
        if (count <= 0).any() or (std <= 0).any() or not np.isfinite([mean, std]).all():
            raise ValueError('Недостаточно обучающих измерений или нулевая дисперсия спутникового канала.')
        payloads[sensor.id] = dict(data._norm_provenance(sensor), mean=mean.tolist(), std=std.tolist())

    completed, refs = [], {}
    for sensor in data.sensors:
        norm = payloads[sensor.id]
        norm_path = norm_paths[sensor.id]
        write_json(norm_path, norm)
        refs[sensor.id] = reference(data.root, norm_path)
        channels = tuple(replace(ch, mean=mean, std=std) for ch, mean, std in zip(sensor.channels, norm['mean'], norm['std']))
        completed.append(replace(sensor, channels=channels, fit_end=norm['fit_end'],
                                 normalization_sha256=refs[sensor.id]['sha256']))
    write_json(registry, {'schema': 'multimodal-sensors-1', 'sensors': [asdict(s) for s in completed], 'statistics': refs})
    write_json(output, dict(data.spec, sensors=reference(data.root, registry)))
    return {'status': 'station_satellite_norms_frozen', 'manifest': str(output), 'sha256': sha256(output),
            'train_issues': data.covered('train'), 'heldout_pixels_read': False, 'meteorologically_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--plan', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--report', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--dataset', required=True)
    fit.add_argument('--manifest', required=True)
    fit.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare_station_sequence(args.plan, args.output, args.report)
    else:
        result = fit_station_sensors(args.dataset, args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
