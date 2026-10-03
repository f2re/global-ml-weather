"""Read-only bridges for pinned f2re projects. Metadata compatibility != calibration.

No upstream Python is imported, no network/credentials/settings are read. CBOR
support is deliberately restricted to nlohmann JSON's definite-length subset.
Native rasters stay native: no resampling, spectral normalization or point
conversion occurs here. Reports are not an admission token for model training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import tempfile

ARKTIKA_REV = '4d744570653ac60c8fd059d10e617c3806a6bafc'
SATDUMP_REV = '394431e11d9fffe1a73d3e0670fb023ad7562241'
MAX_META = 64 * 1024 * 1024


def utc(value):
    if not isinstance(value, str):
        raise ValueError('Ожидается время ISO 8601 с часовым поясом.')
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('Часовой пояс нельзя предполагать.')
    return t.astimezone(timezone.utc)


def iso(t):
    return t.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def local_directory(path):
    p = Path(path).expanduser()
    if '..' in p.parts:
        raise ValueError('Переход к родительскому каталогу запрещён.')
    p = Path(os.path.abspath(p))
    if any(x.is_symlink() for x in (p, *p.parents)) or not p.is_dir():
        raise ValueError('Нужен обычный локальный каталог без символьных ссылок.')
    return p


def local_file(path, root=None):
    """Reject traversal, symlinks and nonregular files, including parent symlinks."""
    p = Path(path).expanduser()
    if '..' in p.parts:
        raise ValueError('Переход к родительскому каталогу запрещён.')
    p = Path(os.path.abspath(p))
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('Символьные ссылки входных файлов запрещены.')
    if root is not None and not p.is_relative_to(Path(root).expanduser().resolve()):
        raise ValueError('Файл вне явно разрешённого каталога данных.')
    if not stat.S_ISREG(p.stat().st_mode):
        raise ValueError('Ожидается обычный локальный файл.')
    return p


def fingerprint(path):
    p = local_file(path)
    before = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Файл изменился во время чтения.')
    return h.hexdigest()


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('Повторяющийся ключ метаданных.')
        result[k] = v
    return result


def load_json(path):
    p = local_file(path)
    if p.stat().st_size > MAX_META:
        raise ValueError('Превышен предел метаданных.')
    return json.loads(p.read_text(encoding='utf-8-sig'), object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NaN/Infinity запрещены.')))


def output_path(path):
    p = Path(path).expanduser()
    if '..' in p.parts:
        raise ValueError('Переходы в пути выхода запрещены.')
    p = Path(os.path.abspath(p))
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('Символьная ссылка в пути выхода.')
    return p


def publish_json(path, payload):
    """Atomic create, never replace an existing report or producer data."""
    p = output_path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('Символьная ссылка в пути отчёта.')
    fd, temp = tempfile.mkstemp(prefix='.bridge-', dir=p.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(text); out.flush(); os.fsync(out.fileno())
        os.link(temp, p)
    finally:
        os.unlink(temp)


def decode_cbor(data, *, max_nodes=1_000_000, max_depth=32):
    """Bounded JSON-like CBOR; no tags, indefinite objects, hooks or execution."""
    if not isinstance(data, bytes) or len(data) > MAX_META:
        raise ValueError('Недопустимый размер CBOR.')
    pos = nodes = 0
    def take(n):
        nonlocal pos
        if n < 0 or pos + n > len(data):
            raise ValueError('Оборванный CBOR.')
        value = data[pos:pos+n]; pos += n
        return value
    def parse(depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError('Превышен предел структуры CBOR.')
        first = take(1)[0]; major, minor = first >> 5, first & 31
        if major == 7:
            if minor in (20, 21, 22): return {20: False, 21: True, 22: None}[minor]
            if minor in (25, 26, 27):
                return struct.unpack({25: '>e', 26: '>f', 27: '>d'}[minor], take({25: 2, 26: 4, 27: 8}[minor]))[0]
            raise ValueError('Неподдерживаемый простой тип CBOR.')
        if major == 6 or minor >= 28:
            raise ValueError('Теги/неопределённые длины CBOR не поддерживаются.')
        n = minor if minor < 24 else int.from_bytes(take({24:1,25:2,26:4,27:8}[minor]), 'big')
        if major == 0: return n
        if major == 1: return -1-n
        if major in (2, 3):
            raw = take(n)
            return raw if major == 2 else raw.decode('utf-8')
        if n > max_nodes - nodes: raise ValueError('Слишком много элементов CBOR.')
        if major == 4: return [parse(depth+1) for _ in range(n)]
        if major == 5:
            pairs = []
            for _ in range(n):
                key = parse(depth+1)
                if not isinstance(key, str): raise ValueError('Ключ CBOR должен быть строкой.')
                pairs.append((key, parse(depth+1)))
            return _pairs(pairs)
        raise ValueError('Неизвестный тип CBOR.')
    try:
        result = parse()
    except (UnicodeError, IndexError, KeyError, struct.error) as exc:
        raise ValueError('Некорректный CBOR.') from exc
    if pos != len(data) or not isinstance(result, dict):
        raise ValueError('Ожидается один CBOR-объект без хвостовых данных.')
    return result


def read_product(path):
    p = local_file(path)
    if p.stat().st_size > MAX_META: raise ValueError('Слишком большой product.cbor.')
    return decode_cbor(p.read_bytes())


def _epoch(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Некорректная эпоха времени SatDump.')
    try: return iso(datetime.fromtimestamp(value, timezone.utc))
    except (ValueError, OverflowError, OSError) as exc: raise ValueError('Некорректная эпоха.') from exc


def _times(values):
    if not isinstance(values, list): values = []
    valid = []; rejected = 0
    for v in values:
        try: valid.append(_epoch(v))
        except ValueError: rejected += 1
    return dict(count=len(values), invalid=rejected, first=min(valid) if valid else None,
                last=max(valid) if valid else None)


def scan_satdump(directory):
    root = local_directory(directory)
    dataset = load_json(local_file(root/'dataset.json', root))
    if not isinstance(dataset, dict) or not isinstance(dataset.get('satellite'), str) or not isinstance(dataset.get('products'), list):
        raise ValueError('Не соответствует dataset.json пользовательской ветки SatDump.')
    stamp = _epoch(dataset['timestamp'])
    status_path = root/'decode-status.json'
    statuses = load_json(status_path) if status_path.exists() else {}
    if not isinstance(statuses, dict) or not isinstance(statuses.get('instruments', []), list):
        raise ValueError('Некорректный decode-status.json.')
    records = []
    for name in dataset['products']:
        if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or ':' in name:
            raise ValueError('Недопустимый путь продукта SatDump.')
        entry = dict(product=name, satellite=dataset['satellite'], nominal_time=stamp,
                     available_at=None, training_ready=False, blockers=[], channels=[])
        records.append(entry)
        try:
            path = local_file(root/name/'product.cbor', root)
            product = read_product(path)
            if product.get('type') != 'image':
                entry['blockers'].append('unsupported_product_type'); continue
            inst = product.get('instrument')
            if inst not in ('msu_mr', 'msu_gs', 'mtvza'):
                entry['blockers'].append('unsupported_instrument')
            images = product.get('images')
            if not isinstance(images, list) or not images or len(images) > 128:
                raise ValueError('Пустой/неверный список каналов.')
            matrix = product.get('save_as_matrix', False)
            if not isinstance(matrix, bool): raise ValueError('Неверный флаг матрицы.')
            quality = product.get('decode_quality', {})
            if not isinstance(quality, dict): raise ValueError('Неверный decode_quality.')
            if not quality:
                quality = next((q for q in statuses.get('instruments', []) if isinstance(q, dict) and q.get('instrument') == inst), {})
            entry.update(instrument=inst, metadata_sha256=fingerprint(path),
                         channel_layout=product.get('channel_layout'),
                         decode_status=quality.get('status'),
                         projection_present=isinstance(product.get('projection_cfg'), dict),
                         calibration_present=isinstance(product.get('calibration'), dict),
                         timestamps_type=product.get('timestamps_type'),
                         time_scope='native_line_or_image; nominal_dataset_time_is_not_pixel_time')
            if matrix: entry['blockers'].append('matrix_unpack_required')
            if product.get('needs_correlation'): entry['blockers'].append('channel_correlation_required')
            if quality.get('status') == 'no_data': entry['blockers'].append('no_complete_scans')
            if inst == 'mtvza' and product.get('channel_layout') == 'hrpt30':
                entry['blockers'] += ['verify_hrpt30_physical_mapping', 'microwave_antenna_operator_required']
            seen = set()
            for image in images:
                if not isinstance(image, dict) or not isinstance(image.get('name'), str) or image['name'] in seen:
                    raise ValueError('Неверный или повторяющийся канал.')
                seen.add(image['name'])
                filename = images[0]['file'] if matrix else image['file']
                if not isinstance(filename, str) or Path(filename).is_absolute() or ':' in filename or '\\' in filename:
                    raise ValueError('Недопустимый файл канала.')
                row = dict(native_channel=image['name'], file=str(Path(name)/filename),
                           time=_times(image.get('timestamps', product.get('timestamps', []))))
                try:
                    pixel = local_file(path.parent/filename, root)
                    row.update(size=pixel.stat().st_size, sha256=fingerprint(pixel))
                except (OSError, ValueError):
                    row['missing_or_unsafe'] = True; entry['blockers'].append('missing_or_unsafe_channel_file')
                entry['channels'].append(row)
            entry['blockers'] += ['physical_export_review_required', 'availability_evidence_required']
        except (OSError, ValueError, TypeError, KeyError) as exc:
            entry['blockers'].append('invalid_native_metadata')
            entry['error_type'] = type(exc).__name__
    return dict(schema='global-weather.native-bridge/1', producer='f2re/SatDump',
                audited_revision=SATDUMP_REV, actual_producer_revision=None,
                producer_revision_note='Audited source revision does not prove the version of these files.',
                data_root=str(root), dataset_sha256=fingerprint(root/'dataset.json'), records=records,
                training_ready=False)


def scan_arktika(database, data_root, *, limit=1000):
    if type(limit) is not int or not 1 <= limit <= 10000: raise ValueError('Предел: 1–10000 записей.')
    db = local_file(database); root = local_directory(data_root)
    connection = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'assets','jobs'} <= names: raise ValueError('Это не catalog.sqlite Арктики.')
        connection.execute('BEGIN')
        rows = connection.execute("SELECT a.id,a.data,j.state,j.path,j.sha256 FROM assets a LEFT JOIN jobs j ON j.asset_id=a.id ORDER BY a.id LIMIT ?", (limit+1,)).fetchall()
        truncated = len(rows) > limit
        records = []
        for row in rows[:limit]:
            if len(row['data']) > MAX_META: raise ValueError('Слишком большой JSON записи каталога.')
            a = json.loads(row['data'], object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NaN/Infinity запрещены.')))
            if not isinstance(a, dict) or a.get('id') != row['id']: raise ValueError('ID каталога не совпадает.')
            entry = dict(asset_id=a['id'], platform=a.get('platform'), native_channel=a.get('channel'),
                         level=a.get('level'), category=a.get('category'), observed_at=a.get('time'),
                         raster_bands=a.get('raster_bands', []), epsg=a.get('epsg'),
                         available_at=None, training_ready=False, blockers=[])
            records.append(entry)
            if a.get('platform') not in ('ARCM1','ARCM2'): entry['blockers'].append('unsupported_platform_in_arktika_catalog')
            try: utc(a.get('time'))
            except (ValueError, TypeError): entry['blockers'].append('observation_time_unknown')
            if a.get('time_assumed'): entry['blockers'].append('observation_timezone_assumed')
            if a.get('category') not in ('channel','science'): entry['blockers'].append('not_a_physical_channel_candidate')
            if row['state'] != 'done' or not row['path']:
                entry['blockers'].append('download_not_complete'); continue
            try:
                p = local_file(row['path'] if Path(row['path']).is_absolute() else root/row['path'], root)
                receipt_path = local_file(str(p)+'.download.json', root)
                receipt = load_json(receipt_path)
                sha = fingerprint(p)
                if (receipt.get('asset_id') != a['id'] or receipt.get('size') != p.stat().st_size or
                        receipt.get('sha256') != sha or (row['sha256'] and row['sha256'] != sha)):
                    raise ValueError('Журнал, БД и файл не совпали.')
                ready = utc(receipt['time'])
                if a.get('time') and ready < utc(a['time']): raise ValueError('Нарушен порядок времени.')
                entry.update(file=str(p.relative_to(root)), sha256=sha, size=p.stat().st_size,
                             available_at=iso(ready), availability_basis='download_receipt_not_physical_processing',
                             receipt_sha256=fingerprint(receipt_path))
                entry['blockers'].append('physical_export_review_required')
            except (OSError, ValueError, TypeError, KeyError):
                entry['blockers'].append('missing_unsafe_or_unverified_download')
        logical = hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return dict(schema='global-weather.native-bridge/1', producer='f2re/arktika-worker',
                    audited_revision=ARKTIKA_REV, actual_producer_revision=None, data_root=str(root),
                    catalog_snapshot_sha256=logical, truncated=truncated, records=records, training_ready=False,
                    note='Read-only assets/jobs snapshot; settings and credentials are not read. Electro-L is not enabled in the audited ARCM-only catalog.')
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    a = sub.add_parser('arktika'); a.add_argument('--database', required=True); a.add_argument('--data-root', required=True)
    a.add_argument('--limit', type=int, default=1000)
    s = sub.add_parser('satdump'); s.add_argument('--dataset-dir', required=True)
    for item in (a,s): item.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = scan_arktika(args.database,args.data_root,limit=args.limit) if args.kind=='arktika' else scan_satdump(args.dataset_dir)
    out = output_path(args.output)
    if out.is_relative_to(Path(report['data_root'])):
        parser.error('Отчёт нельзя писать в каталог данных поставщика.')
    publish_json(out,report)
    print(json.dumps(dict(records=len(report['records']), training_ready=False, output=str(out)),ensure_ascii=False))


if __name__ == '__main__': main()
