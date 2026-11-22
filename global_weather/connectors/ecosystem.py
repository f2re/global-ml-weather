"""Read-only bridges to pinned arktika-worker and SatDump native artifacts.

A compatible transport is not a calibrated observation. No acquisition or
calibration is invented here. Native files are never modified or executed.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit
from .native_cbor import loads as cbor_loads

UPSTREAM = {
    'arktika-worker': dict(repository='f2re/arktika-worker', branch='main', commit='4d744570653ac60c8fd059d10e617c3806a6bafc'),
    'satdump': dict(repository='f2re/SatDump', branch='release/1.2.2', commit='394431e11d9fffe1a73d3e0670fb023ad7562241'),
}
SOURCE_BLOBS = {
    'arktika-worker': {
        'arktika/model.py': '74e938eaee0e7d9b60a3c80b43945496d5370094',
        'arktika/processing.py': '8f9525acae706d2e4dd1343ca1f5094bf5fb2a89',
        'arktika/download.py': 'ead50f5a0d38494862eeb059d342226de229d0b5',
    },
    'satdump': {
        'src-core/products/dataset.cpp': '9b5b8b05566b307db68ef188a9c4a3d564e9301b',
        'src-core/products/image_products.cpp': '365e00067e8879fb6f6f9d79843fa2d878ef48f9',
        'docs/ru/station/MTVZA_PROCESSING.md': 'd77817b9013bad46b69f595fb110404d8b8806a9',
    },
}
SATELLITES = {'arktika_m', 'electro_l', 'meteor_msu_mr', 'meteor_mtvza'}
MAX_METADATA = 8*1024*1024


def utc(value):
    if not isinstance(value, str): raise ValueError('Explicit ISO timestamp required.')
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None or t.utcoffset() is None: raise ValueError('Timezone required.')
    return t.astimezone(timezone.utc)


def iso(value):
    return utc(value).isoformat().replace('+00:00', 'Z')


def regular(path, *, limit=None):
    p = Path(path).absolute()
    if any(q.is_symlink() for q in (p, *p.parents)) or not p.is_file():
        raise ValueError('Expected a regular file without symlink components.')
    if limit is not None and p.stat().st_size > limit: raise ValueError('File exceeds size limit.')
    return p


def child(root, name):
    """Only local relative paths beneath an explicit root; never GDAL URLs/VSI."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Invalid local relative path.')
    part = Path(name)
    if part.is_absolute() or '..' in part.parts: raise ValueError('Path escapes input root.')
    return regular(Path(root)/part)


def sha256(path):
    p = regular(path); before = p.stat(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Input changed while hashing.')
    return h.hexdigest()


def read_json(path):
    p = regular(path, limit=MAX_METADATA)
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out: raise ValueError('Duplicate JSON key.')
            out[k] = v
        return out
    def bad(_): raise ValueError('Nonfinite JSON value.')
    value = json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=bad)
    if not isinstance(value, dict): raise ValueError('Expected JSON object.')
    return value


def metadata_kind(value):
    if not isinstance(value, dict): return None
    if value.get('type') == 'image' and isinstance(value.get('images'), list): return 'satdump_image_product'
    if isinstance(value.get('products'), list) and 'satellite' in value: return 'satdump_dataset'
    if 'scene_id' in value and 'calibration_status' in value and 'legend' in value: return 'arktika_worker_product'
    if all(k in value for k in ('item_id', 'filename', 'raster_bands', 'platform')): return 'arktika_worker_asset'
    if value.get('type') == 'Feature' and isinstance(value.get('assets'), dict): return 'gptl_stac_item'
    return None


def inspect_satdump(path, *, required_instruments=()):
    """Parse actual dataset.json/product.cbor, referenced channels and status reports.

    Complete rasters may still be DN. TLE/proj/calibration presence is inventoried,
    never treated as a calibrated/geolocated field. Unknown timestamp=-1 stays unknown.
    """
    entry = regular(path, limit=MAX_METADATA); refs = {entry.name: sha256(entry)}
    dataset = None
    if entry.name == 'dataset.json':
        dataset = read_json(entry)
        if metadata_kind(dataset) != 'satdump_dataset': raise ValueError('Not a SatDump dataset.')
        if len(dataset['products']) > 128: raise ValueError('Too many products.')
        paths = []
        failures = []
        for name in dataset['products']:
            try: paths.append(child(entry.parent, str(name)+'/product.cbor'))
            except ValueError: failures.append(dict(product=str(name), reasons=['missing_or_unsafe_product']))
    elif entry.suffix.lower() == '.cbor': paths, failures = [entry], []
    else: raise ValueError('Supply native dataset.json or product.cbor, not a presentation.')
    reports = []
    for p in paths:
        try:
            raw = cbor_loads(regular(p, limit=MAX_METADATA).read_bytes())
            if metadata_kind(raw) != 'satdump_image_product': raise ValueError('Not an image product.')
            instrument = raw.get('instrument')
            if not isinstance(instrument, str) or not instrument: raise ValueError('Missing instrument identity.')
            images = raw['images']
            if len(images) > 512: raise ValueError('Too many channels.')
            reasons, channels = [], []
            if not images: reasons.append('no_data')
            if raw.get('save_as_matrix'): reasons.append('matrix_unpack_required')
            if any(not isinstance(im,dict) for im in images): raise ValueError('Invalid channel metadata.')
            if len({str(im.get('name','')) for im in images}) != len(images): reasons.append('duplicate_transport_ids')
            for im in images:
                channel = dict(transport_id=str(im.get('name', '')), present=False)
                try:
                    f = child(p.parent, im['file'])
                    if not f.stat().st_size: raise ValueError('Empty channel file.')
                    channel.update(present=True, file=im['file'], bytes=f.stat().st_size, sha256=sha256(f))
                except (KeyError, TypeError, ValueError): reasons.append('missing_or_unsafe_channel')
                channels.append(channel)
            processing = p.parent/'processing-status.json'
            status = read_json(processing) if processing.exists() else None
            if status is not None and status.get('instrument') != instrument: reasons.append('instrument_status_mismatch')
            if status is not None and status.get('status') in ('no_products', 'failed', 'error', 'no_data'):
                reasons.append('upstream_processing_failed')
            if not status: reasons.append('processing_status_unavailable')
            layout = raw.get('channel_layout')
            expected = {'hrpt30': 30, 'dump46': 46}.get(layout) if instrument == 'mtvza' else None
            if instrument == 'mtvza':
                if expected is None: reasons.append('unknown_mtvza_layout')
                elif len(images) != expected: reasons.append('layout_count_mismatch')
                reasons += ['physical_channel_mapping_required', 'microwave_calibration_required', 'antenna_operator_required']
            # Calibrator parameters may be present in CBOR, but saved pixel values are not automatically physical.
            reasons.append('native_calibration_and_geolocation_execution_required')
            reports.append(dict(instrument=instrument, channel_layout=layout, channel_count=len(images),
                                channels=channels, timestamps_present=raw.get('has_timestamps') is True,
                                timestamps_type=raw.get('timestamps_type'),
                                calibration_metadata_present='calibration' in raw,
                                projection_metadata_present='projection_cfg' in raw,
                                transport_complete=bool(images) and all(x['present'] for x in channels) and not set(reasons).intersection(
                                    {'duplicate_transport_ids','instrument_status_mismatch','upstream_processing_failed','layout_count_mismatch','unknown_mtvza_layout'}),
                                model_ready=False, status='native_transport_only', reasons=sorted(set(reasons))))
            refs[str(p.relative_to(entry.parent))] = sha256(p)
            if processing.exists(): refs[str(processing.relative_to(entry.parent))] = sha256(processing)
        except (ValueError, TypeError, KeyError, UnicodeError):
            failures.append(dict(product=str(p.name), reasons=['invalid_native_metadata']))
    seen = {r['instrument'] for r in reports}
    missing = sorted(set(required_instruments)-seen)
    stamp = dataset.get('timestamp') if dataset else None
    observed = None
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp) and stamp > 0:
        try: observed = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError): pass
    decode_path = entry.parent/'decode-status.json'
    decoded = read_json(decode_path) if decode_path.exists() else None
    if decode_path.exists(): refs[decode_path.name] = sha256(decode_path)
    return dict(schema='ecosystem-inspection-v1', adapter='satdump', upstream=UPSTREAM['satdump'],
                revision_scope='audited_source_not_verified_installed_binary',
                source_files=refs, satellite=dataset.get('satellite') if dataset else None,
                observed_at=observed, available_at=None, products=reports, failures=failures,
                decode_status=decoded, missing_required_instruments=missing,
                model_ready=False, physics_verified=False,
                status='blocked' if not reports or failures or missing or any(not r['transport_complete'] for r in reports) else 'native_transport_only')


def arktika_product(path):
    p = regular(path, limit=MAX_METADATA); meta = read_json(p)
    if metadata_kind(meta) != 'arktika_worker_product': raise ValueError('Not an arktika-worker product.')
    if meta.get('product') != 'channel': raise ValueError('RGB/differences/classes are not source channel measurements.')
    if meta.get('time_assumed', True) is not False: raise ValueError('Assumed observation time is not admitted.')
    if meta.get('calibration_status') not in ('metadata', 'declared'):
        raise ValueError('Unknown/assumed calibration is not admitted.')
    channel = meta['request'].get('channel', 9)
    if type(channel) is not int or not 4 <= channel <= 10: raise ValueError('Expected a calibrated infrared channel 4..10.')
    cal = meta['legend'].get('calibration', [])
    if len(cal) != 1 or cal[0].get('channel') != channel or cal[0].get('units') != 'K':
        raise ValueError('Channel/calibration disagreement.')
    if cal[0].get('status') not in ('metadata', 'declared') or not cal[0].get('reference'):
        raise ValueError('Calibration provenance required.')
    if meta['legend'].get('units') != 'K': raise ValueError('Brightness-temperature units required.')
    platform = meta.get('platform')
    if platform not in ('ARCM1', 'ARCM2'): raise ValueError('This audited worker produces ARCM1/ARCM2 only; use the asset adapter for Electro-L.')
    return dict(source='arktika_m', platform=platform, instrument='MSU-GS/A', channel_id=str(channel),
                native_scene_id=meta['scene_id'],
                observed_at=iso(meta['time']), quantity='brightness_temperature', units='K',
                calibration_reference=cal[0]['reference'], calibration_status=cal[0]['status'],
                scale=1., offset=0., values_already_physical=True,
                raster=child(p.parent, 'values.tif'), quality=child(p.parent, 'quality.tif'),
                inputs=[p], upstream=UPSTREAM['arktika-worker'])


def gptl_asset(path, raster_path, *, source, asset_key=None, channel=None):
    """Read normalize_asset output from arktika-worker, including Electro-L.

    Acquisition remains with the user's existing GPTL client. Its .download.json
    provides checksum and download completion, NOT historical ready time.
    """
    return _gptl_asset(path, raster_path, source=source, asset_key=asset_key, channel=channel)


def _gptl_asset(path, raster_path, *, source, asset_key=None, channel=None, expected_raster_hash=None):
    p = regular(path, limit=MAX_METADATA); a = read_json(p); raster = regular(raster_path)
    if metadata_kind(a) == 'gptl_stac_item':
        if asset_key not in a['assets'] or type(channel) is not int:
            raise ValueError('A STAC item needs an explicit asset key and reviewed physical channel.')
        raw, props, item_id = a['assets'][asset_key], a.get('properties',{}), a.get('id','')
        if not isinstance(raw,dict) or not item_id: raise ValueError('Invalid STAC asset.')
        uri = urlsplit(raw.get('href',''))
        if uri.scheme not in ('https','s3') or not uri.netloc: raise ValueError('Absolute provider asset URI required.')
        canonical = urlunsplit((uri.scheme,uri.netloc,uri.path,'',''))
        level = raw.get('processing:level') or props.get('processing:level') or props.get('processing_level_code')
        stamp = props.get('datetime')
        a = dict(id=hashlib.sha256((item_id+'\n'+canonical).encode()).hexdigest()[:32], item_id=item_id,
                 filename=Path(uri.path).name, platform=props.get('platform') or props.get('platform_identifier'),
                 time=iso(stamp),time_assumed=False,uri=raw.get('href'),level=level,category='channel',channel=channel,
                 raster_bands=raw.get('raster:bands',[]))
    if metadata_kind(a) != 'arktika_worker_asset': raise ValueError('Expected normalized worker asset or explicit STAC item.')
    if source not in ('arktika_m', 'electro_l'): raise ValueError('Choose Arktika or Electro explicitly.')
    platform = a.get('platform')
    if not isinstance(platform, str) or not platform.strip(): raise ValueError('Platform identity required.')
    if source == 'arktika_m' and platform not in ('ARCM1','ARCM2'): raise ValueError('Arktika platform mismatch.')
    if source == 'electro_l' and platform in ('ARCM1','ARCM2'): raise ValueError('Arktika data cannot be labelled Electro-L.')
    if a.get('time_assumed', True) is not False: raise ValueError('Assumed time is not admitted.')
    if a.get('level') != 'L2IR' or a.get('category') != 'channel':
        raise ValueError('Only the audited L2IR numerical channel contract is enabled; other levels require a separate adapter.')
    channel = a.get('channel')
    if type(channel) is not int or not 4 <= channel <= 10: raise ValueError('Physical infrared channel 4..10 required.')
    bands = a['raster_bands']
    if len(bands) != 1 or bands[0].get('unit') != 'K': raise ValueError('Explicit single-band Kelvin metadata required.')
    b = bands[0]
    if 'scale' not in b or 'offset' not in b: raise ValueError('Explicit scale/offset required, even for identity conversion.')
    if isinstance(b['scale'],bool) or isinstance(b['offset'],bool): raise ValueError('Boolean calibration is invalid.')
    scale, offset = float(b['scale']), float(b['offset'])
    if not math.isfinite(scale+offset) or scale <= 0: raise ValueError('Invalid calibration coefficients.')
    nodata = b.get('nodata')
    if nodata is not None and (isinstance(nodata,bool) or not isinstance(nodata,(int,float)) or not math.isfinite(nodata)):
        raise ValueError('Explicit nodata must be a finite number or null.')
    journal = regular(str(raster)+'.download.json', limit=MAX_METADATA); j = read_json(journal)
    raster_hash = sha256(raster) if expected_raster_hash is None else expected_raster_hash
    if j.get('asset_id') != a.get('id') or j.get('size') != raster.stat().st_size or j.get('sha256') != raster_hash:
        raise ValueError('Download journal does not match the physical source file.')
    uri = urlsplit(a.get('uri', ''))
    if uri.username or uri.password: raise ValueError('Credential-bearing URI is not admitted.')
    redacted_uri = urlunsplit((uri.scheme, uri.netloc, uri.path, '', ''))
    return dict(source=source, platform=platform, instrument='MSU-GS/A' if source=='arktika_m' else 'MSU-GS',
                native_scene_id=a['item_id'],
                channel_id=str(channel), observed_at=iso(a['time']), quantity='brightness_temperature', units='K',
                calibration_reference='raster_bands:'+sha256(p), calibration_status='metadata',
                scale=scale, offset=offset, declared_nodata=nodata, values_already_physical=False,
                raster=raster, quality=None, inputs=[p,journal], earliest_available_at=iso(j['time']),
                source_uri=redacted_uri, upstream=UPSTREAM['arktika-worker'])


ELECTRO_L_PLATFORMS = ('EL2', 'EL3', 'EL4')


def native_window_parameters(window, max_pixels):
    fields = ('row_offset', 'column_offset', 'height', 'width')
    if (not isinstance(window, dict) or set(window) != set(fields)
            or any(type(window[field]) is not int for field in fields)):
        raise ValueError('Для нативного окна нужны целочисленные смещения и размеры.')
    row, col, height, width = (window[field] for field in fields)
    if row < 0 or col < 0 or height < 1 or width < 1 or height * width > max_pixels:
        raise ValueError('Размер или смещение нативного окна выходит за пределы.')
    return row, col, height, width


def _electro_kelvin_band(bands):
    if (not isinstance(bands, list) or len(bands) != 1 or not isinstance(bands[0], dict)
            or bands[0].get('unit') != 'K'
            or any(type(bands[0].get(key)) not in (int, float) for key in ('scale', 'offset'))):
        raise ValueError('Нужны явные числовые коэффициенты единственного нативного канала в K.')


def _check_stac_file(asset, ref, path):
    size = asset.get('file:size')
    if size is not None and (type(size) is not int or size != path.stat().st_size):
        raise ValueError('Размер файла отличается от исходных метаданных поставщика.')
    checksum = asset.get('file:checksum')
    if checksum is not None and checksum not in (ref['sha256'], '1220' + ref['sha256'], 'sha256:' + ref['sha256']):
        raise ValueError('Контрольная сумма файла не подтверждена исходными метаданными поставщика.')


def electro_l_native(path, *, verify_raw=True):
    """Read reviewed per-scene evidence for a bounded native Elektro-L L2IR raster."""
    manifest = regular(path, limit=MAX_METADATA)
    digest = sha256(manifest)
    review = read_json(manifest)
    required = {'schema', 'files', 'calibration', 'quality', 'timing'}
    if (set(review) - {'window'} != required
            or review['schema'] != 'electro-l-native-l2ir-1'):
        raise ValueError('Нужен паспорт нативного L2IR «Электро-Л».')
    raster_limit = 32 * 1024**2
    if 'window' in review:
        native_window_parameters(review['window'], 262144)
        raster_limit = 512 * 1024**2
    refs = review['files']
    roles = ('producer_item', 'asset', 'raster', 'download', 'quality', 'geometry')
    if not isinstance(refs, dict) or set(refs) != set(roles):
        raise ValueError('Нужны метаданные поставщика и загрузчика, растр, квитанция, маска качества и геометрия.')
    files = {}
    for role in roles:
        ref = refs[role]
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('Нужны путь и SHA256 нативного файла.')
        expected = ref['sha256']
        if not isinstance(expected, str) or len(expected) != 64 or set(expected)-set('0123456789abcdef'):
            raise ValueError('Неверный SHA256 нативного файла.')
        file = child(manifest.parent, ref['path'])
        limit = raster_limit if role in ('raster', 'quality') else MAX_METADATA
        if not 0 < file.stat().st_size <= limit:
            raise ValueError('Нативный файл пуст или превышает предел размера.')
        if (verify_raw or role not in ('raster', 'quality')) and sha256(file) != expected:
            raise ValueError('SHA256 нативного файла не совпадает с паспортом.')
        files[role] = file
    if len(set(files.values())) != len(roles) or manifest in files.values():
        raise ValueError('Паспорт должен связывать отдельные нативные файлы.')
    if files['download'] != Path(str(files['raster']) + '.download.json'):
        raise ValueError('Квитанция не относится к нативному растру.')
    asset = read_json(files['asset'])
    if (metadata_kind(asset) != 'arktika_worker_asset'
            or asset.get('platform') not in ELECTRO_L_PLATFORMS
            or not all(isinstance(asset.get(key), str) and asset[key].strip() for key in ('id', 'item_id', 'uri'))):
        raise ValueError('Требуется явно идентифицированный актив «Электро-Л» от arktika-worker.')
    uri = urlsplit(asset['uri'])
    if uri.scheme not in ('https', 's3') or not uri.hostname:
        raise ValueError('Требуется абсолютный URI исходного актива поставщика.')
    _electro_kelvin_band(asset['raster_bands'])
    spec = _gptl_asset(files['asset'], files['raster'], source='electro_l',
                       expected_raster_hash=refs['raster']['sha256'])
    calibration = review['calibration']
    quality = review['quality']
    timing = review['timing']
    for section, fields in (
        (calibration, {'revision', 'reference', 'asset_key'}),
        (quality, {'good_flag', 'reference'} | ({'asset_key'} if isinstance(quality, dict) and 'asset_key' in quality else set())),
        (timing, {'scan_start', 'scan_end', 'provider_ready_at', 'reference'}),
    ):
        if (not isinstance(section, dict) or set(section) != fields
                or not isinstance(section['reference'], str) or not section['reference'].strip()):
            raise ValueError('Нужна отдельная проверенная ссылка на калибровку, качество и время.')
    if any(not isinstance(calibration[key], str) or not calibration[key].strip()
           for key in ('revision', 'asset_key')):
        raise ValueError('Нужны стабильная версия калибровки и ключ нативного актива.')
    producer_item = read_json(files['producer_item'])
    key = calibration['asset_key']
    if (metadata_kind(producer_item) != 'gptl_stac_item'
            or not isinstance(producer_item.get('properties'), dict)
            or not isinstance(producer_item['assets'].get(key), dict)
            or not isinstance(producer_item['assets'][key].get('href'), str)):
        raise ValueError('Нужен исходный STAC Item со снимком «Электро-Л».')
    producer_asset = producer_item['assets'][key]
    _electro_kelvin_band(producer_asset.get('raster:bands'))
    producer_spec = _gptl_asset(files['producer_item'], files['raster'], source='electro_l',
                                asset_key=key, channel=asset['channel'], expected_raster_hash=refs['raster']['sha256'])
    fields = ('native_scene_id', 'platform', 'source_uri', 'observed_at', 'scale', 'offset', 'declared_nodata')
    if any(producer_spec[field] != spec[field] for field in fields):
        raise ValueError('Исходный STAC Item и метаданные загрузчика не совпадают.')
    named_channel = re.search(r'(?:^ch(\d+)$|\.ir\.\d+\.(\d+)$)', key, re.IGNORECASE)
    if (named_channel and int(named_channel.group(1) or named_channel.group(2)) != asset['channel']
            or ('channel' in producer_asset and str(producer_asset['channel']) != spec['channel_id'])):
        raise ValueError('Канал не соответствует нативному активу поставщика.')
    _check_stac_file(producer_asset, refs['raster'], files['raster'])
    if type(quality['good_flag']) is not int or quality['good_flag'] != 0:
        raise ValueError('Поддерживается только нативная маска с подтверждённым значением 0 для годных пикселей.')
    quality_key = quality.get('asset_key')
    declared_quality = {
        name for name, candidate in producer_item['assets'].items()
        if isinstance(candidate, dict) and isinstance(candidate.get('roles'), list)
        and any(role in candidate['roles'] for role in ('quality', 'quality-mask'))
    }
    if 'asset_key' in quality or declared_quality:
        if (not isinstance(quality_key, str) or quality_key == key or not quality_key
                or (declared_quality and quality_key not in declared_quality)
                or not isinstance(producer_item['assets'].get(quality_key), dict)):
            raise ValueError('Паспорт маски не связан с активом качества поставщика.')
        _check_stac_file(producer_item['assets'][quality_key], refs['quality'], files['quality'])
    start, nominal, end, provider_ready, downloaded = map(utc, (
        timing['scan_start'], spec['observed_at'], timing['scan_end'],
        timing['provider_ready_at'], spec['earliest_available_at'],
    ))
    if not start < end or not start <= nominal <= end <= provider_ready <= downloaded:
        raise ValueError('Время съёмки, готовности поставщика и загрузки противоречит порядку событий.')
    for product_metadata in (producer_item['properties'], producer_asset):
        for field, expected in (('start_datetime', start), ('end_datetime', end),
                                ('published', provider_ready), ('ready_at', provider_ready)):
            if product_metadata.get(field) is not None and utc(product_metadata[field]) != expected:
                raise ValueError('Границы съёмки или готовность противоречат исходному STAC Item.')
    definition = {key: spec[key] for key in ('platform', 'instrument', 'channel_id', 'units', 'scale', 'offset')}
    definition['revision'] = calibration['revision'].strip()
    calibration_id = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    spec.update(calibration_reference=f'electro-l-gptl-l2ir:{calibration_id}',
                native_manifest_sha256=digest, native_asset_sha256=refs['producer_item']['sha256'],
                worker_asset_sha256=refs['asset']['sha256'],
                calibration_evidence=calibration['reference'], quality_evidence=quality['reference'],
                quality=files['quality'], scan_start=iso(timing['scan_start']), scan_end=iso(timing['scan_end']),
                provider_ready_at=iso(timing['provider_ready_at']), data_kind='real',
                inputs=[files['producer_item'], files['asset'], files['download'], manifest])
    if 'window' in review:
        spec['native_window'] = review['window']
    return spec, files['geometry'], spec['earliest_available_at'], timing['reference']
