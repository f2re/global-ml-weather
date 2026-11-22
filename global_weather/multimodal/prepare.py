"""Сборка последовательностей из численных капсул без повторной шкалы."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from .contracts import Sequence, utc, fingerprint
from .io import sha256, read_json, read_arrays, load_sensors, save_sequence, resolve


def prepare_electro_capsule(native_manifest, output):
    """Import a reviewed native L2IR scene into the usual physical raster capsule."""
    from ..connectors.ecosystem import electro_l_native
    from ..connectors.raster_bridge import import_raster

    spec, geometry, ready, reference = electro_l_native(native_manifest)
    return import_raster(spec, output, available_at=ready, availability_reference=reference,
                         geometry=geometry, max_pixels=262144)


def _verify_electro_scan(capsule, report, metadata, root, evidence, verify_native):
    from ..connectors.ecosystem import electro_l_native
    from ..connectors.raster_bridge import read_capsule

    native_ref = report.get('native_manifest')
    native = resolve(root, native_ref)
    evidence[native] = native_ref['sha256']
    spec, geometry, ready, reference = electro_l_native(native, verify_raw=False)
    expected = {key: value for key, value in spec.items() if key not in ('raster', 'quality', 'inputs')}
    expected.update(available_at=ready, availability_reference=reference, geometry_reference=read_json(geometry)['reference'])
    if (native_ref['sha256'] != spec['native_manifest_sha256']
            or any(metadata.get(key) != value for key, value in expected.items())):
        raise ValueError('Паспорт поставщика «Электро-Л» не соответствует капсуле.')
    if verify_native:
        with TemporaryDirectory(prefix='electro-native-') as cache:
            rebuilt = Path(cache).resolve(strict=True) / 'replayed'
            prepare_electro_capsule(native, rebuilt)
            replay, arrays = read_capsule(rebuilt)
            _, supplied = read_capsule(capsule.parent)
            fields = set(replay) | set(metadata)
            if (any(replay.get(key) != metadata.get(key) for key in fields - {'arrays_sha256'})
                    or set(arrays) != set(supplied)
                    or any(not np.array_equal(value, supplied[k]) for k, value in arrays.items())):
                raise ValueError('Пиксели и маски капсулы «Электро-Л» не воспроизводятся из нативных файлов.')


def _admitted_scan(capsule, manifest_hash, proof, metadata, root, evidence, *, verify_native=False):
    """Bind reviewed external evidence to a capsule without asserting its authenticity."""
    from ..connectors.ecosystem import ELECTRO_L_PLATFORMS, UPSTREAM

    if not isinstance(proof, dict) or set(proof) != {'scan_start', 'scan_end', 'available_at', 'evidence'}:
        raise ValueError('Для реальной капсулы требуется допуск с границами съёмки и временем готовности.')
    refs = proof['evidence']
    if not isinstance(refs, dict) or set(refs) != {'origin', 'timing', 'geometry', 'quality'}:
        raise ValueError('Нужны подтверждения происхождения, времени, геометрии и качества.')
    paths = {}
    for kind, ref in refs.items():
        path = resolve(root, ref)
        paths[kind] = path
        evidence[path] = ref['sha256']
        if not path.stat().st_size or path in (capsule, capsule.with_name('pixels.npz')):
            raise ValueError('Нужно отдельное непустое подтверждение реальной капсулы.')
    if len(set(paths.values())) != len(paths):
        raise ValueError('Нужны отдельные подтверждения происхождения, времени, геометрии и качества.')
    electro = metadata.get('source') == 'electro_l'
    if electro:
        native = (metadata.get('instrument') == 'MSU-GS'
                  and metadata.get('platform') in ELECTRO_L_PLATFORMS
                  and metadata.get('values_already_physical') is False)
    else:
        native = (metadata.get('source') == 'arktika_m'
                  and metadata.get('instrument') == 'MSU-GS/A'
                  and metadata.get('platform') in ('ARCM1', 'ARCM2')
                  and metadata.get('values_already_physical') is True
                  and metadata.get('scale') == 1 and metadata.get('offset') == 0)
    if (not native or metadata.get('upstream') != UPSTREAM['arktika-worker']
            or metadata.get('channel_id') not in ('4', '5', '6', '7', '8', '9', '10')
            or metadata.get('quantity') != 'brightness_temperature' or metadata.get('units') != 'K'
            or metadata.get('calibration_status') not in ('metadata', 'declared')
            or metadata.get('time_support') != 'native_scene_time_not_per_pixel_time'
            or metadata.get('quality_support') != 'paired_quality_raster'
            or metadata.get('observation_export_ready') is not True
            or not metadata.get('availability_reference')
            or not isinstance(metadata.get('native_scene_id'), str) or not metadata['native_scene_id'].strip()):
        raise ValueError('Капсула не содержит подтверждённого нативного происхождения, времени или маски качества.')
    identifiers = {'capsule_manifest_sha256': manifest_hash, 'native_scene_id': metadata['native_scene_id'],
                   'source': metadata['source'], 'platform': metadata['platform']}
    for kind, fields in (
        ('origin', ('upstream', 'instrument', 'channel_id', 'calibration_reference', 'calibration_status')),
        ('geometry', ('geometry_reference',)),
        ('quality', ('quality_support',)),
    ):
        report = read_json(paths[kind])
        expected = identifiers | {field: metadata[field] for field in fields} | {'schema': f'multimodal-ir-{kind}-1'}
        if not isinstance(report, dict) or any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Паспорт {kind} не соответствует капсуле.')
        if electro and kind == 'origin':
            _verify_electro_scan(capsule, report, metadata, root, evidence, verify_native)
    start, end = utc(proof['scan_start']), utc(proof['scan_end'])
    nominal, available = utc(metadata['observed_at']), utc(metadata['available_at'])
    if (not start <= nominal <= end <= available or available != utc(proof['available_at'])
            or (available-start).total_seconds() >= 43200):
        raise ValueError('Границы съёмки противоречат времени сцены или готовности.')
    if electro and (start != utc(metadata['scan_start']) or end != utc(metadata['scan_end'])):
        raise ValueError('Границы съёмки противоречат паспорту поставщика «Электро-Л».')
    timing = read_json(paths['timing'])
    if (not isinstance(timing, dict) or timing.get('schema') != 'multimodal-ir-timing-1'
            or any(timing.get(key) != metadata[key] for key in ('source', 'platform', 'native_scene_id'))
            or any(utc(timing.get(key)) != stamp for key, stamp in (
                ('observed_at', nominal), ('scan_start', start), ('scan_end', end), ('available_at', available)))):
        raise ValueError('Паспорт времени не соответствует допуску сцены.')
    return start.timestamp(), end.timestamp(), available.timestamp()


def _geographic_grid(metadata):
    a, b, c, d, e, f = metadata['transform'][:6]
    if metadata['crs'] not in ('EPSG:4326', 'OGC:CRS84') or b != 0 or d != 0 or a == 0 or e == 0:
        raise ValueError('Автоматическая площадь поддержана только для регулярной географической сетки без поворота.')
    rows, cols = np.indices(metadata['shape'], dtype=float)
    latitude = f + e * (rows + .5)
    longitude = (c + a * (cols + .5) + 180) % 360 - 180
    lo = np.clip(latitude - abs(e) / 2, -90, 90)
    hi = np.clip(latitude + abs(e) / 2, -90, 90)
    from ..grid import EARTH_RADIUS_M
    area = EARTH_RADIUS_M**2 * abs(np.deg2rad(a)) * (np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))
    return latitude, longitude, area


def sequence_from_capsules(plan_path, *, verify_native=True):
    from ..connectors.ecosystem import ELECTRO_L_PLATFORMS
    from ..connectors.raster_bridge import read_capsule
    plan=read_json(plan_path);root=Path(plan_path).absolute().parent
    required={'schema','sensor','sensor_registry','frames','data_kind'}
    if not required.issubset(plan) or set(plan)-required-{'declarations'} or plan['schema']!='multimodal-capsule-plan-1':
        raise ValueError('Неверный план.')
    plan_hash=sha256(plan_path)
    declarations=plan.get('declarations',{})
    if not isinstance(declarations,dict) or set(declarations)-{'data_kind_reference','time_support','time_support_reference','real_admission'}:
        raise ValueError('Неизвестное дополнение к контракту приёмника.')
    declared_refs={}
    for key in ('data_kind_reference','time_support_reference'):
        if key in declarations:declared_refs[key]=resolve(root,declarations[key])
    sensors={s.id:s for s in load_sensors(resolve(root,plan['sensor_registry']))}
    sensor=sensors[plan['sensor']]
    if sensor.kind!='imager':raise ValueError('МТВЗА требует физического экспортёра с антенной поддержкой.')
    if sensor.data_kind!=plan['data_kind']:raise ValueError('Нельзя изменить происхождение данных.')
    if (sensor.data_kind == 'real' and sensor.source == 'electro_l'
            and sensor.platform not in ELECTRO_L_PLATFORMS):
        raise ValueError('Реальные капсулы «Электро-Л» требуют отдельного согласования калибровки GPTL и реестра.')
    real_ir = sensor.data_kind == 'real' and sensor.source in ('arktika_m', 'electro_l')
    admission = {}
    evidence = {}
    if real_ir:
        if 'real_admission' not in declarations:
            raise ValueError('Реальные ИК-капсулы требуют отдельный манифест допуска.')
        declared_refs['real_admission'] = resolve(root, declarations['real_admission'])
        admission = read_json(declared_refs['real_admission'])
        if (not isinstance(admission, dict) or set(admission) != {'schema', 'capsules'}
                or admission['schema'] != 'multimodal-ir-admission-1' or not isinstance(admission['capsules'], dict)):
            raise ValueError('Неверный манифест допуска реальных ИК-капсул.')
        admission = admission['capsules']
    elif 'real_admission' in declarations:
        raise ValueError('Допуск применим только к реальным ИК-капсулам «Арктика-М» и «Электро-Л».')
    if not isinstance(plan['frames'],list) or not 1<=len(plan['frames'])<=48:raise ValueError('Неверное число кадров.')
    xs=[];ms=[];views=[];solars=[];footprints=[];observed=[];available=[];ids=[];geo=None;ident=None
    scan_starts=[];used_capsules=set()
    for frame in plan['frames']:
        if set(frame)!={'channels'} or set(frame['channels'])!=set(c.id for c in sensor.channels):
            raise ValueError('Все каналы должны быть обозначены. Для пропуска задайте null.')
        fields=[];metadata=[];g=None;scans=[];geometry_mask=None;solar=None
        for ch in sensor.channels:
            ref=frame['channels'][ch.id]
            if ref is None:fields.append(None);continue
            if set(ref)!={'manifest','pixels'}:raise ValueError('Нужны проверенные файлы капсулы.')
            mp=resolve(root,ref['manifest']);pp=resolve(root,ref['pixels'])
            if mp.parent!=pp.parent or mp.name!='manifest.json' or pp.name!='pixels.npz':raise ValueError('Неверная капсула.')
            read_arrays(pp)
            m,a=read_capsule(mp.parent)
            if (m.get('source')!=sensor.source or m.get('platform')!=sensor.platform
                    or str(m.get('channel_id'))!=ch.id or m.get('quantity')!=ch.quantity or m.get('units')!=ch.units
                    or m.get('calibration_reference')!=ch.calibration_id):
                raise ValueError('Физический канал или калибровка не совпали с реестром.')
            if m.get('data_kind') is not None and m['data_kind']!=plan['data_kind']:
                raise ValueError('Происхождение капсулы противоречит плану.')
            if m.get('data_kind') is None and not real_ir and 'data_kind_reference' not in declared_refs:
                raise ValueError('Нужно отдельное подтверждение происхождения старой капсулы.')
            if not m.get('geometry_reference') or not {'latitude','longitude','view_zenith_deg','footprint_km'}.issubset(a):
                raise ValueError('Неизвестна геометрия пригодного изображения.')
            grid_id=fingerprint({'crs':m['crs'],'transform':m['transform'],'shape':m['shape']})
            if ident is not None and grid_id!=ident:raise ValueError('Приведите кадры на общую локальную сетку до ConvGRU.')
            ident=grid_id
            if geo is None:
                pixels = a['valid'].size
                if pixels > 262144 or pixels * len(sensor.channels) * len(plan['frames']) > 8_000_000:
                    raise ValueError('Размер входной последовательности превышает бюджет многомодальной выборки.')
                geo = _geographic_grid(m)
            mask = a['valid']
            lat_error = np.abs(a['latitude'][mask]-geo[0][mask])
            lon_error = np.abs((a['longitude'][mask]-geo[1][mask]+180)%360-180)
            if (lat_error>=1e-5).any() or (lon_error>=1e-5).any():
                raise ValueError('Координаты не совпадают с центрами исходной регулярной сетки.')
            if g is None:
                g = {key: a[key].copy() for key in ('view_zenith_deg', 'footprint_km')}
                geometry_mask = mask.copy()
                solar = np.full(mask.shape, np.nan)
            else:
                overlap = geometry_mask & mask
                for key in g:
                    if not np.array_equal(g[key][overlap],a[key][overlap]):
                        raise ValueError('Каналы требуют отдельного согласования геометрии и пятна.')
                    g[key][mask] = a[key][mask]
                geometry_mask |= mask
            if 'solar_zenith_deg' in a:
                overlap = np.isfinite(solar) & mask
                if not np.array_equal(solar[overlap], a['solar_zenith_deg'][overlap]):
                    raise ValueError('Каналы требуют согласования солнечной геометрии.')
                solar[mask] = a['solar_zenith_deg'][mask]
            if real_ir:
                manifest_hash = ref['manifest']['sha256']
                if manifest_hash in used_capsules:
                    raise ValueError('Повторное включение одной капсулы.')
                scans.append(_admitted_scan(mp, manifest_hash, admission.get(manifest_hash), m, root, evidence,
                                            verify_native=verify_native))
                used_capsules.add(manifest_hash)
            fields.append(a);metadata.append((m,pp))
            if sha256(mp)!=ref['manifest']['sha256'] or sha256(pp)!=ref['pixels']['sha256']:raise ValueError('Капсула изменилась при чтении.')
        if not metadata:raise ValueError('Пустой кадр исключите из плана; остальные пропуски сохраните масками.')
        times=[utc(m['observed_at']).timestamp() for m,_ in metadata]
        if max(times)-min(times)>60:raise ValueError('Каналы разделены более чем 60 секундами.')
        if real_ir:
            native_ids = {m['native_scene_id'] for m, _ in metadata}
            if len(native_ids) != 1:
                raise ValueError('Каналы относятся к разным нативным сценам.')
            if any(max(s[i] for s in scans) - min(s[i] for s in scans) > 60 for i in (0, 1)):
                raise ValueError('Границы съёмки каналов разделены более чем 60 секундами.')
            scan_starts.append(min(s[0] for s in scans))
            observed.append(max(s[1] for s in scans))
            available.append(max(s[2] for s in scans))
            ids.append(fingerprint({'source':sensor.source,'platform':sensor.platform,'native_scene_id':next(iter(native_ids))}))
        else:
            for m,_ in metadata:
                support=m.get('time_support')
                if support is None and 'time_support_reference' in declared_refs:support=declarations.get('time_support')
                if support not in ('frame','single_scan'):
                    raise ValueError('Нужно подтверждение временной поддержки кадра; строковый адаптер ещё требуется.')
            observed.append(max(times));available.append(max(utc(m['available_at']).timestamp() for m,_ in metadata))
            ids.append(fingerprint({'pixels':[sha256(p) for _,p in metadata],'grid':ident,'observed':max(times)}))
        shape=metadata[0][0]['shape']
        xs.append(np.stack([a['values'] if a is not None else np.full(shape,np.nan) for a in fields]))
        ms.append(np.stack([a['valid'] if a is not None else np.zeros(shape,bool) for a in fields]))
        views.append(g['view_zenith_deg']);footprints.append(g['footprint_km'])
        solars.append(solar)
    if real_ir and used_capsules != set(admission):
        raise ValueError('Допуск содержит капсулы, которых нет в плане.')
    kw={k:torch.as_tensor(v,dtype=torch.float32) for k,v in dict(values=np.stack(xs),latitude=geo[0],longitude=geo[1],
        view_zenith_deg=np.stack(views),solar_zenith_deg=np.stack(solars),footprint_km=np.stack(footprints),area_m2=geo[2]).items()}
    seq=Sequence(sensor.id,valid=torch.as_tensor(np.stack(ms)),observed_unix=torch.tensor(observed,dtype=torch.float64),
        available_unix=torch.tensor(available,dtype=torch.float64),frame_ids=tuple(ids),channel_ids=tuple(c.id for c in sensor.channels),
        grid_id=ident,geometry_reference='physical-raster-v1; spherical area from geographic transform',
        source_sha256=plan_hash,sensor_signature=sensor.measurement_signature,data_kind=sensor.data_kind,
        scan_start_unix=torch.tensor(scan_starts, dtype=torch.float64) if real_ir else None,**kw).validate(sensor)
    if sha256(plan_path)!=plan_hash:raise ValueError('План изменился во время подготовки.')
    for key,path in declared_refs.items():
        if sha256(path)!=declarations[key]['sha256']:raise ValueError('Подтверждение изменилось.')
    if any(sha256(path) != digest for path,digest in evidence.items()):
        raise ValueError('Подтверждение допуска изменилось во время подготовки.')
    return seq


def from_capsules(plan_path, output):
    seq = sequence_from_capsules(plan_path)
    save_sequence(output,seq)
    return {'status':'sequence_prepared','frames':len(seq.frame_ids),'shape':list(seq.values.shape),'sha256':sha256(output),
            'data_kind':seq.data_kind,'meteorologically_validated':False}
