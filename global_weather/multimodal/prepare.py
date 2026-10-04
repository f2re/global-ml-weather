"""Сборка последовательностей из численных капсул без повторной шкалы."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .contracts import Sequence, utc, fingerprint
from .io import sha256, read_json, read_arrays, load_sensors, save_sequence, resolve


def from_capsules(plan_path,output):
    from ..connectors.raster_bridge import read_capsule
    plan=read_json(plan_path);root=Path(plan_path).absolute().parent
    required={'schema','sensor','sensor_registry','frames','data_kind'}
    if not required.issubset(plan) or set(plan)-required-{'declarations'} or plan['schema']!='multimodal-capsule-plan-1':
        raise ValueError('Неверный план.')
    plan_hash=sha256(plan_path)
    declarations=plan.get('declarations',{})
    if not isinstance(declarations,dict) or set(declarations)-{'data_kind_reference','time_support','time_support_reference'}:
        raise ValueError('Неизвестное дополнение к контракту приёмника.')
    declared_refs={}
    for key in ('data_kind_reference','time_support_reference'):
        if key in declarations:declared_refs[key]=resolve(root,declarations[key])
    sensors={s.id:s for s in load_sensors(resolve(root,plan['sensor_registry']))}
    sensor=sensors[plan['sensor']]
    if sensor.kind!='imager':raise ValueError('МТВЗА требует физического экспортёра с антенной поддержкой.')
    if sensor.data_kind!=plan['data_kind']:raise ValueError('Нельзя изменить происхождение данных.')
    if not isinstance(plan['frames'],list) or not 1<=len(plan['frames'])<=48:raise ValueError('Неверное число кадров.')
    xs=[];ms=[];views=[];solars=[];footprints=[];observed=[];available=[];ids=[];geo=None;ident=None
    for frame in plan['frames']:
        if set(frame)!={'channels'} or set(frame['channels'])!=set(c.id for c in sensor.channels):
            raise ValueError('Все каналы должны быть обозначены. Для пропуска задайте null.')
        fields=[];metadata=[];g=None
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
            if m.get('data_kind') is None and 'data_kind_reference' not in declared_refs:
                raise ValueError('Нужно отдельное подтверждение происхождения старой капсулы.')
            if not m.get('geometry_reference') or not {'latitude','longitude','view_zenith_deg','footprint_km'}.issubset(a):
                raise ValueError('Неизвестна геометрия пригодного изображения.')
            grid_id=fingerprint({'crs':m['crs'],'transform':m['transform'],'shape':m['shape']})
            if ident is not None and grid_id!=ident:raise ValueError('Приведите кадры на общую локальную сетку до ConvGRU.')
            ident=grid_id
            if geo is None:geo=(a['latitude'],a['longitude'])
            if any(not np.array_equal(x,y) for x,y in zip(geo,(a['latitude'],a['longitude']))):
                raise ValueError('Смена геометрии требует явного совмещения кадров.')
            if g is not None and (not np.array_equal(g['view_zenith_deg'],a['view_zenith_deg']) or not np.array_equal(g['footprint_km'],a['footprint_km'])):
                raise ValueError('Каналы требуют отдельного согласования геометрии и пятна.')
            g=a;fields.append(a);metadata.append((m,pp))
            if sha256(mp)!=ref['manifest']['sha256'] or sha256(pp)!=ref['pixels']['sha256']:raise ValueError('Капсула изменилась при чтении.')
        if not metadata:raise ValueError('Пустой кадр исключите из плана; остальные пропуски сохраните масками.')
        times=[utc(m['observed_at']).timestamp() for m,_ in metadata]
        if max(times)-min(times)>60:raise ValueError('Каналы разделены более чем 60 секундами.')
        for m,_ in metadata:
            support=m.get('time_support')
            if support is None and 'time_support_reference' in declared_refs:support=declarations.get('time_support')
            if support not in ('frame','single_scan'):
                raise ValueError('Нужно подтверждение временной поддержки кадра; строковый адаптер ещё требуется.')
        shape=metadata[0][0]['shape']
        xs.append(np.stack([a['values'] if a is not None else np.full(shape,np.nan) for a in fields]))
        ms.append(np.stack([a['valid'] if a is not None else np.zeros(shape,bool) for a in fields]))
        views.append(g['view_zenith_deg']);footprints.append(g['footprint_km'])
        solars.append(g.get('solar_zenith_deg',np.full(shape,np.nan)))
        observed.append(max(times));available.append(max(utc(m['available_at']).timestamp() for m,_ in metadata))
        ids.append(fingerprint({'pixels':[sha256(p) for _,p in metadata],'grid':ident,'observed':max(times)}))
    m=metadata[0][0];a,b,c,d,e,f=m['transform'][:6]
    if m['crs'] not in ('EPSG:4326','OGC:CRS84') or b!=0 or d!=0:
        raise ValueError('Автоматическая площадь поддержана только для регулярной географической сетки без поворота.')
    lat=np.asarray(geo[0]);rows,cols=np.indices(lat.shape,dtype=float)
    expected_lat=f+e*(rows+.5);expected_lon=c+a*(cols+.5)
    actual_lon=np.asarray(geo[1]);lon_error=(actual_lon-expected_lon+180)%360-180
    if not np.allclose(lat,expected_lat,atol=1e-5,rtol=0) or not np.all(np.abs(lon_error)<1e-5):
        raise ValueError('Координаты не совпадают с центрами исходной регулярной сетки.')
    lo=np.clip(lat-abs(e)/2,-90,90);hi=np.clip(lat+abs(e)/2,-90,90)
    from ..grid import EARTH_RADIUS_M
    area=EARTH_RADIUS_M**2*abs(np.deg2rad(a))*(np.sin(np.deg2rad(hi))-np.sin(np.deg2rad(lo)))
    kw={k:torch.as_tensor(v,dtype=torch.float32) for k,v in dict(values=np.stack(xs),latitude=geo[0],longitude=geo[1],
        view_zenith_deg=np.stack(views),solar_zenith_deg=np.stack(solars),footprint_km=np.stack(footprints),area_m2=area).items()}
    seq=Sequence(sensor.id,valid=torch.as_tensor(np.stack(ms)),observed_unix=torch.tensor(observed,dtype=torch.float64),
        available_unix=torch.tensor(available,dtype=torch.float64),frame_ids=tuple(ids),channel_ids=tuple(c.id for c in sensor.channels),
        grid_id=ident,geometry_reference='physical-raster-v1; spherical area from geographic transform',
        source_sha256=sha256(plan_path),sensor_signature=sensor.measurement_signature,data_kind=sensor.data_kind,**kw).validate(sensor)
    if sha256(plan_path)!=plan_hash:raise ValueError('План изменился во время подготовки.')
    for key,path in declared_refs.items():
        if sha256(path)!=declarations[key]['sha256']:raise ValueError('Подтверждение изменилось.')
    save_sequence(output,seq)
    return {'status':'sequence_prepared','frames':len(ids),'shape':list(seq.values.shape),'sha256':sha256(output),
            'data_kind':seq.data_kind,'meteorologically_validated':False}
