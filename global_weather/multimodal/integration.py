"""Явное расширение PreparedDataset. Старый режим остаётся без изменений."""
from __future__ import annotations
from dataclasses import dataclass, fields
from datetime import timedelta
import torch
from ..observations import PackedObservations
from .contracts import utc, fingerprint
from .io import resolve, load_sensors, load_sequence


@dataclass
class MultimodalObservations(PackedObservations):
    sequences: tuple = ()
    sensor_signature: str = ''

    def to(self,device):
        args={f.name:getattr(self,f.name) for f in fields(self)}
        args={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in args.items()}
        args['sequences']=tuple(s.to(device) for s in self.sequences)
        return type(self)(**args)


def specification(ds):
    spec=ds.manifest.get('multimodal')
    if not isinstance(spec,dict) or set(spec)!={'schema','sensors','scenes','radius_km','neighbors','base_channels'}:
        raise ValueError('Неверный контракт multimodal.')
    if spec['schema']!='multimodal-input-1':raise ValueError('Неизвестная схема.')
    if type(spec['neighbors']) is not int or not 1<=spec['neighbors']<=64:
        raise ValueError('Неверное число соседей.')
    if type(spec['radius_km']) not in (int,float) or not 0<spec['radius_km']<=20020:
        raise ValueError('Неверный радиус усвоения.')
    if spec['base_channels'] not in (8,16,32):raise ValueError('Неверная ширина спутникового кодировщика.')
    if set(spec['scenes'])-{s.id for s in ds.samples}:raise ValueError('Неизвестный пример в спутниковом манифесте.')
    sensors=load_sensors(resolve(ds.root,spec['sensors']))
    for sensor in sensors:
        sensor.require_normalization()
        if sensor.data_kind!=ds.kind:raise ValueError('Нормы прибора и выборка имеют разное происхождение.')
        holdout=[s.issue-timedelta(hours=ds.history_hours) for s in ds.samples if s.split in ('validation','test')]
        if holdout and utc(sensor.fit_end)>=min(holdout):raise ValueError('Нормы спутника затрагивают проверочную выборку.')
    sig=fingerprint({'sensors':[s.signature for s in sensors],
                     'radius_km':spec['radius_km'],'neighbors':spec['neighbors'],'base_channels':spec['base_channels']})
    return spec,sensors,sig


def attach(ds,sample,packed):
    """Читает только наблюдения. Никакие будущие цели не нужны для кадров."""
    spec,sensors,sig=specification(ds);by_id={s.id:s for s in sensors}
    seqs=[];seen=set();pixels=0
    refs=spec['scenes'].get(sample.id,[])
    if not isinstance(refs,list) or len(refs)>32:raise ValueError('Превышено число последовательностей.')
    for item in refs:
        if set(item)!={'sensor','file'} or item['sensor'] not in by_id:raise ValueError('Неизвестный адаптер.')
        seq=load_sequence(resolve(ds.root,item['file']),by_id[item['sensor']])
        seq=seq.causal(sample.issue,by_id[item['sensor']])
        if seq is None:continue
        seq.validate(by_id[item['sensor']],n_cells=ds.n_cells,grid_fingerprint=ds.grid_fingerprint)
        identities={(seq.sensor_id,key) for key in seq.frame_ids}
        if seen&identities:raise ValueError('Один кадр повторно включён в пример. Не дублируйте пролёт.')
        seen|=identities;pixels+=seq.values.numel()
        if pixels>8_000_000:raise ValueError('Превышен лимит декодированных спутниковых значений одного примера.')
        seqs.append(seq)
    for seq in seqs:
        sensor=by_id[seq.sensor_id]
        for v in ds.registry.values():
            if not v.get('product') and v.get('source')==sensor.source and v.get('platform')==sensor.platform and v.get('channel_id') in seq.channel_ids:
                raise ValueError('Дублирование исходного канала в растровом и скалярном входе.')
    args={f.name:getattr(packed,f.name) for f in fields(PackedObservations)}
    count=sum(int(s.valid.any().item()) for s in seqs)
    if not packed.accepted_records and not count:raise ValueError('Нет доступных наблюдений для анализа.')
    args['accepted_records']+=count
    return MultimodalObservations(**args,sequences=tuple(seqs),sensor_signature=sig)


def make_model(ds,cfg,observations,grids):
    from .model import MultimodalWeatherModel
    spec,sensors,sig=specification(ds)
    pixels=sum(s.values.shape[0]*s.values.shape[-2]*s.values.shape[-1] for s in observations.sequences)
    estimate=pixels*cfg.hidden*4*32+ds.n_cells*38*cfg.hidden*4*(2+cfg.horizon_hours//ds.step)*16
    if estimate > cfg.memory_budget_mib*1024**2:
        raise ValueError('Оценка спутниковых активаций и динамики превышает бюджет памяти.')
    return MultimodalWeatherModel(grids,observations.vocabulary,observation_schema=observations.schema_fingerprint,
            hidden=cfg.hidden,latent_slots=cfg.latent_slots,step_hours=ds.step,normalization=ds.norm,
            sensors=sensors,sensor_signature=sig,radius_km=spec['radius_km'],neighbors=spec['neighbors'],
            base_channels=spec['base_channels'])
