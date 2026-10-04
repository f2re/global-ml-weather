"""Потоковые нормы спутников только из train. Нет чтения будущих целей."""
from __future__ import annotations
from dataclasses import replace,asdict
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
from .contracts import utc
from .io import read_json,resolve,load_sensors,load_sequence,write_json,sha256,reference


def fit_sensors(dataset_path,output):
    root=Path(dataset_path).absolute().parent;manifest=read_json(dataset_path)
    if manifest.get('schema')!='global-weather-dataset-1':raise ValueError('Нужен манифест обучающей выборки.')
    output=Path(output).absolute()
    if output.parent!=root or output.exists():raise ValueError('Новый манифест должен находиться рядом с исходным.')
    spec=manifest['multimodal'];sensors=load_sensors(resolve(root,spec['sensors']))
    if any(s.data_kind!=manifest['data_kind'] for s in sensors):raise ValueError('Разное происхождение данных и реестра.')
    training=[s for s in manifest['samples'] if s['split']=='train']
    holdout=[utc(s['issue_time'])-timedelta(hours=12) for s in manifest['samples'] if s['split'] in ('validation','test')]
    if not training:raise ValueError('Нет обучающей части.')
    fit_end=max(utc(s['issue_time']) for s in training)
    if holdout and fit_end>=min(holdout):raise ValueError('Период норм пересекает проверочное окно.')
    states={s.id:[np.zeros(len(s.channels),dtype=np.float64) for _ in range(3)] for s in sensors}
    by_id={s.id:s for s in sensors};seen={};source_hashes={}
    for sample in sorted(training,key=lambda s:utc(s['issue_time'])):
        issue=utc(sample['issue_time'])
        for item in spec['scenes'].get(sample['id'],[]):
            sensor=by_id[item['sensor']];path=resolve(root,item['file'])
            seq=load_sequence(path,sensor).causal(issue,sensor)
            if seq is None:continue
            source_hashes[item['file']['path']]=item['file']['sha256']
            for i,key in enumerate(seq.frame_ids):
                identity=(sensor.id,key)
                signature=tuple(v.detach().cpu().numpy().tobytes() for v in (seq.values[i],seq.valid[i],seq.area_m2,seq.latitude,seq.longitude,seq.solar_zenith_deg[i],seq.observed_unix[i]))
                import hashlib
                token=hashlib.sha256(b''.join(signature)).hexdigest()
                if identity in seen:
                    if seen[identity]!=token:raise ValueError('Одна версия кадра имеет разное содержимое.')
                    continue
                seen[identity]=token
                x=seq.values[i].detach().cpu().numpy().astype(np.float64)
                mask=seq.valid[i].cpu().numpy().copy()
                for j,ch in enumerate(sensor.channels):
                    if ch.units=='1':mask[j]&=np.isfinite(seq.solar_zenith_deg[i].numpy()) & (seq.solar_zenith_deg[i].numpy()<90)
                area=seq.area_m2.cpu().numpy().astype(np.float64) if sensor.kind=='imager' else np.ones(x.shape[1:])
                w=np.where(mask,area,0.);safe=np.where(mask,x,0.)
                amount=w.sum((1,2));mean=(safe*w).sum((1,2))/np.maximum(amount,1e-300)
                m2=(w*(safe-mean[:,None,None])**2).sum((1,2))
                old,mu,variance=states[sensor.id];total=old+amount;delta=mean-mu
                mu+=delta*amount/np.maximum(total,1e-300)
                variance+=m2+delta**2*old*amount/np.maximum(total,1e-300)
                states[sensor.id]=[total,mu,variance]
    completed=[];payloads={}
    for sensor in sensors:
        count,mean,m2=states[sensor.id];std=np.sqrt(m2/np.maximum(count,1e-300))
        if (count<=0).any() or (std<=0).any() or not np.isfinite(std).all():raise ValueError('Недостаточно данных или нулевая дисперсия канала.')
        payloads[sensor.id]={'schema':'multimodal-statistics-1','sensor':sensor.id,'mean':mean.tolist(),'std':std.tolist(),
            'fit_end':fit_end.isoformat(),'sample_ids':[s['id'] for s in training],'source_sha256':source_hashes,
            'data_kind':sensor.data_kind,'weighting':'pixel_area' if sensor.kind=='imager' else 'unique_footprint',
            'measurement_signature':sensor.measurement_signature,'source_dataset_sha256':sha256(dataset_path)}
    targets={s.id:root/f'{output.stem}.{s.id}.norms.json' for s in sensors}
    registry_path=root/f'{output.stem}.sensors.json'
    for p in (*targets.values(),registry_path,output):
        if p.exists() or p.is_symlink():raise FileExistsError('Файл результата уже существует.')
    refs={}
    for sensor in sensors:
        norm=payloads[sensor.id];path=targets[sensor.id];write_json(path,norm)
        chans=tuple(replace(ch,mean=mu,std=sd) for ch,mu,sd in zip(sensor.channels,norm['mean'],norm['std']))
        completed.append(replace(sensor,channels=chans,normalization_sha256=sha256(path),fit_end=fit_end.isoformat()))
        refs[sensor.id]=reference(root,path)
    write_json(registry_path,{'schema':'multimodal-sensors-1','sensors':[asdict(s) for s in completed],'statistics':refs})
    new=dict(manifest);new['multimodal']=dict(spec,sensors=reference(root,registry_path))
    write_json(output,new)
    return {'status':'satellite_normalization_frozen','dataset':output.name,'sensors':len(completed),
            'train_frames':len(seen),'test_targets_read':False,'meteorologically_validated':False}
