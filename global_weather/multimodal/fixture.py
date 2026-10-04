"""Только аналитические синтетические примеры. Никаких реальных наблюдений или LUT."""
from __future__ import annotations
from dataclasses import fields
from datetime import datetime,timedelta,timezone
import numpy as np
import torch
from .contracts import Sensor,Channel,Sequence,fingerprint
from .integration import MultimodalObservations
from ..observations import pack_observations,PackedObservations
from ..grid import latlon
from ..vertical import PRESSURE_HPA


def fixture(grids,*,issue=None,height=8,width=8):
    issue=issue or datetime(2020,1,10,12,tzinfo=timezone.utc)
    sensors=[];sequences=[];grid=grids[0]
    lat,lon=latlon(grid.xyz[0]);times=torch.tensor([(issue-timedelta(hours=h)).timestamp() for h in (10,4,1)],dtype=torch.float64)
    for i,source in enumerate(('electro_l','arktika_m','meteor_msu_mr','meteor_mtvza')):
        kind='microwave' if source=='meteor_mtvza' else 'imager'
        c=3 if kind=='imager' else 2
        sensor=Sensor(f'synthetic_{i}',source,'SYNTHETIC',kind,
                      tuple(Channel(str(j),'brightness_temperature','K',250.,25.,'analytic-fixture-only') for j in range(c)),
                      fingerprint({'fixture':i}), '2019-01-01T00:00:00Z','synthetic')
        sensors.append(sensor)
        x=torch.arange(height*width).reshape(height,width).float()/max(1,height*width-1)
        values=torch.stack([torch.stack([245.+3*i+2*j+5*x+2*t for j in range(c)]) for t in range(3)])
        kw={}
        if kind=='microwave':
            pixels=torch.arange(height*width).repeat_interleave(2)
            cells=torch.tensor([0,1]).repeat(height*width)
            kw=dict(link_pixel=pixels,link_cell=cells,link_weight=torch.tensor([.75,.25]).repeat(height*width),
                    link_grid_fingerprint=grid.fingerprint,link_reference='analytic fixture, not an instrument antenna')
        seq=Sequence(sensor.id,values,torch.ones_like(values,dtype=torch.bool),times,times+60,
                     torch.full((height,width),float(lat)),torch.full((height,width),float(lon)),
                     torch.full((3,height,width),25.),torch.full((3,height,width),60.),
                     torch.full((3,height,width),4. if kind=='imager' else 40.),torch.full((height,width),1e6),
                     tuple(fingerprint({'sensor':i,'time':t}) for t in times.tolist()),tuple(str(j) for j in range(c)),
                     'synthetic-grid','analytic fixture geometry',fingerprint({'source':i}),sensor.measurement_signature,'synthetic',**kw)
        sequences.append(seq)
    records=[]
    for j in (0,1):
        lat,lon=latlon(grid.xyz[j])
        for source,name,pressure,value in [('station','t2m',None,279.+j),('radiosonde','temperature',85000.,265.+j),
                                           ('radiosonde','u',70000.,12.+j)]:
            rec=dict(observation_id=f'{j}-{name}',source=source,variable=name,value=value,units='K' if name in ('t2m','temperature') else 'm s-1',
                     latitude=float(lat),longitude=float(lon),observed_at=(issue-timedelta(hours=2)).isoformat(),available_at=issue.isoformat(),valid=True)
            if pressure:rec['pressure_pa']=pressure
            records.append(rec)
    obs=pack_observations(records,grid,np.array(PRESSURE_HPA)*100,issue)
    signature=fingerprint([s.signature for s in sensors])
    args={f.name:getattr(obs,f.name) for f in fields(PackedObservations)}
    packed=MultimodalObservations(**args,sequences=tuple(sequences),sensor_signature=signature)
    return tuple(sensors),packed


def dataset(output,*,horizon_hours=72):
    from ..pipeline.fixture import create_fixture
    from .io import read_json,save_sensors,save_sequence,write_json,reference
    from .normalization import fit_sensors
    from ..grid import build_pyramid
    from pathlib import Path
    from .contracts import utc
    from dataclasses import replace
    root=Path(output).absolute();path=create_fixture(root,horizon_hours=horizon_hours)
    base=read_json(path);grids=build_pyramid(base['mesh_level']);scenes={};registry=None
    for sample in base['samples']:
        sensors,obs=fixture(grids,issue=utc(sample['issue_time']))
        if registry is None:
            registry=tuple(replace(s,channels=tuple(replace(c,mean=None,std=None) for c in s.channels),
                                   normalization_sha256=None,fit_end=None) for s in sensors)
        refs=[]
        for seq in obs.sequences:
            dest=root/f"{sample['id']}-{seq.sensor_id}.npz";save_sequence(dest,seq)
            refs.append({'sensor':seq.sensor_id,'file':reference(root,dest)})
        scenes[sample['id']]=refs
    sensor_path=root/'raw-sensors.json';save_sensors(sensor_path,registry)
    base['multimodal']={'schema':'multimodal-input-1','sensors':reference(root,sensor_path),'scenes':scenes,
                        'radius_km':1000.,'neighbors':8,'base_channels':8}
    raw=root/'multimodal-unscaled.json';write_json(raw,base)
    original=root/'base-dataset.json'
    if original.exists():raise FileExistsError('Исходный демонстрационный манифест уже существует.')
    Path(path).rename(original)
    output=root/'dataset.json';fit_sensors(raw,output)
    return output
