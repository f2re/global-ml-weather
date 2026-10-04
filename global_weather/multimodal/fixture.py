"""Analytic scenes for code tests only. These are NOT downloaded satellite data."""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import numpy as np
from .frames import utc


def specs():
    result={}
    for name,source in [('electro_ir','electro_l'),('arktika_ir','arktika_m'),
                         ('meteor_ir','meteor_msu_mr'),('meteor_mw','meteor_mtvza')]:
        result[name]={'source':source,'platform':'SYNTHETIC','instrument':'SYNTHETIC-'+name,
                       'encoder':'microwave' if name=='meteor_mw' else 'unet',
                       'temporal':'registered' if name in ('electro_ir','arktika_ir') else 'events',
                       'projection':'bounded-centre',
                       'channels':[{'id':str(i),'variable':name+'_tb'+str(i),'quantity':'brightness_temperature',
                                    'units':'K','calibration':'analytic-fixture-not-real'} for i in (1,2)]}
    return {'schema':'global-weather-multimodal-1','sensors':result}


def scene_arrays(spec, issue, *, sample_index=0, age_hours=1, shift=0.):
    y,x=np.mgrid[0:8,0:8]; shape=(8,8)
    values=np.stack([255+i*4+4*np.sin(y*.3+x*.2+sample_index*.3+age_hours*.2) for i in (1,2)]).astype(np.float32)
    valid=np.ones(values.shape,bool);valid[0,0,0]=False;values[0,0,0]=np.nan
    observed=issue-timedelta(hours=age_hours)
    meta={'schema':'satellite-scene-1','data_kind':'synthetic',
          **{k:spec[k] for k in ('source','platform','instrument','channels')},
          'grid_id':'synthetic-grid','available_at':(observed+timedelta(minutes=10)).isoformat(),
          'availability_reference':'analytic-fixture','geometry_reference':'analytic-fixture',
          'source_sha256':[hashlib.sha256(b'analytic input, not satellite observations').hexdigest()],
          'footprint_definition':'bounding_diameter','time_representation':'analytic_fixture'}
    return {'values':values,'valid':valid,'latitude':50+y*.1+shift,'longitude':30+x*.1+shift,
            'view_zenith_deg':np.full(shape,20.),'footprint_major_km':np.full(shape,4.),
            'footprint_minor_km':np.full(shape,4.),'footprint_azimuth_deg':np.zeros(shape),
            'observed_utc_s':np.full(shape,observed.timestamp()),'solar_zenith_deg':np.full(shape,45.),
            'metadata':np.array(json.dumps(meta))}


def create_dataset(output, *, horizon_hours=3):
    from ..pipeline.fixture import create_fixture
    from ..pipeline.io import read_json,reference,atomic_json,write_arrays
    from .prepare import fit_channels
    root=Path(output).absolute();base=create_fixture(root,horizon_hours=horizon_hours)
    data=read_json(base);data['multimodal']=specs()
    for i,sample in enumerate(data['samples']):
        sample['satellite']={}
        for name,spec in data['multimodal']['sensors'].items():
            paths=[]
            for age in (4,1):
                path=root/'scenes'/f'{sample["id"]}-{name}-{age}.npz'
                shift=age*.1 if spec['temporal']=='events' else 0.
                write_arrays(path,**scene_arrays(spec,utc(sample['issue_time']),sample_index=i,age_hours=age,shift=shift))
                paths.append(reference(root,path))
            sample['satellite'][name]=paths
    unscaled=root/'multimodal-unscaled.json';atomic_json(unscaled,data)
    final=root/'multimodal.json';fit_channels(unscaled,final)
    return final
