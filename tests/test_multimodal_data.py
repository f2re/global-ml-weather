"""Контракты выборки и норм; без сетевых запросов и подмены будущих целей."""
from dataclasses import fields,replace
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from global_weather.grid import build_pyramid
from global_weather.observations import PackedObservations
from global_weather.multimodal.fixture import fixture
from global_weather.multimodal.io import save_sensors,save_sequence,reference,write_json,read_json,load_sensors,load_sequence
from global_weather.multimodal.integration import attach,specification
from global_weather.multimodal.normalization import fit_sensors


def on_disk(tmp_path):
    grids=build_pyramid(0);sensors,obs=fixture(grids)
    raw=tuple(replace(s,channels=tuple(replace(c,mean=None,std=None) for c in s.channels),
                      normalization_sha256=None,fit_end=None) for s in sensors)
    save_sensors(tmp_path/'raw.json',raw)
    frames=[]
    for seq in obs.sequences:
        p=tmp_path/f'{seq.sensor_id}.npz';save_sequence(p,seq)
        frames.append({'sensor':seq.sensor_id,'file':reference(tmp_path,p)})
    manifest={'schema':'global-weather-dataset-1','data_kind':'synthetic',
        'samples':[{'id':'train','split':'train','issue_time':obs.issue_time.isoformat()},
                   {'id':'test','split':'test','issue_time':(obs.issue_time+timedelta(days=8)).isoformat()}],
        'multimodal':{'schema':'multimodal-input-1','sensors':reference(tmp_path,tmp_path/'raw.json'),
            'scenes':{'train':frames,'test':[{'sensor':'does-not-exist','file':{'path':'DO-NOT-READ','sha256':'a'*64}}]},
            'radius_km':500.,'neighbors':8,'base_channels':8}}
    write_json(tmp_path/'dataset.json',manifest)
    return grids,sensors,obs,manifest


def test_fit_never_reads_test_scenes(tmp_path):
    _,s,obs,_=on_disk(tmp_path)
    report=fit_sensors(tmp_path/'dataset.json',tmp_path/'fitted.json')
    assert report['train_frames']==12 and report['test_targets_read'] is False
    m=read_json(tmp_path/'fitted.json');registry=tmp_path/m['multimodal']['sensors']['path']
    sensors=load_sensors(registry)
    for original,new,seq in zip(s,sensors,obs.sequences):
        assert new.channels[0].mean==pytest.approx(float(seq.values[:,0].mean()),abs=1e-5)
        assert new.channels[0].std>0 and original.measurement_signature==new.measurement_signature
        assert original.signature!=new.signature
        assert load_sequence(tmp_path/f'{seq.sensor_id}.npz',new).sensor_id==seq.sensor_id


def test_unfitted_registry_cannot_enter_model(tmp_path):
    _,s,obs,_=on_disk(tmp_path)
    raw=load_sensors(tmp_path/'raw.json')[0]
    with pytest.raises(ValueError,match='нормы'):raw.require_normalization()


def test_fit_cannot_overwrite_manifest(tmp_path):
    on_disk(tmp_path)
    with pytest.raises(ValueError):fit_sensors(tmp_path/'dataset.json',tmp_path/'dataset.json')


def test_attached_data_passes_through_existing_packed_contract(tmp_path):
    grids,sensors,obs,m=on_disk(tmp_path)
    fit_sensors(tmp_path/'dataset.json',tmp_path/'fitted.json');m=read_json(tmp_path/'fitted.json')
    ds=SimpleNamespace(manifest=m,root=tmp_path,kind='synthetic',history_hours=12,n_cells=12,
        grid_fingerprint=grids[0].fingerprint,registry={},
        samples=[SimpleNamespace(id=s['id'],split=s['split'],issue=obs.issue_time if s['id']=='train' else obs.issue_time+timedelta(days=8)) for s in m['samples']])
    packed=PackedObservations(**{f.name:getattr(obs,f.name) for f in fields(PackedObservations)})
    result=attach(ds,ds.samples[0],packed)
    assert len(result.sequences)==4 and result.accepted_records==packed.accepted_records+4
    assert torch.equal(result.features,packed.features)


def test_duplicate_frame_is_rejected(tmp_path):
    g,s,o,m=on_disk(tmp_path);fit_sensors(tmp_path/'dataset.json',tmp_path/'fitted.json');m=read_json(tmp_path/'fitted.json')
    m['multimodal']['scenes']['train'].append(m['multimodal']['scenes']['train'][0])
    ds=SimpleNamespace(manifest=m,root=tmp_path,kind='synthetic',history_hours=12,n_cells=12,
        grid_fingerprint=g[0].fingerprint,registry={},samples=[SimpleNamespace(id='train',split='train',issue=o.issue_time),
        SimpleNamespace(id='test',split='test',issue=o.issue_time+timedelta(days=8))])
    p=PackedObservations(**{f.name:getattr(o,f.name) for f in fields(PackedObservations)})
    with pytest.raises(ValueError,match='повторно'):attach(ds,ds.samples[0],p)


def test_changed_statistics_file_is_rejected(tmp_path):
    on_disk(tmp_path);fit_sensors(tmp_path/'dataset.json',tmp_path/'fitted.json')
    m=read_json(tmp_path/'fitted.sensors.json');ref=next(iter(m['statistics'].values()))
    (tmp_path/ref['path']).write_text('{}')
    with pytest.raises(ValueError):load_sensors(tmp_path/'fitted.sensors.json')


def test_norms_block_holdout_leak(tmp_path):
    _,_,o,m=on_disk(tmp_path)
    m['samples'][1]['issue_time']=(o.issue_time+timedelta(hours=1)).isoformat()
    write_json(tmp_path/'overlap.json',m)
    with pytest.raises(ValueError,match='пересекает'):fit_sensors(tmp_path/'overlap.json',tmp_path/'fitted.json')
