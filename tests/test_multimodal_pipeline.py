"""Canonical integration tests run on the complete repository, not substitute stubs."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from global_weather.multimodal.fixture import create_dataset
from global_weather.multimodal.dataset import TrainingDataset
from global_weather.multimodal.model import make_model, MultimodalWeatherModel
from global_weather.pipeline.runner import TrainConfig,train,evaluate,forecast,load_trained,target_tensors
from global_weather.pipeline.io import read_json,atomic_json,reference
from global_weather.training import train_step

@pytest.fixture
def dataset(tmp_path):
    return create_dataset(tmp_path/'data',horizon_hours=3)


def test_all_encoders_connected_and_gradients_finite(dataset):
    ds=TrainingDataset(dataset);cfg=TrainConfig(epochs=1);torch.manual_seed(17);torch.set_num_threads(1)
    model=make_model(ds,cfg);sample=ds.subset('train')[0]
    z,land=(torch.tensor(a,dtype=torch.float32) for a in (ds.elevation,ds.land))
    obs=ds.packed(sample)
    assert isinstance(model,MultimodalWeatherModel) and len(obs.scenes)==8
    report=train_step(model,torch.optim.AdamW(model.parameters(),lr=1e-4),obs,z,land,target_tensors(ds,sample,3))
    assert np.isfinite(report['loss'])
    for name,network in model.satellites.items():
        assert any(p.grad is not None and bool((p.grad!=0).any()) for p in network.parameters()),name
    for name in ('encoder','fusion','compress','processor','profile_head','surface_head'):
        assert any(p.grad is not None and bool((p.grad!=0).any()) for p in getattr(model,name).parameters()),name
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_no_raster_reassimilation_and_missing_sensor(dataset):
    ds=TrainingDataset(dataset);model=make_model(ds,TrainConfig());obs=ds.packed(ds.samples[0])
    static=torch.tensor(ds.elevation,dtype=torch.float32);land=torch.tensor(ds.land,dtype=torch.float32)
    with torch.no_grad():
        a=model.analysis_state(obs,static,land)
        b=model.analysis_state(obs,static,land,background=a)
        assert torch.equal(a.latent,b.latent)
        absent=replace(obs,scenes=tuple(s for s in obs.scenes if s.sensor!='meteor_mw'))
        c=model.analysis_state(absent,static,land)
        assert not torch.allclose(a.latent,c.latent)
        no_scenes=model.analysis_state(replace(obs,scenes=()),static,land)
        assert torch.isfinite(no_scenes.latent).all()


def test_train_resume_and_forecast_without_targets(dataset,tmp_path):
    ds=TrainingDataset(dataset);cfg=TrainConfig(epochs=1,patience=5,threads=1)
    run=tmp_path/'run';train(dataset,run,cfg,progress=lambda *a,**k:None)
    train(dataset,run,replace(cfg,epochs=2),resume=True,progress=lambda *a,**k:None)
    report=evaluate(dataset,run,tmp_path/'test.json')
    assert report['scores'] and report['meteorologically_validated'] is False
    m=read_json(dataset);m['schema']='global-weather-input-1';sample=deepcopy(m['samples'][-1]);sample['split']='inference'
    sample.pop('targets');m['samples']=[sample]
    path=dataset.with_name('input-only.json');atomic_json(path,m)
    # Delete all target files. Forecast must not read them or run evaluation.
    for item in read_json(dataset)['samples']:(dataset.parent/item['targets']['path']).unlink()
    result=forecast(path,run,sample['id'],tmp_path/'forecast',horizon_hours=3)
    assert result['targets_read'] is False and (tmp_path/'forecast'/'frame_003.npz').is_file()


def test_resumed_training_matches_uninterrupted(dataset,tmp_path):
    cfg=TrainConfig(epochs=1,patience=5,threads=1)
    train(dataset,tmp_path/'resumed',cfg,progress=lambda *a,**k:None)
    train(dataset,tmp_path/'resumed',replace(cfg,epochs=2),resume=True,progress=lambda *a,**k:None)
    train(dataset,tmp_path/'full',replace(cfg,epochs=2),progress=lambda *a,**k:None)
    a=torch.load(tmp_path/'resumed'/'epochs'/'000002'/'weights.pt',weights_only=True)
    b=torch.load(tmp_path/'full'/'epochs'/'000002'/'weights.pt',weights_only=True)
    for key,x in a['state_dict'].items():
        if not isinstance(x,torch.Tensor):continue
        y=b['state_dict'][key]
        if x.is_sparse:
            assert torch.equal(x.coalesce().values(),y.coalesce().values()),key
        else:assert torch.equal(x,y),key


def test_full_72h_multimodal_rollout_and_roi(tmp_path):
    path=create_dataset(tmp_path/'long',horizon_hours=72);ds=TrainingDataset(path)
    cfg=TrainConfig(epochs=1,horizon_hours=72,threads=1)
    train(path,tmp_path/'run',cfg,progress=lambda *a,**k:None)
    result=forecast(path,tmp_path/'run','sample-4',tmp_path/'forecast',horizon_hours=72)
    assert result['lead_hours']==list(range(0,73,3))
    model,_,_=load_trained(ds,tmp_path/'run');obs=ds.packed(ds.samples[-1])
    z=torch.tensor(ds.elevation,dtype=torch.float32);land=torch.tensor(ds.land,dtype=torch.float32)
    with torch.no_grad():
        a=list(model(obs,z,land,horizon_hours=3))[-1]
        b=list(model(obs,z,land,horizon_hours=3,product_mask=torch.arange(12)%2==0))[-1]
    assert torch.equal(a.profiles,b.profiles) and a.profiles.shape==(12,37,6)


def test_scene_hash_change_blocks_loading(dataset):
    ds=TrainingDataset(dataset);sample=ds.samples[0]
    ref=ds._satellite[sample.id]['electro_ir'][0]
    with (ds.root/ref['path']).open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='Изменён'):ds.packed(sample)


def test_channel_norms_ignore_heldout_pixels(dataset):
    # Training-only file provenance must contain exactly the 24 train scenes, not 40 scenes.
    ds=TrainingDataset(dataset);component=ds.norm._payload['provenance']['components'][-1]
    assert component['kind']=='unique_training_pixels' and len(component['sha256'])==24
    assert all('sample-3' not in p and 'sample-4' not in p for p in component['sha256'])


def test_unknown_sensor_and_duplicate_scene_rejected(dataset):
    data=read_json(dataset);ref=data['samples'][0]['satellite']['electro_ir'][0]
    data['samples'][0]['satellite']['electro_ir'].append(ref)
    out=dataset.with_name('duplicate.json');atomic_json(out,data)
    with pytest.raises(ValueError,match='повтор'):TrainingDataset(out)


def test_scene_only_training_input_is_allowed(dataset):
    m=read_json(dataset);sample=m['samples'][0]
    empty=dataset.parent/'empty.jsonl';empty.write_text('')
    sample['observations']=reference(dataset.parent,empty)
    out=dataset.with_name('no-stations.json');atomic_json(out,m)
    ds=TrainingDataset(out);obs=ds.packed(ds.samples[0])
    assert len(obs.cells)==0 and obs.accepted_records>0 and len(obs.scenes)==8
