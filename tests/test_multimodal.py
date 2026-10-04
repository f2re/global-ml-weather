"""Аналитическая проверка структуры; не испытание качества реальной погоды."""
from dataclasses import replace
from datetime import timedelta
import json
import numpy as np
import pytest
import torch
from global_weather.grid import build_pyramid
from global_weather.multimodal.fixture import fixture
from global_weather.multimodal.contracts import Channel,Sensor,fingerprint
from global_weather.multimodal.networks import ImagerEncoder,MicrowaveEncoder,ResUNet
from global_weather.multimodal.model import MultimodalWeatherModel
from global_weather.multimodal.fusion import SourceFusion,project_sequence
from global_weather.multimodal.io import save_sequence,load_sequence,read_arrays,sha256,reference,resolve,save_sensors
from global_weather.multimodal.acquisition import plan,execute

torch.set_num_threads(1)

@pytest.fixture
def case():
    g=build_pyramid(0);s,o=fixture(g)
    torch.manual_seed(11)
    m=MultimodalWeatherModel(g,o.vocabulary,observation_schema=o.schema_fingerprint,hidden=16,latent_slots=4,
        sensors=s,sensor_signature=o.sensor_signature,base_channels=8,radius_km=500.,neighbors=8,allow_unscaled_synthetic=True)
    return g,s,o,m


def test_registered_branches(case):
    g,s,o,m=case
    assert len(m.satellite_bank.encoders)==4
    assert isinstance(m.satellite_bank.encoders[s[0].id],ImagerEncoder)
    assert isinstance(m.satellite_bank.encoders[s[3].id],MicrowaveEncoder)


def test_all_networks_receive_gradients(case):
    g,s,o,m=case;z=torch.zeros(g[0].n_cells)
    frame=list(m(o,z,z,horizon_hours=6))[-1]
    target=torch.linspace(260.,280.,g[0].n_cells)[:,None]
    loss=((frame.profiles[...,0]-target)/30).square().mean()+((frame.surface[:,0]-280)/20).square().mean()
    loss.backward()
    modules={'sparse':m.encoder,'fusion':m.source_fusion,'compression':m.compress,
             'dynamics':m.processor,'decoder':m.expand,'profile_head':m.profile_head,'surface_head':m.surface_head}
    modules.update(dict(m.satellite_bank.encoders.items()))
    for name,mod in modules.items():
        grads=[p.grad for p in mod.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads),name
        assert sum(float(v.abs().sum()) for v in grads)>0,name


def test_rollout_72_hours_and_independent_product_mask(case):
    g,s,o,m=case;z=torch.zeros(12);mask=torch.arange(12)%2==0
    with torch.no_grad():
        a=list(m(o,z,z,horizon_hours=72));b=list(m(o,z,z,horizon_hours=72,product_mask=mask))
    assert len(a)==25 and a[-1].profiles.shape==(12,37,6) and a[-1].surface.shape==(12,8)
    assert torch.equal(a[-1].profiles,b[-1].profiles) and not b[-1].profile_mask[~mask].any()


def test_own_background_does_not_reassimilate_frames(case):
    g,s,o,m=case;z=torch.zeros(12)
    with torch.no_grad():a=m.analysis_state(o,z,z);b=m.analysis_state(o,z,z,background=a)
    assert torch.equal(a.latent,b.latent) and a.evidence==b.evidence


def test_new_available_frame_changes_background(case):
    g,s,o,m=case;z=torch.zeros(12)
    older=replace(o,sequences=tuple(replace(q,values=q.values[:2],valid=q.valid[:2],observed_unix=q.observed_unix[:2],
        available_unix=q.available_unix[:2],view_zenith_deg=q.view_zenith_deg[:2],solar_zenith_deg=q.solar_zenith_deg[:2],
        footprint_km=q.footprint_km[:2],frame_ids=q.frame_ids[:2]) for q in o.sequences))
    with torch.no_grad():a=m.analysis_state(older,z,z);b=m.analysis_state(o,z,z,background=a)
    assert not torch.equal(a.latent,b.latent)


def test_future_pixels_cannot_change_analysis(case):
    g,s,o,m=case;z=torch.zeros(12);changed=[]
    for q in o.sequences:
        ready=q.available_unix.clone();ready[-1]=o.issue_time.timestamp()+10
        values=q.values.clone();values[-1]+=1000
        changed.append(replace(q,values=values,available_unix=ready))
    future=replace(o,sequences=tuple(changed));ordinary=replace(o,sequences=tuple(replace(q,available_unix=f.available_unix) for q,f in zip(o.sequences,changed)))
    with torch.no_grad():a=m.analyse(future,z,z);b=m.analyse(ordinary,z,z)
    assert torch.equal(a,b)


def test_masked_pixel_values_have_zero_influence_and_gradients(case):
    _,s,o,_=case;q=o.sequences[0];mask=q.valid.clone();mask[:,:,0,0]=False
    x=q.values.clone().requires_grad_();y=q.values.clone();y[:,:,0,0]=float('nan')
    module=ImagerEncoder(3,16,8)
    a,_,_=module(replace(q,values=x,valid=mask),s[0],o.issue_time.timestamp())
    b,_,_=module(replace(q,values=y,valid=mask),s[0],o.issue_time.timestamp())
    assert torch.equal(a,b)
    a.square().mean().backward();assert torch.isfinite(x.grad).all() and not x.grad[:,:,0,0].any()


def test_all_missing_source_is_exactly_zero(case):
    _,s,o,m=case;q=o.sequences[0];q=replace(q,valid=torch.zeros_like(q.valid),values=torch.full_like(q.values,float('nan')))
    field,support,age=m.satellite_bank.encoders[s[0].id](q,s[0],o.issue_time.timestamp())
    assert not support.any() and not field.any() and torch.isinf(age).all()


def test_missing_source_does_not_change_fusion():
    torch.manual_seed(2);mod=SourceFusion(16);x=torch.randn(12,4,16);v=torch.randn(12,16);mask=torch.ones(12,dtype=torch.bool)
    a=mod(x,[v],[mask]);b=mod(x,[v,torch.zeros_like(v)],[mask,~mask])
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.equal(x,mod(x,[v],[~mask]))


def test_channel_permutation_changes_representation(case):
    _,s,o,m=case;q=o.sequences[0];encoder=m.satellite_bank.encoders[s[0].id]
    with torch.no_grad():
        a=encoder(q,s[0],o.issue_time.timestamp())[0]
        b=encoder(replace(q,values=q.values.flip(1)),s[0],o.issue_time.timestamp())[0]
    assert not torch.allclose(a,b,atol=1e-6)


def test_ir_accepts_unknown_solar_angle_but_reflectance_is_masked(case):
    _,s,o,m=case;q=o.sequences[0]
    q=replace(q,solar_zenith_deg=torch.full_like(q.solar_zenith_deg,float('nan')))
    assert q.validate(s[0]);assert q.normalized(s[0])[1].all()
    channels=tuple(Channel(c.id,'reflectance','1',.4,.1,c.calibration_id) for c in s[0].channels)
    ss=replace(s[0],channels=channels);q=replace(q,sensor_signature=ss.measurement_signature)
    assert not q.normalized(ss)[1].any()


@pytest.mark.parametrize('field,value',[('kind','unknown'),('normalization_sha256','bad'),('data_kind','unknown'),('fit_end','unknown')])
def test_invalid_sensor_rejected(case,field,value):
    _,s,_,_=case
    with pytest.raises(ValueError):replace(s[0],**{field:value})


@pytest.mark.parametrize('mode',['units','zero_sigma','unknown_quantity','calibration'])
def test_invalid_physical_channel(mode):
    kwargs={'id':'9','quantity':'brightness_temperature','units':'K','mean':250.,'std':20.,'calibration_id':'verified'}
    if mode=='units':kwargs['units']='counts'
    if mode=='zero_sigma':kwargs['std']=0
    if mode=='unknown_quantity':kwargs['quantity']='raw_counts'
    if mode=='calibration':kwargs['calibration_id']=''
    with pytest.raises(ValueError):Channel(**kwargs)


def test_microwave_cannot_use_imager_point_operator(case):
    g,s,o,m=case;q=o.sequences[-1]
    with pytest.raises(ValueError):replace(q,link_cell=None).validate(s[-1])
    with pytest.raises(ValueError):replace(q,link_weight=q.link_weight*2).validate(s[-1])
    with pytest.raises(ValueError):replace(q,link_grid_fingerprint='a'*64).validate(s[-1],grid_fingerprint=g[0].fingerprint)


def test_microwave_support_reaches_multiple_cells(case):
    g,s,o,m=case;q=o.sequences[-1]
    val,sup,age=m.satellite_bank.encoders[s[-1].id](q,s[-1],o.issue_time.timestamp())
    f,p,t=project_sequence(val,sup,age,q,s[-1],g[0])
    assert p[:2].all() and p.sum()==2 and torch.isfinite(f).all()


def test_sequence_roundtrip_and_hash(case,tmp_path):
    _,s,o,m=case;q=o.sequences[0];p=tmp_path/'seq.npz'
    save_sequence(p,q);loaded=load_sequence(p,s[0]);assert torch.equal(q.values,loaded.values)
    assert loaded.source_sha256==sha256(p)
    with pytest.raises(FileExistsError):save_sequence(p,q)


def test_corrupted_or_foreign_sequence_rejected(case,tmp_path):
    _,s,o,_=case;p=tmp_path/'seq.npz';save_sequence(p,o.sequences[0])
    with pytest.raises(ValueError):load_sequence(p,s[1])
    ref=reference(tmp_path,p);p.write_bytes(b'changed')
    with pytest.raises(ValueError):resolve(tmp_path,ref)


def test_numpy_objects_are_rejected(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,x=np.array([{}],dtype=object))
    with pytest.raises(ValueError):read_arrays(p)


def test_download_plan_covers_input_and_targets():
    p=plan('2020-01-01T00:00:00Z','2020-01-01T12:00:00Z',stations=['26063099999'])
    assert p['observation_start']=='2019-12-31T12:00:00+00:00'
    assert p['target_end']=='2020-01-04T12:00:00+00:00'
    assert {x['year'] for x in p['jobs'] if x['kind']=='noaa'}=={2019,2020}
    assert len([x for x in p['jobs'] if x['kind']=='era5'])==10


def test_network_is_not_implicit(tmp_path):
    with pytest.raises(ValueError,match='network'):execute(tmp_path/'missing.json',tmp_path/'out')


def test_sequence_to_preserves_all_tensors(case):
    _,s,o,_=case;b=o.to('cpu')
    assert isinstance(b.sequences,tuple) and len(b.sequences)==4
    assert torch.equal(b.sequences[0].values,o.sequences[0].values)


def test_snapshot_schema_includes_sensor_norms(case):
    _,_,_,m=case;schema=m.get_extra_state()
    assert schema['architecture']=='multimodal-adaptive-v1' and len(schema['sensor_contracts'])==4
    with pytest.raises(ValueError):m.set_extra_state(dict(schema,sensor_signature='changed'))
