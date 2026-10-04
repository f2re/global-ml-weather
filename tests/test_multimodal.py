"""Analytic unit tests. No downloaded imagery or weather-skill claim."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from global_weather.grid import build_grid
from global_weather.multimodal.frames import sensor_registry,read_scene,load_scene,project_links
from global_weather.multimodal.fixture import specs,scene_arrays
from global_weather.multimodal.networks import (ResUNet,MicrowaveEncoder,SatelliteEncoder,SourceFusion,
                                                SparseObservationEncoder,scatter_scene)

torch.set_num_threads(1)
TIME=datetime(2020,1,1,12,tzinfo=timezone.utc)

class Norm:
    """Explicit synthetic scale for operator tests only."""
    def get(self,*args):return self
    def at(self):return 260.,10.


def scene(tmp_path,name='electro_ir',*,age=1,shift=0.,modify=None):
    spec=specs()['sensors'][name];a=scene_arrays(spec,TIME,age_hours=age,shift=shift)
    if modify:modify(a)
    path=tmp_path/f'{name}-{age}-{shift}.npz';np.savez(path,**a)
    result=load_scene(path,spec,name,build_grid(0),Norm(),TIME,'synthetic','a'*64)
    return result,spec


@pytest.mark.parametrize('key,value',[('source','unknown'),('encoder','shell'),('temporal','guess'),('projection','bilinear')])
def test_registry_rejects_unknown_modes(key,value):
    r=specs();r['sensors']['electro_ir'][key]=value
    with pytest.raises(ValueError):sensor_registry(r)


def test_raw_counts_cannot_be_declared_as_physical():
    r=specs();r['sensors']['electro_ir']['channels'][0]['quantity']='raw_counts'
    with pytest.raises(ValueError):sensor_registry(r)


def test_duplicate_channel_names_are_rejected():
    r=specs();r['sensors']['electro_ir']['channels'][1]['variable']=r['sensors']['electro_ir']['channels'][0]['variable']
    with pytest.raises(ValueError):sensor_registry(r)


def test_microwave_not_a_visible_image_encoder():
    r=specs();r['sensors']['meteor_mw']['encoder']='unet'
    with pytest.raises(ValueError):sensor_registry(r)


def test_causal_mask_and_partial_scan_times(tmp_path):
    def change(a):a['observed_utc_s'][0,:]=TIME.timestamp()+3600
    # Future pixels cannot precede availability; explicitly late frame is fully excluded.
    def late(a):
        change(a);m=json.loads(str(a['metadata']));m['available_at']=(TIME+timedelta(hours=2)).isoformat();a['metadata']=np.array(json.dumps(m))
    value,_=scene(tmp_path,modify=late)
    assert value is None


def test_old_and_unavailable_scenes_do_not_enter_model(tmp_path):
    assert scene(tmp_path,age=13)[0] is None
    assert scene(tmp_path,age=0)[0] is None  # ten-minute decoding latency


def test_masked_values_and_geometry_do_not_create_nan(tmp_path):
    s,_=scene(tmp_path)
    assert torch.isfinite(s.values).all() and torch.isfinite(s.geometry).all()
    assert s.values[0,0,0]==0 and s.valid[0,0,0]==False


def test_unknown_solar_angle_is_masked_feature_for_ir(tmp_path):
    s,_=scene(tmp_path,modify=lambda a:a['solar_zenith_deg'].fill(np.nan))
    assert s.geometry.shape[0]==8 and not s.geometry[-1].any()


def test_unknown_solar_reflectance_is_not_night_zero(tmp_path):
    spec=specs()['sensors']['electro_ir']
    for ch in spec['channels']:ch.update(quantity='reflectance',units='1')
    a=scene_arrays(spec,TIME);a['values'][:]=.3;a['solar_zenith_deg'][:]=np.nan
    path=tmp_path/'reflect.npz';np.savez(path,**a)
    aa,_=read_scene(path,spec,kind='synthetic',issue=TIME)
    assert not aa['valid'].any()


@pytest.mark.parametrize('change',['unit','calibration','source','kind','nan'])
def test_metadata_and_physical_admission(tmp_path,change):
    spec=specs()['sensors']['electro_ir'];a=scene_arrays(spec,TIME);m=json.loads(str(a['metadata']))
    if change=='unit':m['channels'][0]['units']='degC'
    elif change=='calibration':m['channels'][0]['calibration']='unknown'
    elif change=='source':m['source']='another'
    elif change=='kind':m['data_kind']='real'
    else:a['values'][0,1,1]=np.nan
    a['metadata']=np.array(json.dumps(m));p=tmp_path/'bad.npz';np.savez(p,**a)
    with pytest.raises(ValueError):read_scene(p,spec,kind='synthetic',issue=TIME)


def test_unet_gradients_and_channel_identity(tmp_path):
    a,spec=scene(tmp_path);torch.manual_seed(7);m=ResUNet(2,16)
    v,h=m(a)
    other=replace(a,values=a.values.flip(0),valid=a.valid.flip(0))
    vv,_=m(other)
    assert v.shape==(16,8,8) and not torch.allclose(v,vv)
    v.square().mean().backward()
    assert m.enc0.body[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_unet_ignores_arbitrary_missing_values(tmp_path):
    a,_=scene(tmp_path);m=ResUNet(2,16)
    b=replace(a,values=torch.where(a.valid,a.values,torch.full_like(a.values,1e15)))
    assert torch.equal(m(a)[0],m(b)[0])


def test_registered_temporal_encoder_rejects_moving_pixels(tmp_path):
    a,spec=scene(tmp_path,age=4);b,_=scene(tmp_path,age=1,shift=.1);m=SatelliteEncoder(spec,16)
    with pytest.raises(ValueError):m([a,b],12)


def test_event_encoder_accepts_real_swath_sequence(tmp_path):
    a,spec=scene(tmp_path,'meteor_ir',age=4);b,_=scene(tmp_path,'meteor_ir',age=1,shift=.1)
    m=SatelliteEncoder(spec,16);x,mask=m([a,b],12)
    assert x.shape==(12,16) and mask.any() and torch.isfinite(x).all()


def test_registered_memory_receives_gradients(tmp_path):
    a,spec=scene(tmp_path,age=4);b,_=scene(tmp_path,age=1);m=SatelliteEncoder(spec,16)
    x,mask=m([a,b],12);x.square().sum().backward()
    assert m.image.memory.gates.weight.grad.abs().sum()>0


def test_microwave_encoder_uses_joint_channels(tmp_path):
    a,spec=scene(tmp_path,'meteor_mw');m=MicrowaveEncoder(2,16)
    v,_=m(a);vv,_=m(replace(a,values=a.values.flip(0)))
    assert not torch.allclose(v,vv)


def test_all_missing_source_fusion_is_identity():
    m=SourceFusion(16,4);z=torch.randn(12,8,16);features=torch.randn(12,4,16)
    assert torch.equal(m(z,features,torch.zeros(12,4,dtype=torch.bool)),z)


def test_fusion_has_no_nan_and_missing_sources_have_no_effect():
    m=SourceFusion(16,4);z=torch.randn(12,8,16);features=torch.randn(12,4,16)
    mask=torch.zeros(12,4,dtype=torch.bool);mask[0,0]=True
    a=m(z,features,mask);features[:,1:]=1e6;b=m(z,features,mask)
    assert torch.isfinite(a).all() and torch.allclose(a,b)
    assert torch.equal(a[1:],z[1:])


def test_projection_preserves_constant_features(tmp_path):
    s,_=scene(tmp_path);value,mask=scatter_scene(torch.ones(16,8,8),s,12)
    assert torch.allclose(value[mask],torch.ones_like(value[mask]))


def test_footprint_larger_than_cell_rejected():
    a=scene_arrays(specs()['sensors']['meteor_mw'],TIME);a['footprint_major_km'][:]=4000
    with pytest.raises(ValueError):project_links(a,build_grid(3),'bounded-centre')


def test_gaussian_positive_weights_preserve_constant():
    a=scene_arrays(specs()['sensors']['meteor_mw'],TIME)
    a['valid'][:]=False;a['valid'][:,4,4]=True
    a['footprint_major_km'][:]=3000;a['footprint_minor_km'][:]=1500
    c,p,w=project_links(a,build_grid(3),'gaussian')
    assert len(c)>=3 and (w>0).all() and w.sum()==pytest.approx(1.)


def test_underresolved_gaussian_rejected():
    a=scene_arrays(specs()['sensors']['meteor_mw'],TIME)
    with pytest.raises(ValueError):project_links(a,build_grid(0),'gaussian')


def sparse(grid):
    xyz=torch.tensor(grid.xyz[[0,1]],dtype=torch.float32)
    f=torch.zeros(2,12);f[:,0]=torch.tensor([1.,2.]);f[:,8:11]=xyz
    return SimpleNamespace(features=f,cells=torch.tensor([0,1]),levels=torch.tensor([37,6]),
                slots=torch.tensor([11,10]),sources=torch.tensor([0,1]),variables=torch.tensor([0,1]),weights=torch.ones(2))


def test_sparse_attention_gradients_and_empty_source():
    g=build_grid(0);o=sparse(g);m=SparseObservationEncoder(2,16,g);z=torch.randn(12,38,16)
    a=m(o,z,torch.randn(38,16));a.square().mean().backward()
    assert m.value[0].weight.grad.abs().sum()>0 and torch.isfinite(a).all()
    empty=SimpleNamespace(cells=torch.zeros(0,dtype=torch.long))
    assert torch.equal(m(empty,z,None),z)


def test_sparse_permutation_invariance():
    g=build_grid(0);o=sparse(g);m=SparseObservationEncoder(2,16,g);z=torch.randn(12,38,16)
    other=SimpleNamespace(**{k:v.flip(0) for k,v in vars(o).items()})
    assert torch.allclose(m(o,z,None),m(other,z,None),atol=1e-6)
