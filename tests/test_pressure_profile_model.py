"""Synthetic structural verification, not evidence of weather accuracy."""
from datetime import timedelta
import numpy as np
import pytest
import torch
from global_weather.grid import build_pyramid
from global_weather.profile_normalization import PressureNormalization,SCHEMA,PERIOD
from global_weather.profile_model_v2 import PressureProfileModel
from global_weather.profile_training_v2 import objective
from global_weather.profile_training import VARIABLES,START
from global_weather.vertical import PRESSURE_HPA,PROFILE_UNITS


def norms():
    means=np.zeros((37,5)); std=np.ones((37,5)); means[:,0]=np.linspace(290,210,37)
    means[:,1]=.1; std[:,1]=.05; means[:,4]=np.linspace(0,300000,37);std[:,4]=10000
    return PressureNormalization({'schema':SCHEMA,'split':'train','period':PERIOD,'variables':list(VARIABLES),
        'units':list(PROFILE_UNITS[:5]),'pressure_pa':(np.array(PRESSURE_HPA)*100).tolist(),
        'transforms':['identity','log1p(q/q_scale)','identity','identity','identity'],'q_scale':.003,
        'mean':means.tolist(),'std':std.tolist(),'support':np.ones((37,5),dtype=bool).tolist(),
        'count':np.full((37,5),100).tolist()})


def rows(when):
    return [dict(variable=name,value=[270,.001,4.,-3.,30000][i],units=PROFILE_UNITS[i],latitude=10.,longitude=20.,
                 pressure_pa=70000.,observed_at=when.isoformat(),available_at=when.isoformat(),profile_id='fixture')
            for i,name in enumerate(VARIABLES)]


def test_q_transform_roundtrip_zero_dry_wet_and_supported_gradients():
    n=norms()
    for q in (0.,1e-9,1e-6,.001,.02):
        assert float(n.inverse('specific_humidity',n.normalize('specific_humidity',q,70000),70000))==pytest.approx(q,abs=1e-16)
    transformed=torch.tensor([0.,1e-9,.01],requires_grad=True)
    physical=.003*torch.expm1(transformed).clamp_min(0.)
    assert physical[0]==0
    physical.sum().backward();assert torch.isfinite(transformed.grad).all() and (transformed.grad>0).all()


def test_pressure_basis_survives_dynamics_and_causal_masks():
    torch.manual_seed(29);model=PressureProfileModel(build_pyramid(0)[0],norms(),8)
    issue=START+timedelta(days=3)
    future=rows(issue+timedelta(hours=1))
    with torch.no_grad():frames=model([],issue); other=model(future,issue)
    assert len(frames)==25 and frames[-1].lead_hours==72
    assert frames[-1].profiles.shape==(12,37,6)
    assert torch.isnan(frames[-1].profiles[...,5]).all()
    for a,b in zip(frames,other): assert torch.equal(a.profiles[...,:5],b.profiles[...,:5])
    assert float((frames[-1].profiles[:,0,0]-frames[-1].profiles[:,-1,0]).mean())>70
    assert frames[-1].wind_basis=='local_enu_vector'


def test_all_observation_and_dynamic_branches_have_finite_nonzero_gradients():
    torch.manual_seed(29);model=PressureProfileModel(build_pyramid(0)[0],norms(),8)
    issue=START+timedelta(days=3)
    frames=model(rows(issue-timedelta(hours=6)),issue)
    loss,counts=objective(model,frames,rows(issue+timedelta(hours=12)))
    assert counts==[1]*5; loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None,name
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name


@pytest.mark.parametrize('normalization_mode',['r6','r7'])
def test_fresh_training_exact_resume_norm_drift_and_final_diagnostic(tmp_path,monkeypatch,normalization_mode):
    import json
    import hashlib
    from global_weather import profile_training_v2 as training
    from global_weather.profile_training import prepare,TRAIN_END,VAL_END
    if normalization_mode=='r6':
        from global_weather.profile_normalization import fit
    else:
        from global_weather.profile_graphcast_normalization import create as fit
    from global_weather.profile_verification import _exact_tensor_equal
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    data=[]
    for start,split in ((START,'train'),(TRAIN_END,'validation'),(VAL_END,'test')):
        for day in (2,3):
            when=start+timedelta(days=day)
            for i,row in enumerate(rows(when)):
                row.update(value=row['value']+(day-2)*[2,.0002,2,2,100][i],source='radiosonde',provider='NOAA_IGRA2',valid=True,
                     observation_id=f'{when}/{i}',profile_id=str(when),group_split=split,revision=0,provider_qc={'software_fixture':True},
                     archive_sha256='0'*64,format_sha256='1'*64,time_basis='reported_level_time')
                data.append(row)
    source=tmp_path/'observations.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in data))
    (tmp_path/'manifest.json').write_text(json.dumps({'schema':'igra-observation-archive-1','provider':'NOAA_IGRA2','data_kind':'real',
          'observations_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
    dataset=tmp_path/'dataset';prepare(source,dataset);norm_path=tmp_path/'norms.json';fit(dataset,norm_path)
    config=dict(mesh_level=0,hidden=8,epochs=2,patience=3,threads=1,max_train_issues=1,max_validation_issues=1,max_test_issues=1,max_records_per_window=50)
    training.train(dataset,norm_path,tmp_path/'whole',config)
    original_score=training.score;calls=0
    def interrupted(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('synthetic interruption before publishing second epoch')
        return original_score(*args)
    monkeypatch.setattr(training,'score',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):training.train(dataset,norm_path,tmp_path/'resumed',config)
    monkeypatch.setattr(training,'score',original_score);training.train(dataset,norm_path,tmp_path/'resumed',config)
    a=torch.load(tmp_path/'whole/epoch-0002/state.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/epoch-0002/state.pt',weights_only=True)
    assert all(_exact_tensor_equal(a['model'][name],b['model'][name]) for name in a['model'])
    training.evaluate(dataset,tmp_path/'whole',tmp_path/'test.json')
    report=json.loads((tmp_path/'test.json').read_text());assert report['scientific_acceptance'] is False
    assert all('analysis_persistence_rmse' in r for r in report['metrics'])
    other_norm_path=tmp_path/'other-norms.json'
    if normalization_mode=='r6':
        from global_weather.profile_graphcast_normalization import create as other_norms
    else:
        from global_weather.profile_normalization import fit as other_norms
    other_norms(dataset,other_norm_path)
    with pytest.raises(ValueError,match='resume'):
        training.train(dataset,other_norm_path,tmp_path/'whole',config)
    norm_path.write_text(norm_path.read_text()+'\n')
    with pytest.raises(ValueError,match='resume'):training.train(dataset,norm_path,tmp_path/'resumed',config)


@pytest.mark.parametrize('hours,bin_index',[(.001,0),(3,0),(3.0001,1),(24,7),(48,15),(72,23)])
def test_diagnostic_lead_bin_boundaries(hours,bin_index):
    from global_weather.profile_training_v2 import lead_bin
    assert lead_bin(START+timedelta(hours=hours),START)==bin_index


def test_s1_uses_actual_origin_and_excludes_entire_held_profile(monkeypatch):
    from global_weather import profile_training_v2 as training
    when=START+timedelta(days=3,hours=1); targets=rows(when)
    targets.append(dict(targets[0],observed_at=(when+timedelta(minutes=5)).isoformat()))
    other=dict(targets[0],profile_id='other',observed_at=(when-timedelta(hours=1)).isoformat())
    class Dataset:
        def records(self,start,end,*,issue,split):
            assert start==when-timedelta(hours=12) and end==issue==when and split=='train'
            return targets[:5]+[other]
    class Model:
        normalization=norms()
        def __call__(self,inputs,issue):
            assert issue==when
            assert inputs==[other]
            return ['initial-frame']
    def held_loss(model,frames,held):
        assert frames==['initial-frame'] and held==targets[:5]
        return torch.tensor(1.),[1]*5
    monkeypatch.setattr(training,'objective',held_loss)
    assert float(training.reconstruction(Model(),Dataset(),targets,50,'train'))==1.


def test_unsupported_humidity_is_masked_before_exponential_backward():
    payload=norms().payload
    payload['support'][0][1]=False;payload['mean'][0][1]=None;payload['std'][0][1]=None;payload['count'][0][1]=0
    model=PressureProfileModel(build_pyramid(0)[0],PressureNormalization(payload),8)
    state=torch.zeros(12,37,8,requires_grad=True)
    with torch.no_grad():model.head.weight.zero_();model.head.bias[1]=1000.
    frame=model._decode(state,START,0)
    valid=frame.profile_variable_mask[...,1]
    assert torch.isnan(frame.profiles[:,0,1]).all()
    loss=frame.profiles[...,1][valid].sum()
    loss.backward()
    assert torch.isfinite(model.head.bias.grad).all()
    assert torch.isfinite(state.grad).all()
